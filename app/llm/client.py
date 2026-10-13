from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Iterator
from dataclasses import replace
from typing import Any

import httpx

from app.config import Settings, get_settings
from app.llm.models import EffectiveLLM, ModelChoice, resolve_effective_llm
from app.services.outbound import record_outbound

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(
        self,
        settings: Settings | None = None,
        choice: ModelChoice | None = None,
        effective: EffectiveLLM | None = None,
    ):
        self.settings = settings or get_settings()
        self.effective = effective or resolve_effective_llm()
        if effective is None and settings is not None:
            # Respect an explicitly constructed Settings object over the env cache.
            updates: dict[str, Any] = {
                "api_key": settings.llm_api_key or self.effective.api_key
            }
            if self.effective.provider.id == "deepseek":
                updates["base_url"] = settings.llm_base_url or self.effective.base_url
            self.effective = replace(self.effective, **updates)
        if choice is not None:
            # Explicit choice (mostly tests) wins over prefs.
            self.effective = EffectiveLLM(
                provider=self.effective.provider,
                base_url=self.effective.base_url,
                api_model=choice.api_model,
                api_key=self.effective.api_key,
                thinking=choice.thinking,
                choice_id=choice.id,
            )
        self.choice = choice or ModelChoice(
            id=self.effective.choice_id,
            api_model=self.effective.api_model,
            label=self.effective.api_model,
            hint="",
            thinking=self.effective.thinking,
        )

    @property
    def configured(self) -> bool:
        return self.effective.configured

    @property
    def provider_id(self) -> str:
        return self.effective.provider.id

    @property
    def model(self) -> str:
        return self.effective.api_model

    def test_connection(self, *, timeout: float = 8.0) -> tuple[bool, str]:
        """Probe the OpenAI-compatible /models endpoint. Offline-safe."""
        if not self.effective.base_url:
            return False, "No endpoint URL configured yet"
        if self.effective.provider.needs_key and not self.effective.api_key:
            return False, "Missing API key (set LLM_API_KEY in .env)"
        url = self.effective.base_url.rstrip("/") + "/models"
        try:
            with httpx.Client(timeout=timeout) as client:
                resp = client.get(url, headers=self._headers())
        except httpx.HTTPError as exc:
            return False, f"Cannot reach {self.effective.host}: {type(exc).__name__}"
        if resp.status_code >= 400:
            return False, f"{self.effective.host} returned HTTP {resp.status_code}"
        models = []
        try:
            data = resp.json()
            models = [str(item.get("id")) for item in (data.get("data") or []) if item.get("id")]
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        if models and self.effective.api_model and self.effective.api_model not in models:
            return True, (
                f"Connected, but the endpoint does not list {self.effective.api_model}; "
                f"examples available: {', '.join(models[:3])}"
            )
        return True, (
            f"Connected ({len(models)} models available)" if models else "Connected"
        )

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.1,
        max_tokens: int = 4000,
        timeout: float = 90.0,
        purpose: str = "chat",
        redacted: bool = False,
    ) -> dict[str, Any]:
        raw = self.chat_text(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            purpose=purpose,
            redacted=redacted,
        )
        return _extract_json(raw)

    def chat_text(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: float = 90.0,
        purpose: str = "chat",
        redacted: bool = False,
    ) -> str:
        url, headers, payload, timeout = self._request(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        char_count = len(system) + len(user)
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=timeout) as client:
                    resp = client.post(url, headers=headers, json=payload)
                if resp.status_code >= 400:
                    # Never log request body (may contain chat content)
                    logger.warning("LLM HTTP %s (attempt %s)", resp.status_code, attempt + 1)
                    last_error = LLMError(f"LLM request failed: HTTP {resp.status_code}")
                    continue
                data = resp.json()
                msg = data["choices"][0]["message"]
                content = (msg.get("content") or "").strip()
                if not content:
                    content = (msg.get("reasoning_content") or "").strip()
                self._audit(purpose, char_count, redacted, ok=True, status_code=resp.status_code)
                return content
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
                last_error = LLMError(f"LLM call raised: {exc}")
                logger.warning("LLM call failed attempt %s: %s", attempt + 1, type(exc).__name__)
        self._audit(
            purpose,
            char_count,
            redacted,
            ok=False,
            error=str(last_error or "LLM call failed"),
        )
        raise last_error or LLMError("LLM call failed")

    def chat_stream(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 2000,
        timeout: float = 90.0,
        purpose: str = "chat",
        redacted: bool = False,
    ):
        """Yield (kind, text) where kind is 'thinking' or 'token'."""
        url, headers, payload, timeout = self._request(
            system,
            user,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            stream=True,
        )
        char_count = len(system) + len(user)
        try:
            with httpx.Client(
                timeout=httpx.Timeout(timeout, connect=10.0, read=timeout)
            ) as client:
                with client.stream("POST", url, headers=headers, json=payload) as resp:
                    if resp.status_code >= 400:
                        logger.warning("LLM HTTP %s (stream)", resp.status_code)
                        self._audit(
                            purpose,
                            char_count,
                            redacted,
                            ok=False,
                            status_code=resp.status_code,
                            error=f"HTTP {resp.status_code}",
                        )
                        raise LLMError(f"LLM request failed: HTTP {resp.status_code}")
                    self._audit(
                        purpose, char_count, redacted, ok=True, status_code=resp.status_code
                    )
                    yield from iter_sse_deltas(resp.iter_lines())
        except httpx.HTTPError as exc:
            self._audit(purpose, char_count, redacted, ok=False, error=type(exc).__name__)
            raise LLMError(f"LLM call raised: {exc}") from exc

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.effective.api_key:
            headers["Authorization"] = f"Bearer {self.effective.api_key}"
        return headers

    def _audit(
        self,
        purpose: str,
        char_count: int,
        redacted: bool,
        *,
        ok: bool,
        status_code: int | None = None,
        error: str | None = None,
    ) -> None:
        record_outbound(
            provider=self.effective.provider.id,
            host=self.effective.host,
            model=self.effective.api_model,
            purpose=purpose,
            char_count=char_count,
            redacted=redacted,
            ok=ok,
            status_code=status_code,
            error=error,
        )

    def _request(
        self,
        system: str,
        user: str,
        *,
        temperature: float,
        max_tokens: int,
        timeout: float,
        stream: bool = False,
    ) -> tuple[str, dict[str, str], dict[str, Any], float]:
        if not self.configured:
            raise LLMError("No usable model endpoint is configured; cannot call the model")
        thinking = self.effective.thinking
        payload: dict[str, Any] = {
            "model": self.effective.api_model,
            "max_tokens": max(max_tokens, 4000) if thinking else max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if thinking:
            # Only DeepSeek's reasoning models take this field; local servers
            # reject unknown parameters.
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = "high"
            timeout = max(timeout, 180.0)
        else:
            payload["temperature"] = temperature
        if stream:
            payload["stream"] = True
        url = self.effective.base_url.rstrip("/") + "/chat/completions"
        return url, self._headers(), payload, timeout


def _delta_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
        return "".join(parts)
    return ""


def deltas_from_chunk(chunk: dict[str, Any]) -> list[tuple[str, str]]:
    choices = chunk.get("choices") or []
    if not choices:
        return []
    delta = (choices[0] or {}).get("delta") or {}
    out: list[tuple[str, str]] = []
    thinking = _delta_text(delta.get("reasoning_content") or delta.get("reasoning"))
    if thinking:
        out.append(("thinking", thinking))
    content = _delta_text(delta.get("content"))
    if content:
        out.append(("token", content))
    return out


def iter_sse_deltas(lines: Iterable[str | bytes]) -> Iterator[tuple[str, str]]:
    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        line = (line or "").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        yield from deltas_from_chunk(chunk)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"items": data}
        raise LLMError("The model returned JSON that is not an object")
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise LLMError("Could not parse JSON from the model output") from None
        data = json.loads(match.group(0))
        if not isinstance(data, dict):
            raise LLMError("The model returned JSON that is not an object")
        return data
