from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

ALIASES = {
    "deepseek-chat": "deepseek-v4-flash",
    "deepseek-reasoner": "deepseek-v4-flash-think",
}


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    base_url: str
    hint: str
    needs_key: bool
    local: bool


PROVIDERS: tuple[Provider, ...] = (
    Provider(
        id="deepseek",
        label="DeepSeek (cloud)",
        base_url="https://api.deepseek.com/v1",
        hint="Requires LLM_API_KEY; chat content leaves this device",
        needs_key=True,
        local=False,
    ),
    Provider(
        id="ollama",
        label="Ollama (local)",
        base_url="http://127.0.0.1:11434/v1",
        hint="Local model, data never leaves this device; use the model name from Ollama",
        needs_key=False,
        local=True,
    ),
    Provider(
        id="lmstudio",
        label="LM Studio (local)",
        base_url="http://127.0.0.1:1234/v1",
        hint="Local model; start the local server in LM Studio first",
        needs_key=False,
        local=True,
    ),
    Provider(
        id="custom",
        label="Custom OpenAI-compatible endpoint",
        base_url="",
        hint="Any service compatible with /chat/completions, including one on your LAN",
        needs_key=False,
        local=True,
    ),
)

PROVIDER_BY_ID = {p.id: p for p in PROVIDERS}
DEFAULT_PROVIDER_ID = "deepseek"


@dataclass(frozen=True)
class ModelChoice:
    id: str
    api_model: str
    label: str
    hint: str
    thinking: bool


MODELS: tuple[ModelChoice, ...] = (
    ModelChoice("deepseek-v4-flash", "deepseek-v4-flash", "V4 Flash", "Fast · everyday Q&A", False),
    ModelChoice(
        "deepseek-v4-flash-think",
        "deepseek-v4-flash",
        "V4 Flash Thinking",
        "Slower, steadier reasoning",
        True,
    ),
    ModelChoice("deepseek-v4-pro", "deepseek-v4-pro", "V4 Pro", "Stronger conversation", False),
    ModelChoice(
        "deepseek-v4-pro-think",
        "deepseek-v4-pro",
        "V4 Pro Thinking",
        "Strongest reasoning",
        True,
    ),
)

_MODEL_BY_ID = {m.id: m for m in MODELS}


@dataclass(frozen=True)
class EffectiveLLM:
    """The endpoint actually used for a request, after prefs + env are merged."""

    provider: Provider
    base_url: str
    api_model: str
    api_key: str
    thinking: bool
    choice_id: str

    @property
    def configured(self) -> bool:
        if not self.base_url:
            return False
        if self.provider.needs_key:
            return bool(self.api_key)
        return True

    @property
    def is_local(self) -> bool:
        return self.provider.local

    @property
    def host(self) -> str:
        from urllib.parse import urlparse

        return urlparse(self.base_url).netloc or self.base_url


def prefs_path() -> Path:
    return Path(get_settings().upload_dir).resolve().parent / "prefs.json"


def load_prefs() -> dict:
    path = prefs_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_prefs(updates: dict) -> dict:
    data = load_prefs()
    for key, value in updates.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path = prefs_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def resolve_provider(raw: str | None = None) -> Provider:
    prefs = load_prefs()
    key = (raw or str(prefs.get("llm_provider") or DEFAULT_PROVIDER_ID)).strip()
    return PROVIDER_BY_ID.get(key, PROVIDER_BY_ID[DEFAULT_PROVIDER_ID])


def resolve_model(raw: str | None = None) -> ModelChoice:
    """Resolve the model for the DeepSeek provider (fixed model list)."""
    if not raw:
        raw = str(load_prefs().get("llm_model") or get_settings().llm_model or "")
    key = ALIASES.get(raw, raw)
    return _MODEL_BY_ID.get(key, MODELS[0])


def resolve_effective_llm() -> EffectiveLLM:
    settings = get_settings()
    prefs = load_prefs()
    provider = resolve_provider()

    base_url = str(prefs.get("llm_base_url") or "").strip() or provider.base_url
    if provider.id == "deepseek" and not str(prefs.get("llm_base_url") or "").strip():
        base_url = settings.llm_base_url or provider.base_url

    if provider.id == "deepseek":
        choice = resolve_model()
        return EffectiveLLM(
            provider=provider,
            base_url=base_url,
            api_model=choice.api_model,
            api_key=settings.llm_api_key,
            thinking=choice.thinking,
            choice_id=choice.id,
        )

    model = str(prefs.get("llm_model") or "").strip()
    return EffectiveLLM(
        provider=provider,
        base_url=base_url,
        api_model=model,
        # A key is optional locally; if the user set one we still pass it.
        api_key=settings.llm_api_key,
        thinking=False,
        choice_id=model or provider.id,
    )
