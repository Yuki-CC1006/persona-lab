"""Model endpoint productisation: providers, connectivity, outbound audit."""

from __future__ import annotations

import re

import httpx
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import OutboundEvent
from app.llm.client import LLMClient
from app.llm.models import (
    PROVIDERS,
    resolve_effective_llm,
    resolve_model,
    save_prefs,
)
from app.services.outbound import list_outbound, outbound_summary, record_outbound


def test_local_provider_is_configured_without_api_key(db: Session):
    save_prefs({"llm_provider": "ollama", "llm_model": "qwen3:8b"})
    effective = resolve_effective_llm()
    assert effective.provider.id == "ollama"
    assert effective.is_local is True
    assert effective.configured is True
    assert LLMClient().configured is True
    assert LLMClient().model == "qwen3:8b"


def test_deepseek_requires_api_key(db: Session):
    save_prefs({"llm_provider": "deepseek"})
    effective = resolve_effective_llm()
    assert effective.provider.needs_key is True
    assert effective.configured is False  # tests run with LLM_API_KEY=""


def test_custom_provider_uses_pasted_endpoint(db: Session):
    save_prefs(
        {
            "llm_provider": "custom",
            "llm_base_url": "http://192.168.1.9:8000/v1",
            "llm_model": "my-model",
        }
    )
    effective = resolve_effective_llm()
    assert effective.base_url == "http://192.168.1.9:8000/v1"
    assert effective.host == "192.168.1.9:8000"
    assert effective.api_model == "my-model"


def test_reasoning_payload_only_sent_for_deepseek_thinking(db: Session):
    save_prefs({"llm_provider": "deepseek", "llm_model": "deepseek-v4-flash-think"})
    prev = _settings_with_key("k")
    client = LLMClient(settings=prev)
    _url, _headers, payload, _timeout = client._request(
        "s", "u", temperature=0.2, max_tokens=200, timeout=10
    )
    assert payload["thinking"] == {"type": "enabled"}
    assert resolve_model().thinking is True

    save_prefs({"llm_provider": "ollama", "llm_model": "qwen3:8b"})
    local = LLMClient()
    _url, headers, payload, _timeout = local._request(
        "s", "u", temperature=0.2, max_tokens=200, timeout=10
    )
    # Local servers reject unknown params, and there is no key to send.
    assert "thinking" not in payload
    assert payload["model"] == "qwen3:8b"
    assert "Authorization" not in headers
    assert payload["temperature"] == 0.2


def test_connection_test_reports_unreachable_endpoint(db: Session):
    save_prefs({"llm_provider": "custom", "llm_base_url": "http://127.0.0.1:9/v1"})
    ok, detail = LLMClient().test_connection(timeout=0.5)
    assert ok is False
    # The detail names the endpoint that could not be reached.
    assert "127.0.0.1:9" in detail


def test_connection_test_success(monkeypatch, db: Session):
    save_prefs({"llm_provider": "ollama", "llm_model": "qwen3:8b"})

    class _Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "qwen3:8b"}, {"id": "llama3"}]}

    class _Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def get(self, _url, headers=None):
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    ok, detail = LLMClient().test_connection()
    assert ok is True
    # The detail reports how many models the endpoint returned.
    assert "2" in detail


def test_connection_test_warns_on_missing_model(monkeypatch, db: Session):
    save_prefs({"llm_provider": "ollama", "llm_model": "not-installed"})

    class _Resp:
        status_code = 200

        def json(self):
            return {"data": [{"id": "qwen3:8b"}]}

    class _Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def get(self, _url, headers=None):
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    ok, detail = LLMClient().test_connection()
    assert ok is True
    # The warning names the configured model that the endpoint does not list.
    assert "not-installed" in detail


def test_llm_client_records_outbound_audit(monkeypatch, db: Session):
    save_prefs({"llm_provider": "ollama", "llm_model": "qwen3:8b"})

    class _Resp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": "sure"}}]}

    class _Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(self, _url, headers=None, json=None):
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    client = LLMClient()
    assert client.chat_text("system", "user", purpose="classify") == "sure"

    events = list_outbound(db)
    assert events
    assert events[0].purpose == "classify"
    assert events[0].provider == "ollama"
    assert events[0].ok is True
    assert events[0].redacted is False
    assert events[0].char_count == len("system") + len("user")


def test_outbound_audit_never_stores_prompt_text(db: Session):
    record_outbound(
        provider="deepseek",
        host="api.deepseek.com",
        model="deepseek-v4-flash",
        purpose="extract",
        char_count=1234,
        redacted=True,
        ok=True,
    )
    event = list_outbound(db)[0]
    assert event.char_count == 1234
    assert event.redacted is True
    assert not hasattr(event, "prompt")
    assert outbound_summary(db)["total"] >= 1


def test_settings_page_shows_providers_and_egress(client: TestClient):
    page = client.get("/settings")
    assert page.status_code == 200
    assert "Model endpoint" in page.text
    assert "Ollama" in page.text
    assert "LM Studio" in page.text
    # Every provider card is rendered, whatever language its label is in.
    for provider in PROVIDERS:
        assert provider.label in page.text
    assert "Does data leave this device?" in page.text
    assert "fully local" in page.text


def test_settings_saves_local_provider(client: TestClient):
    resp = client.post(
        "/settings/llm",
        data={
            "provider": "lmstudio",
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "local-model",
            "next": "/settings",
        },
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    effective = resolve_effective_llm()
    assert effective.provider.id == "lmstudio"
    assert effective.api_model == "local-model"
    page = client.get("/settings")
    assert "LM Studio" in page.text


def test_settings_switching_back_to_deepseek_clears_override(client: TestClient):
    client.post(
        "/settings/llm",
        data={"provider": "custom", "base_url": "http://x/v1", "model": "m"},
    )
    client.post("/settings/llm", data={"provider": "deepseek", "next": "/settings"})
    effective = resolve_effective_llm()
    assert effective.provider.id == "deepseek"
    assert "deepseek.com" in effective.base_url


def test_settings_connection_test_endpoint(client: TestClient):
    resp = client.post(
        "/settings/llm/test", data={"next": "/settings"}, follow_redirects=False
    )
    assert resp.status_code in (302, 303)
    assert "llm_test=" in resp.headers["location"]
    page = client.get(resp.headers["location"])
    assert page.status_code == 200
    # No key configured in tests: the probe must fail locally, naming the missing key,
    # without ever sending a request.
    alert = re.search(r'<div class="alert">(.*?)</div>', page.text, re.S)
    assert alert and "LLM_API_KEY" in alert.group(1)


def test_all_providers_declare_required_metadata():
    ids = {p.id for p in PROVIDERS}
    assert {"deepseek", "ollama", "lmstudio", "custom"} <= ids
    for provider in PROVIDERS:
        assert provider.label and provider.hint
        if provider.id != "custom":
            assert provider.base_url.startswith("http")


def test_outbound_events_are_wiped_with_data(client: TestClient, db: Session):
    record_outbound(
        provider="ollama",
        host="127.0.0.1:11434",
        model="qwen3:8b",
        purpose="chat",
        char_count=10,
        redacted=False,
        ok=True,
    )
    assert db.query(OutboundEvent).count() >= 1
    client.post("/settings/wipe", follow_redirects=False)
    assert db.query(OutboundEvent).count() == 0


def _settings_with_key(key: str):
    from app.config import Settings

    return Settings(llm_api_key=key)
