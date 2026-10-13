from __future__ import annotations

from fastapi.testclient import TestClient

from app.llm.client import LLMClient
from app.llm.models import resolve_model, save_prefs


def test_resolve_legacy_aliases():
    assert resolve_model("deepseek-chat").id == "deepseek-v4-flash"
    assert resolve_model("deepseek-chat").thinking is False
    assert resolve_model("deepseek-reasoner").id == "deepseek-v4-flash-think"
    assert resolve_model("deepseek-reasoner").thinking is True
    assert resolve_model("deepseek-v4-pro-think").api_model == "deepseek-v4-pro"


def test_settings_switch_model(client: TestClient):
    page = client.get("/settings")
    assert page.status_code == 200
    assert "V4 Flash" in page.text
    assert "V4 Pro" in page.text

    resp = client.post(
        "/settings/model",
        data={"model": "deepseek-v4-pro-think", "next": "/settings"},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    choice = resolve_model()
    assert choice.id == "deepseek-v4-pro-think"
    assert choice.thinking is True
    home = client.get("/")
    assert "V4 Pro Thinking" in home.text


def test_llm_client_uses_resolved_model(tmp_path, monkeypatch):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    from app.config import get_settings

    get_settings.cache_clear()
    save_prefs({"llm_model": "deepseek-v4-flash-think"})
    client = LLMClient()
    assert client.model == "deepseek-v4-flash"
    assert client.choice.thinking is True
    get_settings.cache_clear()
