from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from app.branding import LEGACY_DB_FILENAME, PRODUCT_NAME
from app.config import ROOT_DIR, get_settings, sqlite_file_path
from app.llm.models import (
    MODELS,
    PROVIDERS,
    load_prefs,
    resolve_effective_llm,
    resolve_model,
    save_prefs,
)
from app.db.migrations import adopt_legacy_database, wipe_all_data
from app.db.session import get_db, get_engine
from app.integrations.hermes import export_profile_for_hermes
from app.llm.client import LLMClient
from app.routes import backup, chat, diary, imports, memories, profile
from app.routes.chat import _chat_page
from app.services.evaluation import EMBEDDING_SWITCH_MIN_DELTA
from app.services.outbound import list_outbound, outbound_summary
from app.services.usage import usage_summary
from app.services.prompts import (
    DEFAULT_ADVISOR_PROMPT,
    MAX_PROMPT_CHARS,
    prompt_is_custom,
    resolve_advisor_prompt,
    save_advisor_prompt,
)
from app.templating import render

BASE_DIR = Path(__file__).resolve().parent
_SAFE_NEXT = {
    "/",
    "/chat",
    "/settings",
    "/profile",
    "/graph",
    "/diary",
    "/imports",
    "/memories",
    "/search",
}


def _safe_next(path: str) -> str:
    path = (path or "").split("?")[0]
    if path in _SAFE_NEXT:
        return path
    if path.startswith("/memories"):
        return "/memories"
    if path.startswith("/chat/"):
        return path
    if path.startswith("/chat"):
        return "/"
    return "/settings"


def create_app() -> FastAPI:
    settings = get_settings()
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    db_path = sqlite_file_path(settings.database_url)
    if db_path is not None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        adopted = adopt_legacy_database(
            db_path,
            (
                db_path.parent / LEGACY_DB_FILENAME,
                ROOT_DIR / "data" / LEGACY_DB_FILENAME,
            ),
        )
        if adopted is not None:
            print(
                f"[{PRODUCT_NAME}] Migrated the legacy database {adopted.name} to {db_path.name}; "
                "existing data is unaffected."
            )
    get_engine()

    app = FastAPI(title=PRODUCT_NAME, docs_url="/api/docs", redoc_url=None)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.include_router(imports.router)
    app.include_router(memories.router)
    app.include_router(chat.router)
    app.include_router(profile.router)
    app.include_router(diary.router)
    app.include_router(backup.router)

    @app.exception_handler(404)
    async def not_found_handler(request: Request, _exc):
        accept = request.headers.get("accept", "")
        if "text/html" in accept:
            return render(request, "not_found.html", {}, status_code=404)
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, db: Session = Depends(get_db), q: str = "", mode: str = "ask"):
        return _chat_page(request, db, prefill=q, mode=mode or "ask")

    @app.get("/settings", response_class=HTMLResponse)
    def settings_page(
        request: Request,
        db: Session = Depends(get_db),
        saved: str = "",
        llm_test: str = "",
    ):
        settings = get_settings()
        effective = resolve_effective_llm()
        return render(
            request,
            "settings.html",
            {
                "settings": settings,
                "llm_configured": LLMClient().configured,
                "llm_choice": resolve_model(),
                "llm_models": MODELS,
                "llm_providers": PROVIDERS,
                "llm_effective": effective,
                "llm_prefs": load_prefs(),
                "llm_test": llm_test,
                "hermes_profile": export_profile_for_hermes(db),
                "saved": saved,
                "system_prompt": resolve_advisor_prompt(),
                "default_system_prompt": DEFAULT_ADVISOR_PROMPT.strip(),
                "prompt_is_custom": prompt_is_custom(),
                "max_prompt_chars": MAX_PROMPT_CHARS,
                "outbound_events": list_outbound(db, limit=20),
                "outbound_stats": outbound_summary(db),
                "usage": usage_summary(db),
                "embedding_switch_delta": EMBEDDING_SWITCH_MIN_DELTA,
            },
            db=db,
        )

    @app.post("/settings/llm")
    def set_llm_endpoint(
        provider: str = Form(...),
        base_url: str = Form(""),
        model: str = Form(""),
        test: str = Form(""),
        next: str = Form("/settings"),
    ):
        provider_ids = {p.id for p in PROVIDERS}
        chosen = provider if provider in provider_ids else "deepseek"
        updates: dict = {"llm_provider": chosen}
        if chosen == "deepseek":
            updates["llm_base_url"] = None
        else:
            updates["llm_base_url"] = base_url.strip() or None
            updates["llm_model"] = model.strip() or None
        save_prefs(updates)
        target = _safe_next(next)
        if target == "/settings":
            target = "/settings?saved=llm"
            if test:
                ok, detail = LLMClient().test_connection()
                target = (
                    f"/settings?saved=llm&llm_test="
                    + quote(("✅ " if ok else "⚠️ ") + detail)
                )
        return RedirectResponse(target, status_code=303)

    @app.post("/settings/llm/test")
    def test_llm_endpoint(next: str = Form("/settings")):
        ok, detail = LLMClient().test_connection()
        target = _safe_next(next)
        if target != "/settings":
            target = "/settings"
        return RedirectResponse(
            "/settings?llm_test=" + quote(("✅ " if ok else "⚠️ ") + detail),
            status_code=303,
        )

    @app.post("/settings/prompt")
    def set_prompt(
        system_prompt: str = Form(""),
        reset: str = Form(""),
        next: str = Form("/settings"),
    ):
        save_advisor_prompt(system_prompt, reset=bool(reset))
        target = _safe_next(next)
        if target == "/settings":
            target = "/settings?saved=prompt"
        return RedirectResponse(target, status_code=303)

    @app.post("/settings/model")
    def set_model(model: str = Form(...), next: str = Form("/settings")):
        choice = resolve_model(model)
        save_prefs({"llm_model": choice.id})
        target = _safe_next(next)
        if target == "/settings":
            target = "/settings?saved=model"
        return RedirectResponse(target, status_code=303)

    @app.post("/settings/wipe")
    def wipe_data():
        wipe_all_data(get_engine())
        return RedirectResponse("/settings", status_code=303)

    @app.get("/api/health")
    def health():
        return {"ok": True, "bind": f"{get_settings().bind_host}:{get_settings().bind_port}"}

    @app.get("/api/hermes/profile")
    def api_hermes_profile(db: Session = Depends(get_db)):
        return export_profile_for_hermes(db)

    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.bind_host,
        port=settings.bind_port,
        reload=False,
    )


if __name__ == "__main__":
    run()
