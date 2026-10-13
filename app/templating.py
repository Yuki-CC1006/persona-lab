from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates
from markupsafe import Markup
from sqlalchemy.orm import Session

from app.branding import PRODUCT_NAME
from app.db.models import MemoryStatus
from app.services.memory import memory_stats

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.globals["brand"] = PRODUCT_NAME


def render_answer_html(text: str) -> Markup:
    raw = (text or "").strip()
    if not raw:
        return Markup("")
    escaped = html.escape(raw)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", escaped)
    blocks = re.split(r"\n\s*\n", escaped)
    parts: list[str] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        if not lines:
            continue
        if all(re.match(r"^[-•]\s+", ln) for ln in lines):
            items = "".join(f"<li>{re.sub(r'^[-•]\s+', '', ln)}</li>" for ln in lines)
            parts.append(f"<ul>{items}</ul>")
        else:
            parts.append("<p>" + "<br>".join(lines) + "</p>")
    return Markup("".join(parts))


templates.env.filters["answer_md"] = render_answer_html


def render(
    request: Request,
    name: str,
    context: dict[str, Any] | None = None,
    *,
    db: Session | None = None,
    status_code: int = 200,
):
    from app.llm.models import MODELS, resolve_model

    ctx: dict[str, Any] = dict(context or {})
    ctx.setdefault("llm_choice", resolve_model())
    ctx.setdefault("llm_models", MODELS)
    if db is not None:
        from app.services.chat import list_threads

        stats = memory_stats(db)
        ctx.setdefault("candidate_count", stats.get("candidates", 0))
        ctx.setdefault("confirmed_count", stats.get("confirmed", 0))
        ctx.setdefault("nav_stats", stats)
        ctx.setdefault("chat_threads", list_threads(db))
        ctx.setdefault("active_thread_id", None)
    else:
        ctx.setdefault("candidate_count", 0)
        ctx.setdefault("confirmed_count", 0)
        ctx.setdefault("chat_threads", [])
    return templates.TemplateResponse(request, name, ctx, status_code=status_code)


def status_label(status: str | None = None) -> str:
    return status or MemoryStatus.CONFIRMED.value
