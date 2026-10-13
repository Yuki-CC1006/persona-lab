from __future__ import annotations

import json
from urllib.parse import urlparse

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.llm.client import LLMClient
from app.services import chat as chat_svc
from app.services.distill import distill_user_message
from app.services.onboarding import example_questions
from app.services.prediction import (
    _humane_unknown,
    _looks_like_refusal,
    answer_question,
    iter_answer_events,
    iter_tendency_events,
    predict_tendency,
    tendency_to_dict,
)
from app.templating import render

router = APIRouter(tags=["chat"])


def _chat_page(
    request: Request,
    db: Session,
    *,
    thread=None,
    prefill: str = "",
    mode: str = "ask",
):
    messages = chat_svc.turns_as_messages(thread) if thread else []
    return render(
        request,
        "index.html",
        {
            "messages": messages,
            "prefill": prefill,
            "mode": mode,
            "thread": thread,
            "active_thread_id": thread.id if thread else None,
            "llm_configured": LLMClient().configured,
            "suggestions": example_questions(db, limit=2) if not thread else [],
        },
        db=db,
    )


@router.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request, db: Session = Depends(get_db)):
    return _chat_page(request, db)


@router.get("/chat/{thread_id}", response_class=HTMLResponse)
def chat_thread_page(thread_id: int, request: Request, db: Session = Depends(get_db)):
    thread = chat_svc.get_thread(db, thread_id)
    if not thread:
        raise HTTPException(404, "Conversation not found")
    return _chat_page(request, db, thread=thread)


@router.post("/chat/{thread_id}/rename")
def chat_rename(
    thread_id: int,
    request: Request,
    title: str = Form(...),
    db: Session = Depends(get_db),
):
    thread = chat_svc.rename_thread(db, thread_id, title)
    if not thread:
        raise HTTPException(404, "Conversation not found")
    if "application/json" in (request.headers.get("accept") or ""):
        return JSONResponse({"ok": True, "id": thread.id, "title": thread.title})
    dest = urlparse(request.headers.get("referer") or "").path or f"/chat/{thread.id}"
    if not dest.startswith("/") or dest.startswith("//"):
        dest = f"/chat/{thread.id}"
    return RedirectResponse(dest, status_code=303)


@router.post("/chat/{thread_id}/delete")
def chat_delete(thread_id: int, request: Request, db: Session = Depends(get_db)):
    if not chat_svc.delete_thread(db, thread_id):
        raise HTTPException(404, "Conversation not found")
    return RedirectResponse(_after_delete(request, thread_id), status_code=303)


def _after_delete(request: Request, thread_id: int) -> str:
    path = urlparse(request.headers.get("referer") or "").path or "/"
    if path == f"/chat/{thread_id}":
        return "/"
    if path.startswith("/") and not path.startswith("//"):
        return path
    return "/"


@router.post("/chat/ask", response_class=HTMLResponse)
def chat_ask(
    request: Request,
    question: str = Form(...),
    mode: str = Form("ask"),
    thread_id: int | None = Form(None),
    db: Session = Depends(get_db),
):
    q = question.strip()
    if not q:
        return RedirectResponse("/", status_code=303)

    thread = chat_svc.get_thread(db, thread_id) if thread_id else None
    history = chat_svc.history_for_llm(thread) if thread else []
    if thread is None:
        thread = chat_svc.create_thread(db, q)

    chat_svc.append_turn(db, thread, role="user", content=q, mode=mode)

    if mode == "tendency":
        result = tendency_to_dict(predict_tendency(db, q, history=history))
        if result["can_answer"]:
            content = result["tendency"] or ""
            chips = [{"text": f"Tendency · {result['confidence']:.0%}", "class": "ok"}]
        else:
            content = result["refusal_reason"] or _humane_unknown(q, [])
            chips = []
        extra = {
            "meta_chips": chips,
            "support": result.get("supporting_evidence") or [],
            "counter": result.get("counter_evidence") or [],
        }
    else:
        qa = answer_question(db, q, history=history)
        chips = []
        content = qa.get("answer") or ""
        if not qa.get("can_answer") and _looks_like_refusal(content):
            chips.append({"text": "Insufficient evidence", "class": "warn"})
        extra = {
            "meta_chips": chips,
            "evidence": qa.get("evidence") or [],
            "thinking": qa.get("thinking") or "",
        }

    extra = _attach_distill(db, q, thread.id, history, extra)
    chat_svc.append_turn(db, thread, role="assistant", content=content, mode=mode, extra=extra)
    return RedirectResponse(f"/chat/{thread.id}", status_code=303)


@router.post("/chat/stream")
def chat_stream(
    question: str = Form(...),
    mode: str = Form("ask"),
    thread_id: int | None = Form(None),
    db: Session = Depends(get_db),
):
    q = question.strip()
    if not q:
        return StreamingResponse(
            iter([_sse("error", {"text": "Empty question"})]),
            media_type="text/event-stream",
        )

    def events():
        thread = chat_svc.get_thread(db, thread_id) if thread_id else None
        history = chat_svc.history_for_llm(thread) if thread else []
        if thread is None:
            thread = chat_svc.create_thread(db, q)
        chat_svc.append_turn(db, thread, role="user", content=q, mode=mode)
        yield _sse("meta", {"thread_id": thread.id, "title": thread.title})

        thinking = ""
        assembled = ""
        final = ""
        extra: dict = {"meta_chips": []}
        source = iter_tendency_events if mode == "tendency" else iter_answer_events
        try:
            for ev in source(db, q, history=history):
                kind = ev.get("type")
                if kind == "status":
                    yield _sse("status", {"text": ev.get("text") or ""})
                elif kind == "thinking":
                    yield _sse("thinking", {"text": ev.get("text") or ""})
                elif kind == "token":
                    assembled += ev.get("text") or ""
                    yield _sse("token", {"text": ev.get("text") or ""})
                elif kind == "done":
                    final = ev.get("answer") or assembled
                    thinking = ev.get("thinking") or ""
                    extra = _extra_from_done(mode, ev)
        except Exception:
            final = assembled or "Something went wrong while generating the answer."
            extra = {"meta_chips": [{"text": "Error", "class": "warn"}]}
            if not assembled:
                yield _sse("token", {"text": final})

        content = final or assembled
        extra["thinking"] = thinking
        extra = _attach_distill(db, q, thread.id, history, extra)
        chat_svc.append_turn(
            db, thread, role="assistant", content=content, mode=mode, extra=extra
        )
        yield _sse(
            "done",
            {
                "thread_id": thread.id,
                "answer": content,
                "thinking": thinking,
                "meta_chips": extra.get("meta_chips") or [],
                "evidence": extra.get("evidence") or [],
                "support": extra.get("support") or [],
                "counter": extra.get("counter") or [],
                "distilled": extra.get("distilled") or [],
            },
        )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extra_from_done(mode: str, ev: dict) -> dict:
    if mode == "tendency":
        if ev.get("can_answer"):
            chips = [{"text": f"Tendency · {float(ev.get('confidence') or 0):.0%}", "class": "ok"}]
        else:
            chips = []
        return {
            "meta_chips": chips,
            "support": ev.get("supporting_evidence") or [],
            "counter": ev.get("counter_evidence") or [],
        }
    chips = []
    answer = ev.get("answer") or ""
    if not ev.get("can_answer") and _looks_like_refusal(answer):
        chips.append({"text": "Insufficient evidence", "class": "warn"})
    return {"meta_chips": chips, "evidence": ev.get("evidence") or []}


def _attach_distill(db: Session, q: str, thread_id: int, history: list, extra: dict) -> dict:
    distilled = distill_user_message(db, q, thread_id=thread_id, history=history)
    if not distilled.changes:
        return extra
    extra.setdefault("meta_chips", [])
    if distilled.corrected_count:
        extra["meta_chips"].append(
            {"text": f"Corrected {distilled.corrected_count} memories", "class": "ok"}
        )
    if distilled.added_count:
        extra["meta_chips"].append(
            {"text": f"Wrote {distilled.added_count} to the profile", "class": "ok"}
        )
    extra["distilled"] = [
        {
            "id": c.memory.id,
            "content": c.memory.content,
            "memory_type": c.memory.memory_type,
            "action": c.action,
            "replaced": c.replaced,
        }
        for c in distilled.changes
    ]
    return extra


@router.post("/chat/tendency", response_class=HTMLResponse)
def chat_tendency(
    request: Request,
    situation: str = Form(...),
    thread_id: int | None = Form(None),
    db: Session = Depends(get_db),
):
    return chat_ask(
        request, question=situation, mode="tendency", thread_id=thread_id, db=db
    )
