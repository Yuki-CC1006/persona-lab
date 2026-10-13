from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session, joinedload

from app.db.models import ChatThread, ChatTurn


def list_threads(db: Session, *, limit: int = 30) -> list[ChatThread]:
    return (
        db.query(ChatThread)
        .order_by(ChatThread.updated_at.desc(), ChatThread.id.desc())
        .limit(limit)
        .all()
    )


def get_thread(db: Session, thread_id: int) -> ChatThread | None:
    return (
        db.query(ChatThread)
        .options(joinedload(ChatThread.turns))
        .filter(ChatThread.id == thread_id)
        .first()
    )


def create_thread(db: Session, title: str) -> ChatThread:
    thread = ChatThread(title=_clip_title(title), updated_at=datetime.utcnow())
    db.add(thread)
    db.commit()
    db.refresh(thread)
    return thread


def rename_thread(db: Session, thread_id: int, title: str) -> ChatThread | None:
    thread = db.get(ChatThread, thread_id)
    if not thread:
        return None
    cleaned = (title or "").strip().replace("\n", " ")
    if cleaned:
        thread.title = _clip_title(cleaned)
        thread.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(thread)
    return thread


def delete_thread(db: Session, thread_id: int) -> bool:
    thread = db.get(ChatThread, thread_id)
    if not thread:
        return False
    db.query(ChatTurn).filter(ChatTurn.thread_id == thread_id).delete(
        synchronize_session=False
    )
    db.delete(thread)
    db.commit()
    return True


def append_turn(
    db: Session,
    thread: ChatThread,
    *,
    role: str,
    content: str,
    mode: str = "ask",
    extra: dict | None = None,
) -> ChatTurn:
    turn = ChatTurn(
        thread_id=thread.id,
        role=role,
        content=content,
        mode=mode,
        extra=extra,
    )
    db.add(turn)
    thread.updated_at = datetime.utcnow()
    if role == "user" and (thread.title == "New chat" or not thread.title):
        thread.title = _clip_title(content)
    db.commit()
    db.refresh(turn)
    return turn


def turns_as_messages(thread: ChatThread) -> list[dict]:
    out = []
    for t in thread.turns or []:
        extra = t.extra or {}
        out.append(
            {
                "role": t.role,
                "content": t.content,
                "meta_chips": extra.get("meta_chips") or [],
                "evidence": extra.get("evidence") or [],
                "support": extra.get("support") or [],
                "counter": extra.get("counter") or [],
                "distilled": extra.get("distilled") or [],
                "thinking": extra.get("thinking") or "",
            }
        )
    return out


def history_for_llm(thread: ChatThread, *, limit: int = 8) -> list[dict]:
    turns = list(thread.turns or [])[-limit:]
    return [{"role": t.role, "content": t.content} for t in turns]


def _clip_title(text: str) -> str:
    title = (text or "").strip().replace("\n", " ")
    if len(title) > 22:
        title = title[:22] + "…"
    return title or "New chat"
