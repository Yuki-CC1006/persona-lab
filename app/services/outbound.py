"""Local outbound audit: what metadata left the machine, and was it redacted."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import OutboundEvent
from app.db.session import session_scope


def record_outbound(
    *,
    provider: str,
    host: str,
    model: str,
    purpose: str,
    char_count: int,
    redacted: bool,
    ok: bool,
    status_code: int | None = None,
    error: str | None = None,
) -> None:
    """Best-effort audit write. Never raises into the caller's request path."""
    try:
        with session_scope() as db:
            db.add(
                OutboundEvent(
                    provider=provider[:32],
                    host=host[:256],
                    model=model[:128],
                    purpose=purpose[:32],
                    char_count=max(0, int(char_count)),
                    redacted=bool(redacted),
                    ok=bool(ok),
                    status_code=status_code,
                    error=(error or None) and str(error)[:256],
                )
            )
    except Exception:  # noqa: BLE001 — auditing must not break the feature
        pass


def list_outbound(db: Session, *, limit: int = 30) -> list[OutboundEvent]:
    return (
        db.query(OutboundEvent)
        .order_by(OutboundEvent.id.desc())
        .limit(limit)
        .all()
    )


def outbound_summary(db: Session) -> dict:
    total = int(db.query(func.count(OutboundEvent.id)).scalar() or 0)
    remote = int(
        db.query(func.count(OutboundEvent.id))
        .filter(OutboundEvent.provider != "ollama", OutboundEvent.provider != "lmstudio")
        .scalar()
        or 0
    )
    last = db.query(OutboundEvent).order_by(OutboundEvent.id.desc()).first()
    return {"total": total, "remote": remote, "last_at": last.created_at if last else None}
