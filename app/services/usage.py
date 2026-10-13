"""Usage rhythm: the retention evidence that decides what to build next.

MCP/sync/collaboration/monetization all stay unbuilt until these numbers show
people coming back. This module only measures; it does not decide.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import ChatTurn, DiaryEntry, ImportJob, Memory, MemoryStatus

WEEKS = 4


def usage_summary(db: Session, *, weeks: int = WEEKS) -> dict:
    """Weekly activity for the last ``weeks`` weeks, newest first."""
    now = datetime.utcnow()
    start = (now - timedelta(days=7 * weeks)).replace(hour=0, minute=0, second=0, microsecond=0)
    buckets = []
    for index in range(weeks):
        end = now - timedelta(days=7 * index)
        begin = end - timedelta(days=7)
        buckets.append({"label": f"{begin:%m-%d} ~ {end:%m-%d}", "begin": begin, "end": end})

    weekly = []
    for bucket in buckets:
        weekly.append(
            {
                "label": bucket["label"],
                "questions": _count(
                    db, ChatTurn, ChatTurn.created_at, bucket["begin"], bucket["end"]
                ),
                "confirmed": _count_memories(db, MemoryStatus.CONFIRMED, bucket["begin"], bucket["end"]),
                "candidates": _count_memories(db, MemoryStatus.CANDIDATE, bucket["begin"], bucket["end"]),
                "imports": _count(db, ImportJob, ImportJob.created_at, bucket["begin"], bucket["end"]),
                "diary": _count(db, DiaryEntry, DiaryEntry.created_at, bucket["begin"], bucket["end"]),
            }
        )

    active_weeks = sum(
        1
        for week in weekly
        if week["questions"] or week["confirmed"] or week["imports"] or week["diary"]
    )
    total_confirmed = int(
        db.query(func.count(Memory.id))
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .scalar()
        or 0
    )
    last_question = (
        db.query(func.max(ChatTurn.created_at)).filter(ChatTurn.role == "user").scalar()
    )
    return {
        "weeks": weekly,
        "active_weeks": active_weeks,
        "total_confirmed": total_confirmed,
        "total_questions": int(db.query(func.count(ChatTurn.id)).scalar() or 0),
        "last_question_at": last_question,
        "since": start,
        "expansion_gates": _expansion_gates(active_weeks, total_confirmed),
    }


def _expansion_gates(active_weeks: int, total_confirmed: int) -> list[dict]:
    """What must be true before MCP/sync/collaboration work is worth doing."""
    return [
        {
            "id": "read-only-mcp",
            "label": "Read-only MCP",
            "ready": True,
            "detail": "Available now (scripts/mcp_server.py); exposes only confirmed memories and the profile summary, with no write tools.",
        },
        {
            "id": "retrieval-upgrade",
            "label": "Retrieval upgrade (vector index)",
            "ready": total_confirmed >= 200,
            "detail": f"{total_confirmed} confirmed memories so far; reassess an incremental vector index above 200.",
        },
        {
            "id": "sync",
            "label": "Cross-device sync",
            "ready": active_weeks >= 4,
            "detail": f"{active_weeks}/4 consecutive active weeks. Fewer than 4 weeks means a usage habit has not formed yet.",
        },
        {
            "id": "collab",
            "label": "Collaboration and subscription",
            "ready": False,
            "detail": "Needs paid-interview evidence (at least 5 users willing to pay); there is none yet.",
        },
    ]


def _count(db: Session, model, column, begin: datetime, end: datetime) -> int:
    return int(
        db.query(func.count(model.id)).filter(column >= begin, column < end).scalar() or 0
    )


def _count_memories(db: Session, status: MemoryStatus, begin: datetime, end: datetime) -> int:
    return int(
        db.query(func.count(Memory.id))
        .filter(
            Memory.status == status.value,
            Memory.created_at >= begin,
            Memory.created_at < end,
        )
        .scalar()
        or 0
    )
