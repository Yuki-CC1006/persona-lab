"""First-use guidance: turn imported material into an evidence-backed answer.

The goal is that a new user never needs the README: import something, review
the candidates it produced, then ask one question that provably cites their
own material.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import ImportJob, Memory, MemoryStatus
from app.services.memory import memory_stats

# Question templates keyed by the memory types actually present.
TYPE_QUESTIONS: list[tuple[str, str]] = [
    ("aversion", "What does {s} dislike?"),
    ("preference", "What does {s} like?"),
    ("habit", "What habits does {s} usually have?"),
    ("communication", "What should I keep in mind when talking to {s}?"),
    ("plan", "What is {s} planning lately?"),
    ("person_place", "Which people or places matter to {s}?"),
    ("event", "What important dates are there for {s}?"),
    ("fact", "What facts about {s} are confirmed?"),
]

DEFAULT_SUBJECT = "the other person"


def dominant_subject(db: Session, fallback: str | None = None) -> str:
    rows = (
        db.query(Memory.subject)
        .filter(
            Memory.status == MemoryStatus.CONFIRMED.value,
            Memory.subject.isnot(None),
        )
        .all()
    )
    counts: dict[str, int] = {}
    for (value,) in rows:
        name = (value or "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    if counts:
        return max(counts.items(), key=lambda kv: kv[1])[0]
    cleaned = (fallback or "").strip()
    return cleaned or DEFAULT_SUBJECT


def example_questions(
    db: Session, *, subject_name: str | None = None, limit: int = 4
) -> list[str]:
    types = [
        row[0]
        for row in db.query(Memory.memory_type)
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .distinct()
        .all()
    ]
    if not types:
        return []

    subject = dominant_subject(db, subject_name)
    present = set(types)
    questions = [
        template.format(s=subject)
        for memory_type, template in TYPE_QUESTIONS
        if memory_type in present
    ]
    if not questions:
        questions = [f"What have I confirmed about {subject}?"]
    return questions[:limit]


def import_next_steps(db: Session, job: ImportJob) -> dict:
    stats = memory_stats(db)
    pending = int(stats.get("candidates") or 0)
    subject = dominant_subject(db, job.subject_name)
    return {
        "subject_name": subject,
        "message_count": job.message_count,
        "pending_candidates": pending,
        "questions": example_questions(db, subject_name=job.subject_name),
        "steps": [
            {
                "label": f"Review the candidate memories from this import ({pending} pending)",
                "href": "/memories/candidates",
                "done": pending == 0,
            },
            {
                "label": "Ask one question to verify the material can be cited",
                "href": "/",
                "done": False,
            },
            {
                "label": "View the profile overview",
                "href": "/profile",
                "done": False,
            },
        ],
    }
