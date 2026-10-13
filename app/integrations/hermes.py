"""Optional Hermes Agent adapter — not required for MVP runtime.

Expose confirmed profile as a compact read-only payload that Hermes
(or any agent) can pull via local HTTP, without syncing raw chat history.
"""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import MemoryStatus
from app.services.memory import list_memories, memory_stats


def export_profile_for_hermes(db: Session, *, max_chars: int = 1200) -> dict:
    """Export a short curated profile suitable for Hermes USER.md-like injection."""
    memories = list_memories(db, status=MemoryStatus.CONFIRMED.value, limit=100)
    lines: list[str] = []
    used = 0
    included = []
    for m in memories:
        line = f"- [{m.memory_type}] {m.content}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
        included.append(m.id)

    text = "\n".join(lines)
    stats = memory_stats(db)
    return {
        "profile_text": text,
        "char_count": len(text),
        "max_chars": max_chars,
        "memory_ids": included,
        "stats": stats,
        "note": (
            "Contains only confirmed memories (including ones the user explicitly asked "
            "to save). Raw chat logs are never synced to Hermes."
        ),
    }


def hermes_tool_spec() -> dict:
    """Describe tools an external Hermes skill could register."""
    return {
        "tools": [
            {
                "name": SEARCH_TOOL,
                "description": (
                    "Search confirmed memories and related evidence in the local "
                    "knowledge base (read-only)"
                ),
                "parameters": {"query": "string"},
            },
            {
                "name": PROFILE_TOOL,
                "description": (
                    "Fetch the concise confirmed profile text (read-only, length-capped)"
                ),
                "parameters": {},
            },
        ],
        "write_policy": (
            "The confirmed profile is only updated directly when the user explicitly says "
            "\"remember\" or \"correct\". Ordinary conversation never writes to it; imported "
            "extractions still go through candidate review."
        ),
    }
