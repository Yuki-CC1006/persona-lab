"""Usage rhythm feeds the build/not-build decision for MCP, sync, collaboration."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import ChatThread, ChatTurn, DiaryEntry, Memory, MemoryStatus
from app.services import memory as memory_svc
from app.services.usage import usage_summary


def test_empty_database_reports_zero_activity(db: Session):
    summary = usage_summary(db)
    assert summary["active_weeks"] == 0
    assert summary["total_questions"] == 0
    assert summary["total_confirmed"] == 0
    assert len(summary["weeks"]) == 4


def test_weeks_are_newest_first_and_cover_four_weeks(db: Session):
    summary = usage_summary(db)
    labels = [week["label"] for week in summary["weeks"]]
    assert len(labels) == 4
    assert labels == sorted(labels, reverse=True)


def test_activity_is_bucketed_into_the_right_week(db: Session):
    thread = ChatThread(title="This week's conversation")
    db.add(thread)
    db.commit()
    db.add(
        ChatTurn(
            thread_id=thread.id,
            role="user",
            content="What does she like?",
            created_at=datetime.utcnow() - timedelta(days=1),
        )
    )
    db.add(
        ChatTurn(
            thread_id=thread.id,
            role="user",
            content="Asked a long time ago",
            created_at=datetime.utcnow() - timedelta(days=20),
        )
    )
    db.commit()

    summary = usage_summary(db)
    assert summary["weeks"][0]["questions"] == 1
    assert summary["total_questions"] == 2
    assert summary["active_weeks"] == 2


def test_old_activity_falls_outside_the_window(db: Session):
    memory = memory_svc.create_manual_memory(db, "a memory confirmed long ago", memory_type="fact")
    memory.created_at = datetime.utcnow() - timedelta(days=120)
    db.commit()
    summary = usage_summary(db)
    assert sum(week["confirmed"] for week in summary["weeks"]) == 0
    assert summary["total_confirmed"] == 1  # lifetime total is still reported


def test_candidates_and_imports_are_counted_separately(db: Session):
    db.add(
        Memory(
            content="a candidate waiting for review",
            memory_type="fact",
            status=MemoryStatus.CANDIDATE.value,
            created_at=datetime.utcnow() - timedelta(days=1),
        )
    )
    db.add(DiaryEntry(entry_date=datetime.utcnow(), content="Good chat today"))
    db.commit()
    summary = usage_summary(db)
    assert summary["weeks"][0]["candidates"] == 1
    assert summary["weeks"][0]["diary"] == 1
    assert summary["weeks"][0]["confirmed"] == 0


def test_expansion_gates_stay_closed_on_thin_data(db: Session):
    gates = {gate["id"]: gate for gate in usage_summary(db)["expansion_gates"]}
    assert gates["read-only-mcp"]["ready"] is True  # the only thing already shipped
    assert gates["sync"]["ready"] is False
    assert gates["collab"]["ready"] is False
    assert gates["retrieval-upgrade"]["ready"] is False


def test_sync_gate_opens_after_four_active_weeks(db: Session):
    thread = ChatThread(title="Sustained use")
    db.add(thread)
    db.commit()
    now = datetime.utcnow()
    for week in range(4):
        db.add(
            ChatTurn(
                thread_id=thread.id,
                role="user",
                content=f"Question from week {week}",
                created_at=now - timedelta(days=7 * week + 1),
            )
        )
    db.commit()
    gates = {gate["id"]: gate for gate in usage_summary(db)["expansion_gates"]}
    assert gates["sync"]["ready"] is True


def test_settings_page_shows_usage_and_gates(client: TestClient, db: Session):
    memory_svc.create_manual_memory(db, "Alex likes unsweetened Americano", memory_type="preference")
    page = client.get("/settings").text
    assert "Usage cadence" in page
    assert "Expansion gates" in page
    assert "Read-only MCP" in page
    assert "Cross-device sync" in page
    assert "Collaboration and subscription" in page
