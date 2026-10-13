from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryStatus
from app.services import memory as memory_svc
from app.services.profile import export_memories_markdown


def test_export_contains_confirmed_only(db: Session):
    memory_svc.create_manual_memory(
        db, "Alex likes unsweetened Americano", memory_type="preference", subject="Alex"
    )
    db.add(
        Memory(
            content="a candidate that should never appear",
            memory_type="fact",
            status=MemoryStatus.CANDIDATE.value,
            subject="Alex",
        )
    )
    db.commit()

    text = export_memories_markdown(db)

    assert "Alex likes unsweetened Americano" in text
    assert "a candidate that should never appear" not in text
    assert "Alex — profile export" in text
    assert "Preference" in text


def test_export_is_empty_safe(db: Session):
    text = export_memories_markdown(db)
    assert "(no confirmed memories yet)" in text


def test_export_route_serves_markdown(client, db: Session):
    memory_svc.create_manual_memory(
        db, "Alex likes unsweetened Americano", memory_type="preference", subject="Alex"
    )
    resp = client.get("/profile/export.md")
    assert resp.status_code == 200
    assert "markdown" in resp.headers["content-type"]
    assert "Alex likes unsweetened Americano" in resp.text
