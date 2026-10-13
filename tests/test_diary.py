from __future__ import annotations

from datetime import datetime

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import MemoryStatus
from app.services import diary as diary_svc


def test_diary_create_and_offline_extract(db: Session):
    entry = diary_svc.create_diary(
        db,
        "Today Alex said they dislike horror movies, and would rather watch a movie at home this weekend.",
        entry_date=datetime(2026, 8, 11),
        title="Friday dinner",
        subject_name="Alex",
    )
    assert entry.id
    updated, created = diary_svc.extract_diary_memories(db, entry.id)
    assert updated.status in ("extracted", "extracted_offline")
    assert len(created) >= 1
    assert all(m.status == MemoryStatus.CANDIDATE.value for m in created)
    assert all(m.source_diary_id == entry.id for m in created)


def test_diary_pages(client: TestClient):
    assert client.get("/diary").status_code == 200
    assert "Notes" in client.get("/diary").text

    resp = client.post(
        "/diary/create",
        data={
            "content": "Alex dislikes cilantro, so watch out when ordering.",
            "entry_date": "2026-08-11",
            "title": "Lunch",
            "subject_name": "Alex",
            "extract_now": "yes",
        },
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    # extract_now redirects to candidates
    assert "/memories/candidates" in resp.headers.get("location", "")
    candidates = client.get("/memories/candidates")
    assert candidates.status_code == 200
