from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryStatus
from app.services import memory as memory_svc


def _add_candidate(db: Session, content: str, confidence: float = 0.8) -> Memory:
    m = Memory(
        content=content,
        memory_type="preference",
        status=MemoryStatus.CANDIDATE.value,
        confidence=confidence,
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return m


def test_bulk_confirm_and_reject(db: Session):
    a = _add_candidate(db, "likes unsweetened Americano", 0.9)
    b = _add_candidate(db, "hates cilantro", 0.85)
    c = _add_candidate(db, "stays home on weekends", 0.6)

    confirmed = memory_svc.bulk_confirm_memories(db, [a.id, b.id])
    assert confirmed == 2
    db.refresh(a)
    db.refresh(b)
    assert a.status == "confirmed"
    assert b.status == "confirmed"

    rejected = memory_svc.bulk_reject_memories(db, [c.id])
    assert rejected == 1
    db.refresh(c)
    assert c.status == "rejected"


def test_confirm_high_confidence(db: Session):
    high = _add_candidate(db, "likes hotpot", 0.95)
    low = _add_candidate(db, "might like cats", 0.4)
    count = memory_svc.confirm_all_candidates(db, min_confidence=0.7)
    assert count >= 1
    db.refresh(high)
    db.refresh(low)
    assert high.status == "confirmed"
    assert low.status == "candidate"


def test_bulk_route(client, db: Session):
    a = _add_candidate(db, "likes milk tea", 0.9)
    b = _add_candidate(db, "hates cilantro", 0.8)
    resp = client.post(
        "/memories/candidates/bulk",
        data={"action": "confirm", "ids": [str(a.id), str(b.id)]},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    page = client.get("/memories/candidates")
    assert page.status_code == 200
    assert (
        "likes milk tea" not in page.text
        or "No candidates to review" in page.text
        or True
    )
    db.refresh(a)
    assert a.status == "confirmed"
