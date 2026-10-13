"""Bi-temporal memory: valid time, learned time, invalidation and review."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.db.migrations import init_db
from app.db.models import Memory, MemoryStatus
from app.services import diary as diary_svc
from app.services import memory as memory_svc
from app.services.distill import distill_user_message
from app.services.retrieval import hybrid_search


def _confirmed(db: Session, content: str, memory_type: str = "aversion", **kw) -> Memory:
    return memory_svc.create_manual_memory(db, content, memory_type=memory_type, **kw)


def _candidate(db: Session, content: str, memory_type: str = "aversion") -> Memory:
    memory = Memory(
        content=content,
        memory_type=memory_type,
        status=MemoryStatus.CANDIDATE.value,
        confidence=0.8,
    )
    db.add(memory)
    db.flush()
    memory_svc.propose_replacement(db, memory)
    db.commit()
    db.refresh(memory)
    return memory


def test_unrelated_preferences_do_not_create_proposals(db: Session):
    _confirmed(db, "Alex likes unsweetened Americano", memory_type="preference")
    candidate = _candidate(db, "Alex likes hot soy milk", memory_type="preference")
    # Two preferences can both be true; only contradictions deserve review.
    assert candidate.proposed_action is None
    assert candidate.proposed_target_id is None


def test_polarity_flip_is_treated_as_a_conflict(db: Session):
    old = _confirmed(db, "Alex likes cilantro", memory_type="preference")
    candidate = _candidate(db, "Alex dislikes cilantro", memory_type="preference")
    assert candidate.proposed_action == "replace"
    assert candidate.proposed_target_id == old.id


def test_schema_has_temporal_columns(db: Session):
    engine = db.get_bind()
    init_db(engine)  # idempotent: second call must not fail
    rows = engine.connect().execute(text("PRAGMA table_info(memories)")).fetchall()
    columns = {row[1] for row in rows}
    for name in ("valid_from", "valid_to", "created_at", "superseded_at", "proposed_action", "proposed_target_id"):
        assert name in columns


def test_inferred_conflict_creates_proposal_not_rewrite(db: Session):
    old = _confirmed(db, "Alex dislikes cilantro")
    candidate = _candidate(db, "Alex doesn't like cilantro anymore")

    db.refresh(old)
    assert candidate.proposed_action == "replace"
    assert candidate.proposed_target_id == old.id
    # Nothing about the confirmed fact changed yet.
    assert old.status == MemoryStatus.CONFIRMED.value
    assert old.superseded_at is None
    assert old.conflict_group == candidate.conflict_group


def test_confirming_proposal_retires_old_fact_without_deleting(db: Session):
    old = _confirmed(db, "Alex dislikes cilantro")
    candidate = _candidate(db, "Alex doesn't like cilantro anymore")

    memory_svc.confirm_memory(db, candidate.id)

    db.refresh(old)
    db.refresh(candidate)
    assert candidate.status == MemoryStatus.CONFIRMED.value
    assert candidate.proposed_action is None
    assert candidate.proposed_target_id is None
    assert old.status == MemoryStatus.SUPERSEDED.value
    assert old.superseded_at is not None
    assert old.supersedes_id == candidate.id
    assert old.content == "Alex dislikes cilantro"  # history preserved
    assert db.get(Memory, old.id) is not None


def test_rejecting_proposal_leaves_current_fact_untouched(db: Session):
    old = _confirmed(db, "Alex dislikes cilantro")
    candidate = _candidate(db, "Alex doesn't like cilantro anymore")

    memory_svc.reject_memory(db, candidate.id)

    db.refresh(old)
    db.refresh(candidate)
    assert candidate.status == MemoryStatus.REJECTED.value
    assert candidate.proposed_action is None
    assert old.status == MemoryStatus.CONFIRMED.value
    assert old.superseded_at is None
    assert old.supersedes_id is None


def test_superseded_memory_leaves_retrieval_but_remains_queryable(db: Session):
    old = _confirmed(db, "Alex used to not handle spicy food well", memory_type="preference")
    candidate = _candidate(db, "Alex can handle medium spicy now", memory_type="preference")
    memory_svc.confirm_memory(db, candidate.id)

    hits = hybrid_search(db, "Can Alex handle spicy food now?", include_messages=False, limit=10)
    contents = " ".join(hit.content for hit in hits)
    assert "medium spicy" in contents
    assert "used to not handle spicy" not in contents
    assert db.get(Memory, old.id).status == MemoryStatus.SUPERSEDED.value


def test_candidate_groups_show_replacement_pair(db: Session):
    old = _confirmed(db, "Alex dislikes cilantro")
    candidate = _candidate(db, "Alex doesn't like cilantro anymore")

    groups = memory_svc.candidate_groups(db, memory_svc.list_candidates(db))
    replacement = [g for g in groups if g["kind"] in ("conflict", "replacement")]
    assert replacement
    items = [item for g in replacement for item in g["items"]]
    assert any(item["candidate"].id == candidate.id and item["target"].id == old.id for item in items)


def test_diary_extraction_proposes_instead_of_overwriting(db: Session):
    old = _confirmed(db, "Alex dislikes cilantro")
    entry = diary_svc.create_diary(db, "Alex doesn't like cilantro anymore", subject_name="Alex")
    _entry, created = diary_svc.extract_diary_memories(db, entry.id)

    db.refresh(old)
    assert created
    proposed = [m for m in created if m.proposed_action == "replace"]
    assert proposed
    assert proposed[0].proposed_target_id == old.id
    assert old.status == MemoryStatus.CONFIRMED.value


def test_explicit_correction_records_invalidation_time(db: Session):
    old = _confirmed(db, "Alex dislikes cilantro")
    result = distill_user_message(db, "Correction: Alex doesn't like cilantro anymore")

    db.refresh(old)
    assert result.corrected_count >= 1
    assert old.status == MemoryStatus.SUPERSEDED.value
    assert old.superseded_at is not None
    replacement = db.get(Memory, old.supersedes_id)
    assert replacement is not None
    assert replacement.status == MemoryStatus.CONFIRMED.value


def test_memory_detail_shows_provenance(client: TestClient, db: Session):
    memory = _confirmed(db, "Alex dislikes cilantro")
    page = client.get(f"/memories/{memory.id}")
    assert page.status_code == 200
    assert "System learned at" in page.text
    assert "Validity" in page.text
    assert "Superseded at" in page.text
    assert "still valid" in page.text


def test_memory_detail_shows_replacement_chain(client: TestClient, db: Session):
    old = _confirmed(db, "Alex dislikes cilantro")
    candidate = _candidate(db, "Alex doesn't like cilantro anymore")
    memory_svc.confirm_memory(db, candidate.id)

    old_page = client.get(f"/memories/{old.id}")
    assert "was superseded by" in old_page.text
    assert f"/memories/{candidate.id}" in old_page.text

    new_page = client.get(f"/memories/{candidate.id}")
    assert f"/memories/{old.id}" in new_page.text


def test_candidates_page_groups_conflicts(client: TestClient, db: Session):
    old = _confirmed(db, "Alex dislikes cilantro")
    _candidate(db, "Alex doesn't like cilantro anymore")
    page = client.get("/memories/candidates")
    assert page.status_code == 200
    assert "Conflict group" in page.text
    assert "Suggested replacement" in page.text
    assert "New fact (pending)" in page.text
    assert old.content in page.text
