"""Deleting a single import must not leave dangling memories or evidence."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import (
    AnalysisRun,
    ImportJob,
    Memory,
    MemoryEvidence,
    MemoryStatus,
    Message,
    MessageChunk,
)
from app.services import memory as memory_svc
from app.services.deletion import delete_import, import_impact


def _import_with_data(db: Session, *, candidates: int = 2) -> dict:
    job = ImportJob(
        filename="chat.txt",
        file_hash="hash-1",
        source_format="text_chat",
        status="completed",
        message_count=2,
        subject_name="Alex",
    )
    db.add(job)
    db.commit()

    run = AnalysisRun(import_id=job.id, status="completed", model="deepseek-v4-flash")
    db.add(run)
    db.commit()

    chunk = MessageChunk(
        import_id=job.id,
        start_message_id=0,
        end_message_id=0,
        message_ids=[],
        redacted_text="Alex: dislikes cilantro",
    )
    db.add(chunk)
    db.commit()

    messages = [
        Message(import_id=job.id, fingerprint=f"fp{i}", sender="Alex", content=text)
        for i, text in enumerate(["dislikes cilantro", "likes unsweetened Americano"])
    ]
    db.add_all(messages)
    db.commit()

    memory = memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion", subject="Alex")
    db.add(
        MemoryEvidence(
            memory_id=memory.id,
            message_id=messages[0].id,
            quote="dislikes cilantro",
        )
    )
    derived = []
    for i in range(candidates):
        candidate = Memory(
            content=f"candidate extracted from the import {i}",
            memory_type="fact",
            status=MemoryStatus.CANDIDATE.value,
            source_run_id=run.id,
            source_chunk_id=chunk.id,
        )
        db.add(candidate)
        derived.append(candidate)
    db.commit()
    return {
        "job": job,
        "run": run,
        "chunk": chunk,
        "messages": messages,
        "memory": memory,
        "derived": derived,
    }


def test_impact_previews_what_disappears(db: Session):
    data = _import_with_data(db)
    impact = import_impact(db, data["job"].id)
    assert impact["message_count"] == 2
    assert impact["chunk_count"] == 1
    assert impact["run_count"] == 1
    assert impact["derived_count"] == 2
    assert impact["evidence_count"] == 1
    assert impact["orphaned_count"] == 1
    assert impact["sample"] == ["dislikes cilantro"]


def test_delete_removes_import_scoped_rows(db: Session):
    data = _import_with_data(db)
    job_id = data["job"].id
    result = delete_import(db, job_id)
    assert result["messages"] == 2
    assert result["candidates"] == 2
    assert result["runs"] == 1
    assert result["filename"] == "chat.txt"

    assert db.query(ImportJob).count() == 0
    assert db.query(Message).count() == 0
    assert db.query(MessageChunk).count() == 0
    assert db.query(AnalysisRun).count() == 0
    assert db.query(MemoryEvidence).count() == 0


def test_delete_keeps_confirmed_memory_but_drops_its_stale_evidence(db: Session):
    data = _import_with_data(db)
    memory_id = data["memory"].id
    delete_import(db, data["job"].id)
    kept = db.get(Memory, memory_id)
    assert kept is not None
    assert kept.status == MemoryStatus.CONFIRMED.value
    assert kept.evidence == []


def test_delete_leaves_no_proposal_pointing_at_a_missing_memory(db: Session):
    data = _import_with_data(db, candidates=1)
    target = data["derived"][0]
    proposal = Memory(
        content="maybe they actually like cilantro",
        memory_type="aversion",
        status=MemoryStatus.CANDIDATE.value,
        proposed_action="replace",
        proposed_target_id=target.id,
    )
    db.add(proposal)
    chain_follower = Memory(
        content="a newer fact afterwards",
        memory_type="aversion",
        status=MemoryStatus.CONFIRMED.value,
        supersedes_id=target.id,
    )
    db.add(chain_follower)
    db.commit()
    proposal_id, follower_id = proposal.id, chain_follower.id

    delete_import(db, data["job"].id)
    assert db.get(Memory, proposal_id).proposed_target_id is None
    assert db.get(Memory, follower_id).supersedes_id is None


def test_delete_only_affects_the_chosen_import(db: Session):
    first = _import_with_data(db)
    other_job = ImportJob(filename="other.txt", file_hash="hash-2", source_format="text_chat")
    db.add(other_job)
    db.commit()
    db.add(Message(import_id=other_job.id, fingerprint="other-1", sender="Me", content="another record"))
    db.commit()
    other_id = other_job.id

    delete_import(db, first["job"].id)
    assert db.query(Message).filter(Message.import_id == other_id).count() == 1
    assert db.get(ImportJob, other_id) is not None


def test_delete_unknown_import_reports_error(db: Session):
    with pytest.raises(ValueError):
        import_impact(db, 9999)


def test_import_detail_shows_delete_preview(client: TestClient, db: Session):
    data = _import_with_data(db)
    page = client.get(f"/imports/{data['job'].id}")
    assert page.status_code == 200
    assert "Delete this import" in page.text
    assert "Confirmed memories kept but left without evidence" in page.text


def test_delete_through_the_web_reports_summary(client: TestClient, db: Session):
    data = _import_with_data(db)
    job_id = data["job"].id
    resp = client.post(
        f"/imports/{job_id}/delete", data={"confirm": "yes"}, follow_redirects=False
    )
    assert resp.status_code in (302, 303)
    location = resp.headers["location"]
    assert "deleted=" in location
    page = client.get(location)
    assert "Deleted import #" in page.text
    assert "lost their evidence references" in page.text
    assert db.query(ImportJob).count() == 0


def test_delete_requires_explicit_confirmation(client: TestClient, db: Session):
    data = _import_with_data(db)
    job_id = data["job"].id
    resp = client.post(f"/imports/{job_id}/delete", data={}, follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert db.query(ImportJob).count() == 1
    assert db.query(Message).count() == 2
