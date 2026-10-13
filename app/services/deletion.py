"""Delete a single import without leaving dangling memories or evidence.

The product rule is "never destroy a fact the user confirmed", so deletion is
split in two:

* rows that only exist because of this import (messages, chunks, runs, and the
  candidates those runs produced) are removed;
* confirmed memories that the user approved are kept, but the evidence that
  pointed at those messages is gone — the preview says so before it happens, so
  the user can decide.
"""

from __future__ import annotations

from sqlalchemy import func, or_
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

SAMPLE_LIMIT = 5


def import_impact(db: Session, import_id: int) -> dict:
    """Counts and samples shown *before* deleting an import."""
    job = db.get(ImportJob, import_id)
    if job is None:
        raise ValueError("Import job not found")

    message_ids = [
        row[0]
        for row in db.query(Message.id).filter(Message.import_id == import_id).all()
    ]
    run_ids = [
        row[0]
        for row in db.query(AnalysisRun.id).filter(AnalysisRun.import_id == import_id).all()
    ]
    chunk_ids = [
        row[0]
        for row in db.query(MessageChunk.id).filter(MessageChunk.import_id == import_id).all()
    ]

    derived = _derived_memories(db, run_ids, chunk_ids)
    derived_ids = [m.id for m in derived]
    evidence_rows = (
        int(
            db.query(func.count(MemoryEvidence.id))
            .filter(MemoryEvidence.message_id.in_(message_ids))
            .scalar()
            or 0
        )
        if message_ids
        else 0
    )
    orphaned = _orphaned_memories(db, message_ids, derived_ids)
    return {
        "job": job,
        "job_id": job.id,
        "filename": job.filename,
        "message_count": len(message_ids),
        "chunk_count": len(chunk_ids),
        "run_count": len(run_ids),
        "derived_candidates": derived,
        "derived_count": len(derived),
        "evidence_count": evidence_rows,
        "orphaned": orphaned,
        "orphaned_count": len(orphaned),
        "sample": [m.content for m in orphaned[:SAMPLE_LIMIT]],
    }


def delete_import(db: Session, import_id: int) -> dict:
    """Remove one import and everything that only depended on it."""
    impact = import_impact(db, import_id)
    message_ids = [
        row[0]
        for row in db.query(Message.id).filter(Message.import_id == import_id).all()
    ]
    run_ids = [
        row[0]
        for row in db.query(AnalysisRun.id).filter(AnalysisRun.import_id == import_id).all()
    ]
    chunk_ids = [
        row[0]
        for row in db.query(MessageChunk.id).filter(MessageChunk.import_id == import_id).all()
    ]
    derived_ids = [m.id for m in impact["derived_candidates"]]

    # Evidence first: it references both memories and messages.
    if message_ids:
        db.query(MemoryEvidence).filter(
            MemoryEvidence.message_id.in_(message_ids)
        ).delete(synchronize_session=False)
    if derived_ids:
        db.query(MemoryEvidence).filter(
            MemoryEvidence.memory_id.in_(derived_ids)
        ).delete(synchronize_session=False)

    # Replacement chains and pending proposals must not point at deleted rows.
    if derived_ids:
        db.query(Memory).filter(Memory.supersedes_id.in_(derived_ids)).update(
            {Memory.supersedes_id: None}, synchronize_session=False
        )
        db.query(Memory).filter(Memory.proposed_target_id.in_(derived_ids)).update(
            {Memory.proposed_target_id: None}, synchronize_session=False
        )
        db.query(Memory).filter(Memory.id.in_(derived_ids)).delete(
            synchronize_session=False
        )

    if chunk_ids:
        db.query(Memory).filter(Memory.source_chunk_id.in_(chunk_ids)).update(
            {Memory.source_chunk_id: None}, synchronize_session=False
        )

    chunk_count = (
        db.query(MessageChunk)
        .filter(MessageChunk.import_id == import_id)
        .delete(synchronize_session=False)
    )
    message_count = (
        db.query(Message)
        .filter(Message.import_id == import_id)
        .delete(synchronize_session=False)
    )
    run_count = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.import_id == import_id)
        .delete(synchronize_session=False)
    )
    db.query(ImportJob).filter(ImportJob.id == import_id).delete(synchronize_session=False)
    db.commit()

    return {
        "import_id": import_id,
        "filename": impact["filename"],
        "messages": message_count,
        "chunks": chunk_count,
        "runs": run_count,
        "candidates": len(derived_ids),
        "evidence": impact["evidence_count"],
        "orphaned": impact["orphaned_count"],
        "orphaned_sample": impact["sample"],
    }


def _derived_memories(db: Session, run_ids: list[int], chunk_ids: list[int]) -> list[Memory]:
    if not run_ids and not chunk_ids:
        return []
    query = db.query(Memory)
    if run_ids and chunk_ids:
        query = query.filter(
            or_(Memory.source_run_id.in_(run_ids), Memory.source_chunk_id.in_(chunk_ids))
        )
    elif run_ids:
        query = query.filter(Memory.source_run_id.in_(run_ids))
    else:
        query = query.filter(Memory.source_chunk_id.in_(chunk_ids))
    return query.all()


def _orphaned_memories(
    db: Session, message_ids: list[int], derived_ids: list[int]
) -> list[Memory]:
    """Confirmed memories that would lose all their evidence."""
    if not message_ids:
        return []
    candidates = (
        db.query(Memory)
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .all()
    )
    orphaned = []
    for memory in candidates:
        if memory.id in derived_ids:
            continue
        links = [e.message_id for e in memory.evidence]
        if links and all(message_id in message_ids for message_id in links):
            orphaned.append(memory)
    return orphaned
