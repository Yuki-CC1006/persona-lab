#!/usr/bin/env python3
"""One-off helper to resume an interrupted extraction run (import_id 1).

Usage: ``uv run python scripts/resume_extraction.py [subject_name]``
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone

from app.db.models import AnalysisRun, MessageChunk
from app.db.session import get_session_factory
from app.llm.client import LLMClient
from app.services.extraction import _attach_evidence, _extract_chunk, _item_to_memory


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def main() -> None:
    factory = get_session_factory()
    db = factory()

    for r in db.query(AnalysisRun).filter(AnalysisRun.status == "running").all():
        r.status = "failed"
        r.error_message = ((r.error_message or "") + " | interrupted; resumed later").strip(" |")
        r.finished_at = utcnow()
    db.commit()

    done = sum(
        int(r.chunk_done or 0)
        for r in db.query(AnalysisRun).filter(AnalysisRun.import_id == 1)
    )
    chunks = (
        db.query(MessageChunk)
        .filter(MessageChunk.import_id == 1)
        .order_by(MessageChunk.id.asc())
        .all()
    )
    skip = min(done, len(chunks))
    remaining = chunks[skip:]
    print(f"resume skip={skip} remaining={len(remaining)} total={len(chunks)}", flush=True)

    run = AnalysisRun(
        import_id=1,
        status="running",
        model="deepseek-chat",
        chunk_total=len(remaining),
        chunk_done=0,
        candidates_created=0,
        started_at=utcnow(),
    )
    db.add(run)
    db.commit()
    print(f"new_run_id={run.id}", flush=True)

    llm = LLMClient()
    subject = (sys.argv[1] if len(sys.argv) > 1 else "").strip() or "them"
    created = 0
    try:
        for i, chunk in enumerate(remaining, 1):
            items = _extract_chunk(llm, chunk, subject_name=subject)
            msg_ids = list(chunk.message_ids or [])
            for item in items:
                memory = _item_to_memory(item, run.id, chunk.id, subject)
                db.add(memory)
                db.flush()
                _attach_evidence(db, memory, item, msg_ids)
                created += 1
            run.chunk_done = i
            run.candidates_created = created
            db.commit()
            if i % 25 == 0 or i == len(remaining):
                print(f"progress {i}/{len(remaining)} candidates={created}", flush=True)
        run.status = "completed"
        run.finished_at = utcnow()
        db.commit()
        print(f"done run#{run.id} candidates={created}", flush=True)
    except Exception as exc:  # noqa: BLE001
        run.status = "failed"
        run.error_message = str(exc)
        run.finished_at = utcnow()
        db.commit()
        print(f"failed: {exc}", flush=True)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
