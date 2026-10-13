from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import (
    AnalysisRun,
    Memory,
    MemoryEvidence,
    MemoryStatus,
    MemoryType,
    Message,
    MessageChunk,
)
from app.llm.client import LLMClient, LLMError
from app.services.memory import propose_replacement
from app.services.redaction import redact_text

logger = logging.getLogger(__name__)

VALID_TYPES = {t.value for t in MemoryType}

EXTRACTION_SYSTEM = """You are a cautious personal-profile memory extractor.
Task: from redacted chat excerpts, extract structured candidate memories about the "portrait subject" (subject).

Hard rules:
1. Only extract content supported by chat evidence; if evidence is insufficient, output nothing.
2. Distinguish facts from inferences: inferences must be marked is_inference=true and given lower confidence.
3. Do not output emotional diagnoses, mental-illness labels, or manipulative advice.
4. Do not mistake the "partner's" preferences for the "subject's" preferences.
5. Every memory must be short, verifiable, and reviewable by a human.
6. Output strictly JSON in this format:
{
  "memories": [
    {
      "content": "short statement",
      "memory_type": "fact|preference|aversion|person_place|habit|event|communication|plan",
      "confidence": 0.0-1.0,
      "is_inference": false,
      "evidence_indices": [0, 3],
      "counterexample_indices": [],
      "valid_from": null,
      "valid_to": null
    }
  ]
}
If there is nothing to extract, return {"memories": []}.
Write each memory's content in the same language as the chat excerpt.
"""


def build_chunks(
    db: Session,
    import_id: int,
    *,
    settings: Settings | None = None,
) -> list[MessageChunk]:
    settings = settings or get_settings()
    messages = (
        db.query(Message)
        .filter(Message.import_id == import_id)
        .order_by(Message.sent_at.asc().nullslast(), Message.id.asc())
        .all()
    )
    if not messages:
        return []

    chunks: list[MessageChunk] = []
    buf: list[Message] = []
    buf_chars = 0

    def flush():
        nonlocal buf, buf_chars
        if not buf:
            return
        lines = []
        for i, m in enumerate(buf):
            ts = m.sent_at.strftime("%Y-%m-%d %H:%M") if m.sent_at else "?"
            role = m.sender_role or "unknown"
            redacted = redact_text(m.content, enabled=settings.redaction_enabled)
            lines.append(f"[{i}] ({ts}) {m.sender}/{role}: {redacted.text}")
        text = "\n".join(lines)
        original = sum(len(m.content) for m in buf)
        chunk = MessageChunk(
            import_id=import_id,
            start_message_id=buf[0].id,
            end_message_id=buf[-1].id,
            message_ids=[m.id for m in buf],
            redacted_text=text,
            original_char_count=original,
            redacted_char_count=len(text),
        )
        db.add(chunk)
        chunks.append(chunk)
        buf = []
        buf_chars = 0

    for msg in messages:
        # Skip non-text heavy types for extraction body but keep text/system
        if msg.msg_type not in ("text", "system", "link") and len(msg.content) < 2:
            continue
        add_len = len(msg.content) + 40
        if buf and (
            len(buf) >= settings.chunk_max_messages
            or buf_chars + add_len > settings.chunk_max_chars
        ):
            flush()
        buf.append(msg)
        buf_chars += add_len
    flush()
    db.flush()
    return chunks


def run_extraction(
    db: Session,
    import_id: int,
    *,
    subject_name: str | None = None,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> AnalysisRun:
    settings = settings or get_settings()
    llm = llm or LLMClient(settings)

    run = AnalysisRun(
        import_id=import_id,
        status="running",
        model=llm.choice.api_model if llm.configured else None,
        started_at=datetime.utcnow(),
    )
    db.add(run)
    db.flush()

    try:
        chunks = build_chunks(db, import_id, settings=settings)
        run.chunk_total = len(chunks)
        db.flush()

        if not chunks:
            run.status = "completed"
            run.finished_at = datetime.utcnow()
            db.commit()
            return run

        if not llm.configured:
            # Offline mode: create no candidates, mark completed with note
            run.status = "completed_no_llm"
            run.error_message = "LLM_API_KEY is not configured; only chunking and redaction were done, no memories extracted"
            run.finished_at = datetime.utcnow()
            db.commit()
            return run

        created = 0
        for chunk in chunks:
            items = _extract_chunk(llm, chunk, subject_name=subject_name)
            msg_id_list = list(chunk.message_ids or [])
            for item in items:
                memory = _item_to_memory(item, run.id, chunk.id, subject_name)
                db.add(memory)
                db.flush()
                _attach_evidence(db, memory, item, msg_id_list)
                # Inferred facts never rewrite a confirmed one on their own:
                # record a reviewable proposal instead.
                propose_replacement(db, memory)
                created += 1
            run.chunk_done += 1
            run.candidates_created = created
            # Commit each chunk so long imports survive interruption and
            # other readers (web UI) can observe progress.
            db.commit()

        run.status = "completed"
        run.finished_at = datetime.utcnow()
        db.commit()
        return run
    except Exception as exc:  # noqa: BLE001
        logger.exception("extraction failed")
        run.status = "failed"
        run.error_message = str(exc)
        run.finished_at = datetime.utcnow()
        db.commit()
        return run


def _extract_chunk(
    llm: LLMClient, chunk: MessageChunk, subject_name: str | None
) -> list[dict[str, Any]]:
    subject = subject_name or "Subject"
    user = (
        f"Subject name/nickname: {subject}\n"
        f"Below is a redacted chat excerpt (the bracketed numbers are evidence indices):\n\n"
        f"{chunk.redacted_text}\n\n"
        f"Extract candidate memories about \"{subject}\"."
    )
    try:
        data = llm.chat_json(
            EXTRACTION_SYSTEM, user, purpose="extract", redacted=True
        )
    except (LLMError, json.JSONDecodeError, ValueError, TypeError) as exc:
        # One bad model response should not abort a multi-thousand-chunk run.
        logger.warning("chunk %s extraction skipped: %s", chunk.id, exc)
        return []
    memories = data.get("memories") or data.get("items") or []
    if not isinstance(memories, list):
        return []
    cleaned = []
    for m in memories:
        if not isinstance(m, dict):
            continue
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        mtype = str(m.get("memory_type") or "fact").strip()
        if mtype not in VALID_TYPES:
            mtype = "fact"
        conf = float(m.get("confidence") or 0.5)
        conf = max(0.0, min(1.0, conf))
        cleaned.append(
            {
                "content": content,
                "memory_type": mtype,
                "confidence": conf,
                "is_inference": bool(m.get("is_inference")),
                "evidence_indices": m.get("evidence_indices") or [],
                "counterexample_indices": m.get("counterexample_indices") or [],
                "valid_from": m.get("valid_from"),
                "valid_to": m.get("valid_to"),
            }
        )
    return cleaned


def _item_to_memory(
    item: dict[str, Any],
    run_id: int,
    chunk_id: int,
    subject_name: str | None,
) -> Memory:
    return Memory(
        content=item["content"],
        memory_type=item["memory_type"],
        status=MemoryStatus.CANDIDATE.value,
        confidence=item["confidence"],
        subject=subject_name,
        is_inference=item["is_inference"],
        source_run_id=run_id,
        source_chunk_id=chunk_id,
    )


def _attach_evidence(
    db: Session,
    memory: Memory,
    item: dict[str, Any],
    msg_id_list: list[int],
) -> None:
    for idx in item.get("evidence_indices") or []:
        if not isinstance(idx, int) or idx < 0 or idx >= len(msg_id_list):
            continue
        mid = msg_id_list[idx]
        msg = db.get(Message, mid)
        quote = (msg.content[:200] if msg else None)
        db.add(
            MemoryEvidence(
                memory_id=memory.id,
                message_id=mid,
                is_counterexample=False,
                quote=quote,
            )
        )
    for idx in item.get("counterexample_indices") or []:
        if not isinstance(idx, int) or idx < 0 or idx >= len(msg_id_list):
            continue
        mid = msg_id_list[idx]
        msg = db.get(Message, mid)
        quote = (msg.content[:200] if msg else None)
        db.add(
            MemoryEvidence(
                memory_id=memory.id,
                message_id=mid,
                is_counterexample=True,
                quote=quote,
            )
        )
