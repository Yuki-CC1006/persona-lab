from __future__ import annotations

import logging
import re
from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import DiaryEntry, Memory, MemoryStatus, MemoryType
from app.llm.client import LLMClient, LLMError
from app.services.memory import propose_replacement
from app.services.redaction import redact_text
from app.services.retrieval import local_embed

logger = logging.getLogger(__name__)

VALID_TYPES = {t.value for t in MemoryType}

DIARY_EXTRACTION_SYSTEM = """You are a cautious personal-profile memory extractor.
Task: from a "written note", extract structured candidate memories about the portrait subject.

Hard rules:
1. Only extract content supported by the source text; if evidence is insufficient, output nothing.
2. Distinguish facts from inferences: inferences must be marked is_inference=true and given lower confidence.
3. Do not output emotional diagnoses, mental-illness labels, or manipulative advice.
4. The note author's observations may be extracted if they clearly point to the subject; skip when unsure.
5. Every memory must be short, verifiable, and reviewable by a human.
6. Output strictly JSON:
{
  "memories": [
    {
      "content": "short statement",
      "memory_type": "fact|preference|aversion|person_place|habit|event|communication|plan",
      "confidence": 0.0-1.0,
      "is_inference": false,
      "quote": "the source sentence supporting this memory"
    }
  ]
}
If there is nothing to extract, return {"memories": []}.
Write each memory's content in the same language as the note.
"""


def list_diaries(db: Session, *, limit: int = 100, q: str | None = None) -> list[DiaryEntry]:
    query = db.query(DiaryEntry)
    if q:
        query = query.filter(
            or_(DiaryEntry.content.contains(q), DiaryEntry.title.contains(q))
        )
    return query.order_by(DiaryEntry.entry_date.desc(), DiaryEntry.id.desc()).limit(limit).all()


def get_diary(db: Session, diary_id: int) -> DiaryEntry | None:
    return db.get(DiaryEntry, diary_id)


def create_diary(
    db: Session,
    content: str,
    *,
    entry_date: datetime | None = None,
    title: str | None = None,
    mood: str | None = None,
    subject_name: str | None = None,
) -> DiaryEntry:
    entry = DiaryEntry(
        content=content.strip(),
        entry_date=entry_date or datetime.utcnow(),
        title=(title or "").strip() or None,
        mood=(mood or "").strip() or None,
        subject_name=(subject_name or "").strip() or None,
        status="draft",
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


def update_diary(
    db: Session,
    diary_id: int,
    *,
    content: str | None = None,
    title: str | None = None,
    mood: str | None = None,
    subject_name: str | None = None,
    entry_date: datetime | None = None,
) -> DiaryEntry:
    entry = db.get(DiaryEntry, diary_id)
    if not entry:
        raise ValueError("Written note not found")
    if content is not None:
        entry.content = content.strip()
    if title is not None:
        entry.title = title.strip() or None
    if mood is not None:
        entry.mood = mood.strip() or None
    if subject_name is not None:
        entry.subject_name = subject_name.strip() or None
    if entry_date is not None:
        entry.entry_date = entry_date
    entry.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(entry)
    return entry


def delete_diary(db: Session, diary_id: int) -> None:
    entry = db.get(DiaryEntry, diary_id)
    if not entry:
        raise ValueError("Written note not found")
    db.delete(entry)
    db.commit()


def extract_diary_memories(
    db: Session,
    diary_id: int,
    *,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> tuple[DiaryEntry, list[Memory]]:
    settings = settings or get_settings()
    llm = llm or LLMClient(settings)
    entry = db.get(DiaryEntry, diary_id)
    if not entry:
        raise ValueError("Written note not found")

    created: list[Memory] = []
    subject = entry.subject_name or "Subject"
    date_str = entry.entry_date.strftime("%Y-%m-%d") if entry.entry_date else "?"
    redacted = redact_text(entry.content, enabled=settings.redaction_enabled)

    if not llm.configured:
        # Offline fallback: split sentences and classify lightly
        from app.services.classify import classify_memory_text

        sentences = [
            s.strip()
            for s in re.split(r"[。！？\n]+", entry.content)
            if len(s.strip()) >= 6
        ]
        for s in sentences[:8]:
            result = classify_memory_text(s, subject_hint=entry.subject_name, use_llm=False)
            if result.confidence < 0.7:
                continue
            mem = Memory(
                content=s,
                memory_type=result.memory_type,
                status=MemoryStatus.CANDIDATE.value,
                confidence=result.confidence,
                subject=result.subject or entry.subject_name,
                is_inference=False,
                source_diary_id=entry.id,
                notes=f"From written note #{entry.id} ({date_str}) · offline rule distillation",
                embedding=local_embed(s),
            )
            db.add(mem)
            db.flush()
            propose_replacement(db, mem)
            created.append(mem)
        entry.status = "extracted_offline"
        entry.candidates_created = len(created)
        entry.last_extracted_at = datetime.utcnow()
        db.commit()
        for m in created:
            db.refresh(m)
        db.refresh(entry)
        return entry, created

    user = (
        f"Subject: {subject}\n"
        f"Note date: {date_str}\n"
        f"Title: {entry.title or 'none'}\n"
        f"Mood tag: {entry.mood or 'none'}\n\n"
        f"Redacted note body:\n{redacted.text}\n\n"
        f"Extract candidate memories about \"{subject}\"."
    )
    try:
        data = llm.chat_json(
            DIARY_EXTRACTION_SYSTEM,
            user,
            temperature=0.1,
            max_tokens=2500,
            purpose="diary_extract",
            redacted=True,
        )
    except LLMError as exc:
        entry.status = "extract_failed"
        db.commit()
        raise ValueError(f"AI extraction failed: {exc}") from exc

    items = data.get("memories") or []
    if not isinstance(items, list):
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        mtype = str(item.get("memory_type") or "fact").strip()
        if mtype not in VALID_TYPES:
            mtype = "fact"
        conf = float(item.get("confidence") or 0.6)
        conf = max(0.0, min(1.0, conf))
        quote = str(item.get("quote") or "").strip()[:200]
        note = f"From written note #{entry.id} ({date_str})"
        if quote:
            note += f" · quote: {quote}"
        mem = Memory(
            content=content,
            memory_type=mtype,
            status=MemoryStatus.CANDIDATE.value,
            confidence=conf,
            subject=entry.subject_name,
            is_inference=bool(item.get("is_inference")),
            source_diary_id=entry.id,
            notes=note,
            embedding=local_embed(content),
        )
        db.add(mem)
        db.flush()
        propose_replacement(db, mem)
        created.append(mem)

    entry.status = "extracted"
    entry.candidates_created = (entry.candidates_created or 0) + len(created)
    entry.last_extracted_at = datetime.utcnow()
    db.commit()
    for m in created:
        db.refresh(m)
    db.refresh(entry)
    return entry, created


def diaries_linked_to_memory(db: Session, memory: Memory) -> DiaryEntry | None:
    if not memory.source_diary_id:
        return None
    return db.get(DiaryEntry, memory.source_diary_id)
