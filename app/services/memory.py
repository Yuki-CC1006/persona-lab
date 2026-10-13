from __future__ import annotations

import hashlib
import re
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.db.models import Memory, MemoryEvidence, MemoryStatus, Message
from app.services.retrieval import cosine, local_embed


def list_candidates(db: Session, *, limit: int = 100, offset: int = 0) -> list[Memory]:
    return (
        db.query(Memory)
        .options(joinedload(Memory.evidence))
        .filter(Memory.status == MemoryStatus.CANDIDATE.value)
        .order_by(Memory.confidence.desc(), Memory.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


SORT_COLUMNS = {
    "id": Memory.id,
    "type": Memory.memory_type,
    "status": Memory.status,
    "confidence": Memory.confidence,
    "content": Memory.content,
    "conflict": Memory.conflict_group,
    "updated": Memory.updated_at,
}


def _filter_memories(
    query,
    *,
    status: str | None,
    memory_type: str | None,
    q: str | None,
):
    if status:
        query = query.filter(Memory.status == status)
    if memory_type:
        query = query.filter(Memory.memory_type == memory_type)
    if q:
        query = query.filter(Memory.content.contains(q))
    return query


def count_memories(
    db: Session,
    *,
    status: str | None = MemoryStatus.CONFIRMED.value,
    memory_type: str | None = None,
    q: str | None = None,
) -> int:
    query = _filter_memories(
        db.query(func.count(Memory.id)),
        status=status,
        memory_type=memory_type,
        q=q,
    )
    return int(query.scalar() or 0)


def list_memories(
    db: Session,
    *,
    status: str | None = MemoryStatus.CONFIRMED.value,
    memory_type: str | None = None,
    q: str | None = None,
    limit: int = 100,
    offset: int = 0,
    sort: str = "updated",
    direction: str = "desc",
) -> list[Memory]:
    query = _filter_memories(
        db.query(Memory).options(joinedload(Memory.evidence)),
        status=status,
        memory_type=memory_type,
        q=q,
    )
    col = SORT_COLUMNS.get(sort, Memory.updated_at)
    order = col.asc() if direction == "asc" else col.desc()
    return (
        query.order_by(order, Memory.id.desc())
        .offset(max(0, offset))
        .limit(limit)
        .all()
    )


def get_memory(db: Session, memory_id: int) -> Memory | None:
    return (
        db.query(Memory)
        .options(joinedload(Memory.evidence).joinedload(MemoryEvidence.message))
        .filter(Memory.id == memory_id)
        .first()
    )


def confirm_memory(
    db: Session,
    memory_id: int,
    *,
    content: str | None = None,
    memory_type: str | None = None,
    notes: str | None = None,
) -> Memory:
    memory = db.get(Memory, memory_id)
    if not memory:
        raise ValueError("Memory not found")
    if content is not None:
        memory.content = content.strip()
    if memory_type is not None:
        memory.memory_type = memory_type
    if notes is not None:
        memory.notes = notes
    memory.status = MemoryStatus.CONFIRMED.value
    memory.reviewed_at = datetime.utcnow()
    memory.embedding = local_embed(memory.content)
    keeper = _merge_if_duplicate(db, memory)
    if keeper.id != memory.id:
        db.commit()
        db.refresh(keeper)
        return keeper
    _apply_proposal(db, keeper)
    _detect_conflicts(db, keeper)
    db.commit()
    db.refresh(keeper)
    return keeper


def reject_memory(db: Session, memory_id: int, notes: str | None = None) -> Memory:
    memory = db.get(Memory, memory_id)
    if not memory:
        raise ValueError("Memory not found")
    memory.status = MemoryStatus.REJECTED.value
    memory.reviewed_at = datetime.utcnow()
    # Rejecting a proposal must never touch the fact it wanted to replace.
    memory.proposed_action = None
    memory.proposed_target_id = None
    if notes:
        memory.notes = notes
    db.commit()
    db.refresh(memory)
    return memory


def supersede_memory(
    db: Session, old: Memory, new_id: int | None, *, note: str = ""
) -> Memory:
    """Retire a memory without deleting it, keeping the replacement chain."""
    old.status = MemoryStatus.SUPERSEDED.value
    old.superseded_at = datetime.utcnow()
    if new_id is not None:
        old.supersedes_id = new_id
    if note:
        old.notes = ((old.notes or "") + f" · {note}").strip(" ·")
    db.flush()
    return old


def propose_replacement(db: Session, memory: Memory) -> Memory | None:
    """Record a reviewable replacement proposal instead of rewriting history.

    Inferred candidates (import extraction, diary, model output) never
    silently overwrite a confirmed fact. They are marked with the memory they
    would replace so a human can compare both side by side.

    A proposal needs both a topical signal (surface overlap or embedding
    similarity) and a contradiction cue. Two facts that merely share a topic
    ("喜欢无糖美式" / "喜欢热豆浆") are not conflicts.
    """
    key = _normalize_key(memory.content)
    if len(key) < 4:
        return None
    others = (
        db.query(Memory)
        .filter(
            Memory.status == MemoryStatus.CONFIRMED.value,
            Memory.memory_type == memory.memory_type,
        )
        .all()
    )
    best: Memory | None = None
    best_signal = 0.0
    for other in others:
        other_key = _normalize_key(other.content)
        if not other_key or other.content.strip() == memory.content.strip():
            continue
        if not _has_contradiction_cue(memory.content, other.content):
            continue
        signal = _topical_signal(memory.content, other.content)
        if signal > best_signal:
            best, best_signal = other, signal
    if not best or best_signal < CONFLICT_SIGNAL_THRESHOLD:
        return None
    memory.proposed_action = "replace"
    memory.proposed_target_id = best.id
    if not memory.conflict_group:
        memory.conflict_group = best.conflict_group or _conflict_group_id(
            memory.memory_type, key
        )
    best.conflict_group = memory.conflict_group
    db.flush()
    return best


def replaced_memories(db: Session, memory: Memory) -> list[Memory]:
    """Memories this one replaced — the other half of the replacement chain."""
    return (
        db.query(Memory)
        .filter(Memory.supersedes_id == memory.id)
        .order_by(Memory.id.asc())
        .all()
    )


def replacement_target(db: Session, memory: Memory) -> Memory | None:
    if not memory.proposed_target_id:
        return None
    return db.get(Memory, memory.proposed_target_id)


def superseded_by(db: Session, memory: Memory) -> Memory | None:
    """Return the memory that replaced this one, if any."""
    if not memory.supersedes_id:
        return None
    return db.get(Memory, memory.supersedes_id)


def candidate_groups(db: Session, candidates: list[Memory]) -> list[dict]:
    """Group candidates for review: new fact vs current fact vs evidence."""
    groups: list[dict] = []
    by_group: dict[str, dict] = {}
    loose: list[dict] = []

    for candidate in candidates:
        target = replacement_target(db, candidate)
        entry = {"candidate": candidate, "target": target, "kind": "single"}
        if target:
            entry["kind"] = "replacement"
        key = candidate.conflict_group
        if key:
            bucket = by_group.get(key)
            if bucket is None:
                bucket = {"key": key, "items": []}
                by_group[key] = bucket
            bucket["items"].append(entry)
        else:
            loose.append(entry)

    for bucket in by_group.values():
        bucket["replacement_count"] = sum(
            1 for item in bucket["items"] if item["kind"] == "replacement"
        )
        groups.append({"kind": "conflict", **bucket})
    for entry in loose:
        groups.append(
            {"kind": "replacement" if entry["kind"] == "replacement" else "single", "items": [entry]}
        )
    groups.sort(key=lambda g: (g["kind"] == "conflict", g["kind"] == "replacement"), reverse=True)
    return groups


def _conflict_group_id(memory_type: str, key: str) -> str:
    return hashlib.md5(  # noqa: S324 — non-crypto grouping id
        f"{memory_type}:{key[:20]}".encode()
    ).hexdigest()[:12]


# Topical signal (surface overlap or embedding similarity) required before a
# conflict is worth a human's attention.
CONFLICT_SIGNAL_THRESHOLD = 0.45

# Wording that signals a fact may have changed rather than being added to.
CHANGE_PAT = re.compile(
    r"现在|以前|不再|已经|改成|变成|开始|后来|如今|目前|以后|之前|"
    r"\b(?:now|used to|no longer|anymore|already|changed|became|becomes|"
    r"started|started to|previously|lately)\b",
    re.IGNORECASE,
)
# Negation that flips the polarity of otherwise identical statements.
NEGATION_PAT = re.compile(
    r"不|没|别|讨厌|受不了|拒绝|忌讳|雷区|"
    r"\b(?:not|no|never|don'?t|doesn'?t|didn'?t|dislike|dislikes|hate|hates|"
    r"refuse|refuses|avoid|avoids|can'?t stand)\b",
    re.IGNORECASE,
)


def _has_contradiction_cue(a: str, b: str) -> bool:
    """Cheap offline check: does either side signal a change or a polarity flip?"""
    if CHANGE_PAT.search(a or "") or CHANGE_PAT.search(b or ""):
        return True
    return bool(NEGATION_PAT.search(a or "")) != bool(NEGATION_PAT.search(b or ""))


def _apply_proposal(db: Session, memory: Memory) -> None:
    """After the human confirms a proposal, retire the old fact in place."""
    if memory.proposed_action != "replace" or not memory.proposed_target_id:
        return
    target = db.get(Memory, memory.proposed_target_id)
    memory.proposed_action = None
    memory.proposed_target_id = None
    if not target or target.id == memory.id:
        return
    if target.status != MemoryStatus.CONFIRMED.value:
        return
    supersede_memory(db, target, memory.id, note="replaced by a newer fact (reviewed)")


def bulk_confirm_memories(db: Session, memory_ids: list[int]) -> int:
    """Confirm many candidates in one transaction. Returns count confirmed."""
    if not memory_ids:
        return 0
    memories = (
        db.query(Memory)
        .filter(
            Memory.id.in_(memory_ids),
            Memory.status == MemoryStatus.CANDIDATE.value,
        )
        .all()
    )
    count = 0
    for memory in memories:
        memory.status = MemoryStatus.CONFIRMED.value
        memory.reviewed_at = datetime.utcnow()
        memory.embedding = local_embed(memory.content)
        keeper = _merge_if_duplicate(db, memory)
        if keeper.id != memory.id:
            continue
        _apply_proposal(db, keeper)
        _detect_conflicts(db, keeper)
        count += 1
    db.commit()
    return count


def bulk_reject_memories(db: Session, memory_ids: list[int]) -> int:
    if not memory_ids:
        return 0
    memories = (
        db.query(Memory)
        .filter(
            Memory.id.in_(memory_ids),
            Memory.status == MemoryStatus.CANDIDATE.value,
        )
        .all()
    )
    count = 0
    now = datetime.utcnow()
    for memory in memories:
        memory.status = MemoryStatus.REJECTED.value
        memory.reviewed_at = now
        count += 1
    db.commit()
    return count


def confirm_all_candidates(
    db: Session, *, min_confidence: float | None = None, limit: int = 500
) -> int:
    query = db.query(Memory).filter(Memory.status == MemoryStatus.CANDIDATE.value)
    if min_confidence is not None:
        query = query.filter(Memory.confidence >= min_confidence)
    ids = [m.id for m in query.order_by(Memory.confidence.desc()).limit(limit).all()]
    return bulk_confirm_memories(db, ids)


def update_memory(
    db: Session,
    memory_id: int,
    *,
    content: str | None = None,
    memory_type: str | None = None,
    status: str | None = None,
    notes: str | None = None,
    confidence: float | None = None,
) -> Memory:
    memory = db.get(Memory, memory_id)
    if not memory:
        raise ValueError("Memory not found")
    if content is not None:
        memory.content = content.strip()
        memory.embedding = local_embed(memory.content)
    if memory_type is not None:
        memory.memory_type = memory_type
    if status is not None:
        memory.status = status
    if notes is not None:
        memory.notes = notes
    if confidence is not None:
        memory.confidence = max(0.0, min(1.0, confidence))
    memory.updated_at = datetime.utcnow()
    if memory.status == MemoryStatus.CONFIRMED.value:
        _detect_conflicts(db, memory)
    db.commit()
    db.refresh(memory)
    return memory


def delete_memory(db: Session, memory_id: int) -> None:
    memory = db.get(Memory, memory_id)
    if not memory:
        raise ValueError("Memory not found")
    db.delete(memory)
    db.commit()


def create_manual_memory(
    db: Session,
    content: str,
    *,
    memory_type: str | None = None,
    confidence: float | None = None,
    subject: str | None = None,
    evidence_message_ids: list[int] | None = None,
    auto_classify: bool = True,
) -> Memory:
    text = content.strip()
    chosen_type = (memory_type or "").strip()
    chosen_subject = (subject or "").strip() or None
    chosen_confidence = confidence
    classify_meta = None

    if auto_classify and (not chosen_type or chosen_type == "auto"):
        from app.services.classify import classify_memory_text

        result = classify_memory_text(text, subject_hint=chosen_subject)
        chosen_type = result.memory_type
        if not chosen_subject:
            chosen_subject = result.subject
        if chosen_confidence is None:
            chosen_confidence = max(0.7, result.confidence)
        classify_meta = {
            "source": result.source,
            "reason": result.reason,
            "confidence": result.confidence,
        }

    if not chosen_type:
        chosen_type = MemoryType.FACT.value
    if chosen_confidence is None:
        chosen_confidence = 1.0

    existing = _find_duplicate(
        db,
        content=text,
        memory_type=chosen_type,
        statuses=(MemoryStatus.CONFIRMED.value,),
    )
    if existing:
        if chosen_confidence > (existing.confidence or 0):
            existing.confidence = chosen_confidence
        if chosen_subject and not existing.subject:
            existing.subject = chosen_subject
        existing.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(existing)
        return existing

    notes = None
    if classify_meta:
        notes = f"Auto-classified [{classify_meta['source']}]: {classify_meta['reason']}"

    memory = Memory(
        content=text,
        memory_type=chosen_type,
        status=MemoryStatus.CONFIRMED.value,
        confidence=chosen_confidence,
        subject=chosen_subject,
        is_inference=False,
        notes=notes,
        reviewed_at=datetime.utcnow(),
        embedding=local_embed(text),
    )
    db.add(memory)
    db.flush()
    for mid in evidence_message_ids or []:
        msg = db.get(Message, mid)
        if not msg:
            continue
        db.add(
            MemoryEvidence(
                memory_id=memory.id,
                message_id=mid,
                is_counterexample=False,
                quote=msg.content[:200],
            )
        )
    _detect_conflicts(db, memory)
    db.commit()
    db.refresh(memory)
    return memory


def memory_stats(db: Session) -> dict:
    rows = (
        db.query(Memory.status, func.count(Memory.id))
        .group_by(Memory.status)
        .all()
    )
    by_status = {s: c for s, c in rows}
    type_rows = (
        db.query(Memory.memory_type, func.count(Memory.id))
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .group_by(Memory.memory_type)
        .all()
    )
    return {
        "by_status": by_status,
        "by_type": {t: c for t, c in type_rows},
        "confirmed": by_status.get(MemoryStatus.CONFIRMED.value, 0),
        "candidates": by_status.get(MemoryStatus.CANDIDATE.value, 0),
    }


def _normalize_key(content: str) -> str:
    text = re.sub(r"\s+", "", content.lower())
    text = re.sub(r"[，。！？、,.!?;；:：\"'“”‘’]", "", text)
    return text[:80]


def _find_duplicate(
    db: Session,
    *,
    content: str,
    memory_type: str,
    statuses: tuple[str, ...],
    exclude_id: int | None = None,
) -> Memory | None:
    key = _normalize_key(content)
    if len(key) < 4:
        return None
    query = db.query(Memory).filter(
        Memory.memory_type == memory_type,
        Memory.status.in_(statuses),
    )
    if exclude_id is not None:
        query = query.filter(Memory.id != exclude_id)
    for other in query.all():
        if _normalize_key(other.content) == key:
            return other
    return None


def _merge_if_duplicate(db: Session, memory: Memory) -> Memory:
    """If an equivalent confirmed memory exists, fold this one into it."""
    other = _find_duplicate(
        db,
        content=memory.content,
        memory_type=memory.memory_type,
        statuses=(MemoryStatus.CONFIRMED.value,),
        exclude_id=memory.id,
    )
    if not other:
        return memory
    keeper, duplicate = other, memory
    if (memory.confidence or 0) > (other.confidence or 0) or (
        memory.confidence == other.confidence and memory.id < other.id
    ):
        keeper, duplicate = memory, other
    _absorb_duplicate(db, keeper, duplicate)
    return keeper


def _absorb_duplicate(db: Session, keeper: Memory, duplicate: Memory) -> None:
    if (duplicate.confidence or 0) > (keeper.confidence or 0):
        keeper.confidence = duplicate.confidence
        keeper.content = duplicate.content
    if duplicate.subject and not keeper.subject:
        keeper.subject = duplicate.subject
    keeper_ids = {e.message_id for e in (keeper.evidence or [])}
    for ev in list(duplicate.evidence or []):
        if ev.message_id in keeper_ids:
            continue
        ev.memory_id = keeper.id
        keeper_ids.add(ev.message_id)
    duplicate.status = MemoryStatus.SUPERSEDED.value
    duplicate.superseded_at = datetime.utcnow()
    duplicate.supersedes_id = keeper.id


def _detect_conflicts(db: Session, memory: Memory) -> None:
    """Mark soft conflicts among confirmed memories with similar keys but different content."""
    key = _normalize_key(memory.content)
    if len(key) < 4:
        return

    # Find other confirmed memories sharing significant overlap
    others = (
        db.query(Memory)
        .filter(
            Memory.status == MemoryStatus.CONFIRMED.value,
            Memory.id != memory.id,
            Memory.memory_type == memory.memory_type,
        )
        .all()
    )
    group = None
    for other in others:
        other_key = _normalize_key(other.content)
        if not other_key:
            continue
        # Conflict heuristic: high token overlap but not identical
        if other.content.strip() == memory.content.strip():
            continue
        overlap = _jaccard(key, other_key)
        if overlap >= 0.45 and overlap < 0.95:
            group = other.conflict_group or memory.conflict_group
            if not group:
                group = _conflict_group_id(memory.memory_type, key)
            other.conflict_group = group
            memory.conflict_group = group


# Common English words carry no topical signal; comparing them would make any
# two English sentences look related.
_TOPIC_STOPWORDS = frozenset(
    "a an and are as at be been by can could did do does for from had has have he her him his "
    "i if in into is it its me my no not now of on or she so that the their them then there "
    "they this to too was we were will with would you your".split()
)


def _content_tokens(text: str) -> set[str]:
    """Meaningful word tokens; empty for text without Latin words (e.g. Chinese)."""
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w for w in words if len(w) > 1 and w not in _TOPIC_STOPWORDS}


def _topical_signal(a: str, b: str) -> float:
    """How strongly two statements are about the same thing, across scripts.

    Character-bigram overlap is dense for Chinese but sparse for English, where
    a single idea spans whole words. A containment ratio over content words
    keeps the signal comparable in both scripts.
    """
    signals = [
        _jaccard(_normalize_key(a), _normalize_key(b)),
        cosine(local_embed(a), local_embed(b)),
    ]
    ta, tb = _content_tokens(a), _content_tokens(b)
    if ta and tb:
        signals.append(len(ta & tb) / min(len(ta), len(tb)))
    return max(signals)


def _jaccard(a: str, b: str) -> float:
    # Character bigram Jaccard for CJK-friendly overlap
    def grams(s: str) -> set[str]:
        if len(s) < 2:
            return {s}
        return {s[i : i + 2] for i in range(len(s) - 1)}

    ga, gb = grams(a), grams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def dedupe_candidates(db: Session) -> int:
    """Merge exact-duplicate candidates; keep highest confidence."""
    return _dedupe_status(db, MemoryStatus.CANDIDATE.value)


def dedupe_confirmed(db: Session) -> int:
    """Merge exact-duplicate confirmed memories; keep highest confidence."""
    return _dedupe_status(db, MemoryStatus.CONFIRMED.value)


def _dedupe_status(db: Session, status: str) -> int:
    rows = (
        db.query(Memory)
        .filter(Memory.status == status)
        .order_by(Memory.confidence.desc(), Memory.id.asc())
        .all()
    )
    seen: dict[str, Memory] = {}
    removed = 0
    for mem in rows:
        key = f"{mem.memory_type}:{_normalize_key(mem.content)}"
        if key in seen:
            _absorb_duplicate(db, seen[key], mem)
            removed += 1
        else:
            seen[key] = mem
    db.commit()
    return removed
