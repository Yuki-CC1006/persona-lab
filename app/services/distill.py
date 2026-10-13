from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryStatus, MemoryType
from app.llm.client import LLMClient, LLMError
from app.services.memory import (
    _detect_conflicts,
    _find_duplicate,
    _jaccard,
    _normalize_key,
    supersede_memory,
)
from app.services.retrieval import hybrid_search, local_embed

VALID_TYPES = {t.value for t in MemoryType}

TEACH_PAT = re.compile(
    r"记住|记一下|补充|其实|另外|告诉你|她还|他还|更新一下|"
    r"\b(?:remember|note that|note down|note this|by the way|actually|let me tell you|"
    r"update the profile|she also|he also|they also|for the record)\b",
    re.IGNORECASE,
)
CORRECT_PAT = re.compile(
    r"不对|不是这样|不是的|说错|搞错|记错|更正|纠正|"
    r"现在不|现在能|现在会|已经不|已经可以|不再|以前是|那是以前|"
    r"过时了|改了|别记|"
    # A leading "No" / "Nope" is the usual way to open a correction.
    r"^\s*(?:no|nope|wrong)\b[\s,，:：—–\-]*|"
    r"\b(?:not right|not correct|incorrect|that'?s wrong|you got (?:that|this|it) wrong|"
    r"got it wrong|i made a mistake|my mistake|correction|correct that|no longer|"
    r"not anymore|used to|that was before|outdated|it changed|don'?t note)\b|"
    # "she can drink matcha now" — a verb plus a trailing "now" marks a shift.
    r"\b(?:can|can'?t|could|is|isn'?t|does|doesn'?t|has|hasn'?t|will|won'?t|"
    r"likes?|loves?|drinks?|eats?|handles?|enjoys?|knows?|prefers?)\b[^.!?]{0,25}\bnow\b",
    re.IGNORECASE | re.MULTILINE,
)

DISTILL_SYSTEM = """You update the profile from the user's message. This is an automatic write: new facts go straight into the archive, corrections replace the old entry.
If the user is only asking a question and provides no new fact or correction, return an empty list.
Do not re-extract the assistant's reply.

Output JSON:
{"memories":[{
  "content":"short corrected statement",
  "memory_type":"fact|preference|aversion|person_place|habit|event|communication|plan",
  "confidence":0.0-1.0,
  "is_inference":false,
  "action":"add|replace|retract",
  "replace_id": null,
  "replaces":"original text of the old fact being replaced, or null"
}]}
action guide:
- add: a brand-new fact
- replace: correct an old memory; fill replace_id with the archive ID when possible
- retract: retire this old memory; content may be left empty, replace_id is required
When there is no new content: {"memories":[]}
Write each memory's content in the same language as the user's message.
"""


@dataclass
class DistillChange:
    memory: Memory
    action: str  # add | replace | retract
    replaced: str | None = None


@dataclass
class DistillResult:
    changes: list[DistillChange] = field(default_factory=list)

    @property
    def memories(self) -> list[Memory]:
        return [c.memory for c in self.changes if c.action != "retract"]

    @property
    def added_count(self) -> int:
        return sum(1 for c in self.changes if c.action == "add")

    @property
    def corrected_count(self) -> int:
        return sum(1 for c in self.changes if c.action in ("replace", "retract"))


def looks_like_teaching(text: str) -> bool:
    text = (text or "").strip()
    if len(text) < 6:
        return False
    # Only explicit teaching/correction should mutate the confirmed profile.
    # Treating every declarative chat message as a write is surprising and
    # contradicts the product's human-controlled memory model.
    return bool(TEACH_PAT.search(text) or CORRECT_PAT.search(text))


def looks_like_correction(text: str) -> bool:
    return bool(CORRECT_PAT.search(text or ""))


def distill_user_message(
    db: Session,
    text: str,
    *,
    thread_id: int | None = None,
    history: list[dict] | None = None,
    llm: LLMClient | None = None,
) -> DistillResult:
    raw = (text or "").strip()
    if not looks_like_teaching(raw) and not looks_like_correction(raw):
        return DistillResult()

    llm = llm or LLMClient()
    related_memories = _gather_related(db, raw, history=history)
    items = (
        _llm_items(llm, raw, history=history, related=related_memories)
        if llm.configured
        else _offline_items(raw, related_memories)
    )

    result = DistillResult()
    correcting = looks_like_correction(raw)
    for item in items:
        change = _apply_item(
            db, item, related_memories, thread_id, correcting=correcting
        )
        if change:
            result.changes.append(change)
    if result.changes:
        db.commit()
        for change in result.changes:
            if change.memory.id:
                db.refresh(change.memory)
    return result


def _gather_related(
    db: Session, text: str, *, history: list[dict] | None
) -> list[Memory]:
    queries = [text]
    if history:
        for item in reversed(history):
            if item.get("role") == "assistant" and (item.get("content") or "").strip():
                queries.append(item["content"][:240])
                break
    seen: set[int] = set()
    out: list[Memory] = []
    for q in queries:
        for h in hybrid_search(db, q, include_messages=False, limit=8):
            if getattr(h, "kind", "memory") != "memory" or h.id in seen:
                continue
            mem = db.get(Memory, h.id)
            if not mem or mem.status != MemoryStatus.CONFIRMED.value:
                continue
            seen.add(mem.id)
            out.append(mem)
    for mem in _overlap_confirmed(db, text, limit=8):
        if mem.id not in seen:
            seen.add(mem.id)
            out.append(mem)
    return out[:10]


def _overlap_confirmed(db: Session, text: str, *, limit: int) -> list[Memory]:
    key = _normalize_key(text)
    if len(key) < 2:
        return []
    scored: list[tuple[float, Memory]] = []
    rows = (
        db.query(Memory)
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .limit(300)
        .all()
    )
    for mem in rows:
        score = _jaccard(key, _normalize_key(mem.content))
        if score < 0.06:
            continue
        scored.append((score, mem))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:limit]]


def _apply_item(
    db: Session,
    item: dict,
    related: list[Memory],
    thread_id: int | None,
    *,
    correcting: bool,
) -> DistillChange | None:
    action = str(item.get("action") or "add").strip()
    mtype = str(item.get("memory_type") or "fact").strip()
    if mtype not in VALID_TYPES:
        mtype = "fact"
    content = str(item.get("content") or "").strip()
    replaces = str(item.get("replaces") or "").strip()
    old = _resolve_old(
        db, item.get("replace_id"), replaces or content, mtype, related
    )
    if not old and related and (action in ("replace", "retract") or correcting):
        old = related[0]

    if action == "retract":
        if not old:
            return None
        supersede_memory(db, old, None, note="voided in chat")
        return DistillChange(memory=old, action="retract", replaced=old.content)

    if len(content) < 4:
        return None

    should_replace = action == "replace" or correcting
    if should_replace and old:
        dup = _find_duplicate(
            db,
            content=content,
            memory_type=mtype,
            statuses=(MemoryStatus.CONFIRMED.value,),
            exclude_id=old.id,
        )
        if dup:
            supersede_memory(db, old, dup.id, note="corrected in chat")
            return DistillChange(
                memory=dup, action="replace", replaced=old.content
            )

        mem = _new_confirmed(
            db, content, mtype, item, thread_id, note_extra="auto-corrected in chat"
        )
        supersede_memory(db, old, mem.id, note="corrected in chat")
        return DistillChange(memory=mem, action="replace", replaced=old.content)

    dup = _find_duplicate(
        db,
        content=content,
        memory_type=mtype,
        statuses=(MemoryStatus.CONFIRMED.value,),
    )
    if dup:
        return None
    mem = _new_confirmed(db, content, mtype, item, thread_id, note_extra="auto-written to profile in chat")
    return DistillChange(memory=mem, action="add")


def _new_confirmed(
    db: Session,
    content: str,
    mtype: str,
    item: dict,
    thread_id: int | None,
    *,
    note_extra: str,
) -> Memory:
    conf = float(item.get("confidence") or 0.9)
    conf = max(0.7, min(1.0, conf))
    note = note_extra
    if thread_id:
        note += f" · chat #{thread_id}"
    mem = Memory(
        content=content,
        memory_type=mtype,
        status=MemoryStatus.CONFIRMED.value,
        confidence=conf,
        is_inference=bool(item.get("is_inference")),
        notes=note,
        reviewed_at=datetime.utcnow(),
        embedding=local_embed(content),
    )
    db.add(mem)
    db.flush()
    _detect_conflicts(db, mem)
    return mem


def _resolve_old(
    db: Session,
    replace_id,
    hint: str,
    memory_type: str,
    related: list[Memory],
) -> Memory | None:
    try:
        rid = int(replace_id) if replace_id is not None else None
    except (TypeError, ValueError):
        rid = None
    if rid:
        mem = db.get(Memory, rid)
        if mem and mem.status == MemoryStatus.CONFIRMED.value:
            return mem
    key = _normalize_key(hint)
    if len(key) < 4:
        return None
    pool = list(related)
    others = (
        db.query(Memory)
        .filter(
            Memory.status == MemoryStatus.CONFIRMED.value,
            Memory.memory_type == memory_type,
        )
        .all()
    )
    for mem in others:
        if mem not in pool:
            pool.append(mem)
    best = None
    best_score = 0.0
    for other in pool:
        score = _jaccard(key, _normalize_key(other.content))
        if score > best_score:
            best = other
            best_score = score
    threshold = 0.12 if related else 0.32
    if best and best_score >= threshold:
        return best
    return None


def _llm_items(
    llm: LLMClient,
    text: str,
    *,
    history: list[dict] | None,
    related: list[Memory],
) -> list[dict]:
    lines = [f"The user said: {text}"]
    if related:
        lines.append("\nPossibly related entries currently in the profile:")
        for m in related[:8]:
            lines.append(f"- ID={m.id} [{m.memory_type}] {m.content}")
    if history:
        lines.append("\nRecent conversation (the user may be correcting what the assistant just said):")
        for item in history[-6:]:
            role = "User" if item.get("role") == "user" else "Assistant"
            content = (item.get("content") or "").strip()[:300]
            if content:
                lines.append(f"{role}: {content}")
    lines.append("\nIf this is merely a question, return an empty list. If correcting the profile, prefer replace and include replace_id.")
    try:
        data = llm.chat_json(
            DISTILL_SYSTEM,
            "\n".join(lines),
            temperature=0.1,
            max_tokens=900,
            timeout=25.0,
            purpose="distill",
        )
    except (LLMError, ValueError, TypeError):
        return _offline_items(text, related)
    items = data.get("memories") or []
    return [i for i in items if isinstance(i, dict)]


_PREFIX = re.compile(
    r"^(记住|记一下|补充一下|告诉你|不对[的]?|不是这样|不是的|不是|"
    r"你说错了|说错了|搞错了|记错了|更正一下|纠正一下|"
    r"remember|note that|note down|note|by the way|update the profile|"
    r"not right|not correct|that'?s wrong|wrong|correction|correct that|"
    r"i made a mistake|my mistake|no|nope)[:：，,。\s—–\-]*",
    re.IGNORECASE,
)


def _offline_items(text: str, related: list[Memory]) -> list[dict]:
    from app.services.classify import classify_memory_text

    cleaned = _PREFIX.sub("", text).strip()
    if len(cleaned) < 4:
        return []
    correcting = looks_like_correction(text)
    if correcting and related:
        result = classify_memory_text(cleaned, use_llm=False)
        old = related[0]
        retract_only = (
            bool(
                re.search(
                    r"现在不|已经不|不再|删掉|作废|"
                    r"\b(?:no longer|not anymore|retract|void|delete that)\b",
                    text,
                    re.IGNORECASE,
                )
            )
            and len(cleaned) < 12
        )
        if retract_only:
            return [
                {
                    "content": "",
                    "memory_type": old.memory_type,
                    "action": "retract",
                    "replace_id": old.id,
                    "replaces": old.content,
                }
            ]
        return [
            {
                "content": cleaned,
                "memory_type": result.memory_type if result.confidence >= 0.7 else old.memory_type,
                "confidence": 0.9,
                "is_inference": False,
                "action": "replace",
                "replace_id": old.id,
                "replaces": old.content,
            }
        ]
    if len(cleaned) < 6:
        return []
    result = classify_memory_text(cleaned, use_llm=False)
    if result.confidence < 0.7:
        return []
    return [
        {
            "content": cleaned,
            "memory_type": result.memory_type,
            "confidence": min(0.95, max(0.85, result.confidence)),
            "is_inference": False,
            "action": "replace" if correcting else "add",
            "replaces": related[0].content if correcting and related else None,
            "replace_id": related[0].id if correcting and related else None,
        }
    ]
