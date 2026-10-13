from __future__ import annotations

import math
import re
from hashlib import blake2b
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import or_, text
from sqlalchemy.orm import Session, joinedload

from app.db.models import Memory, MemoryStatus, Message


@dataclass
class RetrievalHit:
    kind: str  # memory | message
    id: int
    content: str
    score: float
    meta: dict


def local_embed(text_value: str, dim: int = 1024) -> list[float]:
    """Deterministic signed bag-of-bigrams hash embedding — fully local, no API.

    The sign bit matters: without it, unrelated texts collide in the same
    buckets and always gain similarity, so out-of-scope questions looked
    "related". Signed hashing makes collisions cancel out instead.
    """
    vec = [0.0] * dim
    text_value = text_value.lower().strip()
    if not text_value:
        return vec
    tokens = _tokens(text_value)
    if not tokens:
        return vec
    counts = Counter(tokens)
    for tok, cnt in counts.items():
        # Python's hash() is randomized per process. Persisting vectors built
        # from it makes similarity scores invalid after every restart.
        digest = blake2b(tok.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        sign = 1.0 if (h >> 63) & 1 else -1.0
        vec[h % dim] += sign * float(cnt)
    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _tokens(text_value: str) -> list[str]:
    # Split CJK chars individually + latin words
    parts = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text_value.lower())
    grams: list[str] = []
    for i, p in enumerate(parts):
        grams.append(p)
        if i + 1 < len(parts):
            grams.append(p + parts[i + 1])
    return grams


TYPE_HINTS: list[tuple[str, str]] = [
    (r"讨厌|不喜欢|厌恶|害怕|受不了|雷区|忌讳|恐惧|怕什么", "aversion"),
    (r"喜欢什么|爱吃|爱喝|爱看|偏好|钟爱|(?<!不)喜欢", "preference"),
    (r"习惯|通常|经常|每天|总是|一般|周末|平时", "habit"),
    (r"怎么说话|沟通|约会|约(她|他|对方|人)|回复|消息|临时|改计划|注意|要不要", "communication"),
    (r"计划|打算|准备去", "plan"),
    (r"谁|朋友|家人|住在|地点", "person_place"),
    # English equivalents, checked case-insensitively.
    (
        r"(?i:\b(?:dislikes?|hates?|afraid|scared|aversion|allergic|can'?t stand)\b)",
        "aversion",
    ),
    (
        r"(?i:\b(?:favorites?|favourites?|likes?|loves?|prefers?|enjoys?)\b)",
        "preference",
    ),
    (
        r"(?i:\b(?:usually|often|habit|habits|routine|every day|on weekends?|schedule)\b)",
        "habit",
    ),
    (
        r"(?i:\b(?:how to talk|communicat\w*|dating|reply|replies|messages?|last.?minute|"
        r"change (?:of )?plans?|keep in mind|be careful|watch out|heads? up)\b|"
        r"\b(?:ask(?:ing)?|invite|inviting)\b[^.]{0,20}\bout\b)",
        "communication",
    ),
    (r"(?i:\b(?:plans?|planning|intends? to|going to)\b)", "plan"),
    (r"(?i:\b(?:who|friend|friends|family|lives?|where)\b)", "person_place"),
]

QUERY_SYNONYMS: dict[str, list[str]] = {
    "讨厌": ["不喜欢", "厌恶", "害怕", "受不了", "忌讳", "雷区"],
    "不喜欢": ["讨厌", "厌恶", "害怕"],
    "喜欢": ["爱吃", "爱喝", "爱看", "偏好"],
    "习惯": ["经常", "通常", "总是"],
    "dislike": ["hate", "afraid", "allergic"],
    "dislikes": ["hates", "afraid", "allergic"],
    "like": ["love", "prefer", "enjoy", "favorite"],
    "likes": ["loves", "prefers", "enjoys", "favorite"],
    "hate": ["dislike", "afraid"],
    "hates": ["dislikes", "afraid"],
    "habit": ["usually", "often", "routine"],
    "habits": ["usually", "often", "routine"],
}


def hinted_memory_types(query: str) -> list[str]:
    found: list[str] = []
    for pattern, memory_type in TYPE_HINTS:
        if re.search(pattern, query) and memory_type not in found:
            found.append(memory_type)
    return found


def expand_query(query: str) -> str:
    extras: list[str] = []
    for stem, syns in QUERY_SYNONYMS.items():
        if stem in query:
            extras.extend(syns)
    if not extras:
        return query
    return query + " " + " ".join(extras)


def search_memories_by_types(
    db: Session, types: list[str], *, limit: int = 12
) -> list[RetrievalHit]:
    if not types:
        return []
    rows = (
        db.query(Memory)
        .filter(
            Memory.status == MemoryStatus.CONFIRMED.value,
            Memory.memory_type.in_(types),
        )
        .order_by(Memory.confidence.desc(), Memory.id.desc())
        .limit(limit)
        .all()
    )
    return [
        RetrievalHit(
            kind="memory",
            id=m.id,
            content=m.content,
            score=0.88,
            meta={
                "memory_type": m.memory_type,
                "confidence": m.confidence,
                "subject": m.subject,
            },
        )
        for m in rows
    ]


def search_memories_fts(db: Session, query: str, *, limit: int = 20) -> list[RetrievalHit]:
    if not query.strip():
        return []
    # Escape FTS special chars roughly
    safe = re.sub(r'[^\w\u4e00-\u9fff]+', " ", query).strip()
    if not safe:
        return []
    tokens = _fts_tokens(safe)
    if not tokens:
        return []
    fts_q = " OR ".join(f'"{t}"' for t in tokens)
    try:
        rows = db.execute(
            text(
                """
                SELECT m.id, m.content, m.memory_type, m.confidence, m.status,
                       m.subject, bm25(memories_fts) AS rank
                FROM memories_fts
                JOIN memories m ON m.id = memories_fts.rowid
                WHERE memories_fts MATCH :q
                  AND m.status = :status
                ORDER BY rank
                LIMIT :limit
                """
            ),
            {"q": fts_q, "status": MemoryStatus.CONFIRMED.value, "limit": limit},
        ).fetchall()
    except Exception:
        return _search_memories_like(db, query, limit=limit)

    if not rows and _contains_cjk(query):
        # unicode61 does not segment CJK text into useful word tokens.
        return _search_memories_like(db, query, limit=limit)

    hits = []
    for r in rows:
        # bm25 lower is better; convert to score
        rank = float(r.rank) if r.rank is not None else 0.0
        score = 1.0 / (1.0 + max(rank, 0.0))
        hits.append(
            RetrievalHit(
                kind="memory",
                id=r.id,
                content=r.content,
                score=score,
                meta={
                    "memory_type": r.memory_type,
                    "confidence": r.confidence,
                    "status": r.status,
                    "subject": r.subject,
                },
            )
        )
    return hits


def search_messages_fts(db: Session, query: str, *, limit: int = 20) -> list[RetrievalHit]:
    if not query.strip():
        return []
    safe = re.sub(r'[^\w\u4e00-\u9fff]+', " ", query).strip()
    tokens = _fts_tokens(safe)
    if not tokens:
        return []
    fts_q = " OR ".join(f'"{t}"' for t in tokens)
    try:
        rows = db.execute(
            text(
                """
                SELECT m.id, m.content, m.sender, m.sent_at,
                       bm25(messages_fts) AS rank
                FROM messages_fts
                JOIN messages m ON m.id = messages_fts.rowid
                WHERE messages_fts MATCH :q
                ORDER BY rank
                LIMIT :limit
                """
            ),
            {"q": fts_q, "limit": limit},
        ).fetchall()
    except Exception:
        return _search_messages_like(db, query, limit=limit)

    if not rows and _contains_cjk(query):
        return _search_messages_like(db, query, limit=limit)

    hits = []
    for r in rows:
        rank = float(r.rank) if r.rank is not None else 0.0
        score = 1.0 / (1.0 + max(rank, 0.0))
        hits.append(
            RetrievalHit(
                kind="message",
                id=r.id,
                content=r.content,
                score=score * 0.9,
                meta={"sender": r.sender, "sent_at": str(r.sent_at)},
            )
        )
    return hits


def semantic_memory_search(
    db: Session, query: str, *, limit: int = 20
) -> list[RetrievalHit]:
    qvec = local_embed(query)
    memories = (
        db.query(Memory)
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .all()
    )
    scored: list[RetrievalHit] = []
    for m in memories:
        # Recompute here so databases created by the old process-randomized
        # hash implementation remain searchable without a migration.
        emb = local_embed(m.content)
        score = cosine(qvec, emb)
        if score <= 0.05:
            continue
        scored.append(
            RetrievalHit(
                kind="memory",
                id=m.id,
                content=m.content,
                score=score,
                meta={
                    "memory_type": m.memory_type,
                    "confidence": m.confidence,
                    "subject": m.subject,
                },
            )
        )
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored[:limit]


def _fts_tokens(value: str, *, limit: int = 24) -> list[str]:
    """Return bounded CJK-friendly terms for SQLite unicode61 FTS queries."""
    terms: list[str] = []
    seen: set[str] = set()
    for token in _tokens(value):
        if token in seen:
            continue
        seen.add(token)
        terms.append(token.replace('"', '""'))
        if len(terms) >= limit:
            break
    return terms


def _contains_cjk(value: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", value))


def _cjk_bigrams(value: str, *, limit: int = 12) -> list[str]:
    chars = re.findall(r"[\u4e00-\u9fff]", value)
    if len(chars) < 2:
        return chars
    return list(dict.fromkeys(chars[i] + chars[i + 1] for i in range(len(chars) - 1)))[:limit]


def _search_memories_like(
    db: Session, query: str, *, limit: int
) -> list[RetrievalHit]:
    terms = _cjk_bigrams(query) or [query]
    rows = (
        db.query(Memory)
        .filter(
            Memory.status == MemoryStatus.CONFIRMED.value,
            or_(*(Memory.content.contains(term) for term in terms)),
        )
        .limit(limit)
        .all()
    )
    return [
        RetrievalHit(
            kind="memory",
            id=m.id,
            content=m.content,
            score=0.5,
            meta={
                "memory_type": m.memory_type,
                "confidence": m.confidence,
                "subject": m.subject,
            },
        )
        for m in rows
    ]


def _search_messages_like(
    db: Session, query: str, *, limit: int
) -> list[RetrievalHit]:
    terms = _cjk_bigrams(query) or [query]
    rows = (
        db.query(Message)
        .filter(or_(*(Message.content.contains(term) for term in terms)))
        .limit(limit)
        .all()
    )
    return [
        RetrievalHit(
            kind="message",
            id=m.id,
            content=m.content,
            score=0.4,
            meta={"sender": m.sender, "sent_at": str(m.sent_at)},
        )
        for m in rows
    ]


def hybrid_search(
    db: Session,
    query: str,
    *,
    include_messages: bool = True,
    limit: int = 15,
) -> list[RetrievalHit]:
    expanded = expand_query(query)
    fts_mem = search_memories_fts(db, expanded, limit=limit)
    sem_mem = semantic_memory_search(db, expanded, limit=limit)
    type_mem = search_memories_by_types(db, hinted_memory_types(query), limit=limit)
    merged: dict[tuple[str, int], RetrievalHit] = {}

    def add(hit: RetrievalHit, weight: float = 1.0):
        key = (hit.kind, hit.id)
        if key in merged:
            merged[key].score = max(merged[key].score, hit.score * weight)
        else:
            merged[key] = RetrievalHit(
                kind=hit.kind,
                id=hit.id,
                content=hit.content,
                score=hit.score * weight,
                meta=hit.meta,
            )

    for h in fts_mem:
        add(h, 1.0)
    for h in sem_mem:
        add(h, 1.1)
    for h in type_mem:
        add(h, 1.2)

    if include_messages:
        for h in search_messages_fts(db, query, limit=limit):
            add(h, 0.85)

    results = _apply_subject_policy(db, query, merged.values())
    return results[:limit]


def _apply_subject_policy(
    db: Session, query: str, hits
) -> list[RetrievalHit]:
    """Keep memories about another person from leaking into the answer.

    A memory is only dropped when it has an explicit subject that is not the
    subject asked about. Unlabelled memories keep their previous behaviour so
    existing single-person databases are unaffected.
    """
    asked = query_subjects(db, query)
    scored: list[RetrievalHit] = []
    for hit in hits:
        subject = hit.meta.get("subject") if hit.kind == "memory" else None
        if subject and asked and subject not in asked:
            continue
        score = hit.score
        if subject and asked and subject in asked:
            score *= 1.25
        scored.append(
            RetrievalHit(
                kind=hit.kind,
                id=hit.id,
                content=hit.content,
                score=score,
                meta=hit.meta,
            )
        )
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored


def known_subjects(db: Session) -> list[str]:
    rows = (
        db.query(Memory.subject)
        .filter(
            Memory.status == MemoryStatus.CONFIRMED.value,
            Memory.subject.isnot(None),
        )
        .distinct()
        .all()
    )
    subjects = []
    for (value,) in rows:
        text = (value or "").strip()
        if text and text not in subjects:
            subjects.append(text)
    return subjects


def query_subjects(db: Session, query: str) -> list[str]:
    text = query or ""
    return [name for name in known_subjects(db) if name and name in text]


def load_confirmed_profile(db: Session, *, limit: int = 80) -> list[Memory]:
    return (
        db.query(Memory)
        .options(joinedload(Memory.evidence))
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .order_by(Memory.confidence.desc(), Memory.updated_at.desc())
        .limit(limit)
        .all()
    )
