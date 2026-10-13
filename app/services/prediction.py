from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.models import Memory, MemoryEvidence, MemoryStatus
from app.llm.client import LLMClient, LLMError, _extract_json
from app.services.prompts import build_qa_system_prompt
from app.services.retrieval import hinted_memory_types, hybrid_search, load_confirmed_profile
from app.services.scope import classify_scope


@dataclass
class TendencyResult:
    can_answer: bool
    tendency: str | None
    confidence: float
    rationale: str
    supporting_evidence: list[dict] = field(default_factory=list)
    counter_evidence: list[dict] = field(default_factory=list)
    refusal_reason: str | None = None


TENDENCY_SYSTEM = """You are a conversational situation-tendency assistant. Confirmed memories are reference material only, not permission to answer.
You must output JSON:
{
  "can_answer": true/false,
  "tendency": "a natural paragraph describing the tendency, written like chat; do not use numbered lists or memory IDs",
  "confidence": 0.0-1.0,
  "rationale": "one or two sentences of internal reasoning; the UI may not show it",
  "supporting_ids": [memory IDs],
  "counter_ids": [memory IDs],
  "refusal_reason": "an explanation the user can see; never write only 'cannot tell'"
}
Rules:
- When relevant memories exist, can_answer=true and write the tendency as a short passage.
- When no relevant memories exist, you may still set can_answer=true: answer normally in an advisor's tone, state clearly that this is an opinion rather than a documented fact, and keep confidence low.
- Set can_answer=false only when the request involves manipulation or harm. refusal_reason must still be warm; never write only 'cannot tell'.
- Do not pretend to read minds. confidence must be honest.
- Reply in the same language the user wrote in.
"""


def answer_question(
    db: Session,
    question: str,
    *,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    history: list[dict] | None = None,
) -> dict:
    done = {
        "answer": _humane_unknown(question, []),
        "can_answer": True,
        "evidence": [],
        "thinking": "",
    }
    for event in iter_answer_events(
        db, question, llm=llm, settings=settings, history=history
    ):
        if event.get("type") == "done":
            done = {
                "answer": event.get("answer") or "",
                "can_answer": bool(event.get("can_answer")),
                "evidence": event.get("evidence") or [],
                "thinking": event.get("thinking") or "",
            }
    return done


def iter_answer_events(
    db: Session,
    question: str,
    *,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    history: list[dict] | None = None,
) -> Iterator[dict]:
    settings = settings or get_settings()
    llm = llm or LLMClient(settings)
    verdict = classify_scope(db, question, history=history)
    if not verdict.in_scope:
        yield {"type": "status", "text": "This question is outside the knowledge base…"}
        yield {"type": "token", "text": verdict.refusal}
        yield {
            "type": "done",
            "answer": verdict.refusal,
            "can_answer": False,
            "evidence": [],
            "thinking": "",
            "scope": verdict.reason,
        }
        return

    yield {"type": "status", "text": "Searching memories…"}
    hits = hybrid_search(db, question, include_messages=True, limit=16)
    memory_hits = [h for h in hits if h.kind == "memory"]
    message_hits = [h for h in hits if h.kind == "message"]
    typed = hinted_memory_types(question)
    evidence = [_hit_dict(h) for h in hits[:8]]
    profile_memories = []
    if not memory_hits:
        profile_memories = load_confirmed_profile(db, limit=16)

    n = len(memory_hits) + len(message_hits)
    if n:
        yield {"type": "status", "text": f"Found {n} relevant items, answering…"}
    elif llm.configured:
        yield {"type": "status", "text": "No direct match in the knowledge base; answering as an advisor…"}

    if not llm.configured:
        answer = _extractive_answer(question, memory_hits, message_hits)
        yield {"type": "token", "text": answer}
        yield {
            "type": "done",
            "answer": answer,
            "can_answer": True,
            "evidence": evidence,
            "thinking": "",
        }
        return

    evidence_block = _format_hits(memory_hits, message_hits, profile_memories)
    type_note = ""
    if typed and memory_hits:
        type_note = (
            f"\nQuestion type hint: {', '.join(typed)}. "
            "Prefer confirmed memories of these types when discussing concrete facts about the subject."
        )
    history_block = _format_history(history)
    kb_note = (
        "Use relevant entries if there are any; if not, answer normally and do not refuse just because the knowledge base is empty."
        if n
        else "No knowledge-base entries directly matched this question. Answer normally as an advisor; do not invent private details about the subject, and mark opinions as inferences."
    )
    user = (
        f"Question: {question}{type_note}\n"
        f"{history_block}"
        f"Internal knowledge base (reference, not permission to answer):\n{evidence_block}\n\n"
        f"{kb_note}\n"
        "Answer in natural language. If this is a follow-up, connect it to the earlier conversation."
    )
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    system = build_qa_system_prompt()
    try:
        for kind, text in llm.chat_stream(
            system, user, temperature=0.45, purpose="answer"
        ):
            if kind == "thinking":
                thinking_parts.append(text)
                yield {"type": "thinking", "text": text}
            elif kind == "token":
                answer_parts.append(text)
                yield {"type": "token", "text": text}
    except LLMError as exc:
        answer = f"Model call failed: {exc}"
        yield {"type": "token", "text": answer}
        yield {
            "type": "done",
            "answer": answer,
            "can_answer": False,
            "evidence": evidence,
            "thinking": "".join(thinking_parts),
        }
        return

    answer = _strip_citation_noise("".join(answer_parts).strip())
    if not answer:
        answer = "The model returned no content."
    yield {
        "type": "done",
        "answer": answer,
        "can_answer": True,
        "evidence": evidence,
        "thinking": "".join(thinking_parts).strip(),
    }


def iter_tendency_events(
    db: Session,
    situation: str,
    *,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    history: list[dict] | None = None,
) -> Iterator[dict]:
    yield {"type": "status", "text": "Analyzing tendency…"}
    settings = settings or get_settings()
    llm = llm or LLMClient(settings)
    verdict = classify_scope(db, situation, history=history)
    if not verdict.in_scope:
        result = {
            "can_answer": False,
            "tendency": None,
            "confidence": 0.0,
            "rationale": "Outside the knowledge base's scope.",
            "supporting_evidence": [],
            "counter_evidence": [],
            "refusal_reason": verdict.refusal,
        }
        yield {"type": "token", "text": verdict.refusal}
        yield {**result, "type": "done", "answer": verdict.refusal, "thinking": ""}
        return
    if not llm.configured:
        result = tendency_to_dict(predict_tendency(db, situation, llm=llm, settings=settings, history=history))
        text = _soften_tendency_text(situation, result)
        yield {"type": "token", "text": text}
        yield {"type": "done", **result, "answer": text, "thinking": ""}
        return

    hits = hybrid_search(db, situation, include_messages=False, limit=15)
    memory_hits = [h for h in hits if h.kind == "memory"]
    mem_lines = []
    id_map: dict[int, Memory] = {}
    for h in memory_hits:
        mem = db.get(Memory, h.id)
        if not mem or mem.status != MemoryStatus.CONFIRMED.value:
            continue
        id_map[mem.id] = mem
        mem_lines.append(
            f"ID={mem.id} type={mem.memory_type} conf={mem.confidence:.2f} "
            f"inference={mem.is_inference}: {mem.content}"
        )
    kb_block = "\n".join(mem_lines) if mem_lines else "(no relevant confirmed memories)"
    user = (
        f"Situation: {situation}\n"
        f"{_format_history(history)}"
        f"Confirmed memories (reference, may be empty):\n"
        + kb_block
        + "\nIf no relevant memories exist, still answer normally in an advisor's tone and make clear this is an opinion."
    )
    thinking_parts: list[str] = []
    answer_parts: list[str] = []
    try:
        for kind, text in llm.chat_stream(TENDENCY_SYSTEM, user, purpose="tendency"):
            if kind == "thinking":
                thinking_parts.append(text)
                yield {"type": "thinking", "text": text}
            elif kind == "token":
                answer_parts.append(text)
    except LLMError as exc:
        text = f"Model call failed: {exc}"
        yield {"type": "token", "text": text}
        yield {
            "type": "done",
            "can_answer": False,
            "tendency": None,
            "confidence": 0.0,
            "rationale": "",
            "supporting_evidence": [],
            "counter_evidence": [],
            "refusal_reason": text,
            "answer": text,
            "thinking": "".join(thinking_parts),
        }
        return

    try:
        data = _extract_json("".join(answer_parts))
    except (LLMError, ValueError, TypeError):
        result = tendency_to_dict(predict_tendency(db, situation, llm=llm, settings=settings, history=history))
        text = _soften_tendency_text(situation, result, memory_hits)
        yield {"type": "token", "text": text}
        yield {"type": "done", **result, "answer": text, "thinking": "".join(thinking_parts)}
        return

    can_answer = bool(data.get("can_answer"))
    confidence = max(0.0, min(1.0, float(data.get("confidence") or 0.0)))
    support = _ids_to_evidence(id_map, data.get("supporting_ids") or [])
    counter = _ids_to_evidence(id_map, data.get("counter_ids") or [])
    if not can_answer:
        text = _soften_tendency_text(
            situation,
            {
                "tendency": data.get("tendency"),
                "refusal_reason": data.get("refusal_reason")
                or "This is not something to give manipulative or harmful advice about.",
            },
            memory_hits,
        )
        yield {"type": "token", "text": text}
        yield {
            "type": "done",
            "can_answer": False,
            "tendency": None,
            "confidence": confidence,
            "rationale": str(data.get("rationale") or ""),
            "supporting_evidence": support,
            "counter_evidence": counter,
            "refusal_reason": text,
            "answer": text,
            "thinking": "".join(thinking_parts).strip(),
        }
        return
    text = str(data.get("tendency") or "").strip() or _humane_unknown(situation, memory_hits)
    yield {"type": "token", "text": text}
    yield {
        "type": "done",
        "can_answer": True,
        "tendency": text,
        "confidence": confidence,
        "rationale": str(data.get("rationale") or ""),
        "supporting_evidence": support,
        "counter_evidence": counter,
        "refusal_reason": None,
        "answer": text,
        "thinking": "".join(thinking_parts).strip(),
    }


def predict_tendency(
    db: Session,
    situation: str,
    *,
    llm: LLMClient | None = None,
    settings: Settings | None = None,
    history: list[dict] | None = None,
) -> TendencyResult:
    settings = settings or get_settings()
    llm = llm or LLMClient(settings)

    verdict = classify_scope(db, situation, history=history)
    if not verdict.in_scope:
        return TendencyResult(
            can_answer=False,
            tendency=None,
            confidence=0.0,
            rationale="Outside the knowledge base's scope.",
            refusal_reason=verdict.refusal,
        )

    hits = hybrid_search(db, situation, include_messages=False, limit=15)
    memory_hits = [h for h in hits if h.kind == "memory"]

    if not llm.configured:
        return _offline_tendency(db, situation, memory_hits, settings)

    mem_lines = []
    id_map: dict[int, Memory] = {}
    for h in memory_hits:
        mem = db.get(Memory, h.id)
        if not mem or mem.status != MemoryStatus.CONFIRMED.value:
            continue
        id_map[mem.id] = mem
        mem_lines.append(
            f"ID={mem.id} type={mem.memory_type} conf={mem.confidence:.2f} "
            f"inference={mem.is_inference}: {mem.content}"
        )

    kb_block = "\n".join(mem_lines) if mem_lines else "(no relevant confirmed memories)"
    user = (
        f"Situation: {situation}\n"
        f"{_format_history(history)}"
        f"Confirmed memories (reference, may be empty):\n"
        + kb_block
        + "\nIf no relevant memories exist, still answer normally in an advisor's tone and make clear this is an opinion."
    )
    try:
        data = llm.chat_json(TENDENCY_SYSTEM, user, purpose="tendency")
    except LLMError as exc:
        return TendencyResult(
            can_answer=False,
            tendency=None,
            confidence=0.0,
            rationale="",
            refusal_reason=f"Model call failed: {exc}",
        )

    can_answer = bool(data.get("can_answer"))
    confidence = float(data.get("confidence") or 0.0)
    confidence = max(0.0, min(1.0, confidence))

    if not can_answer:
        return TendencyResult(
            can_answer=False,
            tendency=None,
            confidence=confidence,
            rationale=str(data.get("rationale") or ""),
            refusal_reason=_soften_tendency_text(
                situation,
                {
                    "tendency": data.get("tendency"),
                    "refusal_reason": data.get("refusal_reason"),
                },
                memory_hits,
            ),
            supporting_evidence=_ids_to_evidence(id_map, data.get("supporting_ids") or []),
            counter_evidence=_ids_to_evidence(id_map, data.get("counter_ids") or []),
        )

    return TendencyResult(
        can_answer=True,
        tendency=str(data.get("tendency") or "").strip() or None,
        confidence=confidence,
        rationale=str(data.get("rationale") or ""),
        supporting_evidence=_ids_to_evidence(id_map, data.get("supporting_ids") or []),
        counter_evidence=_ids_to_evidence(id_map, data.get("counter_ids") or []),
    )


def _offline_tendency(
    db: Session, situation: str, memory_hits, settings: Settings
) -> TendencyResult:
    if len(memory_hits) < settings.min_prediction_evidence:
        return TendencyResult(
            can_answer=False,
            tendency=None,
            confidence=0.0,
            rationale="",
            refusal_reason=_humane_unknown(situation, memory_hits),
        )
    support = []
    for h in memory_hits[:5]:
        support.append(
            {
                "memory_id": h.id,
                "content": h.content,
                "memory_type": h.meta.get("memory_type"),
                "confidence": h.meta.get("confidence"),
            }
        )
    avg_conf = sum(float(h.meta.get("confidence") or 0.5) for h in memory_hits[:5]) / min(
        5, len(memory_hits)
    )
    # Discount offline heuristic
    confidence = min(avg_conf * 0.7, 0.7)
    if confidence < settings.min_prediction_confidence:
        return TendencyResult(
            can_answer=False,
            tendency=None,
            confidence=confidence,
            rationale="Offline heuristic confidence too low.",
            refusal_reason=_humane_unknown(situation, memory_hits),
            supporting_evidence=support,
        )
    tendency = (
        f"Based on {len(memory_hits)} relevant confirmed memories, past tendencies may align with: "
        + "; ".join(h.content for h in memory_hits[:3])
    )
    return TendencyResult(
        can_answer=True,
        tendency=tendency,
        confidence=confidence,
        rationale="Offline mode: a heuristic hint based only on retrieval similarity, not model reasoning.",
        supporting_evidence=support,
    )


def _ids_to_evidence(id_map: dict[int, Memory], ids) -> list[dict]:
    out = []
    for i in ids:
        try:
            mid = int(i)
        except (TypeError, ValueError):
            continue
        mem = id_map.get(mid)
        if not mem:
            continue
        out.append(
            {
                "memory_id": mem.id,
                "content": mem.content,
                "memory_type": mem.memory_type,
                "confidence": mem.confidence,
            }
        )
    return out


def _strip_citation_noise(answer: str) -> str:
    text = answer or ""
    text = re.sub(r"[（(]\s*记忆\s*#?\s*\d+\s*[）)]", "", text)
    text = re.sub(r"记忆\s*#\s*\d+", "", text)
    text = re.sub(r"[（(]\s*#\s*\d+\s*[）)]", "", text)
    text = re.sub(r"[（(]memory#\d+[^）)]*[）)]", "", text)
    text = re.sub(r"[（(]\s*memory\s*#?\s*\d+\s*[）)]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(结论|依据|根据已确认记忆|conclusion|basis|based on confirmed memories)"
                  r"\s*[:：]\s*", "", text, flags=re.M | re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


RELATIONAL_PAT = re.compile(
    r"喜欢我|爱我|不喜欢我|会不会不爱|对我(的态度|怎么看)|"
    r"感情|心动|共鸣|异地|会不会(变|腻|淡|冷|分手)|"
    r"还喜不喜欢|那你觉得|你觉得呢|你怎么看|"
    r"\b(?:like me|likes me|loves? me|love me|hate me|feel about me|feelings? for me|"
    r"into me|interested in me|we|us|relationship|long distance|drift apart|break up)\b",
    re.IGNORECASE,
)


def _is_relational_question(text: str) -> bool:
    return bool(RELATIONAL_PAT.search(text or ""))


def _looks_like_refusal(answer: str) -> bool:
    text = (answer or "").strip()
    if not text:
        return True
    tokens = (
        "无法判断", "没法判断", "无法给出可靠", "证据不足", "无法回答", "没有任何一条",
        "not enough evidence", "can't tell", "cannot tell", "can't say", "cannot say",
        "no way to tell", "the records don't say", "the records have nothing",
        "i can't answer", "no direct record",
    )
    lowered = text.lower()
    if not any(token in text or token in lowered for token in tokens):
        return False
    # Long companion replies that mention uncertainty are still answers.
    return len(text) < 60


def _humane_unknown(question: str, memory_hits) -> str:
    related = []
    for h in memory_hits or []:
        content = (getattr(h, "content", None) or "").rstrip("。.")
        if content and content not in related:
            related.append(content)
        if len(related) >= 4:
            break
    if _is_relational_question(question):
        if related:
            return (
                "I can't answer for what they feel inside, and the records don't state directly how they feel about you. "
                f"The related impressions I do have are: {', '.join(related)}. "
                "Different interests, little time together, or long distance can genuinely wear a relationship down, "
                "but none of that alone tells me whether they like you. "
                "You're asking this now, so something probably left you unsure recently. Want to say what set it off?"
            )
        return (
            "I can't make that call for them — whether someone likes you isn't something the records can prove. "
            "There's also no direct record in the knowledge base of how they feel about you. "
            "If you're willing, tell me which recent interaction left you unsure, and I can look at it with you."
        )
    if related:
        return (
            "The records don't say this directly, and I won't make it up. "
            f"The closest impressions I have are: {', '.join(related)}."
        )
    return "The records have nothing on this, and guessing would only mislead you. If you know more clues, tell me and I'll help you check."


def _soften_tendency_text(situation: str, result: dict, memory_hits=None) -> str:
    text = (result.get("tendency") or result.get("refusal_reason") or "").strip()
    if text and not _looks_like_refusal(text):
        return text
    return _humane_unknown(situation, memory_hits or [])


def _extractive_answer(question: str, memory_hits, message_hits) -> str:
    if _is_relational_question(question):
        return _humane_unknown(question, memory_hits)
    typed = hinted_memory_types(question)
    bits = []
    for h in memory_hits:
        if typed and h.meta.get("memory_type") not in typed:
            continue
        bits.append(h.content.rstrip("。."))
        if len(bits) >= 6:
            break
    if not bits:
        bits = [h.content.rstrip("。.") for h in memory_hits[:6]]
    if not bits:
        return _humane_unknown(question, memory_hits)
    joined = ", ".join(bits)
    if typed == ["aversion"]:
        return f"From what I can tell, they can't stand these: {joined}."
    if typed == ["preference"]:
        return f"From what I can tell, they lean toward: {joined}."
    return f"Putting the impressions together: {joined}."


def _format_history(history: list[dict] | None) -> str:
    if not history:
        return ""
    lines = ["Recent conversation (for understanding follow-ups, not memories):"]
    for item in history[-8:]:
        role = "User" if item.get("role") == "user" else "Assistant"
        content = (item.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:400]}")
    return "\n".join(lines) + "\n\n"


def _format_hits(memory_hits, message_hits, profile_memories=None) -> str:
    lines = ["## Related knowledge-base entries (for reasoning only, do not write into the answer)"]
    if memory_hits:
        for h in memory_hits:
            lines.append(f"- [{h.meta.get('memory_type')}] {h.content}")
    else:
        lines.append("- (no direct match for this question)")
        if profile_memories:
            lines.append("## Profile summary (may be unrelated to this question; reference as needed)")
            for m in profile_memories[:12]:
                lines.append(f"- [{m.memory_type}] {m.content}")
    lines.append("## Related raw messages")
    if message_hits:
        for h in message_hits[:6]:
            lines.append(f"- (message#{h.id}, {h.meta.get('sender')}) {h.content[:200]}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def _hit_dict(h) -> dict:
    return {
        "kind": h.kind,
        "id": h.id,
        "content": h.content[:300],
        "score": round(h.score, 4),
        "meta": h.meta,
    }


def tendency_to_dict(result: TendencyResult) -> dict:
    return {
        "can_answer": result.can_answer,
        "tendency": result.tendency,
        "confidence": result.confidence,
        "rationale": result.rationale,
        "supporting_evidence": result.supporting_evidence,
        "counter_evidence": result.counter_evidence,
        "refusal_reason": result.refusal_reason,
    }
