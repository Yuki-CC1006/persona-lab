from __future__ import annotations

import re
from dataclasses import dataclass

from app.db.models import MemoryType
from app.llm.client import LLMClient, LLMError
from app.services.profile import TYPE_META

VALID_TYPES = {t.value for t in MemoryType}

CLASSIFY_SYSTEM = """You are a memory-type classifier. Given one sentence, decide the memory type and extract the portrait subject's nickname if a name appears in the sentence.
Output JSON only:
{
  "memory_type": "fact|preference|aversion|person_place|habit|event|communication|plan",
  "subject": "subject nickname or null",
  "confidence": 0.0-1.0,
  "reason": "one-sentence reason"
}
Type guide:
- preference: likes, preferences
- aversion: dislikes, hates, pet peeves
- habit: habits, usually, often
- communication: communication style, speaking style, dating/social preferences
- person_place: people relationships or places
- event: concrete events, birthdays, anniversaries
- plan: plans, intentions
- fact: other stable facts
"""


@dataclass
class ClassifyResult:
    memory_type: str
    subject: str | None
    confidence: float
    reason: str
    source: str  # rule | llm


def classify_memory_text(
    content: str,
    *,
    subject_hint: str | None = None,
    llm: LLMClient | None = None,
    use_llm: bool = True,
) -> ClassifyResult:
    text = (content or "").strip()
    if not text:
        return ClassifyResult("fact", subject_hint, 0.3, "empty content, defaulted to fact", "rule")

    rule = _rule_classify(text, subject_hint=subject_hint)
    if not use_llm:
        return rule

    client = llm or LLMClient()
    if not client.configured:
        return rule

    # Prefer fast local rules for interactive form; only call LLM when unsure
    if rule.confidence >= 0.75:
        return rule

    try:
        data = client.chat_json(
            CLASSIFY_SYSTEM,
            f"Sentence: {text}\nOptional subject hint: {subject_hint or 'none'}",
            temperature=0.0,
            max_tokens=300,
            timeout=12.0,
            purpose="classify",
        )
        mtype = str(data.get("memory_type") or "").strip()
        if mtype not in VALID_TYPES:
            return rule
        subject = data.get("subject")
        if isinstance(subject, str):
            subject = subject.strip() or None
        else:
            subject = None
        conf = float(data.get("confidence") or 0.75)
        conf = max(0.0, min(1.0, conf))
        reason = str(data.get("reason") or "model classification").strip()
        return ClassifyResult(
            memory_type=mtype,
            subject=subject or subject_hint or rule.subject,
            confidence=conf,
            reason=reason,
            source="llm",
        )
    except (LLMError, ValueError, TypeError):
        return rule


def _rule_classify(text: str, *, subject_hint: str | None = None) -> ClassifyResult:
    subject = subject_hint or _guess_subject(text)

    aversion_pat = r"(不喜欢|讨厌|厌恶|受不了|别再|不要给我|雷区|忌讳|拒绝)"
    preference_pat = r"(喜欢|爱喝|爱吃|偏爱|钟爱|更想要|想要|爱看)"
    habit_pat = r"(通常|一般|经常|总是|习惯|每周|周末|每天|早上|晚上)"
    communication_pat = r"(提前|临时改|连发|消息|沟通|说话|语气|回复|约会|约她|约他)"
    plan_pat = r"(打算|计划|准备|下周|明天要|想去)"
    event_pat = r"(生日|纪念日|婚礼|毕业|见面了|去了|来了)"
    person_place_pat = r"(朋友|同事|妈妈|爸爸|家人|住在|在.{0,8}(市|区|路|店))"

    # English equivalents. Inline (?i:...) keeps them case-insensitive without
    # changing the flags of the surrounding patterns.
    aversion_pat += (
        r"|(?i:\b(?:dislikes?|hates?|can'?t stand|cannot stand|avoids?|allergic to)\b)"
        r"|(?i:\b(?:do(?:es)? not|don'?t|doesn'?t|never)\s+(?:really\s+)?like\b)"
    )
    preference_pat += r"|(?i:\b(?:likes?|loves?|prefers?|enjoys?|favorites?|favourites?|is into)\b)"
    habit_pat += (
        r"|(?i:\b(?:usually|often|always|every (?:day|week|weekend|morning|evening)|"
        r"on weekends?|habits?|routine)\b)"
    )
    communication_pat += (
        r"|(?i:\b(?:in advance|ahead of time|last.?minute|messages?|repl(?:y|ies)|"
        r"talk|tone|a day ahead)\b)"
    )
    plan_pat += r"|(?i:\b(?:plans?|planning|going to|intends? to|next week)\b)"
    event_pat += r"|(?i:\b(?:birthday|anniversary|wedding|graduation)\b)"
    person_place_pat += (
        r"|(?i:\b(?:friends?|colleagues?|moms?|mums?|dad|mother|father|family|lives? in)\b)"
    )

    # Negation guard: "does not like" must not read as a preference.
    negated = r"(不喜欢|讨厌|(?i:\b(?:dislikes?|hates?|not|never|n'?t|no longer|can'?t stand)\b))"

    if re.search(aversion_pat, text):
        # "不喜欢" should beat bare "喜欢"
        return ClassifyResult("aversion", subject, 0.92, "matched aversion/dislike keywords", "rule")
    if re.search(preference_pat, text) and not re.search(negated, text):
        return ClassifyResult("preference", subject, 0.9, "matched preference keywords", "rule")
    if re.search(communication_pat, text) and re.search(
        r"(不|别|最好|提前|临时|(?i:\b(?:not|don'?t|best|better|in advance|ahead|last.?minute|prefer)\b))",
        text,
    ):
        return ClassifyResult("communication", subject, 0.82, "matched communication preference", "rule")
    if re.search(habit_pat, text):
        return ClassifyResult("habit", subject, 0.8, "matched habit keywords", "rule")
    if re.search(plan_pat, text):
        return ClassifyResult("plan", subject, 0.78, "matched plan keywords", "rule")
    if re.search(event_pat, text):
        return ClassifyResult("event", subject, 0.75, "matched event keywords", "rule")
    if re.search(person_place_pat, text):
        return ClassifyResult("person_place", subject, 0.72, "matched person/place", "rule")
    return ClassifyResult("fact", subject, 0.55, "no strong feature matched; recorded as fact", "rule")


def _guess_subject(text: str) -> str | None:
    # 中文姓名常见 2-3 字 + 谓语；英文姓名常见首字母大写 + 谓语。
    m = re.match(
        r"^([\u4e00-\u9fff]{2,3})"
        r"(?=不喜欢|不爱|讨厌|厌恶|很|挺|特别|其实|一般|通常|经常|喜欢|爱看|爱吃|爱喝|会|要|的)",
        text,
    )
    if m:
        return m.group(1)
    m = re.match(
        r"^([A-Z][a-zA-Z'’\-]{1,20})\s+"
        r"(?:doesn'?t|does not|does|isn'?t|is not|is|was|likes?|loves?|hates?|dislikes?|"
        r"prefers?|enjoys?|usually|often|always|can|can'?t|cannot|will|plans?|lives?|"
        r"wants?|needs?|has|have|had)\b",
        text,
    )
    if m:
        return m.group(1)
    return None


def type_label(memory_type: str) -> str:
    return TYPE_META.get(memory_type, {}).get("label", memory_type)
