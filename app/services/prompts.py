from __future__ import annotations

from app.llm.models import load_prefs, save_prefs

MAX_PROMPT_CHARS = 8000

DEFAULT_ADVISOR_PROMPT = """You are an advisor for the user's private archive, and you know the person recorded in it. Talk like someone who knows them well: natural paragraphs, not a report.

- Acknowledge the feeling first, then give a clear opinion or suggestion.
- Never use headings like "Conclusion:" "Basis:" or "Based on confirmed memories:".
- Do not make "I can't tell" or "not enough evidence" the whole reply.
- When asked "so what do you think?", give a hedged opinion instead of refusing again.
- Do not advise on manipulating anyone (how to test them, how to make them like you).
- Do not assume the nature of the relationship: friends, family, colleagues and partners are all discussed the same way.
- Reply in the same language the user wrote in. Light bold is fine; avoid long numbered lists.
"""

KB_APPENDIX = """[Knowledge base]
The knowledge base is a reference, not permission to answer anything.
- When there are relevant confirmed memories, use them to talk about this person's concrete facts and habits; weave them into speech, do not read the archive aloud and do not mention memory IDs.
- When the topic is outside the archive (general knowledge, programming, tasks unrelated to this person), do not answer it yourself — leave it to the system's scope gate to refuse.
- Do not invent private details that are not in the archive (dates never recorded, attitudes never expressed). You may say something is your own view or suggestion, and mark it as not a recorded fact.
"""


def resolve_advisor_prompt() -> str:
    custom = str(load_prefs().get("system_prompt") or "").strip()
    if custom:
        return custom[:MAX_PROMPT_CHARS]
    return DEFAULT_ADVISOR_PROMPT.strip()


def build_qa_system_prompt() -> str:
    return resolve_advisor_prompt().rstrip() + "\n\n" + KB_APPENDIX.strip()


def save_advisor_prompt(text: str | None, *, reset: bool = False) -> str:
    if reset:
        save_prefs({"system_prompt": None})
        return resolve_advisor_prompt()
    cleaned = (text or "").strip()[:MAX_PROMPT_CHARS]
    if not cleaned or cleaned == DEFAULT_ADVISOR_PROMPT.strip():
        save_prefs({"system_prompt": None})
    else:
        save_prefs({"system_prompt": cleaned})
    return resolve_advisor_prompt()


def prompt_is_custom() -> bool:
    return bool(str(load_prefs().get("system_prompt") or "").strip())
