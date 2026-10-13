from __future__ import annotations

from html import unescape

from fastapi.testclient import TestClient

from app.services.prediction import answer_question, iter_answer_events
from app.services.prompts import (
    DEFAULT_ADVISOR_PROMPT,
    KB_APPENDIX,
    build_qa_system_prompt,
    prompt_is_custom,
    resolve_advisor_prompt,
    save_advisor_prompt,
)


class _FakeLLM:
    configured = True

    def __init__(self) -> None:
        self.system = ""
        self.user = ""

    def chat_stream(self, system: str, user: str, temperature: float = 0.45, **_kwargs):
        self.system = system
        self.user = user
        yield (
            "token",
            "An anniversary doesn't have to land on a lucky date — pick the day you both "
            "remember and actually want to celebrate.",
        )


def test_default_prompt_includes_kb_appendix(db):
    save_advisor_prompt(None, reset=True)
    text = build_qa_system_prompt()
    assert DEFAULT_ADVISOR_PROMPT.strip() in text
    assert KB_APPENDIX.strip() in text
    assert not prompt_is_custom()


def test_default_prompt_does_not_assume_romance(db):
    """The product is a persona lab, not a romance chatbot."""
    text = build_qa_system_prompt()
    assert "girlfriend" not in text.lower()
    assert "boyfriend" not in text.lower()
    # No gendered pronoun either: any subject must fit.
    assert " she " not in text.lower()
    assert " he " not in text.lower()
    assert "them" in text.lower() or "the other person" in text.lower()


def test_default_subject_is_not_gendered():
    from app.services.onboarding import DEFAULT_SUBJECT

    assert DEFAULT_SUBJECT == "the other person"


def test_custom_prompt_is_used(db):
    save_advisor_prompt("You are a very direct relationship coach. Less comfort, more concrete advice.")
    assert prompt_is_custom()
    assert resolve_advisor_prompt().startswith("You are a very direct relationship coach")
    system = build_qa_system_prompt()
    assert "relationship coach" in system
    assert KB_APPENDIX.strip() in system
    save_advisor_prompt(None, reset=True)
    assert resolve_advisor_prompt() == DEFAULT_ADVISOR_PROMPT.strip()


def test_qa_calls_llm_without_kb_hits(db):
    llm = _FakeLLM()
    events = list(iter_answer_events(db, "Which day in September works for our anniversary?", llm=llm))
    done = [e for e in events if e.get("type") == "done"][-1]
    assert llm.system
    assert KB_APPENDIX.strip() in llm.system
    assert "No knowledge-base entries directly matched this question" in llm.user
    assert "lucky date" in done["answer"]
    assert done["can_answer"] is True

    qa = answer_question(db, "How should a regular friend start a date?", llm=llm)
    assert "I can't tell" not in qa["answer"]
    assert qa["can_answer"] is True


def test_settings_prompt_roundtrip(client: TestClient):
    page = client.get("/settings")
    assert page.status_code == 200
    assert "Advisor prompt" in page.text
    # The textarea escapes quotes, so compare against the unescaped page.
    assert DEFAULT_ADVISOR_PROMPT.strip() in unescape(page.text)

    resp = client.post(
        "/settings/prompt",
        data={
            "system_prompt": "You are a blunt but kind advisor.",
            "next": "/settings",
        },
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    assert resolve_advisor_prompt() == "You are a blunt but kind advisor."

    saved = client.get("/settings?saved=prompt")
    assert "blunt but kind" in saved.text
    assert "Advisor prompt saved" in saved.text

    reset = client.post(
        "/settings/prompt",
        data={
            "system_prompt": "You are a blunt but kind advisor.",
            "reset": "1",
            "next": "/settings",
        },
        follow_redirects=False,
    )
    assert reset.status_code in (302, 303)
    assert resolve_advisor_prompt() == DEFAULT_ADVISOR_PROMPT.strip()
    assert not prompt_is_custom()
