"""The scope gate must refuse out-of-scope questions on every path.

Regression guard for a real defect: the golden set marked two questions as
"must refuse", but that only held when no API key was configured. With a key,
the prompt told the model to answer anything, so the baseline's 100% refusal
accuracy described a code path users with a key never hit.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.services import memory as memory_svc
from app.services.evaluation import (
    _OFFLINE_LLM,
    evaluate,
    format_report,
    load_golden_set,
    seed_profile,
)
from app.services.prediction import answer_question, predict_tendency
from app.services.scope import REFUSAL_MESSAGE, classify_scope, known_subjects


class _AnswerAnythingLLM:
    """Stands in for the API-key path, where the model used to decide alone."""

    configured = True
    calls = 0

    def chat_json(self, *_args, **_kwargs):
        type(self).calls += 1
        return {"can_answer": True, "tendency": "Sure, I can answer that directly.", "confidence": 0.8}

    def chat_text(self, *_args, **_kwargs):
        type(self).calls += 1
        return "A generic answer unrelated to the profile."

    def chat_stream(self, *_args, **_kwargs):
        type(self).calls += 1
        yield ("token", "A generic answer unrelated to the profile.")


@pytest.fixture()
def profile(db: Session) -> None:
    seed_profile(db, load_golden_set())


OUT_OF_SCOPE = ["What is the decoherence mechanism in quantum computing?", "Write me a quicksort in python"]
IN_SCOPE = [
    "What does Alex dislike?",
    "What does she like to drink?",
    "What should I keep in mind when asking Alex out?",
    "Where should we go this weekend?",
    "Has she seemed a bit unhappy lately?",
]


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_out_of_scope_is_refused_even_when_a_model_is_configured(profile, db, question):
    llm = _AnswerAnythingLLM()
    llm.calls = 0
    assert answer_question(db, question, llm=llm)["can_answer"] is False
    assert predict_tendency(db, question, llm=llm).can_answer is False
    # The decision is local: the model is never consulted.
    assert llm.calls == 0


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_out_of_scope_is_refused_offline(profile, db, question):
    assert answer_question(db, question, llm=_OFFLINE_LLM)["can_answer"] is False
    assert predict_tendency(db, question, llm=_OFFLINE_LLM).can_answer is False


@pytest.mark.parametrize("question", IN_SCOPE)
def test_in_scope_questions_are_never_refused(profile, db, question):
    assert classify_scope(db, question).in_scope is True
    assert answer_question(db, question, llm=_AnswerAnythingLLM())["can_answer"] is True


def test_in_scope_signal_wins_over_a_task_request(profile, db):
    # A writing request about her is still about her.
    verdict = classify_scope(db, "Help me write a message to get to know them better")
    assert verdict.in_scope is True
    assert verdict.reason == "portrait-vocab"


def test_pronoun_only_questions_stay_in_scope(profile, db):
    assert classify_scope(db, "What about them?").in_scope is True


def test_greetings_are_not_refused(profile, db):
    for text in ("hello", "hey", "thanks", "good evening~"):
        assert classify_scope(db, text).in_scope is True, text


def test_short_follow_up_is_in_scope_only_with_history(profile, db):
    assert classify_scope(db, "Why is that?").in_scope is False
    assert classify_scope(db, "Why is that?", history=[{"role": "user", "content": "What does she like?"}]).in_scope is True


def test_short_follow_up_does_not_excuse_a_foreign_request(profile, db):
    history = [{"role": "user", "content": "What does she like?"}]
    # Longer than the follow-up window, so the foreign cue is judged on its own.
    verdict = classify_scope(db, "Then write me a quicksort in python", history=history)
    assert verdict.in_scope is False
    assert verdict.reason == "foreign-cue"


def test_unrecognised_text_is_refused_with_guidance(profile, db):
    verdict = classify_scope(db, "Nice weather today")
    assert verdict.in_scope is False
    assert verdict.reason == "unrecognised"


def test_lexical_knowledge_base_evidence_is_an_in_scope_signal(profile, db):
    # No pronoun, no portrait word — only the memory itself matches.
    assert classify_scope(db, "unsweetened Americano").in_scope is True


def test_semantic_noise_alone_does_not_open_the_gate(profile, db):
    """The hash embedding returns ~0.05 noise for unrelated text.

    If the gate trusted semantic hits, every question would look in scope.
    """
    verdict = classify_scope(db, "How do I solve the Schrodinger equation")
    assert verdict.in_scope is False


def test_empty_question_is_left_to_the_normal_path(profile, db):
    assert classify_scope(db, "   ").in_scope is True


def test_known_subjects_come_from_confirmed_memories(db):
    memory_svc.create_manual_memory(db, "Sam dislikes spring onion", memory_type="aversion", subject="Sam")
    assert known_subjects(db) == ["Sam"]


def test_refusal_message_explains_the_boundary(profile, db):
    message = answer_question(db, "Write me a quicksort in python", llm=_AnswerAnythingLLM())["answer"]
    assert "I can't tell" not in message
    assert message == REFUSAL_MESSAGE
    # It still explains the boundary and offers a way forward instead of a dead end.
    assert len(message) >= 40


def test_gate_survives_a_broken_search_index(profile, db, monkeypatch):
    from app.services import scope as scope_svc

    def _boom(*_args, **_kwargs):
        raise RuntimeError("index down")

    monkeypatch.setattr("app.services.retrieval.search_memories_fts", _boom)
    # Falls back to the remaining signals rather than crashing.
    assert scope_svc.classify_scope(db, "What does Alex dislike?").in_scope is True
    assert scope_svc.classify_scope(db, "Write me some code").in_scope is False


def test_golden_baseline_still_passes_with_the_gate(profile, db):
    metrics = evaluate(db)
    report = format_report(metrics)
    assert metrics.refusal_accuracy >= 1.0, report
    assert metrics.recall_at_k >= 1.0, report


def test_both_paths_agree_on_every_golden_case(profile, db):
    """The gap that caused the defect: offline and configured must not diverge."""
    for case in load_golden_set().cases:
        if case.mode == "tendency":
            offline = predict_tendency(db, case.question, llm=_OFFLINE_LLM).can_answer
            configured = predict_tendency(db, case.question, llm=_AnswerAnythingLLM()).can_answer
        else:
            offline = answer_question(db, case.question, llm=_OFFLINE_LLM)["can_answer"]
            configured = answer_question(db, case.question, llm=_AnswerAnythingLLM())["can_answer"]
        expected = not case.expect_refusal
        assert offline == expected, f"{case.id} offline path disagreed"
        assert configured == expected, f"{case.id} disagreed once a model was configured"


def test_web_refuses_an_out_of_scope_question(client: TestClient, db: Session):
    memory_svc.create_manual_memory(db, "Sam dislikes spring onion", memory_type="aversion", subject="Sam")
    resp = client.post(
        "/chat/ask",
        data={"question": "Write me a quicksort in python", "mode": "ask"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    from html import unescape

    assert REFUSAL_MESSAGE in unescape(resp.text)
