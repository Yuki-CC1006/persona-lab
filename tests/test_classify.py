from __future__ import annotations

from app.services.classify import classify_memory_text
from app.services import memory as memory_svc


def test_rule_classify_aversion():
    result = classify_memory_text(
        "Alex dislikes horror movies", use_llm=False
    )
    assert result.memory_type == "aversion"
    assert result.subject == "Alex"
    assert result.source == "rule"


def test_rule_classify_preference():
    result = classify_memory_text("Alex likes unsweetened Americano", use_llm=False)
    assert result.memory_type == "preference"


def test_create_auto_classify(db):
    mem = memory_svc.create_manual_memory(
        db,
        "Alex dislikes horror movies",
        memory_type="auto",
        auto_classify=True,
    )
    assert mem.memory_type == "aversion"
    assert mem.subject == "Alex"
    assert mem.notes and "Auto-classified" in mem.notes
