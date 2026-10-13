from app.services.prediction import (
    _extractive_answer,
    _humane_unknown,
    _is_relational_question,
    _looks_like_refusal,
    _strip_citation_noise,
    answer_question,
)
from app.services.retrieval import RetrievalHit
from app.templating import render_answer_html


def test_strip_memory_ids():
    raw = (
        "She hates matcha (memory#2985), and she won't eat rice noodles (#2707).\n"
        "Conclusion: quite the picky eater."
    )
    cleaned = _strip_citation_noise(raw)
    assert "2985" not in cleaned
    assert "2707" not in cleaned
    assert not cleaned.startswith("Conclusion")


def test_extractive_is_prose_not_dump():
    hits = [
        RetrievalHit(
            "memory", 1, "Alex is afraid of horror movies", 1.0, {"memory_type": "aversion"}
        ),
        RetrievalHit("memory", 2, "dislikes cilantro", 0.9, {"memory_type": "aversion"}),
    ]
    text = _extractive_answer("What does Alex dislike?", hits, [])
    assert "horror" in text and "cilantro" in text
    assert "[aversion]" not in text
    assert "memory#" not in text


def test_answer_md_paragraphs():
    html = str(render_answer_html("First paragraph.\n\n**Key point** here."))
    assert "<p>" in html
    assert "<strong>Key point</strong>" in html
    assert "&lt;" not in html


def test_looks_like_refusal_only_short():
    assert _looks_like_refusal("I can't tell.")
    assert _looks_like_refusal("Not enough evidence — I cannot answer.")
    prose = (
        "I can't make that call for them, and the records don't state directly how "
        "they feel about you. Different interests don't mean they dislike you. "
        "What happened recently that left you unsure?"
    )
    assert not _looks_like_refusal(prose)


def test_relational_question_is_not_bare_refusal():
    assert _is_relational_question("Do you think Alex might not like me?")
    hits = [
        RetrievalHit(
            "memory", 1, "Family is in Anhui; studying in Xi'an now", 0.8, {"memory_type": "fact"}
        ),
        RetrievalHit("memory", 2, "not much into horror movies", 0.7, {"memory_type": "aversion"}),
    ]
    text = _humane_unknown(
        "Could it just be resonance at the consumer level? Would long distance wear the affection away?",
        hits,
    )
    assert "I can't tell" not in text
    assert "like you" in text or "feel about you" in text
    assert "Anhui" in text or "Xi'an" in text or "horror" in text
    extractive = _extractive_answer("Do you think she might not like me?", hits, [])
    assert "I can't tell" not in extractive
    assert len(extractive) > 40


def test_empty_qa_is_humane(db):
    qa = answer_question(db, "Which day in September works for our anniversary?")
    assert qa["can_answer"] is True
    assert "I can't tell" not in qa["answer"]
    assert "I cannot answer" not in qa["answer"]
    assert qa["answer"]


def test_relational_qa_uses_related_memories(db):
    from app.services import memory as memory_svc

    memory_svc.create_manual_memory(db, "Family is in Anhui; studying in Xi'an now", memory_type="fact")
    memory_svc.create_manual_memory(db, "not much into horror movies", memory_type="aversion")
    qa = answer_question(
        db, "Do you think she might not like me? Would long distance wear the affection away?"
    )
    assert "I can't tell" not in qa["answer"]
    assert qa["can_answer"] is True
    assert (
        "Anhui" in qa["answer"]
        or "Xi'an" in qa["answer"]
        or "feel about you" in qa["answer"]
    )
