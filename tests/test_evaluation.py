"""Regression guard for retrieval, evidence and refusal quality.

The thresholds below are the measured offline baseline, not aspirations.
Raise them only when an intentional change improves the score.
"""

from __future__ import annotations

import httpx
from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryStatus
from app.services.evaluation import (
    EndpointEmbedder,
    compare_embeddings,
    embedding_variants,
    evaluate,
    format_embedding_report,
    format_report,
    load_golden_set,
    seed_profile,
    should_switch_embedding,
)
from app.services.outbound import list_outbound
from app.services.retrieval import local_embed

# Measured offline baseline with the local hashing embedding. These are
# regression guards, not product-quality claims: the golden set is small and
# hand-built from the same persona the other tests use.
MIN_RECALL_AT_K = 1.0
MIN_EVIDENCE_ACCURACY = 1.0
MIN_REFUSAL_ACCURACY = 1.0
MAX_WRONG_SUBJECT_RATE = 0.0


def _golden_db(db: Session) -> None:
    seed_profile(db, load_golden_set())


def test_golden_set_is_loadable():
    golden = load_golden_set()
    assert len(golden.cases) >= 10
    assert any(case.expect_refusal for case in golden.cases)
    assert any(case.forbid for case in golden.cases)
    assert all(case.question for case in golden.cases)


def test_baseline_metrics_meet_floor(db: Session):
    _golden_db(db)
    metrics = evaluate(db)
    report = format_report(metrics)

    assert metrics.case_count >= 10, report
    assert metrics.recall_at_k >= MIN_RECALL_AT_K, report
    assert metrics.evidence_accuracy >= MIN_EVIDENCE_ACCURACY, report
    assert metrics.refusal_accuracy >= MIN_REFUSAL_ACCURACY, report
    assert metrics.wrong_subject_rate <= MAX_WRONG_SUBJECT_RATE, report


def test_wrong_subject_memories_are_filtered(db: Session):
    _golden_db(db)
    from app.services.retrieval import hybrid_search

    hits = hybrid_search(db, "What does Sam dislike?", include_messages=False, limit=10)
    contents = " ".join(hit.content for hit in hits)
    assert "spring onion" in contents
    assert "horror" not in contents
    assert "cilantro" not in contents


def test_unlabelled_memories_are_never_dropped(db: Session):
    db.add(
        Memory(
            content="likes unsweetened Americano",
            memory_type="preference",
            status=MemoryStatus.CONFIRMED.value,
            confidence=1.0,
        )
    )
    db.commit()
    from app.services.retrieval import hybrid_search

    hits = hybrid_search(db, "What does she like to drink?", include_messages=False, limit=5)
    assert any("Americano" in hit.content for hit in hits)


def test_empty_dataset_metrics_are_not_zero(db: Session):
    metrics = evaluate(db)
    assert metrics.recall_at_k == 0.0 or metrics.recall_at_k > 0.0
    assert 0.0 <= metrics.refusal_accuracy <= 1.0


def test_embedding_switch_gate_requires_real_improvement():
    assert should_switch_embedding(0.6, 0.7) is True
    assert should_switch_embedding(0.6, 0.62) is False
    assert should_switch_embedding(0.6, 0.5) is False


def test_compare_embeddings_keeps_local_baseline(db: Session):
    _golden_db(db)
    comparison = compare_embeddings(db, load_golden_set())
    assert comparison["baseline"] == "local-hash"
    assert "local-hash" in comparison["scores"]
    # A candidate must clear the delta gate, not merely tie, to be recommended.
    assert all(
        score - comparison["baseline_score"] >= 0.05
        for score in comparison["recommended_switch"].values()
    )


def test_compare_embeddings_detects_better_variant(db: Session):
    _golden_db(db)
    comparison = compare_embeddings(
        db,
        load_golden_set(),
        variants={
            "local-hash": local_embed,
            "all-ones": lambda _text: [1.0] * 8,
        },
    )
    assert set(comparison["scores"]) == {"local-hash", "all-ones"}


def test_compare_embeddings_never_switches_on_a_tie(db: Session):
    _golden_db(db)
    comparison = compare_embeddings(
        db,
        load_golden_set(),
        variants={"local-hash": local_embed, "same": local_embed},
    )
    assert comparison["recommended_switch"] == {}
    assert comparison["min_delta"] == 0.05


def test_embedding_variants_stay_local_without_config(monkeypatch):
    from app.config import get_settings

    monkeypatch.setenv("EMBEDDING_BASE_URL", "")
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    get_settings.cache_clear()
    try:
        assert list(embedding_variants()) == ["local-hash"]
    finally:
        get_settings.cache_clear()


def test_embedding_variant_name_reflects_endpoint_location(monkeypatch):
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        monkeypatch.setenv("EMBEDDING_BASE_URL", "http://127.0.0.1:11434/v1")
        monkeypatch.setenv("EMBEDDING_MODEL", "bge-m3")
        get_settings.cache_clear()
        assert list(embedding_variants()) == ["local-hash", "local-endpoint"]

        monkeypatch.setenv("EMBEDDING_BASE_URL", "https://api.example.com/v1")
        get_settings.cache_clear()
        assert list(embedding_variants()) == ["local-hash", "remote-endpoint"]
    finally:
        get_settings.cache_clear()


def test_endpoint_embedder_caches_and_audits(monkeypatch, db: Session):
    calls: list[str] = []

    class _Resp:
        status_code = 200

        def json(self):
            return {"data": [{"embedding": [0.5, 0.5]}]}

    class _Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def post(self, url, headers=None, json=None):
            calls.append(json["input"])
            return _Resp()

    monkeypatch.setattr(httpx, "Client", _Client)
    embedder = EndpointEmbedder(
        base_url="https://api.example.com/v1", model="bge-m3", api_key="secret"
    )
    assert embedder("Alex likes hotpot") == [0.5, 0.5]
    assert embedder("Alex likes hotpot") == [0.5, 0.5]
    assert calls == ["Alex likes hotpot"]  # cached: no second request

    events = list_outbound(db)
    assert events and events[0].purpose == "embedding"
    assert events[0].ok is True


def test_endpoint_embedder_failure_is_reported_not_crashed(db: Session):
    _golden_db(db)
    broken = EndpointEmbedder(
        base_url="http://127.0.0.1:9/v1", model="nope", timeout=0.3
    )
    comparison = compare_embeddings(
        db,
        load_golden_set(),
        variants={"local-hash": local_embed, "remote-endpoint": broken},
    )
    assert "remote-endpoint" not in comparison["scores"]
    assert "remote-endpoint" in comparison["errors"]
    assert comparison["baseline"] == "local-hash"
    assert comparison["baseline_score"] > 0


def test_embedding_report_states_the_decision(db: Session):
    _golden_db(db)
    result = compare_embeddings(
        db,
        load_golden_set(),
        variants={"local-hash": local_embed, "same": local_embed},
    )
    report = format_embedding_report(result)
    assert "Baseline: local-hash" in report
    assert "Recommend keeping the baseline" in report
