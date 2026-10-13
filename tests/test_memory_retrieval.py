from __future__ import annotations

import json
import os
import subprocess
import sys

from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryStatus
from app.services import memory as memory_svc
from app.services.extraction import build_chunks, run_extraction
from app.services.import_service import commit_import
from app.services.prediction import answer_question, predict_tendency
from app.services.retrieval import hybrid_search, local_embed, search_memories_fts


def test_memory_confirm_and_list(db: Session):
    a = memory_svc.create_manual_memory(db, "likes unsweetened Americano", memory_type="preference")
    b = memory_svc.create_manual_memory(db, "likes unsweetened Americano coffee", memory_type="preference")
    assert a.status == "confirmed"
    assert b.status == "confirmed"
    listed = memory_svc.list_memories(db, status="confirmed")
    assert len(listed) == 2
    by_id = memory_svc.list_memories(db, status="confirmed", sort="id", direction="asc")
    assert [m.id for m in by_id] == sorted(m.id for m in by_id)
    page1 = memory_svc.list_memories(db, status="confirmed", sort="id", direction="asc", limit=1, offset=0)
    page2 = memory_svc.list_memories(db, status="confirmed", sort="id", direction="asc", limit=1, offset=1)
    assert len(page1) == 1 and len(page2) == 1
    assert page1[0].id != page2[0].id
    assert memory_svc.count_memories(db, status="confirmed") == 2


def test_candidate_reject_not_in_profile(db: Session):
    cand = Memory(
        content="dislikes cilantro",
        memory_type="aversion",
        status=MemoryStatus.CANDIDATE.value,
        confidence=0.9,
    )
    db.add(cand)
    db.commit()
    memory_svc.reject_memory(db, cand.id)
    profile = memory_svc.list_memories(db, status="confirmed")
    assert all(m.id != cand.id for m in profile)


def test_confirm_candidate_with_edit(db: Session):
    cand = Memory(
        content="likes Americano",
        memory_type="preference",
        status=MemoryStatus.CANDIDATE.value,
        confidence=0.7,
    )
    db.add(cand)
    db.commit()
    confirmed = memory_svc.confirm_memory(db, cand.id, content="likes unsweetened Americano")
    assert confirmed.status == "confirmed"
    assert confirmed.content == "likes unsweetened Americano"
    assert confirmed.embedding is not None


def test_extraction_offline_chunks(db: Session, tmp_path):
    sample = tmp_path / "chat.json"
    sample.write_text(
        '[{"sender":"Alex","content":"wants to watch movies at home this weekend, phone 13800138000","timestamp":"2026-03-01 10:00:00"},'
        '{"sender":"Alex","content":"dislikes changing plans last minute","timestamp":"2026-03-01 10:05:00"}]',
        encoding="utf-8",
    )
    job = commit_import(db, sample, subject_name="Alex")
    chunks = build_chunks(db, job.id)
    assert len(chunks) >= 1
    assert "13800138000" not in chunks[0].redacted_text
    assert "[phone number]" in chunks[0].redacted_text
    run = run_extraction(db, job.id, subject_name="Alex")
    assert run.status == "completed_no_llm"


def test_prediction_refuses_without_evidence(db: Session):
    result = predict_tendency(db, "A last-minute dinner invite on Friday")
    assert result.can_answer is False
    assert result.refusal_reason
    reason = (result.refusal_reason or "").strip()
    # A humane refusal: never a bare "can't tell" and long enough to explain itself.
    assert reason.lower() not in {"cannot tell", "can't tell", "no idea"}
    assert len(reason) > 20


def test_prediction_offline_with_memories(db: Session):
    memory_svc.create_manual_memory(
        db, "does not want to change plans last minute after working late on Friday", memory_type="habit", confidence=0.9
    )
    memory_svc.create_manual_memory(
        db, "best to mention dinner a day ahead", memory_type="preference", confidence=0.85
    )
    memory_svc.create_manual_memory(db, "prefers staying home on weekends", memory_type="preference", confidence=0.8)
    result = predict_tendency(db, "A last-minute dinner invite on Friday night")
    assert result.can_answer in (True, False)
    if result.can_answer:
        assert result.supporting_evidence
        assert result.confidence > 0


def test_qa_uses_aversion_memories(db: Session):
    memory_svc.create_manual_memory(
        db, "Alex is afraid of horror movies", memory_type="aversion", confidence=1.0
    )
    memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion", confidence=0.95)
    memory_svc.create_manual_memory(
        db, "replies to messages while gaming", memory_type="habit", confidence=0.7
    )
    hits = hybrid_search(db, "What does Alex dislike?", include_messages=False, limit=8)
    assert any("horror" in h.content or "cilantro" in h.content for h in hits)
    qa = answer_question(db, "What does Alex dislike?")
    assert "horror" in qa["answer"] or "cilantro" in qa["answer"]
    assert "cannot tell" not in qa["answer"].lower()
    assert qa["can_answer"] is True
    assert "memory#" not in qa["answer"].lower()
    assert "[aversion]" not in qa["answer"]


def test_hybrid_search_and_qa(db: Session):
    memory_svc.create_manual_memory(db, "likes unsweetened Americano", memory_type="preference")
    memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion")
    hits = hybrid_search(db, "Americano coffee", include_messages=False, limit=5)
    assert any("Americano" in h.content for h in hits)
    qa = answer_question(db, "What does she like to drink?")
    assert qa["evidence"] or "Americano" in qa["answer"]


def test_cjk_fts_falls_back_to_bigrams(db: Session):
    # Deliberately CJK: unicode61 does not segment Chinese into word tokens, so
    # this is the only test covering the LIKE/bigram fallback path.
    memory_svc.create_manual_memory(
        db,
        "她周末喜欢在家看电影",
        memory_type="preference",
    )
    hits = search_memories_fts(db, "她周末通常做什么", limit=5)
    assert any("周末" in hit.content for hit in hits)


def test_local_embed_stable():
    a = local_embed("likes unsweetened Americano")
    b = local_embed("likes unsweetened Americano")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_local_embed_stable_across_process_hash_seeds():
    code = (
        "import json; "
        "from app.services.retrieval import local_embed; "
        "print(json.dumps(local_embed('likes unsweetened Americano')))"
    )
    vectors = []
    for seed in ("1", "999"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        output = subprocess.check_output(
            [sys.executable, "-c", code],
            text=True,
            env=env,
        )
        vectors.append(json.loads(output))
    assert vectors[0] == vectors[1]


def test_create_manual_merges_duplicate(db: Session):
    a = memory_svc.create_manual_memory(
        db, "Alex is afraid of horror movies", memory_type="aversion", confidence=0.95
    )
    b = memory_svc.create_manual_memory(
        db, "Alex is afraid of horror movies", memory_type="aversion", confidence=1.0
    )
    assert a.id == b.id
    db.refresh(a)
    assert a.confidence == 1.0
    confirmed = memory_svc.list_memories(db, status="confirmed")
    assert len(confirmed) == 1


def test_dedupe_confirmed_identical(db: Session):
    db.add(
        Memory(
            content="Alex is afraid of horror movies",
            memory_type="aversion",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.95,
        )
    )
    db.add(
        Memory(
            content="Alex is afraid of horror movies",
            memory_type="aversion",
            status=MemoryStatus.CONFIRMED.value,
            confidence=1.0,
        )
    )
    db.commit()
    removed = memory_svc.dedupe_confirmed(db)
    assert removed == 1
    kept = memory_svc.list_memories(db, status="confirmed")
    assert len(kept) == 1
    assert kept[0].confidence == 1.0


def test_dedupe_candidates(db: Session):
    db.add(
        Memory(
            content="dislikes cilantro",
            memory_type="aversion",
            status=MemoryStatus.CANDIDATE.value,
            confidence=0.6,
        )
    )
    db.add(
        Memory(
            content="dislikes cilantro",
            memory_type="aversion",
            status=MemoryStatus.CANDIDATE.value,
            confidence=0.9,
        )
    )
    db.commit()
    removed = memory_svc.dedupe_candidates(db)
    assert removed == 1
