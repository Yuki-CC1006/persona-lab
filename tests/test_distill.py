from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryStatus
from app.services.distill import distill_user_message, looks_like_correction, looks_like_teaching


def test_questions_are_not_teaching():
    assert looks_like_teaching("What does Alex dislike?") is False
    assert looks_like_teaching("What does she like to drink?") is False


def test_statements_are_teaching():
    assert looks_like_teaching("Remember: Alex dislikes cilantro") is True
    assert looks_like_teaching("Actually she doesn't like horror movies") is True


def test_ordinary_statement_does_not_write_profile():
    assert looks_like_teaching("She went to the movies with friends today") is False


def test_correction_phrases():
    assert looks_like_correction("No — she can drink matcha now") is True
    assert looks_like_correction("You got that wrong, that was before") is True
    assert looks_like_teaching("No — she can drink matcha now") is True


def test_distill_writes_confirmed_profile(db: Session):
    result = distill_user_message(db, "Remember: Alex dislikes cilantro")
    assert result.added_count == 1
    mem = result.memories[0]
    assert mem.status == MemoryStatus.CONFIRMED.value
    assert mem.embedding is not None
    assert "cilantro" in mem.content


def test_distill_skips_duplicate(db: Session):
    db.add(
        Memory(
            content="Alex dislikes cilantro",
            memory_type="aversion",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.9,
        )
    )
    db.commit()
    result = distill_user_message(db, "Remember: Alex dislikes cilantro")
    assert result.changes == []


def test_distill_skips_pure_question(db: Session):
    assert distill_user_message(db, "What does Alex dislike?").changes == []


def test_distill_corrects_existing_memory(db: Session):
    old = Memory(
        content="Alex dislikes matcha",
        memory_type="aversion",
        status=MemoryStatus.CONFIRMED.value,
        confidence=0.9,
    )
    db.add(old)
    db.commit()
    db.refresh(old)
    result = distill_user_message(
        db,
        "No — she can drink matcha now",
        history=[{"role": "assistant", "content": "Alex dislikes matcha."}],
    )
    assert result.corrected_count >= 1
    db.refresh(old)
    assert old.status == MemoryStatus.SUPERSEDED.value
    assert old.supersedes_id is not None
    new = db.get(Memory, old.supersedes_id)
    assert new is not None
    assert new.status == MemoryStatus.CONFIRMED.value
    assert "matcha" in new.content


def test_new_fact_does_not_clobber_unrelated(db: Session):
    other = Memory(
        content="Alex dislikes cilantro",
        memory_type="aversion",
        status=MemoryStatus.CONFIRMED.value,
        confidence=0.9,
    )
    db.add(other)
    db.commit()
    result = distill_user_message(db, "Remember: she usually sleeps in until noon on weekends")
    db.refresh(other)
    assert other.status == MemoryStatus.CONFIRMED.value
    assert result.added_count == 1
