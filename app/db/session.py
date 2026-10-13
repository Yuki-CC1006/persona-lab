from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.migrations import init_db
from app.db.models import create_db_engine, make_session_factory

_engine = None
_SessionLocal = None


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = create_db_engine(settings.database_url)
        _SessionLocal = make_session_factory(_engine)
        init_db(_engine)
    return _engine


def get_session_factory():
    get_engine()
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    factory = get_session_factory()
    db = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reset_engine_for_tests(database_url: str) -> None:
    """Replace the global engine — used by tests only."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = create_db_engine(database_url)
    _SessionLocal = make_session_factory(_engine)
    init_db(_engine)
