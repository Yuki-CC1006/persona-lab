from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_db, reset_engine_for_tests
from app.main import create_app


@pytest.fixture()
def db(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    upload = tmp_path / "uploads"
    upload.mkdir(exist_ok=True)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("UPLOAD_DIR", str(upload))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("LLM_API_KEY", "")
    get_settings.cache_clear()
    reset_engine_for_tests(f"sqlite:///{db_path}")
    from app.db.session import get_session_factory

    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
        get_settings.cache_clear()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "web.db"
    upload = tmp_path / "uploads"
    upload.mkdir(exist_ok=True)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("UPLOAD_DIR", str(upload))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.setenv("LLM_API_KEY", "")
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    get_settings.cache_clear()
    reset_engine_for_tests(f"sqlite:///{db_path}")

    app = create_app()

    def _override_db():
        from app.db.session import get_session_factory

        session = get_session_factory()()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = _override_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
    get_settings.cache_clear()
