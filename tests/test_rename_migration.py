"""The rename must not look like data loss.

Renaming the project also renamed the database file. If the app simply started
using the new filename while a user's only database was still called
``rxy_wiki.db``, the UI would come up empty and the data would look deleted.
These tests pin the one-time adoption of the legacy file, plus the identifiers
the new name is made of.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from app.branding import DB_FILENAME, LEGACY_DB_FILENAME, PRODUCT_NAME, PRODUCT_SLUG
from app.config import ROOT_DIR, Settings, sqlite_file_path
from app.db.migrations import adopt_legacy_database


def _make_db(path: Path, marker: str = "kept") -> None:
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE memories (content TEXT)")
    connection.execute("INSERT INTO memories VALUES (?)", (marker,))
    connection.commit()
    connection.close()


def _contents(path: Path) -> str:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return connection.execute("SELECT content FROM memories").fetchone()[0]
    finally:
        connection.close()


def test_legacy_file_is_adopted_when_the_new_one_is_absent(tmp_path):
    legacy = tmp_path / LEGACY_DB_FILENAME
    target = tmp_path / DB_FILENAME
    _make_db(legacy)

    adopted = adopt_legacy_database(target, [legacy])

    assert adopted == legacy
    assert not legacy.exists()
    assert _contents(target) == "kept"


def test_sidecar_files_follow_the_database(tmp_path):
    """A ``-wal`` left behind holds committed pages that would never be read."""
    legacy = tmp_path / LEGACY_DB_FILENAME
    target = tmp_path / DB_FILENAME
    _make_db(legacy)
    (tmp_path / (LEGACY_DB_FILENAME + "-wal")).write_text("wal", encoding="utf-8")
    (tmp_path / (LEGACY_DB_FILENAME + "-shm")).write_text("shm", encoding="utf-8")

    adopt_legacy_database(target, [legacy])

    assert (tmp_path / (DB_FILENAME + "-wal")).read_text(encoding="utf-8") == "wal"
    assert (tmp_path / (DB_FILENAME + "-shm")).read_text(encoding="utf-8") == "shm"


def test_an_existing_database_is_never_clobbered(tmp_path):
    legacy = tmp_path / LEGACY_DB_FILENAME
    target = tmp_path / DB_FILENAME
    _make_db(legacy, "old")
    _make_db(target, "current")

    assert adopt_legacy_database(target, [legacy]) is None
    assert _contents(target) == "current"
    assert _contents(legacy) == "old"


@pytest.mark.parametrize("candidates", [[], [Path("does-not-exist")]])
def test_nothing_happens_without_a_legacy_file(tmp_path, candidates):
    target = tmp_path / DB_FILENAME
    assert adopt_legacy_database(target, candidates) is None
    assert not target.exists()


def test_the_target_is_not_treated_as_its_own_legacy(tmp_path):
    target = tmp_path / DB_FILENAME
    _make_db(target)
    assert adopt_legacy_database(target, [target]) is None
    assert _contents(target) == "kept"


@pytest.mark.parametrize(
    "url", ["sqlite:///:memory:", "sqlite://", "postgresql://localhost/db"]
)
def test_non_file_databases_have_no_path(url):
    assert sqlite_file_path(url) is None


def test_sqlite_url_yields_a_path():
    assert str(sqlite_file_path("sqlite:///./data/persona_lab.db")) == "data/persona_lab.db"


def test_default_database_uses_the_new_filename():
    url = Settings().database_url
    assert url.endswith(f"/data/{DB_FILENAME}")
    assert LEGACY_DB_FILENAME not in url


def test_the_previous_name_is_distinct_and_only_used_for_migration():
    assert LEGACY_DB_FILENAME != DB_FILENAME


def test_backup_format_and_extension_are_renamed():
    from app.services import backup

    assert backup.FORMAT == f"{PRODUCT_SLUG}-backup"
    assert backup.BACKUP_EXT == ".personabak"
    assert backup.FORMAT != "rxy-wiki-backup"


def test_mcp_surface_is_renamed():
    from app.mcp import readonly

    assert readonly.SERVER_NAME == f"{PRODUCT_SLUG}-readonly"
    assert readonly.SEARCH_TOOL == "persona_search"
    assert readonly.PROFILE_TOOL == "persona_profile"
    assert len(readonly.TOOL_NAMES) == 2


def test_product_identity_is_centralised():
    assert PRODUCT_NAME == "Persona Lab"
    assert PRODUCT_SLUG == "persona-lab"


def test_no_stale_name_survives_in_shipped_source():
    """A leftover old name is the likeliest way a rename half-lands."""
    offenders = []
    for folder in ("app", "scripts"):
        for path in (ROOT_DIR / folder).rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".html", ".js", ".css"}:
                continue
            if path.name == "branding.py":
                continue
            if "rxy" in path.read_text(encoding="utf-8").lower():
                offenders.append(str(path.relative_to(ROOT_DIR)))
    assert offenders == []
