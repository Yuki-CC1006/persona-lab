from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db.models import Base

SCHEMA_VERSION = 4

FTS_MESSAGES_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    sender,
    content='messages',
    content_rowid='id',
    tokenize='unicode61'
);
"""

FTS_MEMORIES_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    memory_type,
    content='memories',
    content_rowid='id',
    tokenize='unicode61'
);
"""

FTS_DIARY_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS diary_fts USING fts5(
    title,
    content,
    content='diary_entries',
    content_rowid='id',
    tokenize='unicode61'
);
"""

TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
      INSERT INTO messages_fts(rowid, content, sender)
      VALUES (new.id, new.content, new.sender);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
      INSERT INTO messages_fts(messages_fts, rowid, content, sender)
      VALUES ('delete', old.id, old.content, old.sender);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
      INSERT INTO messages_fts(messages_fts, rowid, content, sender)
      VALUES ('delete', old.id, old.content, old.sender);
      INSERT INTO messages_fts(rowid, content, sender)
      VALUES (new.id, new.content, new.sender);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
      INSERT INTO memories_fts(rowid, content, memory_type)
      VALUES (new.id, new.content, new.memory_type);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
      INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
      VALUES ('delete', old.id, old.content, old.memory_type);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
      INSERT INTO memories_fts(memories_fts, rowid, content, memory_type)
      VALUES ('delete', old.id, old.content, old.memory_type);
      INSERT INTO memories_fts(rowid, content, memory_type)
      VALUES (new.id, new.content, new.memory_type);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS diary_ai AFTER INSERT ON diary_entries BEGIN
      INSERT INTO diary_fts(rowid, title, content)
      VALUES (new.id, coalesce(new.title, ''), new.content);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS diary_ad AFTER DELETE ON diary_entries BEGIN
      INSERT INTO diary_fts(diary_fts, rowid, title, content)
      VALUES ('delete', old.id, coalesce(old.title, ''), old.content);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS diary_au AFTER UPDATE ON diary_entries BEGIN
      INSERT INTO diary_fts(diary_fts, rowid, title, content)
      VALUES ('delete', old.id, coalesce(old.title, ''), old.content);
      INSERT INTO diary_fts(rowid, title, content)
      VALUES (new.id, coalesce(new.title, ''), new.content);
    END;
    """,
]


def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    names = {r[1] for r in rows}
    if column not in names:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))


def init_db(engine: Engine) -> None:
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
        )
        row = conn.execute(
            text("SELECT value FROM schema_meta WHERE key = 'version'")
        ).fetchone()
        if row is None:
            conn.execute(
                text("INSERT INTO schema_meta(key, value) VALUES ('version', :v)"),
                {"v": str(SCHEMA_VERSION)},
            )
        else:
            conn.execute(
                text("UPDATE schema_meta SET value = :v WHERE key = 'version'"),
                {"v": str(SCHEMA_VERSION)},
            )

        _ensure_column(conn, "memories", "source_diary_id", "source_diary_id INTEGER")
        _ensure_column(conn, "memories", "superseded_at", "superseded_at DATETIME")
        _ensure_column(conn, "memories", "proposed_action", "proposed_action TEXT")
        _ensure_column(conn, "memories", "proposed_target_id", "proposed_target_id INTEGER")

        conn.execute(text(FTS_MESSAGES_SQL))
        conn.execute(text(FTS_MEMORIES_SQL))
        conn.execute(text(FTS_DIARY_SQL))
        for trigger_sql in TRIGGERS:
            conn.execute(text(trigger_sql))


def adopt_legacy_database(db_path: Path, legacy_paths: Iterable[Path]) -> Path | None:
    """Move a pre-rename database to its new filename. Returns the source path.

    A rename must not look like data loss: if the app simply started using the
    new filename while the user's only database still carried the old one, every
    screen would come up empty. So when the target does not exist yet and a
    legacy candidate does, the file is moved (never copied, never deleted)
    together with SQLite's ``-wal`` and ``-shm`` sidecars. The previous filename
    is recorded in :mod:`app.branding` as ``LEGACY_DB_FILENAME``.

    Does nothing if the target already exists, so it is safe on every startup.
    """
    if db_path.exists():
        return None
    for legacy in legacy_paths:
        if legacy == db_path or not legacy.exists():
            continue
        legacy.replace(db_path)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(legacy) + suffix)
            if sidecar.exists():
                sidecar.replace(Path(str(db_path) + suffix))
        return legacy
    return None


def wipe_all_data(engine: Engine) -> None:
    """Delete all application data while keeping schema."""
    with engine.begin() as conn:
        for table in (
            "memory_evidence",
            "memories",
            "message_chunks",
            "analysis_runs",
            "messages",
            "imports",
            "diary_entries",
            "chat_turns",
            "chat_threads",
            "outbound_events",
        ):
            conn.execute(text(f"DELETE FROM {table}"))
        for fts in ("messages_fts", "memories_fts", "diary_fts"):
            try:
                conn.execute(text(f"INSERT INTO {fts}({fts}) VALUES('rebuild')"))
            except Exception:
                pass
