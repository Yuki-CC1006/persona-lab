"""Encrypted local backup, restore preview and retention for the whole dataset.

Design notes (stdlib only, no new dependency):

* The snapshot is JSON, then encrypted with a keystream derived from
  HMAC-SHA256 (a PRF-based stream cipher) and authenticated with
  encrypt-then-MAC over the envelope. The key comes from ``hashlib.scrypt``.
* Plaintext backups are never written to disk. Only the envelope is stored, and
  the passphrase is never persisted.
* Restoring is destructive by design, so it always goes through a preview that
  shows how many rows exist now versus in the backup.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.migrations import wipe_all_data
from app.db.models import (
    AnalysisRun,
    ChatThread,
    ChatTurn,
    DiaryEntry,
    ImportJob,
    Memory,
    MemoryEvidence,
    Message,
    MessageChunk,
    OutboundEvent,
)
from app.branding import PRODUCT_NAME, PRODUCT_SLUG
from app.db.session import get_engine

FORMAT = f"{PRODUCT_SLUG}-backup"
FORMAT_VERSION = 1
CIPHER = "hmac-sha256-keystream"
BACKUP_EXT = ".personabak"
KDF_N = 2**14
KDF_R = 8
KDF_P = 1
KDF_MEMLIMIT = 64 * 1024 * 1024
PENDING_TTL_SECONDS = 600

# Insert order matters: children reference parents.
SNAPSHOT_TABLES = (
    ImportJob,
    Message,
    MessageChunk,
    AnalysisRun,
    Memory,
    MemoryEvidence,
    ChatThread,
    ChatTurn,
    DiaryEntry,
    OutboundEvent,
)

_DATETIME_COLUMNS = {
    model.__tablename__: {
        column.name
        for column in model.__table__.columns
        if str(column.type).startswith("DATETIME")
    }
    for model in SNAPSHOT_TABLES
}


class BackupError(RuntimeError):
    """User-facing backup problem: wrong passphrase, corrupt or unknown file."""


def build_snapshot(db: Session) -> dict:
    """Everything the user created, in a single JSON-able dict."""
    tables: dict[str, list[dict]] = {}
    for model in SNAPSHOT_TABLES:
        table = model.__table__
        date_cols = _DATETIME_COLUMNS[model.__tablename__]
        rows: list[dict] = []
        for row in db.execute(select(table)).mappings():
            item = {}
            for key, value in row.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()
                elif key in date_cols and value is not None:
                    item[key] = str(value)
                else:
                    item[key] = value
            rows.append(item)
        tables[model.__tablename__] = rows

    counts = {name: len(rows) for name, rows in tables.items()}
    return {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "counts": counts,
        "total_rows": sum(counts.values()),
        "tables": tables,
    }


def encrypt_snapshot(snapshot: dict, passphrase: str) -> dict:
    """Return a JSON-able envelope. ``passphrase`` is never stored."""
    if not passphrase or len(passphrase) < 6:
        raise BackupError("Passphrase must be at least 6 characters: it is the only way to unlock the backup, and if it is lost the backup truly cannot be opened")
    plaintext = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(16)
    key = _derive_key(passphrase, salt)
    body = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "cipher": CIPHER,
        "created_at": datetime.utcnow().isoformat(),
        "kdf": {"name": "scrypt", "n": KDF_N, "r": KDF_R, "p": KDF_P, "salt": _b64(salt)},
        "nonce": _b64(nonce),
        "ciphertext": _b64(_xor(plaintext, _keystream(key[:32], nonce, len(plaintext)))),
    }
    body["mac"] = _b64(hmac.new(key[32:], _canonical(body), hashlib.sha256).digest())
    return body


def decrypt_envelope(envelope: dict, passphrase: str) -> dict:
    """Verify then decrypt. Raises ``BackupError`` with a clear reason."""
    if not isinstance(envelope, dict) or envelope.get("format") != FORMAT:
        raise BackupError(f"This is not a {PRODUCT_NAME} backup file")
    if envelope.get("cipher") != CIPHER:
        raise BackupError("The backup uses an unsupported encryption method")
    if int(envelope.get("version") or 0) > FORMAT_VERSION:
        raise BackupError("The backup comes from a newer version; upgrade the app before restoring")

    try:
        kdf = envelope["kdf"]
        salt = _unb64(kdf["salt"])
        nonce = _unb64(envelope["nonce"])
        ciphertext = _unb64(envelope["ciphertext"])
        expected = _unb64(envelope["mac"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackupError("The backup file structure is incomplete or corrupted") from exc

    key = _derive_key(
        passphrase, salt, n=int(kdf.get("n") or KDF_N), r=int(kdf.get("r") or KDF_R), p=int(kdf.get("p") or KDF_P)
    )
    body = {k: v for k, v in envelope.items() if k != "mac"}
    actual = hmac.new(key[32:], _canonical(body), hashlib.sha256).digest()
    if not hmac.compare_digest(actual, expected):
        raise BackupError("Incorrect passphrase, or the backup file has been modified")

    plaintext = _xor(ciphertext, _keystream(key[:32], nonce, len(ciphertext)))
    try:
        snapshot = json.loads(plaintext.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackupError("Decryption succeeded but the content could not be parsed; the backup may be corrupted") from exc
    if snapshot.get("format") != FORMAT:
        raise BackupError(f"The decrypted content is not a {PRODUCT_NAME} backup")
    return snapshot


def dumps_envelope(envelope: dict) -> str:
    return json.dumps(envelope, ensure_ascii=False, indent=2)


def loads_envelope(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise BackupError(f"Backup file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise BackupError("Backup file content has the wrong format")
    return data


def preview_restore(db: Session, snapshot: dict) -> dict:
    """What would change if this backup were restored, shown before doing it."""
    tables: dict[str, dict] = {}
    for model in SNAPSHOT_TABLES:
        name = model.__tablename__
        in_backup = len((snapshot.get("tables") or {}).get(name) or [])
        now = int(
            db.execute(
                select(func.count()).select_from(model.__table__)
            ).scalar()
            or 0
        )
        tables[name] = {"in_backup": in_backup, "now": now}
    warnings = ["Restoring first wipes all current data, then replaces it wholesale with the backup contents."]
    if not snapshot.get("tables"):
        warnings.append("The backup contains no data tables; restoring will leave an empty database.")
    return {
        "created_at": snapshot.get("created_at"),
        "version": snapshot.get("version"),
        "tables": tables,
        "total_in_backup": sum(t["in_backup"] for t in tables.values()),
        "total_now": sum(t["now"] for t in tables.values()),
        "warnings": warnings,
    }


def restore_snapshot(db: Session, snapshot: dict) -> dict:
    """Destructive full restore. Callers must run ``preview_restore`` first."""
    tables = snapshot.get("tables") or {}
    if snapshot.get("format") != FORMAT:
        raise BackupError(f"This is not {PRODUCT_NAME} backup content")

    wipe_all_data(get_engine())
    restored: dict[str, int] = {}
    for model in SNAPSHOT_TABLES:
        name = model.__tablename__
        rows = tables.get(name) or []
        if not rows:
            continue
        date_cols = _DATETIME_COLUMNS[name]
        payload = [_coerce_row(row, date_cols) for row in rows]
        db.execute(model.__table__.insert(), payload)
        restored[name] = len(payload)
    db.commit()
    _rebuild_search_indexes()
    return {"restored": restored, "total": sum(restored.values())}


def backup_dir() -> Path:
    path = Path(get_settings().backup_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_backup_file(envelope: dict, *, keep: int | None = None) -> Path:
    """Write the envelope and apply the retention policy. Returns the new path."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = backup_dir() / f"{PRODUCT_SLUG}-{stamp}{BACKUP_EXT}"
    path.write_text(dumps_envelope(envelope), encoding="utf-8")
    prune_backups(keep=keep)
    return path


def list_backups() -> list[dict]:
    items = []
    for path in sorted(backup_dir().glob(f"{PRODUCT_SLUG}-*{BACKUP_EXT}"), reverse=True):
        stat = path.stat()
        items.append(
            {
                "name": path.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return items


def prune_backups(*, keep: int | None = None) -> list[str]:
    """Keep the newest ``backup_keep`` files; return the names that were removed."""
    limit = int(keep if keep is not None else get_settings().backup_keep)
    removed = []
    for path in sorted(backup_dir().glob(f"{PRODUCT_SLUG}-*{BACKUP_EXT}"), reverse=True)[max(0, limit) :]:
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def resolve_backup_path(name: str) -> Path:
    """Reject anything that is not a plain file inside the backup directory."""
    candidate = Path(name).name
    if not candidate or candidate != name or not candidate.endswith(BACKUP_EXT):
        raise BackupError("Invalid backup filename")
    path = backup_dir() / candidate
    if not path.exists():
        raise BackupError("Backup file not found")
    return path


_PENDING: dict[str, tuple[float, dict]] = {}


def stash_pending(snapshot: dict) -> str:
    """Hold a decrypted snapshot in memory between preview and confirm."""
    _expire_pending()
    token = secrets.token_urlsafe(16)
    _PENDING[token] = (time.time() + PENDING_TTL_SECONDS, snapshot)
    return token


def take_pending(token: str) -> dict:
    _expire_pending()
    entry = _PENDING.pop(token or "", None)
    if not entry:
        raise BackupError("The preview has expired; please upload the backup file again")
    return entry[1]


def _expire_pending() -> None:
    now = time.time()
    for key in [k for k, (deadline, _) in _PENDING.items() if deadline < now]:
        _PENDING.pop(key, None)


def _coerce_row(row: dict, date_cols: set[str]) -> dict:
    item = {}
    for key, value in row.items():
        if key in date_cols and isinstance(value, str):
            item[key] = datetime.fromisoformat(value) if value else None
        else:
            item[key] = value
    return item


def table_columns(table_name: str) -> set[str]:
    table = Base.metadata.tables.get(table_name)
    return {column.name for column in table.columns} if table is not None else set()


def _rebuild_search_indexes() -> None:
    from sqlalchemy import text

    with get_engine().begin() as conn:
        for fts in ("messages_fts", "memories_fts", "diary_fts"):
            try:
                conn.execute(text(f"INSERT INTO {fts}({fts}) VALUES('rebuild')"))
            except Exception:  # noqa: BLE001 — index is a cache, never fatal
                pass


def _derive_key(passphrase: str, salt: bytes, *, n: int = KDF_N, r: int = KDF_R, p: int = KDF_P) -> bytes:
    return hashlib.scrypt(
        passphrase.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=64,
        maxmem=KDF_MEMLIMIT,
    )


def _keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    if length <= 0:
        return b""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(key, nonce + counter.to_bytes(8, "big"), hashlib.sha256).digest()
        counter += 1
    return bytes(out[:length])


def _xor(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a, b in zip(data, stream))


def _canonical(body: dict) -> bytes:
    return json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _unb64(value: str) -> bytes:
    return base64.b64decode(value, validate=True)
