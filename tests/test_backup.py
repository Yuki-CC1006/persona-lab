"""Encrypted backup, restore preview and retention."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import ImportJob, Memory, MemoryEvidence, Message
from app.services import backup as backup_svc
from app.services import memory as memory_svc


def test_roundtrip_encrypts_and_recovers_the_same_data(db: Session):
    memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion", subject="Alex")
    snapshot = backup_svc.build_snapshot(db)
    envelope = backup_svc.encrypt_snapshot(snapshot, "correct-horse")

    text = backup_svc.dumps_envelope(envelope)
    assert "cilantro" not in text  # plaintext must not appear on disk

    restored = backup_svc.decrypt_envelope(backup_svc.loads_envelope(text), "correct-horse")
    assert restored["counts"] == snapshot["counts"]
    assert restored["tables"]["memories"][0]["content"] == "dislikes cilantro"


def test_wrong_passphrase_is_rejected(db: Session):
    envelope = backup_svc.encrypt_snapshot(backup_svc.build_snapshot(db), "correct-horse")
    with pytest.raises(backup_svc.BackupError) as exc:
        backup_svc.decrypt_envelope(envelope, "wrong-horse")
    assert "Incorrect passphrase" in str(exc.value)


def test_tampered_ciphertext_is_rejected(db: Session):
    envelope = backup_svc.encrypt_snapshot(backup_svc.build_snapshot(db), "correct-horse")
    envelope["ciphertext"] = envelope["ciphertext"][:-4] + "AAAA"
    with pytest.raises(backup_svc.BackupError):
        backup_svc.decrypt_envelope(envelope, "correct-horse")


def test_short_passphrase_is_refused(db: Session):
    with pytest.raises(backup_svc.BackupError) as exc:
        backup_svc.encrypt_snapshot(backup_svc.build_snapshot(db), "123")
    assert "at least 6 characters" in str(exc.value)


def test_non_backup_file_is_refused():
    with pytest.raises(backup_svc.BackupError):
        backup_svc.decrypt_envelope({"format": "something-else"}, "correct-horse")


def test_preview_reports_before_and_after_counts(db: Session):
    memory_svc.create_manual_memory(db, "likes unsweetened Americano", memory_type="preference")
    snapshot = backup_svc.build_snapshot(db)
    memory_svc.create_manual_memory(db, "a memory added later", memory_type="fact")

    preview = backup_svc.preview_restore(db, snapshot)
    assert preview["tables"]["memories"]["in_backup"] == 1
    assert preview["tables"]["memories"]["now"] == 2
    assert preview["warnings"]


def test_restore_replaces_current_data(db: Session):
    memory_svc.create_manual_memory(db, "the memory inside the backup", memory_type="fact")
    snapshot = backup_svc.build_snapshot(db)
    db.query(Memory).delete()
    db.commit()
    memory_svc.create_manual_memory(db, "written after the backup", memory_type="fact")

    result = backup_svc.restore_snapshot(db, snapshot)
    assert result["restored"]["memories"] == 1
    contents = [m.content for m in db.query(Memory).all()]
    assert contents == ["the memory inside the backup"]


def test_restore_keeps_evidence_links_intact(db: Session):
    job = ImportJob(filename="c.txt", file_hash="h1", source_format="text_chat", status="done")
    db.add(job)
    db.commit()
    message = Message(
        import_id=job.id, fingerprint="f1", sender="Alex", content="dislikes cilantro"
    )
    db.add(message)
    db.commit()
    memory = memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion")
    db.add(MemoryEvidence(memory_id=memory.id, message_id=message.id, quote="dislikes cilantro"))
    db.commit()

    snapshot = backup_svc.build_snapshot(db)
    backup_svc.restore_snapshot(db, snapshot)

    assert db.query(MemoryEvidence).count() == 1
    assert db.query(ImportJob).count() == 1
    assert db.query(Message).count() == 1
    assert db.query(MemoryEvidence).one().memory.content == "dislikes cilantro"


def test_restore_is_searchable_afterwards(db: Session):
    from app.services.retrieval import hybrid_search

    memory_svc.create_manual_memory(db, "Alex likes mildly spicy hotpot", memory_type="preference", subject="Alex")
    snapshot = backup_svc.build_snapshot(db)
    backup_svc.restore_snapshot(db, snapshot)

    hits = hybrid_search(db, "Do they like hotpot?", include_messages=False, limit=5)
    assert any("hotpot" in hit.content for hit in hits)


def test_backup_files_are_pruned_to_keep_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))
    from app.config import get_settings

    get_settings.cache_clear()
    try:
        directory = backup_svc.backup_dir()
        for stamp in ("20260101-000001", "20260101-000002", "20260101-000003", "20260101-000004"):
            (directory / f"persona-lab-{stamp}{backup_svc.BACKUP_EXT}").write_text(
                "{}", encoding="utf-8"
            )

        removed = backup_svc.prune_backups(keep=2)
        assert sorted(removed) == [
            "persona-lab-20260101-000001.personabak",
            "persona-lab-20260101-000002.personabak",
        ]
        # Newest first, and only the retained files are listed.
        assert [b["name"] for b in backup_svc.list_backups()] == [
            "persona-lab-20260101-000004.personabak",
            "persona-lab-20260101-000003.personabak",
        ]
    finally:
        get_settings.cache_clear()


def test_resolve_backup_path_rejects_traversal():
    with pytest.raises(backup_svc.BackupError):
        backup_svc.resolve_backup_path("../secrets.personabak")
    with pytest.raises(backup_svc.BackupError):
        backup_svc.resolve_backup_path("notes.txt")


def test_backup_page_renders(client: TestClient):
    page = client.get("/settings/backup")
    assert page.status_code == 200
    assert "Encrypted export" in page.text
    assert "Restore preview" not in page.text


def test_export_then_restore_through_the_web(client: TestClient, db: Session):
    memory_svc.create_manual_memory(db, "the memory backed up from the web page", memory_type="fact")
    export = client.post(
        "/settings/backup/export",
        data={"passphrase": "correct-horse", "passphrase2": "correct-horse"},
        follow_redirects=False,
    )
    assert export.status_code in (302, 303)
    assert "saved=export" in export.headers["location"]

    listing = client.get(export.headers["location"])
    assert "the memory backed up from the web page" not in listing.text
    assert backup_svc.BACKUP_EXT in listing.text

    name = backup_svc.list_backups()[0]["name"]
    envelope_text = (backup_svc.backup_dir() / name).read_text(encoding="utf-8")

    db.query(Memory).delete()
    db.commit()
    assert db.query(Memory).count() == 0

    preview = client.post(
        "/settings/backup/restore/preview",
        files={"file": (f"backup{backup_svc.BACKUP_EXT}", envelope_text.encode("utf-8"), "text/plain")},
        data={"passphrase": "correct-horse"},
    )
    assert preview.status_code == 200
    assert "Restore preview" in preview.text
    token = _extract_token(preview.text)
    assert token

    confirm = client.post(
        "/settings/backup/restore", data={"token": token}, follow_redirects=False
    )
    assert confirm.status_code in (302, 303)
    assert "saved=restore" in confirm.headers["location"]
    assert [m.content for m in db.query(Memory).all()] == ["the memory backed up from the web page"]


def test_restore_preview_rejects_wrong_passphrase(client: TestClient):
    envelope_text = backup_svc.dumps_envelope(
        backup_svc.encrypt_snapshot({"format": backup_svc.FORMAT, "tables": {}}, "correct-horse")
    )
    resp = client.post(
        "/settings/backup/restore/preview",
        files={"file": (f"b{backup_svc.BACKUP_EXT}", envelope_text.encode("utf-8"), "text/plain")},
        data={"passphrase": "nope-nope"},
    )
    assert resp.status_code == 200
    assert "Incorrect passphrase" in resp.text
    assert "Restore preview" not in resp.text


def test_confirm_restore_without_preview_fails(client: TestClient):
    resp = client.post(
        "/settings/backup/restore", data={"token": "forged"}, follow_redirects=False
    )
    assert resp.status_code in (302, 303)
    assert "error=" in resp.headers["location"]


def test_delete_backup_file(client: TestClient, db: Session):
    envelope = backup_svc.encrypt_snapshot(backup_svc.build_snapshot(db), "correct-horse")
    name = backup_svc.save_backup_file(envelope).name
    resp = client.post(f"/settings/backup/{name}/delete", follow_redirects=False)
    assert resp.status_code in (302, 303)
    assert backup_svc.list_backups() == []


def test_snapshot_excludes_binary_and_covers_meta(db: Session):
    memory_svc.create_manual_memory(db, "an exportable memory", memory_type="fact")
    snapshot = backup_svc.build_snapshot(db)
    assert snapshot["format"] == backup_svc.FORMAT
    assert snapshot["counts"]["memories"] == 1
    assert set(snapshot["counts"]) >= {"memories", "messages", "imports", "outbound_events"}


def _extract_token(html: str) -> str:
    marker = 'name="token" value="'
    start = html.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    return html[start : html.find('"', start)]
