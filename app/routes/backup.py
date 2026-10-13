"""Encrypted backup / restore endpoints. Local-first: nothing here leaves the box."""

from __future__ import annotations

from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_db, session_scope
from app.services import backup as backup_svc
from app.templating import render

router = APIRouter(tags=["backup"])


@router.get("/settings/backup", response_class=HTMLResponse)
def backup_page(
    request: Request,
    db: Session = Depends(get_db),
    saved: str = "",
    error: str = "",
):
    return _render_page(request, db, saved=saved, error=error)


@router.post("/settings/backup/export")
def create_backup(passphrase: str = Form(""), passphrase2: str = Form("")):
    if passphrase != passphrase2:
        return _redirect_error("The two passphrases do not match")
    try:
        with session_scope() as session:
            snapshot = backup_svc.build_snapshot(session)
        envelope = backup_svc.encrypt_snapshot(snapshot, passphrase)
        name = backup_svc.save_backup_file(envelope).name
    except backup_svc.BackupError as exc:
        return _redirect_error(str(exc))
    return _redirect("saved=export&name=" + quote(name))


@router.post("/settings/backup/restore/preview", response_class=HTMLResponse)
async def preview_restore(
    request: Request,
    file: UploadFile | None = File(None),
    passphrase: str = Form(""),
    db: Session = Depends(get_db),
):
    if file is None or not file.filename:
        return _render_page(request, db, error="Please choose a backup file to restore")
    raw = await file.read()
    try:
        envelope = backup_svc.loads_envelope(raw.decode("utf-8", errors="replace"))
        snapshot = backup_svc.decrypt_envelope(envelope, passphrase)
    except backup_svc.BackupError as exc:
        return _render_page(request, db, error=str(exc))

    preview = backup_svc.preview_restore(db, snapshot)
    # Decrypted data stays in memory only, and only until the user confirms.
    token = backup_svc.stash_pending(snapshot)
    return _render_page(request, db, preview=preview, token=token)


@router.post("/settings/backup/restore")
def restore_backup(token: str = Form(""), db: Session = Depends(get_db)):
    try:
        snapshot = backup_svc.take_pending(token)
        result = backup_svc.restore_snapshot(db, snapshot)
    except backup_svc.BackupError as exc:
        return _redirect_error(str(exc))
    except Exception as exc:  # noqa: BLE001 — never fail silently on a destructive action
        db.rollback()
        return _redirect_error(f"Restore failed: {type(exc).__name__}; keep the original database file before asking for help")
    return _redirect(f"saved=restore&rows={result['total']}")


@router.post("/settings/backup/{name}/delete")
def delete_backup(name: str):
    try:
        path = backup_svc.resolve_backup_path(name)
    except backup_svc.BackupError as exc:
        return _redirect_error(str(exc))
    path.unlink(missing_ok=True)
    return _redirect(f"saved=deleted&name={quote(path.name)}")


def _render_page(
    request: Request,
    db: Session,
    *,
    saved: str = "",
    error: str = "",
    preview: dict | None = None,
    token: str = "",
) -> HTMLResponse:
    return render(
        request,
        "backup.html",
        {
            "backups": backup_svc.list_backups(),
            "keep": max(1, int(get_settings().backup_keep)),
            "saved": saved,
            "error": error,
            "preview": preview,
            "token": token,
        },
        db=db,
    )


def _redirect(query: str) -> RedirectResponse:
    return RedirectResponse(f"/settings/backup?{query}", status_code=303)


def _redirect_error(message: str) -> RedirectResponse:
    return _redirect("error=" + quote(message))
