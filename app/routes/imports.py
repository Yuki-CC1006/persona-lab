from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.config import ROOT_DIR, get_settings
from app.db.models import AnalysisRun, ImportJob, Message
from app.db.session import get_db
from app.importers.text_chat import SOURCE_TEXT
from app.services.extraction import run_extraction
from app.services.deletion import delete_import, import_impact
from app.services.import_service import commit_import, preview_import
from app.services.onboarding import import_next_steps
from app.services.redaction import preview_redaction
from app.templating import render

router = APIRouter(tags=["imports"])

# Bundled, fictional demo files so an empty install can try the whole loop.
SAMPLE_FILES = {
    "wechat": "wechat_sample.json",
    "weflow": "weflow_sample.json",
}


def _upload_dir() -> Path:
    settings = get_settings()
    path = Path(settings.upload_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fix_hint(message: str, path: Path) -> str:
    """Turn a parser error into a concrete next action."""
    name = path.name
    hint = "Write each line as \"nickname: content\", for example: Alex: likes mild hot pot"
    if name.lower().endswith((".json", ".jsonl")):
        hint = (
            "JSON must be an array of objects, or an object containing a messages/data/chats field; "
            "you can also switch to the paste box and use the plain \"nickname: content\" text format."
        )
    return f"{message}\nHow to fix: {hint}"


@router.get("/imports", response_class=HTMLResponse)
def imports_page(
    request: Request,
    db: Session = Depends(get_db),
    deleted: str = "",
    messages: str = "",
    candidates: str = "",
    orphans: str = "",
):
    jobs = db.query(ImportJob).order_by(ImportJob.id.desc()).limit(50).all()
    return render(
        request,
        "imports.html",
        {
            "jobs": jobs,
            "preview": None,
            "error": None,
            "deleted": deleted,
            "deleted_summary": {
                "messages": messages,
                "candidates": candidates,
                "orphans": orphans,
            },
        },
        db=db,
    )


@router.post("/imports/preview", response_class=HTMLResponse)
async def imports_preview(
    request: Request,
    file: UploadFile | None = File(None),
    text: str = Form(""),
    source: str = Form(""),
    db: Session = Depends(get_db),
):
    pasted = (text or "").strip()
    if pasted:
        dest = _upload_dir() / f"pasted-{datetime.now():%Y%m%d-%H%M%S}.txt"
        dest.write_text(pasted, encoding="utf-8")
    elif file is not None and file.filename:
        dest = _upload_dir() / Path(file.filename).name
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    else:
        jobs = db.query(ImportJob).order_by(ImportJob.id.desc()).limit(50).all()
        return render(
            request,
            "imports.html",
            {
                "jobs": jobs,
                "preview": None,
                "error": "Please choose a file or paste some chat text.",
                "upload_path": None,
            },
            db=db,
            status_code=400,
        )

    resolved_source = source or (SOURCE_TEXT if pasted else None)
    return _render_preview(request, dest, db, resolved_source)


@router.post("/imports/sample", response_class=HTMLResponse)
def imports_sample(
    request: Request,
    sample: str = Form("wechat"),
    db: Session = Depends(get_db),
):
    """Load a bundled fictional demo file so a fresh install can try the loop."""
    name = SAMPLE_FILES.get(sample, SAMPLE_FILES["wechat"])
    source = ROOT_DIR / "samples" / name
    if not source.exists():
        jobs = db.query(ImportJob).order_by(ImportJob.id.desc()).limit(50).all()
        return render(
            request,
            "imports.html",
            {
                "jobs": jobs,
                "preview": None,
                "error": "The bundled sample file is missing; please upload or paste instead.",
                "upload_path": None,
            },
            db=db,
            status_code=400,
        )

    dest = _upload_dir() / f"sample-{datetime.now():%Y%m%d-%H%M%S}-{name}"
    shutil.copyfile(source, dest)
    return _render_preview(request, dest, db, None)


def _render_preview(
    request: Request,
    dest: Path,
    db: Session,
    source: str | None,
    status_code: int = 200,
) -> HTMLResponse:
    try:
        preview = preview_import(dest, source=source)
    except ValueError as exc:
        jobs = db.query(ImportJob).order_by(ImportJob.id.desc()).limit(50).all()
        return render(
            request,
            "imports.html",
            {
                "jobs": jobs,
                "preview": None,
                "error": _fix_hint(str(exc), dest),
                "upload_path": None,
            },
            db=db,
            status_code=400,
        )

    settings = get_settings()
    sample_texts = [
        str(s.get("content") or "") for s in preview.get("sample") or [] if s.get("content")
    ]
    redaction_preview = preview_redaction(
        sample_texts, enabled=settings.redaction_enabled, limit=5
    )

    jobs = db.query(ImportJob).order_by(ImportJob.id.desc()).limit(50).all()
    return render(
        request,
        "imports.html",
        {
            "jobs": jobs,
            "preview": preview,
            "redaction_preview": redaction_preview,
            "upload_path": str(dest),
            "error": None,
            "mapping_json": json.dumps(
                preview.get("suggested_mapping") or {}, ensure_ascii=False
            ),
        },
        db=db,
        status_code=status_code,
    )


@router.post("/imports/commit")
async def imports_commit(
    upload_path: str = Form(...),
    subject_name: str = Form(""),
    partner_name: str = Form(""),
    mapping_json: str = Form("{}"),
    run_analysis: str = Form("no"),
    source: str = Form(""),
    db: Session = Depends(get_db),
):
    path = Path(upload_path)
    if not path.exists():
        raise HTTPException(400, "The uploaded file does not exist; please preview again")

    try:
        mapping = json.loads(mapping_json) if mapping_json else {}
    except json.JSONDecodeError as exc:
        raise HTTPException(400, f"Invalid field-mapping JSON: {exc}") from exc

    try:
        job = commit_import(
            db,
            path,
            mapping=mapping or None,
            subject_name=subject_name.strip() or None,
            partner_name=partner_name.strip() or None,
            source=source or None,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    if run_analysis == "yes" and job.message_count > 0:
        run_extraction(
            db,
            job.id,
            subject_name=subject_name.strip() or None,
        )

    return RedirectResponse(f"/imports/{job.id}", status_code=303)


@router.get("/imports/{job_id}", response_class=HTMLResponse)
def import_detail(job_id: int, request: Request, db: Session = Depends(get_db)):
    job = db.get(ImportJob, job_id)
    if not job:
        raise HTTPException(404, "Import job not found")
    messages = (
        db.query(Message)
        .filter(Message.import_id == job_id)
        .order_by(Message.sent_at.asc().nullslast(), Message.id.asc())
        .limit(100)
        .all()
    )
    runs = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.import_id == job_id)
        .order_by(AnalysisRun.id.desc())
        .all()
    )
    return render(
        request,
        "import_detail.html",
        {
            "job": job,
            "messages": messages,
            "runs": runs,
            "next_steps": import_next_steps(db, job),
            "impact": import_impact(db, job_id),
        },
        db=db,
    )


@router.post("/imports/{job_id}/delete")
def import_delete(job_id: int, confirm: str = Form(""), db: Session = Depends(get_db)):
    if not db.get(ImportJob, job_id):
        raise HTTPException(404, "Import job not found")
    if confirm != "yes":
        return RedirectResponse(f"/imports/{job_id}", status_code=303)
    result = delete_import(db, job_id)
    query = (
        f"/imports?deleted={job_id}&messages={result['messages']}"
        f"&candidates={result['candidates']}&orphans={result['orphaned']}"
    )
    return RedirectResponse(query, status_code=303)


@router.post("/imports/{job_id}/analyze")
def import_analyze(job_id: int, db: Session = Depends(get_db)):
    job = db.get(ImportJob, job_id)
    if not job:
        raise HTTPException(404, "Import job not found")
    run_extraction(db, job.id, subject_name=job.subject_name)
    return RedirectResponse(f"/imports/{job_id}", status_code=303)
