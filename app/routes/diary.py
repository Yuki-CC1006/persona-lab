from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.db.models import Memory
from app.db.session import get_db
from app.llm.client import LLMClient
from app.services import diary as diary_svc
from app.templating import render

router = APIRouter(tags=["diary"])


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


@router.get("/diary", response_class=HTMLResponse)
def diary_list(request: Request, q: str = "", db: Session = Depends(get_db)):
    entries = diary_svc.list_diaries(db, q=q or None)
    today = datetime.now().strftime("%Y-%m-%d")
    return render(
        request,
        "diary.html",
        {
            "entries": entries,
            "q": q,
            "today": today,
            "llm_configured": LLMClient().configured,
            "flash": None,
        },
        db=db,
    )


@router.post("/diary/create")
def diary_create(
    content: str = Form(...),
    entry_date: str = Form(""),
    title: str = Form(""),
    mood: str = Form(""),
    subject_name: str = Form(""),
    extract_now: str = Form("no"),
    db: Session = Depends(get_db),
):
    entry = diary_svc.create_diary(
        db,
        content,
        entry_date=_parse_date(entry_date) or datetime.now(),
        title=title,
        mood=mood,
        subject_name=subject_name,
    )
    if extract_now == "yes":
        try:
            diary_svc.extract_diary_memories(db, entry.id)
        except ValueError:
            pass
        return RedirectResponse("/memories/candidates", status_code=303)
    return RedirectResponse(f"/diary/{entry.id}", status_code=303)


@router.get("/diary/{diary_id}", response_class=HTMLResponse)
def diary_detail(diary_id: int, request: Request, db: Session = Depends(get_db)):
    entry = diary_svc.get_diary(db, diary_id)
    if not entry:
        raise HTTPException(404, "Written note not found")
    linked = (
        db.query(Memory)
        .filter(Memory.source_diary_id == diary_id)
        .order_by(Memory.id.desc())
        .all()
    )
    return render(
        request,
        "diary_detail.html",
        {
            "entry": entry,
            "linked": linked,
            "llm_configured": LLMClient().configured,
            "entry_date_value": entry.entry_date.strftime("%Y-%m-%d") if entry.entry_date else "",
        },
        db=db,
    )


@router.post("/diary/{diary_id}/update")
def diary_update(
    diary_id: int,
    content: str = Form(...),
    entry_date: str = Form(""),
    title: str = Form(""),
    mood: str = Form(""),
    subject_name: str = Form(""),
    db: Session = Depends(get_db),
):
    try:
        diary_svc.update_diary(
            db,
            diary_id,
            content=content,
            title=title,
            mood=mood,
            subject_name=subject_name,
            entry_date=_parse_date(entry_date),
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return RedirectResponse(f"/diary/{diary_id}", status_code=303)


@router.post("/diary/{diary_id}/extract")
def diary_extract(diary_id: int, db: Session = Depends(get_db)):
    try:
        diary_svc.extract_diary_memories(db, diary_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return RedirectResponse("/memories/candidates", status_code=303)


@router.post("/diary/{diary_id}/delete")
def diary_delete(diary_id: int, db: Session = Depends(get_db)):
    try:
        diary_svc.delete_diary(db, diary_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return RedirectResponse("/diary", status_code=303)
