from __future__ import annotations

from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session

from app.db.models import DiaryEntry, MemoryStatus, MemoryType
from app.db.session import get_db
from app.services import memory as memory_svc
from app.services.retrieval import hybrid_search
from app.templating import render

router = APIRouter(tags=["memories"])


PAGE_SIZE = 50


def _memories_query_string(**params: str | int) -> str:
    items = []
    for key, value in params.items():
        if value is None or value == "":
            continue
        items.append((key, str(value)))
    return urlencode(items)


@router.get("/memories", response_class=HTMLResponse)
def memories_page(
    request: Request,
    status: str = "confirmed",
    q: str = "",
    memory_type: str = "",
    sort: str = "id",
    dir: str = "desc",
    page: int = 1,
    created: int | None = None,
    error: str = "",
    db: Session = Depends(get_db),
):
    status_filter = status if status != "all" else None
    sort_key = sort if sort in memory_svc.SORT_COLUMNS else "id"
    direction = "asc" if dir == "asc" else "desc"
    total = memory_svc.count_memories(
        db,
        status=status_filter,
        memory_type=memory_type or None,
        q=q or None,
    )
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(1, min(page, total_pages))
    memories = memory_svc.list_memories(
        db,
        status=status_filter,
        memory_type=memory_type or None,
        q=q or None,
        limit=PAGE_SIZE,
        offset=(page - 1) * PAGE_SIZE,
        sort=sort_key,
        direction=direction,
    )
    stats = memory_svc.memory_stats(db)
    created_memory = memory_svc.get_memory(db, created) if created else None
    filter_qs = _memories_query_string(q=q, status=status, memory_type=memory_type)
    qs = _memories_query_string(
        q=q, status=status, memory_type=memory_type, sort=sort_key, dir=direction
    )
    return render(
        request,
        "memories.html",
        {
            "memories": memories,
            "stats": stats,
            "status": status,
            "q": q,
            "memory_type": memory_type,
            "sort": sort_key,
            "dir": direction,
            "page": page,
            "total": total,
            "total_pages": total_pages,
            "page_size": PAGE_SIZE,
            "qs": qs,
            "filter_qs": filter_qs,
            "types": [t.value for t in MemoryType],
            "statuses": [s.value for s in MemoryStatus],
            "created_memory": created_memory,
            "error": error,
        },
        db=db,
    )


@router.get("/memories/candidates", response_class=HTMLResponse)
def candidates_page(
    request: Request,
    flash: str = "",
    db: Session = Depends(get_db),
):
    candidates = memory_svc.list_candidates(db, limit=500)
    stats = memory_svc.memory_stats(db)
    return render(
        request,
        "candidates.html",
        {
            "candidates": candidates,
            "groups": memory_svc.candidate_groups(db, candidates),
            "stats": stats,
            "types": [t.value for t in MemoryType],
            "flash": flash,
        },
        db=db,
    )


@router.post("/memories/candidates/bulk")
async def candidates_bulk(
    request: Request,
    db: Session = Depends(get_db),
):
    form = await request.form()
    action = str(form.get("action") or "").strip()
    raw_ids = form.getlist("ids")
    ids: list[int] = []
    for value in raw_ids:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue

    if action == "confirm_all":
        count = memory_svc.confirm_all_candidates(db)
        msg = f"Confirmed all {count} candidates in bulk"
    elif action == "confirm_high":
        count = memory_svc.confirm_all_candidates(db, min_confidence=0.7)
        msg = f"Confirmed high-confidence (≥0.7) candidates in bulk: {count}"
    elif action == "confirm":
        count = memory_svc.bulk_confirm_memories(db, ids)
        msg = f"Confirmed {count} in bulk"
    elif action == "reject":
        count = memory_svc.bulk_reject_memories(db, ids)
        msg = f"Rejected {count} in bulk"
    else:
        msg = "Unknown action"
    return RedirectResponse(
        "/memories/candidates?flash=" + quote(msg),
        status_code=303,
    )


@router.get("/memories/{memory_id}", response_class=HTMLResponse)
def memory_detail(memory_id: int, request: Request, db: Session = Depends(get_db)):
    memory = memory_svc.get_memory(db, memory_id)
    if not memory:
        raise HTTPException(404, "Memory not found")
    return render(
        request,
        "memory_detail.html",
        {
            "memory": memory,
            "types": [t.value for t in MemoryType],
            "statuses": [s.value for s in MemoryStatus],
            "superseded_by": memory_svc.superseded_by(db, memory),
            "replaced": memory_svc.replaced_memories(db, memory),
            "replacement_target": memory_svc.replacement_target(db, memory),
            "source_diary": (
                db.get(DiaryEntry, memory.source_diary_id)
                if memory.source_diary_id
                else None
            ),
        },
        db=db,
    )


@router.post("/memories/{memory_id}/confirm")
def confirm_memory(
    memory_id: int,
    content: str = Form(None),
    memory_type: str = Form(None),
    notes: str = Form(None),
    db: Session = Depends(get_db),
):
    try:
        memory_svc.confirm_memory(
            db,
            memory_id,
            content=content,
            memory_type=memory_type,
            notes=notes,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return RedirectResponse("/memories/candidates", status_code=303)


@router.post("/memories/{memory_id}/reject")
def reject_memory(
    memory_id: int,
    notes: str = Form(None),
    db: Session = Depends(get_db),
):
    try:
        memory_svc.reject_memory(db, memory_id, notes=notes)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return RedirectResponse("/memories/candidates", status_code=303)


@router.post("/memories/{memory_id}/update")
def update_memory(
    memory_id: int,
    content: str = Form(...),
    memory_type: str = Form(...),
    status: str = Form(...),
    notes: str = Form(""),
    confidence: float = Form(0.8),
    db: Session = Depends(get_db),
):
    try:
        memory_svc.update_memory(
            db,
            memory_id,
            content=content,
            memory_type=memory_type,
            status=status,
            notes=notes,
            confidence=confidence,
        )
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return RedirectResponse(f"/memories/{memory_id}", status_code=303)


@router.post("/memories/{memory_id}/delete")
def delete_memory(memory_id: int, db: Session = Depends(get_db)):
    try:
        memory_svc.delete_memory(db, memory_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc
    return RedirectResponse("/memories", status_code=303)


@router.post("/memories/create")
def create_memory(
    content: str = Form(...),
    memory_type: str = Form("auto"),
    subject: str = Form(""),
    db: Session = Depends(get_db),
):
    text = (content or "").strip()
    if not text:
        return RedirectResponse(
            "/memories?status=confirmed&error=" + quote("Content cannot be empty"),
            status_code=303,
        )
    try:
        mem = memory_svc.create_manual_memory(
            db,
            text,
            memory_type=memory_type,
            subject=subject.strip() or None,
            auto_classify=True,
        )
    except Exception as exc:  # noqa: BLE001 — surface to user
        return RedirectResponse(
            "/memories?status=confirmed&error=" + quote(f"Add failed: {exc}"),
            status_code=303,
        )
    return RedirectResponse(f"/memories?status=confirmed&created={mem.id}", status_code=303)


@router.get("/search", response_class=HTMLResponse)
def search_page(request: Request, q: str = "", db: Session = Depends(get_db)):
    hits = hybrid_search(db, q, include_messages=True, limit=30) if q.strip() else []
    return render(
        request,
        "search.html",
        {"q": q, "hits": hits},
        db=db,
    )
