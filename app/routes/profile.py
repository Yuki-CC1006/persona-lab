from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.profile import (
    build_memory_graph,
    build_profile_overview,
    export_memories_markdown,
    graph_text_outline,
)
from app.templating import render

router = APIRouter(tags=["profile"])


@router.get("/profile", response_class=HTMLResponse)
def profile_page(request: Request, db: Session = Depends(get_db)):
    overview = build_profile_overview(db)
    return render(request, "profile.html", {"overview": overview}, db=db)


@router.get("/profile/export.md")
def export_profile_markdown(db: Session = Depends(get_db)):
    """Download confirmed memories as Markdown (no candidates, no raw messages)."""
    text = export_memories_markdown(db)
    return Response(
        content=text,
        media_type="text/markdown; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="persona-lab-memories.md"'
        },
    )


@router.get("/graph", response_class=HTMLResponse)
def graph_page(request: Request, db: Session = Depends(get_db)):
    graph = build_memory_graph(db)
    return render(
        request,
        "graph.html",
        {
            "graph": graph,
            "graph_outline": graph_text_outline(graph),
            "graph_data": json.dumps(graph, ensure_ascii=False).replace("<", "\\u003c"),
        },
        db=db,
    )


@router.get("/api/graph")
def api_graph(db: Session = Depends(get_db)):
    return JSONResponse(build_memory_graph(db))
