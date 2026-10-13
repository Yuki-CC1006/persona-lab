from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryStatus
from app.services import memory as memory_svc
from app.services.profile import build_memory_graph, build_profile_overview
from app.services.retrieval import local_embed


def test_profile_overview_empty(db: Session):
    overview = build_profile_overview(db)
    assert overview["total_confirmed"] == 0
    assert overview["sections"] == []


def test_profile_highlights_skip_duplicates(db: Session):
    db.add(
        Memory(
            content="Alex is afraid of horror movies",
            memory_type="aversion",
            status=MemoryStatus.CONFIRMED.value,
            confidence=1.0,
        )
    )
    db.add(
        Memory(
            content="Alex is afraid of horror movies",
            memory_type="aversion",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.95,
        )
    )
    db.add(
        Memory(
            content="won't do long distance",
            memory_type="preference",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.9,
        )
    )
    db.commit()
    overview = build_profile_overview(db)
    aversion_highlights = [h for h in overview["highlights"] if h["type"] == "aversion"]
    assert len(aversion_highlights) == 1
    assert aversion_highlights[0]["content"] == "Alex is afraid of horror movies"


def test_profile_and_graph_with_memories(db: Session):
    memory_svc.create_manual_memory(db, "likes unsweetened Americano", memory_type="preference", subject="Alex")
    memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion", subject="Alex")
    memory_svc.create_manual_memory(db, "prefers staying home on weekends", memory_type="habit", subject="Alex")

    overview = build_profile_overview(db)
    assert overview["total_confirmed"] == 3
    assert overview["subject_name"] == "Alex"
    assert len(overview["type_bars"]) >= 2
    assert overview["highlights"]

    graph = build_memory_graph(db)
    assert graph["meta"]["memory_count"] == 3
    assert graph["meta"]["facet_count"] >= 1
    assert any(n["kind"] == "subject" for n in graph["nodes"])
    assert any(n["kind"] == "type" for n in graph["nodes"])
    assert any(n["kind"] == "facet" for n in graph["nodes"])
    assert not any(n["kind"] == "memory" for n in graph["nodes"])
    assert any(l["kind"] == "contains" for l in graph["links"])


def test_graph_clusters_similar_preferences(db: Session):
    text = "likes unsweetened Americano coffee"
    emb = local_embed(text)
    db.add(
        Memory(
            content=text,
            memory_type="preference",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.95,
            subject="Alex",
            embedding=emb,
        )
    )
    db.add(
        Memory(
            content="unsweetened Americano coffee",
            memory_type="preference",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.9,
            subject="Alex",
            embedding=local_embed("unsweetened Americano coffee"),
        )
    )
    db.commit()

    graph = build_memory_graph(db)
    facets = [n for n in graph["nodes"] if n["kind"] == "facet" and n["group"] == "preference"]
    assert len(facets) == 1
    assert facets[0]["member_count"] == 2
    assert len(facets[0]["memory_ids"]) == 2
    assert len(facets[0]["memories"]) == 2


def test_graph_hides_events(db: Session):
    memory_svc.create_manual_memory(db, "lives in Xi'an now", memory_type="fact", subject="Alex")
    db.add(
        Memory(
            content="Sent a message on 2026-04-04",
            memory_type="event",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.8,
            subject="Alex",
        )
    )
    db.add(
        Memory(
            content="Visited Anhui during the 2025 National Day holiday",
            memory_type="event",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.8,
            subject="Alex",
        )
    )
    db.commit()

    graph = build_memory_graph(db)
    assert graph["meta"]["hidden_event_count"] == 2
    assert not any(n.get("group") == "event" for n in graph["nodes"] if n["kind"] != "subject")
    assert any(n["kind"] == "facet" and n["group"] == "fact" for n in graph["nodes"])


def test_graph_conflict_group_becomes_has_conflict_facet(db: Session):
    db.add(
        Memory(
            content="won't do long distance",
            memory_type="preference",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.95,
            subject="Alex",
            conflict_group="cg-ldr",
        )
    )
    db.add(
        Memory(
            content="okay with short-term long distance",
            memory_type="preference",
            status=MemoryStatus.CONFIRMED.value,
            confidence=0.9,
            subject="Alex",
            conflict_group="cg-ldr",
        )
    )
    db.commit()

    graph = build_memory_graph(db)
    facets = [n for n in graph["nodes"] if n["kind"] == "facet" and n["group"] == "preference"]
    assert len(facets) == 1
    assert facets[0]["has_conflict"] is True
    assert facets[0]["member_count"] == 2
    # Same conflict_group merged into one facet → no inter-facet conflict edge needed
    assert graph["meta"]["conflict_edges"] == 0


def test_profile_pages(client: TestClient, db: Session):
    memory_svc.create_manual_memory(db, "likes mildly spicy hotpot", memory_type="preference", subject="Alex")
    page = client.get("/profile")
    assert page.status_code == 200
    assert "Profile overview" in page.text
    assert "likes mildly spicy hotpot" in page.text
    assert 'class="type-fold"' in page.text
    assert "Click to expand" in page.text

    graph_page = client.get("/graph")
    assert graph_page.status_code == 200
    assert "Profile graph" in graph_page.text
    assert "memory-graph" in graph_page.text
    assert "Text version of the graph" in graph_page.text

    api = client.get("/api/graph")
    assert api.status_code == 200
    payload = api.json()
    assert "nodes" in payload and "links" in payload
    assert "facet_count" in payload["meta"]
    assert "hidden_event_count" in payload["meta"]
    assert any(n["kind"] == "facet" for n in payload["nodes"])
