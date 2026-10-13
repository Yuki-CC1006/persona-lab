from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session, joinedload

from app.branding import PRODUCT_NAME
from app.config import get_settings
from app.db.models import ImportJob, Memory, MemoryStatus, Message
from app.services.memory import _normalize_key, memory_stats
from app.services.redaction import redact_text
from app.services.retrieval import cosine, local_embed

TYPE_META: dict[str, dict[str, str]] = {
    "fact": {"label": "Fact", "color": "#8ab4f8"},
    "preference": {"label": "Preference", "color": "#81c995"},
    "aversion": {"label": "Aversion", "color": "#f28b82"},
    "person_place": {"label": "Person/Place", "color": "#c58af9"},
    "habit": {"label": "Habit", "color": "#fdd663"},
    "event": {"label": "Event", "color": "#78d9ec"},
    "communication": {"label": "Communication", "color": "#f6aea9"},
    "plan": {"label": "Plan", "color": "#a8dab5"},
}

TYPE_ORDER = list(TYPE_META.keys())

# Portrait graph: events are timeline noise, not identity facets.
PORTRAIT_TYPES = (
    "fact",
    "preference",
    "aversion",
    "person_place",
    "habit",
    "communication",
    "plan",
)
SIMILARITY_THRESHOLD = 0.55
MAX_FACETS_PER_TYPE = 8
MAX_CROSS_SIMILAR = 24


class _UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_profile_overview(db: Session) -> dict[str, Any]:
    memories = (
        db.query(Memory)
        .options(joinedload(Memory.evidence))
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .order_by(Memory.confidence.desc(), Memory.updated_at.desc())
        .all()
    )

    by_type: dict[str, list[Memory]] = {t: [] for t in TYPE_ORDER}
    subjects: set[str] = set()
    conflict_groups: set[str] = set()
    inference_count = 0
    evidence_count = 0
    high_confidence = 0

    for m in memories:
        by_type.setdefault(m.memory_type, []).append(m)
        if m.subject:
            subjects.add(m.subject)
        if m.conflict_group:
            conflict_groups.add(m.conflict_group)
        if m.is_inference:
            inference_count += 1
        if (m.confidence or 0) >= 0.8:
            high_confidence += 1
        evidence_count += len(m.evidence or [])

    sections = []
    for t in TYPE_ORDER:
        items = by_type.get(t) or []
        if not items:
            continue
        meta = TYPE_META.get(t, {"label": t, "color": "#8ab4f8"})
        sections.append(
            {
                "type": t,
                "label": meta["label"],
                "color": meta["color"],
                "count": len(items),
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "confidence": m.confidence,
                        "is_inference": m.is_inference,
                        "conflict_group": m.conflict_group,
                        "evidence_count": len(m.evidence or []),
                    }
                    for m in items[:12]
                ],
            }
        )

    stats = memory_stats(db)
    message_count = db.query(func.count(Message.id)).scalar() or 0
    import_count = db.query(func.count(ImportJob.id)).scalar() or 0
    subject_name = next(iter(subjects), None) or "Subject"

    highlights = []
    seen_keys: set[str] = set()
    for key in ("preference", "aversion", "habit", "communication", "fact"):
        taken = 0
        for m in by_type.get(key) or []:
            nk = _normalize_key(m.content)
            if nk in seen_keys:
                continue
            seen_keys.add(nk)
            highlights.append(
                {
                    "id": m.id,
                    "type": m.memory_type,
                    "label": TYPE_META.get(m.memory_type, {}).get("label", m.memory_type),
                    "color": TYPE_META.get(m.memory_type, {}).get("color", "#8ab4f8"),
                    "content": m.content,
                    "confidence": m.confidence,
                }
            )
            taken += 1
            if taken >= 2:
                break
        if len(highlights) >= 8:
            break

    type_bars = [
        {
            "type": t,
            "label": TYPE_META[t]["label"],
            "color": TYPE_META[t]["color"],
            "count": len(by_type.get(t) or []),
        }
        for t in TYPE_ORDER
        if by_type.get(t)
    ]
    max_count = max((b["count"] for b in type_bars), default=1)

    return {
        "subject_name": subject_name,
        "subjects": sorted(subjects),
        "total_confirmed": len(memories),
        "high_confidence": high_confidence,
        "inference_count": inference_count,
        "evidence_count": evidence_count,
        "conflict_groups": len(conflict_groups),
        "message_count": message_count,
        "import_count": import_count,
        "candidate_count": stats.get("candidates", 0),
        "sections": sections,
        "highlights": highlights[:8],
        "type_bars": type_bars,
        "max_type_count": max_count,
        "type_meta": TYPE_META,
    }


def export_memories_markdown(db: Session) -> str:
    """Human-readable export of confirmed memories only.

    Markdown is the portable, diffable form of the profile: candidates and raw
    messages never enter it, and content is redacted the same way outbound
    requests are, so the file is safe to review or share.
    """
    settings = get_settings()
    memories = (
        db.query(Memory)
        .options(joinedload(Memory.evidence))
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .order_by(Memory.confidence.desc(), Memory.updated_at.desc())
        .all()
    )
    subjects = sorted({m.subject for m in memories if m.subject})
    subject = subjects[0] if subjects else PRODUCT_NAME

    lines = [
        f"# {subject} — profile export",
        "",
        f"Generated at: {datetime.utcnow():%Y-%m-%d %H:%M} UTC",
        f"Confirmed memories: {len(memories)}",
        "",
    ]
    if not memories:
        lines.append("(no confirmed memories yet)")
        return "\n".join(lines) + "\n"

    grouped: dict[str, list[Memory]] = defaultdict(list)
    for m in memories:
        grouped[m.memory_type].append(m)

    for t in TYPE_ORDER:
        items = grouped.get(t)
        if not items:
            continue
        lines.append(f"## {TYPE_META.get(t, {}).get('label', t)}")
        lines.append("")
        for m in items:
            content = redact_text(
                m.content, enabled=settings.redaction_enabled
            ).text
            meta = [
                f"#{m.id}",
                f"confidence {m.confidence:.2f}",
                f"{len(m.evidence or [])} evidence",
            ]
            if m.is_inference:
                meta.append("inference")
            if m.conflict_group:
                meta.append("conflict")
            lines.append(f"- {content}  _({' · '.join(meta)})_")
        lines.append("")

    lines.append("> Confirmed memories only; candidate memories and raw chat logs are excluded.")
    return "\n".join(lines) + "\n"


def build_memory_graph(db: Session, *, limit: int = 120) -> dict[str, Any]:
    """Build a portrait graph: subject → types → clustered facets (not raw memories)."""
    memories = (
        db.query(Memory)
        .options(joinedload(Memory.evidence))
        .filter(Memory.status == MemoryStatus.CONFIRMED.value)
        .order_by(Memory.confidence.desc(), Memory.updated_at.desc())
        .all()
    )

    hidden_events = [m for m in memories if m.memory_type == "event"]
    portrait = [m for m in memories if m.memory_type in PORTRAIT_TYPES]
    # Soft cap: prefer high-confidence items if there are many
    if len(portrait) > max(limit * 3, 200):
        portrait = portrait[: max(limit * 3, 200)]

    subject = "Subject"
    for m in memories:
        if m.subject:
            subject = m.subject
            break

    nodes: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    nodes.append(
        {
            "id": "subject",
            "kind": "subject",
            "label": subject,
            "full": subject,
            "color": "#e8eaed",
            "size": 28,
            "group": "subject",
        }
    )

    by_type: dict[str, list[Memory]] = defaultdict(list)
    for m in portrait:
        by_type[m.memory_type].append(m)

    present_types = [t for t in PORTRAIT_TYPES if by_type.get(t)]
    facet_records: list[dict[str, Any]] = []

    for t in present_types:
        meta = TYPE_META.get(t, {"label": t, "color": "#8ab4f8"})
        type_memories = by_type[t]
        nodes.append(
            {
                "id": f"type:{t}",
                "kind": "type",
                "label": meta["label"],
                "full": meta["label"],
                "color": meta["color"],
                "size": 18,
                "group": t,
                "facet_count": 0,  # filled after clustering
            }
        )
        links.append(
            {
                "source": "subject",
                "target": f"type:{t}",
                "kind": "owns",
                "strength": 0.4,
            }
        )

        clusters = _cluster_memories(type_memories)
        # Prefer larger / higher-confidence clusters
        clusters.sort(
            key=lambda c: (
                len(c),
                max((m.confidence or 0) for m in c),
            ),
            reverse=True,
        )
        shown = clusters[:MAX_FACETS_PER_TYPE]
        overflow = clusters[MAX_FACETS_PER_TYPE:]
        overflow_mem_count = sum(len(c) for c in overflow)

        for idx, cluster in enumerate(shown):
            facet = _facet_from_cluster(t, idx, cluster, meta)
            facet_records.append(facet)
            nodes.append(facet)
            links.append(
                {
                    "source": f"type:{t}",
                    "target": facet["id"],
                    "kind": "contains",
                    "strength": 0.7,
                }
            )

        if overflow_mem_count:
            overflow_id = f"overflow:{t}"
            nodes.append(
                {
                    "id": overflow_id,
                    "kind": "overflow",
                    "label": f"+{overflow_mem_count} more",
                    "full": f"{overflow_mem_count} more {meta['label']} memories not expanded",
                    "color": meta["color"],
                    "size": 10,
                    "group": t,
                    "type_label": meta["label"],
                    "href": f"/memories?status=confirmed&memory_type={t}",
                    "memory_count": overflow_mem_count,
                }
            )
            links.append(
                {
                    "source": f"type:{t}",
                    "target": overflow_id,
                    "kind": "contains",
                    "strength": 0.35,
                }
            )

        # Update type node facet_count
        for n in nodes:
            if n["id"] == f"type:{t}":
                n["facet_count"] = len(shown) + (1 if overflow_mem_count else 0)
                break

    # Conflict edges: facets that share a conflict_group across clusters
    # (same-group memories already merged into one has_conflict facet)
    conflict_index: dict[str, list[str]] = defaultdict(list)
    for f in facet_records:
        for cg in f.get("conflict_groups") or []:
            conflict_index[cg].append(f["id"])
    conflict_pairs: set[tuple[str, str]] = set()
    for facet_ids in conflict_index.values():
        uniq = sorted(set(facet_ids))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                conflict_pairs.add((uniq[i], uniq[j]))
    for a, b in conflict_pairs:
        links.append(
            {
                "source": a,
                "target": b,
                "kind": "conflict",
                "strength": 0.2,
            }
        )

    # Cross-type similarity between facet centroids
    sim_added = 0
    for i in range(len(facet_records)):
        for j in range(i + 1, len(facet_records)):
            a, b = facet_records[i], facet_records[j]
            if a["group"] == b["group"]:
                continue
            score = cosine(a["centroid"], b["centroid"])
            if score >= SIMILARITY_THRESHOLD:
                links.append(
                    {
                        "source": a["id"],
                        "target": b["id"],
                        "kind": "similar",
                        "strength": 0.15,
                        "score": round(score, 3),
                    }
                )
                sim_added += 1
            if sim_added >= MAX_CROSS_SIMILAR:
                break
        if sim_added >= MAX_CROSS_SIMILAR:
            break

    # Strip centroids from API payload (internal only)
    for n in nodes:
        n.pop("centroid", None)
        n.pop("conflict_groups", None)

    facet_count = sum(1 for n in nodes if n["kind"] == "facet")
    return {
        "nodes": nodes,
        "links": links,
        "meta": {
            "memory_count": len(portrait),
            "facet_count": facet_count,
            "type_count": len(present_types),
            "conflict_edges": sum(1 for l in links if l["kind"] == "conflict"),
            "similar_edges": sum(1 for l in links if l["kind"] == "similar"),
            "hidden_event_count": len(hidden_events),
            "types": [
                {
                    "type": t,
                    "label": TYPE_META.get(t, {}).get("label", t),
                    "color": TYPE_META.get(t, {}).get("color", "#8ab4f8"),
                }
                for t in present_types
            ],
        },
    }


def graph_text_outline(graph: dict[str, Any]) -> dict[str, Any]:
    """Text equivalent of the canvas graph.

    The canvas is a picture: keyboard users and screen readers cannot read it,
    so the same nodes and links are exposed as nested lists.
    """
    nodes = {node["id"]: node for node in graph.get("nodes") or []}
    subjects = [n for n in nodes.values() if n.get("kind") == "subject"]
    groups: list[dict[str, Any]] = []
    for node in graph.get("nodes") or []:
        if node.get("kind") != "type":
            continue
        items = [
            {
                "text": (nodes[target].get("full") or nodes[target].get("label") or ""),
                "href": nodes[target].get("href") or "",
                "meta": _facet_meta(nodes[target]),
            }
            for link in graph.get("links") or []
            if link.get("kind") == "contains" and link.get("source") == node["id"]
            for target in [link.get("target")]
            if target in nodes
        ]
        groups.append(
            {
                "label": node.get("label") or node.get("id"),
                "count": len(items),
                "entries": items,
            }
        )

    relations: list[str] = []
    for link in graph.get("links") or []:
        if link.get("kind") not in ("conflict", "similar"):
            continue
        source = nodes.get(link.get("source"))
        target = nodes.get(link.get("target"))
        if not source or not target:
            continue
        kind = "Conflict" if link["kind"] == "conflict" else "Related"
        relations.append(f"{kind}: {source.get('full') or source.get('label')} ↔ {target.get('full') or target.get('label')}")

    return {
        "subject": subjects[0].get("label") if subjects else "Subject",
        "groups": groups,
        "relations": relations,
    }


def _facet_meta(node: dict[str, Any]) -> str:
    parts = []
    count = node.get("member_count") or node.get("memory_count")
    if count:
        parts.append(f"{count} memories")
    if node.get("has_conflict"):
        parts.append("conflict")
    if node.get("evidence_count"):
        parts.append(f"{node['evidence_count']} evidence")
    return " · ".join(parts)


def _cluster_memories(memories: list[Memory]) -> list[list[Memory]]:
    if not memories:
        return []
    ids = [m.id for m in memories]
    by_id = {m.id: m for m in memories}
    uf = _UnionFind(ids)

    # Same conflict_group → same facet
    by_conflict: dict[str, list[int]] = defaultdict(list)
    for m in memories:
        if m.conflict_group:
            by_conflict[m.conflict_group].append(m.id)
    for group_ids in by_conflict.values():
        root = group_ids[0]
        for other in group_ids[1:]:
            uf.union(root, other)

    # Always derive vectors from content. Older databases may contain vectors
    # produced with Python's process-randomized hash and cannot be trusted.
    embeds = {m.id: local_embed(m.content) for m in memories}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if uf.find(a) == uf.find(b):
                continue
            if cosine(embeds[a], embeds[b]) >= SIMILARITY_THRESHOLD:
                uf.union(a, b)

    clusters: dict[int, list[Memory]] = defaultdict(list)
    for mid in ids:
        clusters[uf.find(mid)].append(by_id[mid])
    return list(clusters.values())


def _facet_from_cluster(
    memory_type: str,
    idx: int,
    cluster: list[Memory],
    meta: dict[str, str],
) -> dict[str, Any]:
    ranked = sorted(cluster, key=lambda m: (m.confidence or 0), reverse=True)
    primary = ranked[0]
    content = primary.content
    short = content if len(content) <= 18 else content[:16] + "…"
    embeds = [local_embed(m.content) for m in cluster]
    dim = len(embeds[0]) if embeds else 0
    if dim:
        centroid = [sum(e[i] for e in embeds) / len(embeds) for i in range(dim)]
        norm = sum(v * v for v in centroid) ** 0.5 or 1.0
        centroid = [v / norm for v in centroid]
    else:
        centroid = []

    conflict_groups = sorted({m.conflict_group for m in cluster if m.conflict_group})
    has_conflict = bool(conflict_groups)

    max_conf = max((m.confidence or 0.5) for m in cluster)
    return {
        "id": f"facet:{memory_type}:{idx}",
        "kind": "facet",
        "label": short,
        "full": content,
        "color": meta["color"],
        "size": 10 + min(10, len(cluster) * 2) + min(4, int(max_conf * 4)),
        "group": memory_type,
        "type_label": meta["label"],
        "confidence": primary.confidence,
        "has_conflict": has_conflict,
        "memory_ids": [m.id for m in ranked],
        "memories": [
            {
                "id": m.id,
                "content": m.content,
                "confidence": m.confidence,
                "is_inference": m.is_inference,
                "conflict_group": m.conflict_group,
            }
            for m in ranked
        ],
        "member_count": len(cluster),
        "evidence_count": sum(len(m.evidence or []) for m in cluster),
        "centroid": centroid,
        "conflict_groups": conflict_groups,
    }
