from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import ImportJob, Message
from app.importers.base import file_sha256
from app.importers.text_chat import (
    SOURCE_TEXT,
    SOURCE_WECHAT,
    import_text_file,
    looks_like_text_source,
    preview_text_file,
)
from app.importers.wechat_json import import_wechat_file, preview_wechat_file
from app.importers.weflow import (
    SOURCE_WEFLOW,
    import_weflow_file,
    looks_like_weflow,
    preview_weflow_file,
)


def resolve_source(path: Path, source: str | None = None) -> str:
    """Explicit source wins; otherwise guess from the file extension."""
    if source in (SOURCE_TEXT, SOURCE_WECHAT, SOURCE_WEFLOW):
        return source
    if looks_like_text_source(path):
        return SOURCE_TEXT
    return SOURCE_WEFLOW if looks_like_weflow(path) else SOURCE_WECHAT


def preview_import(
    path: Path, mapping: dict[str, str] | None = None, source: str | None = None
) -> dict:
    resolved = resolve_source(path, source)
    if resolved == SOURCE_TEXT:
        preview = preview_text_file(path, mapping=mapping)
    elif resolved == SOURCE_WEFLOW:
        preview = preview_weflow_file(path, mapping=mapping)
    else:
        preview = preview_wechat_file(path, mapping=mapping)
    return {
        "total_rows": preview.total_rows,
        "sample": preview.sample,
        "detected_fields": preview.detected_fields,
        "suggested_mapping": preview.suggested_mapping,
        "errors": preview.errors,
        "format_name": preview.format_name,
        "source": resolved,
        "senders": _distinct_senders(preview.sample),
        "file_hash": file_sha256(path),
    }


def _distinct_senders(sample: list[dict]) -> list[str]:
    """Distinct senders in preview order, for pre-filling the role inputs."""
    seen: list[str] = []
    for row in sample or []:
        name = str((row or {}).get("sender") or "").strip()
        if name and name not in seen:
            seen.append(name)
    return seen


def commit_import(
    db: Session,
    path: Path,
    *,
    mapping: dict[str, str] | None = None,
    subject_name: str | None = None,
    partner_name: str | None = None,
    source: str | None = None,
) -> ImportJob:
    file_hash = file_sha256(path)
    resolved = resolve_source(path, source)
    if resolved == SOURCE_TEXT:
        importer = import_text_file
    elif resolved == SOURCE_WEFLOW:
        importer = import_weflow_file
    else:
        importer = import_wechat_file
    result = importer(
        path,
        mapping=mapping,
        subject_name=subject_name,
        partner_name=partner_name,
    )

    job = ImportJob(
        filename=path.name,
        file_hash=file_hash,
        source_format=resolved,
        status="importing",
        field_mapping=result.mapping_used,
        subject_name=subject_name or result.subject_name,
        partner_name=partner_name or result.partner_name,
        error_report={"errors": result.errors} if result.errors else None,
    )
    db.add(job)
    db.flush()

    inserted = 0
    skipped_dup = 0
    for nm in result.messages:
        fp = nm.fingerprint()
        exists = db.query(Message.id).filter(Message.fingerprint == fp).first()
        if exists:
            skipped_dup += 1
            continue
        db.add(
            Message(
                import_id=job.id,
                fingerprint=fp,
                sender=nm.sender,
                sender_role=nm.sender_role,
                content=nm.content,
                msg_type=nm.msg_type,
                sent_at=nm.sent_at,
                raw=nm.raw,
            )
        )
        inserted += 1

    job.message_count = inserted
    job.skipped_count = result.skipped + skipped_dup
    job.status = "completed"
    job.completed_at = datetime.utcnow()
    db.commit()
    db.refresh(job)
    return job


def content_fingerprint(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
