from __future__ import annotations

from pathlib import Path
from typing import Any

from app.importers.base import (
    CANONICAL_FIELDS,
    ImportPreview,
    ImportResult,
    NormalizedMessage,
    load_json_records,
    parse_timestamp,
    suggest_mapping,
)


def preview_wechat_file(path: Path, mapping: dict[str, str] | None = None) -> ImportPreview:
    records, format_name = load_json_records(path)
    return preview_records(records, format_name, mapping=mapping)


def preview_records(
    records: list[dict[str, Any]],
    format_name: str,
    mapping: dict[str, str] | None = None,
) -> ImportPreview:
    if not records:
        return ImportPreview(
            total_rows=0,
            sample=[],
            detected_fields=[],
            suggested_mapping={},
            errors=["The file contains no message records"],
            format_name=format_name,
        )

    fields = sorted({k for row in records[:50] for k in row.keys()})
    suggested = mapping or suggest_mapping(fields)
    errors: list[str] = []
    missing = [f for f in ("sender", "content") if f not in suggested]
    if missing:
        errors.append(
            f"Could not auto-detect fields: {', '.join(missing)}. "
            f"Detected fields: {', '.join(fields)}. Please map them manually."
        )

    sample = []
    for row in records[:5]:
        sample.append(_project_row(row, suggested))

    return ImportPreview(
        total_rows=len(records),
        sample=sample,
        detected_fields=fields,
        suggested_mapping=suggested,
        errors=errors,
        format_name=format_name,
    )


def import_wechat_file(
    path: Path,
    mapping: dict[str, str] | None = None,
    subject_name: str | None = None,
    partner_name: str | None = None,
) -> ImportResult:
    records, _format_name = load_json_records(path)
    return normalize_records(
        records,
        mapping=mapping,
        subject_name=subject_name,
        partner_name=partner_name,
    )


def normalize_records(
    records: list[dict[str, Any]],
    *,
    mapping: dict[str, str] | None = None,
    subject_name: str | None = None,
    partner_name: str | None = None,
) -> ImportResult:
    if not records:
        return ImportResult(messages=[], skipped=0, errors=[], mapping_used={})

    fields = sorted({k for row in records[:200] for k in row.keys()})
    used = mapping or suggest_mapping(fields)

    missing = [f for f in ("sender", "content") if f not in used]
    if missing:
        raise ValueError(
            f"Missing required field mapping: {', '.join(missing)}. "
            f"Detected fields: {', '.join(fields)}"
        )

    messages: list[NormalizedMessage] = []
    errors: list[dict[str, Any]] = []
    skipped = 0

    for idx, row in enumerate(records):
        try:
            sender = _get(row, used.get("sender"))
            content = _get(row, used.get("content"))
            if sender is None or content is None:
                skipped += 1
                errors.append(
                    {
                        "row": idx,
                        "error": "missing sender or content",
                        "raw_keys": list(row.keys()),
                    }
                )
                continue

            sender_s = str(sender).strip()
            content_s = str(content).strip()
            if not content_s:
                skipped += 1
                continue

            ts_raw = _get(row, used.get("timestamp")) if used.get("timestamp") else None
            sent_at = parse_timestamp(ts_raw)
            msg_type_raw = _get(row, used.get("msg_type")) if used.get("msg_type") else "text"
            msg_type = _normalize_msg_type(msg_type_raw)

            role = _resolve_role(sender_s, subject_name, partner_name)
            messages.append(
                NormalizedMessage(
                    sender=sender_s,
                    content=content_s,
                    sent_at=sent_at,
                    msg_type=msg_type,
                    sender_role=role,
                    raw=row,
                )
            )
        except Exception as exc:  # noqa: BLE001 — collect per-row errors
            skipped += 1
            errors.append({"row": idx, "error": str(exc)})

    return ImportResult(
        messages=messages,
        skipped=skipped,
        errors=errors[:100],
        mapping_used={k: v for k, v in used.items() if k in CANONICAL_FIELDS and v},
    )


def _get(row: dict[str, Any], key: str | None) -> Any:
    if not key:
        return None
    return row.get(key)


def _project_row(row: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for canonical, source in mapping.items():
        out[canonical] = row.get(source)
    out["_raw_keys"] = list(row.keys())
    return out


def _normalize_msg_type(value: Any) -> str:
    if value is None:
        return "text"
    text = str(value).lower().strip()
    mapping = {
        "1": "text",
        "text": "text",
        "3": "image",
        "image": "image",
        "img": "image",
        "34": "voice",
        "voice": "voice",
        "audio": "voice",
        "43": "video",
        "video": "video",
        "47": "sticker",
        "sticker": "sticker",
        "emoji": "sticker",
        "49": "link",
        "link": "link",
        "app": "link",
        "10000": "system",
        "system": "system",
    }
    return mapping.get(text, text or "text")


def _resolve_role(
    sender: str, subject_name: str | None, partner_name: str | None
) -> str:
    if subject_name and sender == subject_name:
        return "subject"
    if partner_name and sender == partner_name:
        return "partner"
    # Soft match
    if subject_name and subject_name in sender:
        return "subject"
    if partner_name and partner_name in sender:
        return "partner"
    return "unknown"
