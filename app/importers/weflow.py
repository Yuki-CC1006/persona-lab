"""WeFlow session JSON adapter.

WeChat has no official text export, and WeFlow (a widely used local exporter) is
the format most people already have on disk. A WeFlow file is one session plus
its messages::

    {
      "session": {"displayName": "Alex", "remark": "Alex", ...},
      "messages": [
        {"createTime": 1704067200, "localType": "文本", "content": "…",
         "isSend": false, "senderDisplayName": "Alex"},
        ...
      ]
    }

``isSend`` is authoritative for who is who, which the generic importer cannot
express. This adapter maps the shape onto the same canonical records the shared
pipeline already consumes; it does not touch storage, review, or retrieval.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.importers.base import ImportPreview, ImportResult
from app.importers.wechat_json import normalize_records, preview_records

SOURCE_WEFLOW = "weflow_json"
FMT_WEFLOW = "weflow.session"

# localType / type values that still carry real text worth remembering.
_TEXT_LOCAL_TYPES = {"", "文本", "引用", "文本消息", "引用消息"}
_TEXT_TYPES = {"文本消息", "引用消息", "文本", "引用"}
_SELF_LABEL = "Me"
_FALLBACK_SUBJECT = "the other person"


def looks_like_weflow(path: Path) -> bool:
    """Cheap shape probe used to route a .json file to this adapter."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return _is_weflow(data)


def preview_weflow_file(path: Path, mapping: dict[str, str] | None = None) -> ImportPreview:
    records, _subject, _partner = _prepare(path)
    return preview_records(records, FMT_WEFLOW, mapping=mapping)


def import_weflow_file(
    path: Path,
    mapping: dict[str, str] | None = None,
    subject_name: str | None = None,
    partner_name: str | None = None,
) -> ImportResult:
    records, subject, partner = _prepare(path, subject_name, partner_name)
    result = normalize_records(
        records,
        mapping=mapping,
        subject_name=subject,
        partner_name=partner,
    )
    result.subject_name = subject
    result.partner_name = partner
    return result


def _is_weflow(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("session"), dict)
        and isinstance(data.get("messages"), list)
    )


def _read(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not _is_weflow(data):
        raise ValueError("Not a WeFlow session JSON: missing session / messages fields")
    return data


def _contact_name(session: dict[str, Any]) -> str | None:
    for key in ("displayName", "remark", "nickname", "wxid"):
        value = str(session.get(key) or "").strip()
        if value:
            return value
    return None


def _prepare(
    path: Path,
    subject_name: str | None = None,
    partner_name: str | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    """Return canonical records plus the labels roles were resolved against."""
    data = _read(path)
    session = data.get("session") or {}
    subject = (subject_name or "").strip() or _contact_name(session) or _FALLBACK_SUBJECT
    partner = (partner_name or "").strip() or _SELF_LABEL
    return _records(data, subject, partner), subject, partner


def _records(data: dict[str, Any], subject: str, partner: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for msg in data.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        local = str(msg.get("localType") or "").strip()
        declared = str(msg.get("type") or "").strip()
        if local not in _TEXT_LOCAL_TYPES and declared not in _TEXT_TYPES:
            continue
        content = str(msg.get("content") or "").strip()
        if not content:
            continue
        sender = partner if bool(msg.get("isSend")) else subject
        records.append(
            {
                "sender": sender,
                "content": content,
                "timestamp": msg.get("createTime") or msg.get("timestamp"),
                "msg_type": "text",
            }
        )
    return records
