from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class NormalizedMessage:
    sender: str
    content: str
    sent_at: datetime | None = None
    msg_type: str = "text"
    sender_role: str = "unknown"
    raw: dict[str, Any] | None = None

    def fingerprint(self) -> str:
        ts = self.sent_at.isoformat() if self.sent_at else ""
        payload = f"{self.sender}|{ts}|{self.msg_type}|{self.content}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ImportPreview:
    total_rows: int
    sample: list[dict[str, Any]]
    detected_fields: list[str]
    suggested_mapping: dict[str, str]
    errors: list[str] = field(default_factory=list)
    format_name: str = "unknown"


@dataclass
class ImportResult:
    messages: list[NormalizedMessage]
    skipped: int
    errors: list[dict[str, Any]]
    mapping_used: dict[str, str]
    # Resolved identity when the adapter can infer it (e.g. WeFlow ``isSend``).
    subject_name: str | None = None
    partner_name: str | None = None


# Canonical field names the importer needs
CANONICAL_FIELDS = ("sender", "content", "timestamp", "msg_type")

# Common aliases seen in WeChat export tools
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "sender": (
        "sender",
        "from",
        "from_user",
        "talker",
        "nickname",
        "sender_name",
        "speaker",
        "name",
        "user",
    ),
    "content": (
        "content",
        "text",
        "message",
        "msg",
        "body",
        "msg_content",
    ),
    "timestamp": (
        "timestamp",
        "time",
        "create_time",
        "msg_time",
        "sent_at",
        "datetime",
        "date",
        "ts",
        "createTime",
    ),
    "msg_type": (
        "msg_type",
        "type",
        "message_type",
        "msgtype",
        "kind",
        "content_type",
    ),
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def suggest_mapping(fields: list[str]) -> dict[str, str]:
    lower_map = {f.lower(): f for f in fields}
    mapping: dict[str, str] = {}
    for canonical, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            if alias.lower() in lower_map:
                mapping[canonical] = lower_map[alias.lower()]
                break
    return mapping


def parse_timestamp(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        # Heuristic: ms vs s
        ts = float(value)
        if ts > 1e12:
            ts /= 1000.0
        elif ts > 1e10:
            ts /= 1000.0
        try:
            return datetime.utcfromtimestamp(ts)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return parse_timestamp(int(text))
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
        ):
            try:
                return datetime.strptime(text.replace("+08:00", "").replace("Z", ""), fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def load_json_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load JSON array, object-with-messages, or JSONL. Returns (records, format)."""
    raw = path.read_text(encoding="utf-8-sig").strip()
    if not raw:
        raise ValueError("The file is empty")

    # JSONL detection: first non-empty line is object, and multiple lines
    if "\n" in raw and raw.lstrip().startswith("{"):
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if len(lines) > 1:
            try:
                records = [json.loads(ln) for ln in lines]
                if all(isinstance(r, dict) for r in records):
                    return records, "jsonl"
            except json.JSONDecodeError:
                pass

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot parse JSON: {exc}") from exc

    if isinstance(data, list):
        if not data:
            return [], "json_array"
        if all(isinstance(x, dict) for x in data):
            return data, "json_array"
        raise ValueError("JSON array elements must be objects")

    if isinstance(data, dict):
        for key in ("messages", "data", "chats", "records", "items"):
            if key in data and isinstance(data[key], list):
                records = data[key]
                if records and not all(isinstance(x, dict) for x in records):
                    raise ValueError(f"Field `{key}` must be an array of objects")
                return records, f"json_object.{key}"
        # Single message object
        if any(k in data for k in ("content", "text", "message")):
            return [data], "json_object.single"
        raise ValueError(
            "Could not recognize a message list. Provide a JSON array, or an object containing a messages/data/chats field."
        )

    raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")
