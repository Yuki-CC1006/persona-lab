"""Import plain text / Markdown chat logs by pasting or uploading.

WeChat has no official text export, so most people arrive with whatever they
can copy: a txt dump, a Markdown export from another tool, or hand-typed
lines. This importer accepts the tolerant shapes people actually have.

Supported line shapes (all optional parts are detected):
    2026-03-01 10:00 Alex: 喜欢火锅微辣
    [2026-03-01 10:00] Alex：喜欢火锅微辣
    Alex: 喜欢火锅微辣
    **Alex**            <- speaker header, following lines belong to Alex
    ## Alex
Lines without a speaker prefix continue the previous message.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.importers.base import ImportPreview, ImportResult
from app.importers.wechat_json import normalize_records, preview_records

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".log"}
SOURCE_TEXT = "text_chat"
SOURCE_WECHAT = "wechat_json"

_BRACKET_TIME = re.compile(
    r"^\s*[\[(]\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?)\s*[\])]\s*"
)
_PREFIX_TIME = re.compile(
    r"^\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?)\s+"
)
_SPEAKER = re.compile(r"^(?:\*\*)?([^:：\n]{1,32}?)(?:\*\*)?\s*[:：]\s*(.*)$")
_HEADER = re.compile(r"^\s*(?:\*\*|#{1,6}\s*)([^:*#\n]{1,12})\*{0,2}\s*$")

UNKNOWN_SENDER = "Unknown"


def looks_like_text_source(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES


def parse_chat_text(text: str) -> list[dict[str, Any]]:
    """Turn pasted text into canonical records the shared pipeline can map."""
    records: list[dict[str, Any]] = []
    current_sender: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue

        timestamp: str | None = None
        body = line

        bracket = _BRACKET_TIME.match(body)
        if bracket:
            timestamp = bracket.group(1)
            body = body[bracket.end() :]

        if timestamp is None:
            prefix = _PREFIX_TIME.match(body)
            if prefix:
                timestamp = prefix.group(1)
                body = body[prefix.end() :]

        speaker_match = _SPEAKER.match(body)
        if speaker_match:
            sender = speaker_match.group(1).strip()
            content = speaker_match.group(2).strip()
            if content:
                current_sender = sender
                records.append(_record(sender, content, timestamp))
                continue
            # A bare "名字:" line names the following speaker.
            if sender:
                current_sender = sender
            continue

        header = _HEADER.match(line)
        if header:
            current_sender = header.group(1).strip()
            continue

        sender = current_sender or UNKNOWN_SENDER
        if records and sender == records[-1]["sender"]:
            records[-1]["content"] = f"{records[-1]['content']}\n{line.strip()}"
            continue
        records.append(_record(sender, line.strip(), timestamp))

    return records


def load_text_records(path: Path) -> tuple[list[dict[str, Any]], str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if not text.strip():
        raise ValueError("The file is empty")
    return parse_chat_text(text), SOURCE_TEXT


def preview_text_file(path: Path, mapping: dict[str, str] | None = None) -> ImportPreview:
    records, format_name = load_text_records(path)
    return preview_records(records, format_name, mapping=mapping)


def import_text_file(
    path: Path,
    mapping: dict[str, str] | None = None,
    subject_name: str | None = None,
    partner_name: str | None = None,
) -> ImportResult:
    records, _format_name = load_text_records(path)
    return normalize_records(
        records,
        mapping=mapping,
        subject_name=subject_name,
        partner_name=partner_name,
    )


def _record(sender: str, content: str, timestamp: str | None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sender": sender,
        "content": content,
        "msg_type": "text",
    }
    if timestamp:
        row["timestamp"] = timestamp
    return row
