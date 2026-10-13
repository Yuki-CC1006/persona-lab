from __future__ import annotations

import re
from dataclasses import dataclass


PHONE_RE = re.compile(r"(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)")
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
ID_CARD_RE = re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)")
BANK_RE = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
URL_RE = re.compile(r"https?://[^\s]+")
WXID_RE = re.compile(r"\bwxid_[a-zA-Z0-9_]+\b", re.IGNORECASE)


@dataclass
class RedactionResult:
    text: str
    original_len: int
    redacted_len: int
    replacements: int


def redact_text(text: str, enabled: bool = True) -> RedactionResult:
    if not enabled:
        return RedactionResult(
            text=text, original_len=len(text), redacted_len=len(text), replacements=0
        )

    replacements = 0

    def _sub(pattern: re.Pattern[str], repl: str, src: str) -> str:
        nonlocal replacements

        def _count(m: re.Match[str]) -> str:
            nonlocal replacements
            replacements += 1
            return repl

        return pattern.sub(_count, src)

    out = text
    out = _sub(ID_CARD_RE, "[ID number]", out)
    out = _sub(PHONE_RE, "[phone number]", out)
    out = _sub(EMAIL_RE, "[email]", out)
    out = _sub(BANK_RE, "[bank card number]", out)
    out = _sub(URL_RE, "[link]", out)
    out = _sub(WXID_RE, "[wechat id]", out)
    return RedactionResult(
        text=out,
        original_len=len(text),
        redacted_len=len(out),
        replacements=replacements,
    )


def preview_redaction(messages: list[str], enabled: bool = True, limit: int = 10) -> list[dict]:
    previews = []
    for msg in messages[:limit]:
        result = redact_text(msg, enabled=enabled)
        previews.append(
            {
                "original": msg,
                "redacted": result.text,
                "replacements": result.replacements,
            }
        )
    return previews
