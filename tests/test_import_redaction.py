from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.services.import_service import commit_import, preview_import
from app.services.redaction import redact_text


def test_redaction_phone_and_email():
    text = "Phone 13800138000 email alice@example.com see https://example.com/x"
    result = redact_text(text, enabled=True)
    assert "13800138000" not in result.text
    assert "alice@example.com" not in result.text
    assert "https://" not in result.text
    assert result.replacements >= 3


def test_redaction_disabled():
    text = "Phone 13800138000"
    result = redact_text(text, enabled=False)
    assert result.text == text
    assert result.replacements == 0


def test_preview_and_idempotent_import(db: Session, tmp_path):
    sample = tmp_path / "chat.json"
    sample.write_text(
        '[{"sender":"Alex","content":"likes unsweetened Americano","timestamp":"2026-03-01 10:00:00","type":"text"},'
        '{"sender":"Sam","content":"okay","timestamp":"2026-03-01 10:01:00","type":"text"}]',
        encoding="utf-8",
    )
    preview = preview_import(sample)
    assert preview["total_rows"] == 2
    assert "sender" in preview["suggested_mapping"]
    assert "content" in preview["suggested_mapping"]

    job1 = commit_import(db, sample, subject_name="Alex", partner_name="Sam")
    assert job1.message_count == 2
    job2 = commit_import(db, sample, subject_name="Alex", partner_name="Sam")
    assert job2.message_count == 0
    assert job2.skipped_count >= 2


def test_bad_json_reports_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"foo": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        preview_import(bad)


def test_jsonl_import(db: Session, tmp_path):
    sample = tmp_path / "chat.jsonl"
    sample.write_text(
        '{"sender":"Alex","content":"does not want to be woken up on Monday","timestamp":1741132800}\n'
        '{"sender":"Sam","content":"okay","timestamp":1741132860}\n',
        encoding="utf-8",
    )
    preview = preview_import(sample)
    assert preview["format_name"] == "jsonl"
    job = commit_import(db, sample, subject_name="Alex", partner_name="Sam")
    assert job.message_count == 2
