from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.importers.base import parse_timestamp
from app.importers.text_chat import parse_chat_text
from app.services import memory as memory_svc
from app.services.import_service import commit_import, preview_import
from app.services.onboarding import example_questions, import_next_steps


def test_parse_speaker_prefixed_lines():
    records = parse_chat_text("Alex: likes mildly spicy hotpot\nMe: then I'll take you next time\n")
    assert [r["sender"] for r in records] == ["Alex", "Me"]
    assert records[0]["content"] == "likes mildly spicy hotpot"


def test_parse_speaker_prefixed_chinese_lines():
    """Chinese line shapes (full-width colon, no space) must still parse."""
    records = parse_chat_text("Alex: 喜欢火锅微辣\nMe：那下次带你去\n")
    assert [r["sender"] for r in records] == ["Alex", "Me"]
    assert records[0]["content"] == "喜欢火锅微辣"


def test_parse_timestamp_brackets_and_prefix():
    records = parse_chat_text(
        "[2026-03-01 21:00] Alex: sounds good\n2026-03-02 09:30 Me: morning\n"
    )
    assert len(records) == 2
    assert parse_timestamp(records[0]["timestamp"]) is not None
    assert parse_timestamp(records[1]["timestamp"]) is not None


def test_parse_header_and_continuation_lines():
    records = parse_chat_text("## Alex\nlikes hotpot\nmildly spicy\n\nMe: noted\n")
    assert records[0]["sender"] == "Alex"
    assert "mildly spicy" in records[0]["content"]
    assert records[1]["sender"] == "Me"


def test_parse_markdown_bold_speaker():
    records = parse_chat_text("**Alex**\ndislikes cilantro\n")
    assert records[0]["sender"] == "Alex"
    assert records[0]["content"] == "dislikes cilantro"


def test_parse_chinese_header_and_continuation_lines():
    """A Chinese speaker header must still label the following lines."""
    records = parse_chat_text("## Alex\n喜欢火锅\n微辣的那种\n\nMe: 记住了\n")
    assert records[0]["sender"] == "Alex"
    assert "微辣" in records[0]["content"]
    assert records[1]["sender"] == "Me"


def test_plain_lines_without_speaker_are_not_lost():
    records = parse_chat_text("We talked about a lot today\nAlex: likes hotpot\n")
    assert records[0]["sender"] == "Unknown"
    assert records[0]["content"] == "We talked about a lot today"


def test_preview_detects_text_source_by_suffix(tmp_path):
    sample = tmp_path / "chat.txt"
    sample.write_text("Alex: likes mildly spicy hotpot\n", encoding="utf-8")
    preview = preview_import(sample)
    assert preview["source"] == "text_chat"
    assert preview["total_rows"] == 1


def test_preview_keeps_json_source_for_json(tmp_path):
    sample = tmp_path / "chat.json"
    sample.write_text('[{"sender":"Alex","content":"likes hotpot"}]', encoding="utf-8")
    preview = preview_import(sample)
    assert preview["source"] == "wechat_json"


def test_commit_text_import_roles(db: Session, tmp_path):
    sample = tmp_path / "chat.md"
    sample.write_text(
        "Alex: likes mildly spicy hotpot\nMe: then I'll take you next time\n", encoding="utf-8"
    )
    job = commit_import(db, sample, subject_name="Alex", partner_name="Me")
    assert job.source_format == "text_chat"
    assert job.message_count == 2


def test_empty_text_file_reports_error(tmp_path):
    sample = tmp_path / "empty.txt"
    sample.write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError):
        preview_import(sample)


def test_bad_json_error_suggests_text_fallback(client: TestClient, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"foo": 1}', encoding="utf-8")
    with bad.open("rb") as f:
        resp = client.post(
            "/imports/preview", files={"file": ("bad.json", f, "application/json")}
        )
    assert resp.status_code == 400
    assert "How to fix" in resp.text
    assert "paste box" in resp.text


def test_paste_text_preview_and_commit(client: TestClient):
    preview = client.post(
        "/imports/preview",
        data={"text": "Alex: likes unsweetened Americano\nMe: noted", "source": "text_chat"},
    )
    assert preview.status_code == 200
    assert "Preview result" in preview.text
    assert "text_chat" in preview.text

    upload_path = _extract_upload_path(preview.text)
    assert upload_path
    commit = client.post(
        "/imports/commit",
        data={
            "upload_path": upload_path,
            "subject_name": "Alex",
            "partner_name": "Me",
            "mapping_json": "{}",
            "source": "text_chat",
            "run_analysis": "no",
        },
        follow_redirects=False,
    )
    assert commit.status_code in (302, 303)
    detail = client.get(commit.headers["location"])
    assert detail.status_code == 200
    assert "What to do next" in detail.text


def test_preview_without_input_asks_for_something(client: TestClient):
    resp = client.post("/imports/preview", data={"text": "", "source": ""})
    assert resp.status_code == 400
    assert "Please choose a file or paste" in resp.text


def test_next_steps_include_questions_after_confirmation(db: Session, tmp_path):
    sample = tmp_path / "chat.txt"
    sample.write_text("Alex: dislikes cilantro\n", encoding="utf-8")
    job = commit_import(db, sample, subject_name="Alex")
    before = import_next_steps(db, job)
    assert before["questions"] == []
    assert before["steps"][0]["label"].startswith("Review the candidate memories from this import")

    memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion", subject="Alex")
    after = import_next_steps(db, job)
    assert after["subject_name"] == "Alex"
    assert any("Alex" in q for q in after["questions"])
    assert after["steps"][0]["done"] is True


def test_example_questions_follow_available_types(db: Session):
    memory_svc.create_manual_memory(db, "likes unsweetened Americano", memory_type="preference", subject="Alex")
    memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion", subject="Alex")
    questions = example_questions(db)
    assert any("dislike" in q for q in questions)
    assert any("like" in q for q in questions)
    assert len(questions) <= 4


def _extract_upload_path(html: str) -> str:
    marker = 'name="upload_path" value="'
    start = html.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    return html[start : html.find('"', start)]


def test_chat_page_uses_generated_suggestions(client: TestClient, db: Session):
    memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion", subject="Alex")
    home = client.get("/")
    assert "What does Alex dislike?" in home.text
