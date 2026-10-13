from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from app.config import ROOT_DIR
from app.importers.weflow import looks_like_weflow
from app.services.import_service import commit_import, preview_import

WEFLOW_SAMPLE = ROOT_DIR / "samples" / "weflow_sample.json"


def test_detects_weflow_shape():
    assert looks_like_weflow(WEFLOW_SAMPLE)


def test_generic_object_is_not_weflow(tmp_path):
    other = tmp_path / "plain.json"
    other.write_text(
        '{"messages": [{"sender": "Alex", "content": "hi"}]}', encoding="utf-8"
    )
    assert not looks_like_weflow(other)


def test_preview_routes_weflow_and_skips_non_text():
    preview = preview_import(WEFLOW_SAMPLE)
    assert preview["source"] == "weflow_json"
    assert preview["format_name"] == "weflow.session"
    # 8 messages in the file, one image is dropped.
    assert preview["total_rows"] == 7
    assert preview["senders"] == ["Alex", "Me"]
    assert not preview["errors"]


def test_commit_resolves_roles_from_is_send(db: Session):
    job = commit_import(
        db,
        WEFLOW_SAMPLE,
        subject_name="Alex",
        partner_name="Sam",
    )
    assert job.source_format == "weflow_json"
    assert job.message_count == 7

    from app.db.models import Message

    roles = {m.sender: m.sender_role for m in db.query(Message).all()}
    assert roles["Alex"] == "subject"
    assert roles["Sam"] == "partner"


def test_subject_defaults_to_session_display_name(db: Session):
    job = commit_import(db, WEFLOW_SAMPLE)
    # The adapter keeps one label per side even without explicit names.
    assert job.subject_name == "Alex"


def test_weflow_file_needs_session_and_messages(tmp_path):
    broken = tmp_path / "broken.json"
    broken.write_text('{"session": {}, "messages": {}}', encoding="utf-8")
    with pytest.raises(ValueError):
        preview_import(broken, source="weflow_json")


def test_sample_endpoint_previews_weflow(client):
    resp = client.post("/imports/sample", data={"sample": "weflow"})
    assert resp.status_code == 200
    assert "weflow.session" in resp.text
    assert "Preview result" in resp.text
    # Roles are pre-filled from the two distinct senders.
    assert 'name="subject_name" value="Alex"' in resp.text
    assert 'name="partner_name" value="Me"' in resp.text


def test_sample_endpoint_previews_generic_json(client):
    resp = client.post("/imports/sample", data={"sample": "wechat"})
    assert resp.status_code == 200
    assert "Preview result" in resp.text
    assert 'name="subject_name" value="Alex"' in resp.text
