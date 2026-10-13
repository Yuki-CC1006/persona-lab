"""Acceptance test: the core loop works with no API key and no network.

Import, candidate review and retrieval are the parts a user cannot live
without. If any of them silently needs the model, the product is unusable
offline — so this test blocks outbound HTTP entirely.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.config import get_settings
from app.services import memory as memory_svc
from app.services.retrieval import hybrid_search


@pytest.fixture()
def no_network(monkeypatch):
    """Any outbound HTTP call raises, proving the flow stayed local."""

    def _blocked(*_args, **_kwargs):
        raise AssertionError("The offline flow must not make network requests")

    monkeypatch.setattr(httpx, "Client", _blocked)
    monkeypatch.setattr(httpx, "post", _blocked)
    monkeypatch.setattr(httpx, "get", _blocked)


def test_import_review_search_work_without_api_key(client: TestClient, no_network):
    assert get_settings().llm_api_key == ""

    # 1. Import a pasted chat log.
    preview = client.post(
        "/imports/preview",
        data={
            "text": "Alex: dislikes cilantro\nAlex: likes unsweetened Americano\nMe: noted",
            "source": "text_chat",
        },
    )
    assert preview.status_code == 200
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

    # 2. Review: a manually added memory is immediately usable, no model needed.
    review = client.post(
        "/memories/create",
        data={"content": "Alex won't eat cilantro", "memory_type": "aversion", "subject": "Alex"},
        follow_redirects=False,
    )
    assert review.status_code in (302, 303)

    # 3. The candidate queue renders even with nothing pending.
    assert client.get("/memories/candidates").status_code == 200


def test_search_stays_local_without_api_key(db: Session, no_network):
    memory_svc.create_manual_memory(db, "Alex dislikes cilantro", memory_type="aversion", subject="Alex")
    hits = hybrid_search(db, "What does Alex dislike?", include_messages=False, limit=5)
    assert any("cilantro" in hit.content for hit in hits)


def test_question_answers_offline_with_evidence(db: Session, no_network):
    from app.services.prediction import answer_question

    memory_svc.create_manual_memory(db, "Alex likes unsweetened Americano", memory_type="preference", subject="Alex")
    result = answer_question(db, "What does Alex like to drink?")
    assert result["can_answer"] is True
    assert any("Americano" in str(item.get("content")) for item in result["evidence"])


def _extract_upload_path(html: str) -> str:
    marker = 'name="upload_path" value="'
    start = html.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    return html[start : html.find('"', start)]
