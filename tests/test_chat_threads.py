from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.services import memory as memory_svc


def test_thread_persists_across_pages(client: TestClient, db: Session):
    memory_svc.create_manual_memory(db, "dislikes cilantro", memory_type="aversion")
    # Kept under the 22-character thread-title clip so the title is shown intact.
    created = client.post(
        "/chat/ask",
        data={"question": "Does Alex drink tea?", "mode": "ask"},
        follow_redirects=False,
    )
    assert created.status_code in (302, 303)
    loc = created.headers["location"]
    assert loc.startswith("/chat/")
    profile = client.get("/profile")
    assert profile.status_code == 200
    assert "Does Alex drink tea?" in profile.text
    opened = client.get(loc)
    assert opened.status_code == 200
    assert "Does Alex drink tea?" in opened.text


def test_delete_thread(client: TestClient):
    created = client.post(
        "/chat/ask",
        data={"question": "Where do they go?", "mode": "ask"},
        follow_redirects=False,
    )
    loc = created.headers["location"]
    thread_id = loc.rsplit("/", 1)[-1]
    deleted = client.post(f"/chat/{thread_id}/delete", follow_redirects=False)
    assert deleted.status_code in (302, 303)
    assert deleted.headers["location"] == "/"
    assert client.get(loc).status_code == 404
    home = client.get("/")
    assert "Where do they go?" not in home.text


def test_rename_thread(client: TestClient):
    created = client.post(
        "/chat/ask",
        data={"question": "What do they do?", "mode": "ask"},
        follow_redirects=False,
    )
    thread_id = created.headers["location"].rsplit("/", 1)[-1]
    renamed = client.post(
        f"/chat/{thread_id}/rename",
        data={"title": "Weekend habits"},
        headers={"Accept": "application/json"},
    )
    assert renamed.status_code == 200
    assert renamed.json()["title"] == "Weekend habits"
    page = client.get(f"/chat/{thread_id}")
    assert "Weekend habits" in page.text
