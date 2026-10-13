from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.services import memory as memory_svc


def test_health_and_bind(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["bind"].startswith("127.0.0.1")


def test_pages_render(client: TestClient):
    for path in (
        "/",
        "/imports",
        "/memories",
        "/memories/candidates",
        "/chat",
        "/search",
        "/settings",
        "/profile",
        "/graph",
        "/diary",
    ):
        resp = client.get(path)
        assert resp.status_code == 200, path
    home = client.get("/")
    assert "prompt-input" in home.text
    assert "Persona Lab" in home.text
    assert "Start new chat" in home.text
    assert "What can I help you with?" in home.text
    assert "Notes" in home.text
    assert "Quick questions" not in home.text
    assert "What details about her" not in home.text
    assert "Profile overview" in home.text
    assert "Profile graph" not in home.text  # demoted to an advanced view inside the profile page
    assert "Open graph" in client.get("/profile").text
    assert "Profile graph" in client.get("/graph").text
    settings = client.get("/settings")
    assert "DeepSeek model" in settings.text
    assert "V4 Flash" in settings.text
    assert "Advisor prompt" in settings.text


def test_chat_ask_renders_thread(client: TestClient, db: Session):
    from app.services import memory as memory_svc

    memory_svc.create_manual_memory(db, "likes coffee", memory_type="preference")
    # Kept under the 22-character thread-title clip so the sidebar shows it intact.
    resp = client.post(
        "/chat/ask",
        data={"question": "Does Alex like coffee?", "mode": "ask"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "likes coffee" in resp.text
    assert "msg user" in resp.text or "You" in resp.text
    assert "/chat/" in str(resp.url)

    first_url = str(resp.url)
    again = client.post(
        "/chat/ask",
        data={
            "question": "So what does she dislike?",
            "mode": "ask",
            "thread_id": first_url.rsplit("/", 1)[-1],
        },
        follow_redirects=True,
    )
    assert again.status_code == 200
    assert "Does Alex like coffee?" in again.text
    assert "So what does she dislike?" in again.text
    home = client.get("/")
    assert "Does Alex like coffee?" in home.text


def test_web_import_preview(client: TestClient, tmp_path):
    sample = tmp_path / "up.json"
    sample.write_text(
        '[{"sender":"Alex","content":"likes mildly spicy hotpot","timestamp":"2026-03-02 21:00:00","type":"text"}]',
        encoding="utf-8",
    )
    with sample.open("rb") as f:
        resp = client.post("/imports/preview", files={"file": ("up.json", f, "application/json")})
    assert resp.status_code == 200
    assert "Preview result" in resp.text


def test_manual_memory_via_form(client: TestClient):
    resp = client.post(
        "/memories/create",
        data={"content": "Alex dislikes horror movies", "memory_type": "auto", "subject": ""},
        follow_redirects=False,
    )
    assert resp.status_code in (302, 303)
    loc = resp.headers.get("location", "")
    assert "created=" in loc
    page = client.get(loc if loc.startswith("/") else "/memories?status=confirmed")
    assert "horror movies" in page.text
    assert "aversion" in page.text
    assert "Added:" in page.text


def test_memories_sort_by_id(client: TestClient, db: Session):
    memory_svc.create_manual_memory(db, "written first", memory_type="fact")
    memory_svc.create_manual_memory(db, "written later", memory_type="habit")
    desc = client.get("/memories?sort=id&dir=desc")
    assert desc.status_code == 200
    assert desc.text.find("written later") < desc.text.find("written first")
    asc = client.get("/memories?sort=id&dir=asc")
    assert asc.text.find("written first") < asc.text.find("written later")
    assert 'class="sort-link active"' in desc.text


def test_memories_pagination(client: TestClient, db: Session):
    for i in range(55):
        memory_svc.create_manual_memory(db, f"paged memory {i:02d}", memory_type="fact")
    first = client.get("/memories?status=confirmed&sort=id&dir=asc&page=1")
    assert first.status_code == 200
    assert "paged memory 00" in first.text
    assert "paged memory 54" not in first.text
    assert "page 1/2" in first.text
    second = client.get("/memories?status=confirmed&sort=id&dir=asc&page=2")
    assert "paged memory 54" in second.text
    assert "paged memory 00" not in second.text
    assert "Next" in first.text


def test_wipe_data(client: TestClient, db: Session):
    memory_svc.create_manual_memory(db, "temporary memory", memory_type="fact")
    resp = client.post("/settings/wipe", follow_redirects=False)
    assert resp.status_code in (303, 302)
    page = client.get("/memories?status=all")
    assert page.status_code == 200
    assert "temporary memory" not in page.text


def test_hermes_profile_api(client: TestClient, db: Session):
    memory_svc.create_manual_memory(db, "likes unsweetened Americano", memory_type="preference")
    resp = client.get("/api/hermes/profile")
    assert resp.status_code == 200
    data = resp.json()
    assert "profile_text" in data
    assert "Americano" in data["profile_text"]
