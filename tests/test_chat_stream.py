from fastapi.testclient import TestClient

from app.llm.client import deltas_from_chunk, iter_sse_deltas


def test_deltas_from_chunk_splits_thinking_and_answer():
    assert deltas_from_chunk(
        {"choices": [{"delta": {"reasoning_content": "Check memories first"}}]}
    ) == [("thinking", "Check memories first")]
    assert deltas_from_chunk(
        {"choices": [{"delta": {"content": "She likes Americano."}}]}
    ) == [("token", "She likes Americano.")]


def test_iter_sse_deltas_stops_at_done():
    lines = [
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"Think\"}}]}",
        "data: {\"choices\":[{\"delta\":{\"content\":\"Answer\"}}]}",
        "data: [DONE]",
        "data: {\"choices\":[{\"delta\":{\"content\":\"should not appear\"}}]}",
    ]
    assert list(iter_sse_deltas(lines)) == [("thinking", "Think"), ("token", "Answer")]


def test_chat_stream_emits_thinking_then_answer(client: TestClient, monkeypatch):
    def fake(_db, _q, history=None, **_kwargs):
        yield {"type": "status", "text": "Searching memories…"}
        yield {"type": "thinking", "text": "She loves Americano"}
        yield {"type": "token", "text": "She likes unsweetened Americano."}
        yield {
            "type": "done",
            "answer": "She likes unsweetened Americano.",
            "thinking": "She loves Americano",
            "can_answer": True,
            "evidence": [],
        }

    monkeypatch.setattr("app.routes.chat.iter_answer_events", fake)
    resp = client.post(
        "/chat/stream",
        data={"question": "What does she like to drink?", "mode": "ask"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    body = resp.text
    assert "event: meta" in body
    assert "event: thinking" in body
    assert "event: token" in body
    assert "event: done" in body
    assert "unsweetened Americano" in body
    assert "She loves Americano" in body

    thread_id = None
    for block in body.split("\n\n"):
        if block.startswith("event: meta"):
            import json

            data_line = [ln for ln in block.split("\n") if ln.startswith("data:")][0]
            thread_id = json.loads(data_line[5:])["thread_id"]
    assert thread_id
    page = client.get(f"/chat/{thread_id}")
    assert page.status_code == 200
    assert "Thinking" in page.text
    assert "She loves Americano" in page.text
    assert "unsweetened Americano" in page.text
