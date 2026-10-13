"""The MCP surface is read-only: confirmed memories out, nothing in."""

from __future__ import annotations

import io
import json

import pytest
from sqlalchemy.orm import Session

from app.db.models import Memory, MemoryStatus
from app.mcp.readonly import (
    MAX_LIMIT,
    PROFILE_TOOL,
    SEARCH_TOOL,
    TOOLS,
    McpError,
    handle_message,
)
from app.services import memory as memory_svc


def _confirmed(db: Session, content: str, memory_type: str = "preference") -> Memory:
    return memory_svc.create_manual_memory(db, content, memory_type=memory_type, subject="Alex")


def _call(db: Session, name: str, arguments: dict | None = None) -> str:
    response = handle_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments or {}},
        },
        db,
    )
    return response["result"]["content"][0]["text"]


def test_initialize_announces_a_read_only_server(db: Session):
    response = handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, db)
    result = response["result"]
    assert result["serverInfo"]["name"] == "persona-lab-readonly"
    assert "read-only" in result["instructions"].lower()
    assert "tools" in result["capabilities"]


def test_tool_list_has_no_write_tool(db: Session):
    response = handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, db)
    names = [tool["name"] for tool in response["result"]["tools"]]
    assert names == [SEARCH_TOOL, PROFILE_TOOL]
    for tool in TOOLS:
        assert tool["inputSchema"]["type"] == "object"
    forbidden = ("create", "update", "delete", "write", "confirm", "reject", "remove")
    assert not any(word in name.lower() for name in names for word in forbidden)


def test_search_returns_confirmed_memories_only(db: Session):
    _confirmed(db, "Alex likes unsweetened Americano")
    db.add(
        Memory(
            content="Alex seems to like cilantro",
            memory_type="preference",
            status=MemoryStatus.CANDIDATE.value,
        )
    )
    db.commit()

    text = _call(db, SEARCH_TOOL, {"query": "What does Alex like to drink?"})
    assert "unsweetened Americano" in text
    assert "cilantro" not in text


def test_search_reports_no_match_without_inventing_content(db: Session):
    text = _call(db, SEARCH_TOOL, {"query": "What does Alex dislike?"})
    assert "no confirmed memories" in text.lower()


def test_search_requires_a_query(db: Session):
    with pytest.raises(McpError) as exc:
        _call(db, SEARCH_TOOL, {})
    assert exc.value.code == -32602


def test_search_clamps_the_limit(db: Session):
    for index in range(MAX_LIMIT + 5):
        _confirmed(db, f"Alex likes shop number {index}")
    text = _call(db, SEARCH_TOOL, {"query": "Alex likes", "limit": 999})
    assert len(text.splitlines()) <= MAX_LIMIT


def test_profile_summary_uses_confirmed_memories(db: Session):
    _confirmed(db, "Alex dislikes cilantro", memory_type="aversion")
    text = _call(db, PROFILE_TOOL)
    assert "[aversion] Alex dislikes cilantro" in text
    assert "confirmed memories" in text


def test_profile_summary_handles_an_empty_profile(db: Session):
    assert "empty" in _call(db, PROFILE_TOOL).lower()


def test_unknown_tool_is_rejected_as_read_only(db: Session):
    with pytest.raises(McpError) as exc:
        _call(db, "memory_create", {"content": "write this in secret"})
    assert exc.value.code == -32602
    assert "read-only" in exc.value.message.lower()


def test_unknown_method_is_rejected(db: Session):
    with pytest.raises(McpError) as exc:
        handle_message({"jsonrpc": "2.0", "id": 3, "method": "resources/list"}, db)
    assert exc.value.code == -32601


def test_notifications_get_no_response(db: Session):
    assert handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"}, db) is None


def test_calls_never_change_the_database(db: Session):
    _confirmed(db, "Alex likes mildly spicy hotpot")
    before = db.query(Memory).count()
    _call(db, SEARCH_TOOL, {"query": "hotpot"})
    _call(db, PROFILE_TOOL)
    db.expire_all()
    assert db.query(Memory).count() == before


def test_stdio_server_answers_and_survives_bad_input(db: Session):
    from scripts.mcp_server import serve

    _confirmed(db, "Alex likes unsweetened Americano")
    stdin = io.StringIO(
        "\n".join(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
                "{not json",
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": PROFILE_TOOL, "arguments": {}},
                    }
                ),
            ]
        )
        + "\n"
    )
    stdout = io.StringIO()
    serve(stdin, stdout)

    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["result"]["serverInfo"]["name"] == "persona-lab-readonly"
    assert responses[1]["error"]["code"] == -32700
    assert "unsweetened Americano" in responses[2]["result"]["content"][0]["text"]
