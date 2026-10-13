"""Read-only MCP surface for the knowledge base.

Only two tools exist, both reads: searching confirmed memories and fetching the
curated profile summary. There is deliberately no write tool, so an external
agent cannot confirm, edit or delete anything — the user stays the only author.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from app.branding import PRODUCT_SLUG
from app.integrations.hermes import export_profile_for_hermes
from app.services.retrieval import hybrid_search

PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = f"{PRODUCT_SLUG}-readonly"
MAX_LIMIT = 20
SEARCH_TOOL = "persona_search"
PROFILE_TOOL = "persona_profile"

TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": SEARCH_TOOL,
        "description": (
            "Search confirmed memories (read-only). Only human-confirmed entries are "
            "returned; candidates are excluded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords or a question cue"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_LIMIT,
                    "default": 8,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": PROFILE_TOOL,
        "description": (
            "Return a concise profile summary built from confirmed memories (read-only)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
)

TOOL_NAMES = frozenset(tool["name"] for tool in TOOLS)


class McpError(RuntimeError):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def handle_message(message: dict[str, Any], db: Session) -> dict[str, Any] | None:
    """Return a JSON-RPC response, or ``None`` for notifications."""
    request_id = message.get("id")
    method = message.get("method")

    if method == "initialize":
        return _result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": "1.0.0"},
                "instructions": (
                    "Read-only: you may search confirmed memories and read the profile "
                    "summary, but you cannot write or modify any data."
                ),
            },
        )
    if method == "tools/list":
        return _result(request_id, {"tools": [dict(tool) for tool in TOOLS]})
    if method == "tools/call":
        params = message.get("params") or {}
        name = str(params.get("name") or "")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise McpError(-32602, "arguments must be an object")
        return _result(
            request_id,
            {
                "content": [{"type": "text", "text": call_tool(name, arguments, db)}],
                "isError": False,
            },
        )
    if isinstance(method, str) and method.startswith("notifications/"):
        return None
    raise McpError(-32601, f"Unsupported method: {method}")


def error_response(request_id: Any, error: McpError) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": error.code, "message": error.message},
    }


def call_tool(name: str, arguments: dict[str, Any], db: Session) -> str:
    if name not in TOOL_NAMES:
        raise McpError(
            -32602,
            f"Unknown tool: {name}. This server is read-only and only exposes "
            f"{', '.join(sorted(TOOL_NAMES))}",
        )
    if name == SEARCH_TOOL:
        return _search(arguments, db)
    return _profile(db)


def _search(arguments: dict[str, Any], db: Session) -> str:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise McpError(-32602, "query must not be empty")
    try:
        limit = int(arguments.get("limit") or 8)
    except (TypeError, ValueError):
        raise McpError(-32602, "limit must be an integer") from None
    limit = max(1, min(MAX_LIMIT, limit))

    hits = hybrid_search(db, query, include_messages=False, limit=limit)
    if not hits:
        return "No confirmed memories match this query."
    lines = [
        f"- [{hit.meta.get('memory_type') or 'fact'}] {hit.content} "
        f"(memory #{hit.id}, confidence {hit.meta.get('confidence', '—')})"
        for hit in hits
    ]
    return "\n".join(lines)


def _profile(db: Session) -> str:
    export = export_profile_for_hermes(db)
    text = str(export.get("profile_text") or "").strip()
    if not text:
        return "No memories have been confirmed yet; the profile is empty."
    count = len(export.get("memory_ids") or [])
    return f"{text}\n\n({count} confirmed memories)"


def _result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}
