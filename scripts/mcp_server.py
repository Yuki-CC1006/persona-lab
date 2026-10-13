"""Run the read-only MCP server over stdio.

Usage (register in any MCP client):

    uv run python scripts/mcp_server.py

It exposes confirmed memories and the curated profile summary, and nothing
else: every tool is a read, so an external agent cannot change your data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import IO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db.session import get_session_factory  # noqa: E402
from app.mcp.readonly import McpError, error_response, handle_message  # noqa: E402


def serve(stdin: IO[str], stdout: IO[str]) -> None:
    session = get_session_factory()()
    try:
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                stdout.write(
                    json.dumps(
                        error_response(None, McpError(-32700, "JSON parse error")),
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                stdout.flush()
                continue
            if not isinstance(message, dict):
                continue
            try:
                response = handle_message(message, session)
            except McpError as exc:
                response = error_response(message.get("id"), exc)
            except Exception as exc:  # noqa: BLE001 — never leak a traceback to the client
                response = error_response(
                    message.get("id"), McpError(-32603, f"Internal error: {type(exc).__name__}")
                )
            if response is not None:
                stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
                stdout.flush()
    finally:
        session.close()


def main() -> int:
    serve(sys.stdin, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
