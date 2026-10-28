"""Session 2 lab: your hardened music MCP server. This is the ONE file you edit. Start it with `uv run my_mcp_server.py`.

Typed errors, the `idempotency_key` argument and the tool bodies are given. You fill in FIVE type annotations
(the lines marked TODO). Until you do, the server starts and works, but the tools are not constrained.
"""

import os
import sys
from typing import Annotated, Literal  # noqa: F401  (you will need both)

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations  # noqa: F401
from pydantic import Field  # noqa: F401

from music_service import BackendError, MusicService, RateLimited, tool_error

svc = MusicService(chaos=os.getenv("LAB_CHAOS") == "1", ledger_path=os.getenv("LAB_LEDGER"))
mcp = MCPServer("music-server")

# ---- TODO 1: annotations. Tell the client what each kind of tool does. -------------------------------------------
# ToolAnnotations(readOnlyHint=..., idempotentHint=..., destructiveHint=..., openWorldHint=...), each True or False.
# READ: the two search tools. They change nothing, and repeating them is harmless.
# WRITE: add_to_playlist. It changes the playlist. Is it safe to repeat? Only because of the idempotency key.
# Both talk to a closed catalogue, not to the open web.
READ = ToolAnnotations()
WRITE = ToolAnnotations()

# ---- TODO 2: constrained arguments. The schema the model sees is built from these types. -------------------------
# Track ids look like T001 (the letter T and three digits): Annotated[str, Field(pattern=r"...", description="...")]
TrackId = str
# A playlist is one of exactly three names: Literal["Road trip", "Focus", "Sunday morning"]
Playlist = str
# A search query is a non-empty string: Annotated[str, Field(min_length=1, description="...")]
Query = str


@mcp.tool(annotations=READ)
def search_tracks(query: Query) -> list[dict]:
    """Search the catalogue by title, artist or album (case-insensitive). Use it to find track ids and to list
    an artist's or album's tracks. Returns [] when nothing matches: try fewer keywords."""
    try:
        return svc.search_tracks(query)
    except RateLimited as e:
        raise tool_error("rate_limited", str(e), retryable=True)


@mcp.tool(annotations=READ)
def get_track(track_id: TrackId) -> dict:
    """Full record of one track, including duration in SECONDS and tempo in BPM (search_tracks does not return them)."""
    try:
        return svc.get_track(track_id)
    except RateLimited as e:
        raise tool_error("rate_limited", str(e), retryable=True)
    except BackendError as e:
        raise tool_error("not_found", str(e))


@mcp.tool(annotations=WRITE)
def add_to_playlist(
    playlist: Playlist,
    track_id: TrackId,
    idempotency_key: Annotated[str | None, Field(description="Set by the harness; never by the model")] = None,
) -> dict:
    """Append a track to a playlist. Call it once, when the user has clearly asked. If it answers with a timeout,
    the track MAY have been added: repeating the exact same call is safe, it never adds the track twice."""
    try:
        return svc.add_to_playlist(playlist, track_id, idempotency_key)
    except TimeoutError:
        raise tool_error("timeout", "The playlist service did not confirm. Repeating the same call is safe.",
                         retryable=True, outcome_unknown=True)
    except BackendError as e:
        raise tool_error("not_found", str(e))


if __name__ == "__main__":
    if sys.stdin.isatty():
        print("MCP server on stdio, waiting for a client. Open 02_agent_in_the_loop.ipynb to test it (Ctrl+C to stop).",
              file=sys.stderr)
    mcp.run()
