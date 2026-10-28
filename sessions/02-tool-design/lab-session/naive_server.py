"""Session 2 lab, given code: the NAIVE music server (v0.1). Do not edit: you harden `my_mcp_server.py` instead.

It works on a good day, but: no validation, no idempotency, opaque errors.
"""

import os
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from music_service import BackendError, MusicService

svc = MusicService(chaos=os.getenv("LAB_CHAOS") == "1", ledger_path=os.getenv("LAB_LEDGER"))
mcp = MCPServer("music-server")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def search_tracks(query: str) -> list[dict]:
    """Search the music catalogue by title, artist or album (case-insensitive substring).
    Use it to find a track id or to list an artist's or an album's tracks. Returns id, title, artist, album, year."""
    try:
        return svc.search_tracks(query)
    except BackendError as e:
        raise ToolError(str(e))


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def get_track(track_id: str) -> dict:
    """Full record of one track: duration in seconds and tempo in BPM, on top of the search fields.
    `track_id` comes from search_tracks, e.g. 'T001'."""
    try:
        return svc.get_track(track_id)
    except BackendError as e:
        raise ToolError(str(e))


@mcp.tool()
def add_to_playlist(playlist: str, track_id: str) -> dict:
    """Append a track to one of the user's playlists ('Road trip', 'Focus', 'Sunday morning').
    This changes the playlist: call it once the user has clearly asked for it."""
    try:
        return svc.add_to_playlist(playlist, track_id)
    except BackendError as e:
        raise ToolError(str(e))


if __name__ == "__main__":
    if sys.stdin.isatty():
        print("MCP server on stdio, waiting for a client. Open 02_agent_in_the_loop.ipynb to test it (Ctrl+C to stop).",
              file=sys.stderr)
    mcp.run()
