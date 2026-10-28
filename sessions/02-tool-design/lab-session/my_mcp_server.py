"""Session 2 lab: the music MCP server. This is the ONE file you edit. Start it with `uv run my_mcp_server.py`.

Do not edit the block between the two ===== lines (the simulated service). Everything below it is yours.
It works on a good day, but: no validation, no idempotency, opaque errors. Harden it (see the TODOs).

With LAB_CHAOS=1 (set by the notebook) every 4th read raises a 429, and the 1st, 4th, 7th... write is EXECUTED and
then raises a timeout. The provider replays the first result when it sees the same idempotency_key twice.
"""

import itertools
import json
import os
import sys

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

# ===== simulated service (given, do not edit) =====================================================
TRACKS = {  # id: (title, artist, album, year, duration in seconds, tempo in BPM)
    "T001": ("Evenfall", "Lumen", "Evenfall", 2021, 214, 92),
    "T002": ("Glass Harbour", "Lumen", "Evenfall", 2021, 187, 118),
    "T003": ("Night Bus", "Lumen", "Evenfall", 2021, 243, 76),
    "T004": ("Paper Moons", "Nova Quartet", "Small Hours", 2019, 301, 64),
    "T005": ("Slow Tide", "Nova Quartet", "Small Hours", 2019, 276, 70),
    "T006": ("Copper Sky", "Nova Quartet", "Small Hours", 2019, 198, 104),
    "T007": ("Sunday Light", "Maya Ortega", "Windows Open", 2023, 225, 88),
    "T008": ("Bicycle Bell", "Maya Ortega", "Windows Open", 2023, 172, 124),
    "T009": ("Long Road Home", "Maya Ortega", "Windows Open", 2023, 262, 96),
    "T010": ("Static Garden", "The Fallow", "Rooms", 2018, 208, 132),
    "T011": ("Blue Hour", "The Fallow", "Rooms", 2018, 233, 110),
    "T012": ("Last Train", "The Fallow", "Rooms", 2018, 250, 84),
}
DEMO_ARGS = {"playlist": "Focus", "track_id": "T003"}


class BackendError(Exception):
    """The request was understood but cannot be done (unknown id, unknown playlist...)."""


class RateLimited(BackendError):
    """HTTP 429: too many requests. Waiting and repeating the same read is safe."""


class MusicService:
    def __init__(self, chaos: bool = False, ledger_path: str | None = None):
        self.chaos, self.ledger_path = chaos, ledger_path
        self.playlists = {"Road trip": [], "Focus": [], "Sunday morning": []}
        self.effects: list[dict] = []          # every real change, in order
        self.replay: dict[str, dict] = {}      # idempotency_key -> first result
        self.reads = self.writes = 0
        self._reads, self._writes = itertools.count(1), itertools.count(1)   # atomic: calls may run in parallel

    def _read(self) -> None:
        self.reads = n = next(self._reads)
        if self.chaos and n % 4 == 0:
            raise RateLimited("429 Too Many Requests: slow down")

    def search_tracks(self, query: str) -> list[dict]:
        self._read()
        q = query.lower()
        return [{"id": i, "title": t[0], "artist": t[1], "album": t[2], "year": t[3]}
                for i, t in TRACKS.items() if q in " ".join(t[:3]).lower()]

    def get_track(self, track_id: str) -> dict:
        self._read()
        if track_id not in TRACKS:
            raise BackendError(f"Unknown track id '{track_id}'")
        t = TRACKS[track_id]
        return {"id": track_id, "title": t[0], "artist": t[1], "album": t[2], "year": t[3],
                "duration_s": t[4], "bpm": t[5]}

    def add_to_playlist(self, playlist: str, track_id: str, idempotency_key: str | None = None) -> dict:
        self.writes = n = next(self._writes)
        if idempotency_key in self.replay:
            result = {**self.replay[idempotency_key], "replayed": True}
        else:
            if track_id not in TRACKS:
                raise BackendError(f"Unknown track id '{track_id}'")
            result = {"id": f"E{len(self.effects) + 1:03d}", "playlist": playlist, "track_id": track_id,
                      "title": TRACKS[track_id][0], "replayed": False}
            self.playlists.setdefault(playlist, []).append(track_id)
            self.effects.append(result)
            if self.ledger_path:
                with open(self.ledger_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
            if idempotency_key:
                self.replay[idempotency_key] = result
        if self.chaos and n % 3 == 1:        # the change happened, the answer is lost
            raise TimeoutError("playlist service timed out")
        return result


def tool_error(code: str, message: str, retryable: bool = False, outcome_unknown: bool = False) -> ToolError:
    """A typed error: `message` is for the model, `retryable` and `outcome_unknown` are for the harness."""
    return ToolError(json.dumps({"code": code, "message": message,
                                 "retryable": retryable, "outcome_unknown": outcome_unknown}))
# ===== end of the simulated service ===============================================================

svc = MusicService(chaos=os.getenv("LAB_CHAOS") == "1", ledger_path=os.getenv("LAB_LEDGER"))
mcp = MCPServer("music-server")


# TODO 1: annotations. Which tools are read-only? Which one is safe to repeat, and what makes it safe?
# TODO 2: constrain the arguments: Annotated[str, Field(pattern=..., description=...)], Literal[...] for playlists.
# TODO 3: typed errors: catch RateLimited and TimeoutError, then raise tool_error(code, message, retryable=...,
#         outcome_unknown=...). The message is for the model, `retryable` is for the harness.
# TODO 4: add `idempotency_key: str | None = None` to add_to_playlist and pass it to svc.add_to_playlist.
#         The model never fills it in: the harness does.
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
