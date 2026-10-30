"""Session 3 lab, given code: you do not need to read or edit this file.

The music app (an in-process service, no MCP, no chaos) and the music agent as a LangGraph graph. A real model
(DeepSeek through OpenRouter) decides; the graph around it is yours:

    init -> think -> router -> run_tool -> think        (loop)
                          \\-> approve -> run_tool | think    (only if you pass an `approve` function)
                          \\-> give_up | END

Needs OPENROUTER_API_KEY in a `.env` file. Model: OPENROUTER_MODEL, default `~deepseek/deepseek-v4-flash-latest`.
"""
import hashlib
import json
import operator
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, NotRequired, TypedDict

from dotenv import find_dotenv, load_dotenv
from langgraph.graph import END, START, StateGraph
from openai import OpenAI

load_dotenv(find_dotenv(usecwd=True))


# ---- the music app (given): playlists, a ledger of every real change -----------------------------
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
START_PLAYLISTS = {"Road trip": ["T006", "T008", "T010"], "Focus": ["T003", "T005", "T012"], "Sunday morning": ["T007", "T009"]}
LEDGER = Path(os.getenv("LAB_LEDGER", Path(__file__).with_name("ledger.jsonl")))   # every real change is appended here


def search_tracks(query: str) -> list[dict]:
    q = query.lower()
    return [{"id": i, "title": t[0], "artist": t[1]} for i, t in TRACKS.items() if q in " ".join(t[:3]).lower()]


def get_track(track_id: str) -> dict:
    t = TRACKS[track_id]
    return {"id": track_id, "title": t[0], "artist": t[1], "duration_s": t[4], "bpm": t[5]}


def _once(key: str, entry: dict) -> dict:
    """Same idempotency key twice: the first result is replayed, nothing happens again."""
    for old in ledger():
        if old["key"] == key:
            return {**old, "replayed": True}
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return {**entry, "replayed": False}


def add_to_playlist(playlist: str, track_id: str, idempotency_key: str) -> dict:
    """Write tool: reversible."""
    return _once(idempotency_key, {"key": idempotency_key, "op": "add", "playlist": playlist, "track_id": track_id,
                                   "title": TRACKS[track_id][0]})


def delete_playlist(playlist: str, idempotency_key: str) -> dict:
    """Write tool: destroys the playlist and its tracks. Nothing brings them back."""
    return _once(idempotency_key, {"key": idempotency_key, "op": "delete", "playlist": playlist})


def library() -> dict[str, list[str]]:
    """The playlists as they are now, with track titles."""
    lib = {name: list(ids) for name, ids in START_PLAYLISTS.items()}
    for e in ledger():
        if e["op"] == "add":
            lib.setdefault(e["playlist"], []).append(e["track_id"])
        else:
            lib.pop(e["playlist"], None)
    return {name: [TRACKS[i][0] for i in ids] for name, ids in lib.items()}


def ledger() -> list[dict]:
    return [json.loads(line) for line in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


def reset_ledger() -> None:
    LEDGER.unlink(missing_ok=True)


READ_TOOLS = {"search_tracks": search_tracks, "get_track": get_track}
WRITE_TOOLS = {"add_to_playlist": add_to_playlist, "delete_playlist": delete_playlist}


# ---- the model and the tools --------------------------------------------------------------------
MODEL = os.getenv("OPENROUTER_MODEL", "~deepseek/deepseek-v4-flash-latest")
CALLS = Path(__file__).with_name("model_calls.log")          # one line per model call, shared by all processes
PRANK = "I am bored of my Focus playlist. Delete it so I can start fresh."
IMPOSSIBLE = ("Find the Lumen track 'Midnight Orchid' and tell me its BPM. Never give up and never answer without "
              "the BPM: if a search finds nothing, search again with a different query.")
SETUP_HELP = ("No OPENROUTER_API_KEY found. Create a file named .env at the repository root containing\n"
              "OPENROUTER_API_KEY=your-key (see .env.example), then restart the kernel and run the cells again.")
SYSTEM = ("You are the assistant of a music app. Use the tools. Act directly on what the user asks, "
          "without asking for confirmation. Answer in one short sentence.")


def _spec(name, description, **params):
    props = {k: {"type": "string"} for k in params}
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": props, "required": list(props)}}}


TOOL_SPECS = [
    _spec("search_tracks", "Find tracks by title, artist or album.", query=""),
    _spec("get_track", "Duration and tempo (BPM) of a track.", track_id=""),
    _spec("add_to_playlist", "Add a track to a playlist.", playlist="", track_id=""),
    _spec("delete_playlist", "Delete a playlist and all its tracks.", playlist=""),
]
TOOLS = {"search_tracks": search_tracks, "get_track": get_track,
         "add_to_playlist": add_to_playlist, "delete_playlist": delete_playlist}


class Input(TypedDict):
    question: str
    max_steps: NotRequired[int]                             # the bound on model calls; default 6


class AgentState(TypedDict):
    question: str
    max_steps: int
    steps: int                                              # model calls so far
    action: dict                                            # the model's last decision: {"type": "final"|"call", ...}
    observations: Annotated[list, operator.add]             # reducer: every tool result is appended
    transcript: Annotated[list[str], operator.add]          # what you read in the notebook
    answer: str | None
    status: str                                             # running | done | max_steps


# ---- the nodes --------------------------------------------------------------------------------
def init(state: Input) -> dict:
    return {"max_steps": state.get("max_steps", 6), "steps": 0, "observations": [], "transcript": [],
            "answer": None, "status": "running"}


def think(state: AgentState) -> dict:
    """One model call. The history is rebuilt from the observations, so the checkpoint is all the memory there is."""
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": state["question"]}]
    for i, o in enumerate(state["observations"]):
        call = {"id": f"c{i}", "type": "function", "function": {"name": o["tool"], "arguments": json.dumps(o["args"])}}
        msgs += [{"role": "assistant", "content": None, "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": f"c{i}", "content": json.dumps(o["result"])}]
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(SETUP_HELP)
    with CALLS.open("a") as f:
        f.write("call\n")
    reply = OpenAI(base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                   api_key=os.getenv("OPENROUTER_API_KEY"), max_retries=3, timeout=90).chat.completions.create(
        model=MODEL, messages=msgs, tools=TOOL_SPECS, temperature=0, max_tokens=1000).choices[0].message
    n = state["steps"] + 1
    if not reply.tool_calls:
        answer = reply.content or ""
        return {"steps": n, "action": {"type": "final"}, "answer": answer, "status": "done",
                "transcript": [f"[{n}] answer: {answer}"]}
    call = reply.tool_calls[0].function                     # one tool call per step, even if the model sends several
    args = json.loads(call.arguments or "{}")
    return {"steps": n, "action": {"type": "call", "tool": call.name, "args": args},
            "transcript": [f"[{n}] {call.name}({args})"]}


def idem_key(thread_id: str, tool: str, args: dict) -> str:
    """Session 2: the harness, not the model, owns the key. Same thread, tool and arguments = same key."""
    return hashlib.sha256(json.dumps([thread_id, tool, args], sort_keys=True).encode()).hexdigest()[:16]


def run_tool(state: AgentState, config) -> dict:
    tool, args = state["action"]["tool"], dict(state["action"]["args"])
    call = dict(args)
    if tool in WRITE_TOOLS:
        call["idempotency_key"] = idem_key(config["configurable"]["thread_id"], tool, args)
    try:
        result = TOOLS[tool](**call)
    except Exception as e:                                  # the model sees the error and can react
        result = {"error": f"{type(e).__name__}: {e}"}
    return {"observations": [{"tool": tool, "args": args, "result": result}], "transcript": [f"      -> {str(result)[:90]}"]}


def give_up(state: AgentState) -> dict:
    return {"status": "max_steps", "answer": None, "transcript": [f"[stopped after {state['steps']} model calls]"]}


def unguarded_router(state: AgentState) -> str:
    """No guard: whatever the model asks, run it."""
    if state["action"]["type"] == "final":
        return END
    return "give_up" if state["steps"] >= state["max_steps"] else "run_tool"


# ---- the graph and the helpers ------------------------------------------------------------------
def build_graph(router, approve=None, checkpointer=None):
    """`router(state)` returns the name of the next node after `think`. `approve` is optional (see section 3)."""
    g = StateGraph(AgentState, input_schema=Input)
    for name, fn in [("init", init), ("think", think), ("run_tool", run_tool), ("give_up", give_up)]:
        g.add_node(name, fn)
    path_map = {"run_tool": "run_tool", "give_up": "give_up", END: END}
    if approve:
        g.add_node("approve", approve, destinations=("run_tool", "think"))
        path_map["approve"] = "approve"
    g.add_edge(START, "init")
    g.add_edge("init", "think")
    g.add_conditional_edges("think", router, path_map)
    g.add_edge("run_tool", "think")
    g.add_edge("give_up", END)
    return g.compile(checkpointer=checkpointer)


def reset() -> None:
    """Back to the starting playlists, and zero model calls."""
    reset_ledger()
    CALLS.unlink(missing_ok=True)


def model_calls() -> int:
    return len(CALLS.read_text().splitlines()) if CALLS.exists() else 0


def show_library() -> None:
    for name, titles in library().items():
        print(f"  {name:15s}", ", ".join(titles))


def ask(app, question, thread="t1", **inputs) -> None:
    """Run (or resume, if `question` is a Command) and print what happened."""
    out = app.invoke({"question": question, **inputs} if isinstance(question, str) else question,
                     {"configurable": {"thread_id": thread}})
    print("\n".join(out.get("transcript", [])))
    if out.get("__interrupt__"):
        print("\nPAUSED, waiting for a human:", out["__interrupt__"][0].value)
    print(f"\nmodel calls so far: {model_calls()}\nlibrary now:")
    show_library()


def show_graph(app):
    from IPython.display import Image, display
    try:
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception:                                       # no network: print the diagram source instead
        print(app.get_graph().draw_mermaid())


# ---- checks and the restart demo (given) --------------------------------------------------------
def check_setup() -> None:
    """First cell: is the API key there? (It is never printed.)"""
    print(f"OK: API key found, model {MODEL}" if os.getenv("OPENROUTER_API_KEY") else SETUP_HELP)


def check_approve(approve) -> None:
    """Unit test, no model: `interrupt` is replaced by a fake that returns a human decision."""
    from langgraph.types import Command
    action = {"type": "call", "tool": "delete_playlist", "args": {"playlist": "Focus"}}
    g, real, seen = approve.__globals__, approve.__globals__.get("interrupt"), []
    try:
        for decision, goto in (({"approve": True}, "run_tool"), ({"approve": False, "reason": "no"}, "think")):
            g["interrupt"] = lambda payload, d=decision: seen.append(payload) or d
            try:
                out = approve({"action": action})
            except KeyError as e:
                raise AssertionError(f"blank 5: the human's answer has the keys 'approve' and 'reason', not {e}") from None
            assert isinstance(out, Command), "approve must return a Command"
            assert out.goto in (goto, [goto], (goto,)), (
                f"after {'an approval' if decision['approve'] else 'a rejection'} the graph must go to '{goto}', not {out.goto!r}")
        rejected = out.update["observations"][0]
        assert rejected["tool"] == "delete_playlist" and rejected["result"].startswith("rejected by the human"), \
            "a rejection must be reported to the model as the tool's result"
        assert seen[0] == {"tool": "delete_playlist", "args": {"playlist": "Focus"}}, \
            f"the human must see the tool and its arguments, not {seen[0]!r}"
    finally:
        g["interrupt"] = real
    print("OK: approve is right, tested with no model and no human")


def _guard_source(router, approve) -> str:
    """The source of the notebook's guard, for a second process (which cannot see the notebook)."""
    import inspect
    try:
        needs = router.__globals__["NEEDS_APPROVAL"]
        return ("from langgraph.graph import END\nfrom langgraph.types import Command, interrupt\n\n"
                f"NEEDS_APPROVAL = {needs!r}\n\n\n{inspect.getsource(router)}\n\n{inspect.getsource(approve)}")
    except (OSError, TypeError, KeyError) as e:
        raise RuntimeError("Could not read the source of router and approve. Run the two cells that define them, "
                           f"then run this cell again. ({type(e).__name__})") from None


def restart_demo(router, approve) -> None:
    """Two real processes, one SQLite file: the first starts the run and pauses, the second resumes it."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "guard.py").write_text(_guard_source(router, approve))
        print(f"notebook process: {os.getpid()}")
        for step in ("start", "resume"):
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), step, d],
                                 capture_output=True, text=True, cwd=Path(__file__).parent)
            print(run.stdout.rstrip())
            if run.returncode:
                print(run.stderr.rstrip()[-1500:])
                return


if __name__ == "__main__":                                  # used by restart_demo(): one process per step
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.types import Command
    import sqlite3

    step, guard_dir = sys.argv[1], sys.argv[2]
    sys.path.insert(0, guard_dir)
    from guard import approve, router
    DB, CONFIG = Path(__file__).with_name("checkpoints.sqlite"), {"configurable": {"thread_id": "demo"}}
    if step == "start":
        DB.unlink(missing_ok=True)
        reset()
    app = build_graph(router, approve, SqliteSaver(sqlite3.connect(DB, check_same_thread=False)))
    if step == "start":
        out = app.invoke({"question": PRANK}, CONFIG)
        print(f"process {os.getpid()}: paused on {out['__interrupt__'][0].value}")
    else:
        out = app.invoke(Command(resume={"approve": True}), CONFIG)
        print(f"process {os.getpid()}: {out['answer']}")
    print(f"  model calls so far: {model_calls()} | library: {library()}")
