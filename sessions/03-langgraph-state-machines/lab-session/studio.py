"""LangGraph Studio demo: the guarded agent and the unguarded one, in the browser.

Do the notebook first (sections 2 and 3). This file holds a finished router and approval step
(the same logic as the notebook's blanks), so open it after you have written your own.

Run from this folder:    uv run langgraph dev
Then open the Studio URL printed in the terminal (Chrome, Firefox or Edge).
"""
import json

import lab
from langgraph.graph import END
from langgraph.types import Command, interrupt

NEEDS_APPROVAL = {"delete_playlist"}


def router(state):
    if state["action"]["type"] == "final":
        return END
    if state["steps"] >= state["max_steps"]:
        return "give_up"
    if state["action"]["tool"] in NEEDS_APPROVAL:
        return "approve"
    return "run_tool"


def as_decision(value) -> dict:
    """Studio's resume box may send text: accept {"approve": true}, true, yes, no, or any other text (a rejection)."""
    if isinstance(value, dict):
        return value
    text = str(value).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        text = str(parsed)
    except ValueError:
        pass
    return {"approve": text.lower() in {"true", "yes", "y", "ok", "approve"}, "reason": text}


def approve(state):
    action = state["action"]
    decision = as_decision(interrupt({"tool": action["tool"], "args": action["args"]}))
    if decision.get("approve"):
        return Command(goto="run_tool")
    rejected = {"tool": action["tool"], "args": action["args"],
                "result": "rejected by the human: " + decision.get("reason", "")}
    return Command(goto="think", update={"observations": [rejected]})


agent = lab.build_graph(router, approve)
agent_unguarded = lab.build_graph(lab.unguarded_router)
