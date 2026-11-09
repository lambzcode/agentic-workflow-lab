"""LangGraph Studio: the supervisor team and the handoff team in the browser.

Do the notebook first (section 3): this file holds a finished version of the cell you wrote.

Run from this folder:    uv run langgraph dev
Then open the Studio URL printed in the terminal, pick a graph and submit {"question": "..."}.
Open a new thread for each question: a thread keeps its state, so the transcript and the hops would carry over.
"""
import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

import lab

MAX_HOPS = 6


class TeamState(TypedDict):
    question: str
    transcript: Annotated[list[str], operator.add]
    hops: int


def supervisor(state: TeamState) -> Command:
    choice = lab.ask_router(state)
    hops = state.get("hops", 0)                              # Studio starts with the question only
    if choice == "FINISH" or hops >= MAX_HOPS:
        return Command(goto=END)
    return Command(goto=choice, update={"hops": hops + 1})


def make_worker(name, brief, tools):
    def worker(state: TeamState) -> Command:
        text = lab.agent_loop(brief, lab.view(state), tools, name)
        return Command(goto="supervisor", update={"transcript": [f"[{name}] {text}"]})
    return worker


WORKERS = {
    "researcher": ("Find the figures with the search tool. Quote them with their document id. Do not compute.", ["search"]),
    "analyst": ("Compute what the question asks from the figures found so far, with the calculator. Reply with the result only.", ["calculator"]),
    "writer": ("Write the final answer in one sentence from the work so far. Cite the document id of each figure in square brackets.", []),
}

builder = StateGraph(TeamState)
builder.add_node("supervisor", supervisor, destinations=(*WORKERS, END))
for name, (brief, tools) in WORKERS.items():
    builder.add_node(name, make_worker(name, brief, tools), destinations=("supervisor",))
builder.add_edge(START, "supervisor")

supervisor_team = builder.compile()
handoffs = lab.build_handoffs(WORKERS, MAX_HOPS)
