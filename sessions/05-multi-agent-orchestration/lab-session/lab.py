"""Session 5 lab, given code: you do not need to read or edit this file.

Real country facts, four questions, and three ways to answer them: one agent, a supervisor team (which you build in
the notebook), and a handoff team. Every model call and every tool call is recorded, so each run can be drawn as a
timeline. A real model (DeepSeek through OpenRouter) does every step. Needs OPENROUTER_API_KEY in a `.env` file.
"""
import ast
import html
import json
import operator
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict, get_type_hints

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
from dotenv import find_dotenv, load_dotenv
from IPython.display import HTML, Image, display
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(find_dotenv(usecwd=True))
MODEL = os.getenv("OPENROUTER_MODEL", "~deepseek/deepseek-v4-flash-latest")
PRICE_IN = float(os.getenv("PRICE_IN", "0.05"))              # USD per million tokens (the default model's price)
PRICE_OUT = float(os.getenv("PRICE_OUT", "0.16"))
SETUP_HELP = ("No OPENROUTER_API_KEY found. Create a file named .env at the repository root containing\n"
              "OPENROUTER_API_KEY=your-key (see .env.example), then restart the kernel and run the cells again.")
___ = "___"                                                  # the blank marker of the notebook

# ---- the facts ---------------------------------------------------------------------------------
# Population on 1 January 2025 (Eurostat); total area in km² (UN Statistics Division, Demographic Yearbook).
FACTS = {
    "Belgium": (11_900_100, 30_528),
    "Netherlands": (18_044_000, 41_543),
    "Germany": (83_577_100, 357_581),
    "France": (68_635_900, 633_109),                         # area includes the overseas departments
    "Spain": (49_078_000, 506_009),
    "Portugal": (10_749_600, 92_225),
}
# Document ids carry no country name: a worker that never saw an id cannot guess it (section 6).
DOC_IDS = {"Belgium": "doc-417", "Netherlands": "doc-283", "Germany": "doc-152", "France": "doc-609",
           "Spain": "doc-538", "Portugal": "doc-371"}
DOCS = {DOC_IDS[name]: f"{name}. Population on 1 January 2025: {pop:,} (Eurostat). Area: {area:,} km² (UN Statistics "
                       f"Division)." for name, (pop, area) in FACTS.items()}
GUIDE = {"travel-blog": "Travel blog, 'Belgium in a weekend'. Belgium is one of the most crowded countries in Europe, "
                        "with about 470 inhabitants per km²."}          # deliberately wrong: used in the bonus only

QUESTIONS = [
    {"q": "What is the population density of Belgium, in people per km²?", "ok": r"389\.8|\b390\b",
     "src": ["doc-417"]},
    {"q": "How many times larger is Germany's area than Belgium's?", "ok": r"11\.7", "src": ["doc-152", "doc-417"]},
    {"q": "Which country is more densely populated, the Netherlands or Belgium? Give both densities.",
     # the conclusion and both densities must be right
     "ok": r"(?is)^(?=.*netherlands)(?=.*434)(?=.*(389\.8|\b390\b))"
           r"(?!.*belgium\W+is\s+(the\s+)?(more|denser))(?!.*netherlands\W+is\s+less)",
     "src": ["doc-283", "doc-417"]},
    {"q": "How many more people live in Spain than in Portugal?", "ok": r"38[ ,.]?3", "src": ["doc-538", "doc-371"]},
]
FIRST = QUESTIONS[0]["q"]


def show_facts() -> pd.DataFrame:
    return pd.DataFrame([{"country": n, "population (1 Jan 2025)": f"{p:,}", "area (km²)": f"{a:,}", "document id": DOC_IDS[n]}
                         for n, (p, a) in FACTS.items()])


# ---- the tools ---------------------------------------------------------------------------------
def search(query: str, docs: dict | None = None) -> str:
    """Keyword search over the documents. Returns the 2 best passages with their ids."""
    docs = docs or DOCS
    words = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = sorted(((len(words & set(re.findall(r"[a-z0-9]+", t.lower()))), k) for k, t in docs.items()),
                    reverse=True)
    return "\n".join(f"[{k}] {docs[k]}" for s, k in scored[:2] if s) or "No document found."


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression: str) -> str:
    """Arithmetic only: numbers, + - * / and parentheses."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only arithmetic is allowed")
    try:
        return str(round(ev(ast.parse(expression.replace(",", "").replace("_", ""), mode="eval")), 4))
    except Exception as e:
        return f"Error: {e}"


SCHEMAS = {
    "search": {"type": "function", "function": {
        "name": "search", "description": "Search the country documents. Returns up to 2 passages.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    "calculator": {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression, e.g. 11900100 / 30528.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
}


# ---- the recorder: every model call and tool call of the current run -----------------------------
@dataclass
class Meter:
    calls: int = 0
    prompt: int = 0
    completion: int = 0
    t0: float = field(default_factory=time.time)
    events: list = field(default_factory=list)               # (agent, kind, detail, prompt tokens, prompt text)

    @property
    def cost(self) -> float:
        return (self.prompt * PRICE_IN + self.completion * PRICE_OUT) / 1e6


RUN = {"meter": Meter(), "sees": "full", "docs": DOCS}


def _log(agent: str, kind: str, detail: str, tokens: int = 0, prompt: str = "") -> None:
    RUN["meter"].events.append((agent, kind, detail, tokens, prompt))


def _prompt_text(messages: list[dict], tools: list[dict] | None = None) -> str:
    """The prompt of one model call, as readable text."""
    parts = [f"[tools offered] {', '.join(t['function']['name'] for t in tools)}"] if tools else []
    for m in messages:
        calls = [f"calls {c['function']['name']}({c['function'].get('arguments', '')})" for c in m.get("tool_calls", [])]
        parts.append(f"[{m['role']}] " + "\n".join(filter(None, [m.get("content") or "", *calls])))
    return "\n\n".join(parts)


def _client() -> OpenAI:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(SETUP_HELP)
    return OpenAI(base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                  api_key=os.getenv("OPENROUTER_API_KEY"), max_retries=3, timeout=90)


def _count(agent: str, usage, what: str, prompt: str = ""):
    m = RUN["meter"]
    m.calls += 1
    m.prompt += usage.prompt_tokens
    m.completion += usage.completion_tokens
    _log(agent, "model", what, usage.prompt_tokens, prompt)


def chat(messages: list[dict], agent: str, tools: list[dict] | None = None):
    params = {"model": MODEL, "messages": messages, "temperature": 0, "max_tokens": 2000}
    if tools:
        params["tools"] = tools
    resp = _client().chat.completions.create(**params)
    if not resp.choices:
        raise RuntimeError(f"The provider returned no reply for the {agent} (empty choices). Run the cell again.")
    msg = resp.choices[0].message
    names = ", ".join(c.function.name for c in msg.tool_calls or [])
    _count(agent, resp.usage, f"asks for {names}" if names else "writes its reply", _prompt_text(messages, tools))
    return msg


def run_tool(name: str, args: dict, agent: str) -> str:
    if name == "search":
        out = search(args.get("query", ""), RUN["docs"])
        _log(agent, "tool", f'search("{args.get("query", "")}")')
    elif name == "calculator":
        out = calculator(args.get("expression", ""))
        _log(agent, "tool", f'calculator("{args.get("expression", "")}") = {out}')
    else:
        out = f"Error: unknown tool {name}"
    return out


def agent_loop(brief: str, task: str, tools: list[str], agent: str = "agent", max_steps: int = 6) -> str:
    """One agent: call the model, run the tools it asks for, repeat until it answers."""
    msgs = [{"role": "system", "content": brief}, {"role": "user", "content": task}]
    for _ in range(max_steps):
        m = chat(msgs, agent, tools=[SCHEMAS[n] for n in tools] or None)
        msgs.append(m.model_dump(exclude_none=True))
        if not m.tool_calls:
            return m.content or ""
        for c in m.tool_calls:
            try:
                out = run_tool(c.function.name, json.loads(c.function.arguments or "{}"), agent)
            except Exception as e:
                out = f"Error: {e}"
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": out})
    return "(step limit reached)"


# ---- the supervisor's two helpers (used by the code you write) ---------------------------------
ROUTER = ("You coordinate three workers: the researcher finds figures with a search tool, the analyst computes with "
          "a calculator, the writer writes the final answer. Read the question and the transcript, then choose who "
          "works next. Choose FINISH once the writer has answered.")


class Route(BaseModel):
    next: Literal["researcher", "analyst", "writer", "FINISH"]


def view(state: dict) -> str:
    """What a worker sees: the question and the transcript (or only its last entry, see section 6)."""
    transcript = state.get("transcript", [])
    if RUN["sees"] == "last":
        transcript = transcript[-1:]
    return f"Question: {state['question']}\n\nTranscript so far:\n" + ("\n".join(transcript) or "(empty)")


def ask_router(state: dict) -> str:
    """One model call: the supervisor reads the question and the transcript and names the next worker."""
    transcript = "\n".join(state.get("transcript", [])) or "(empty)"
    messages = [{"role": "system", "content": ROUTER},
                {"role": "user", "content": f"Question: {state['question']}\n\nTranscript so far:\n{transcript}"}]
    try:
        reply = _client().chat.completions.parse(model=MODEL, response_format=Route, temperature=0, max_tokens=2000,
                                                 messages=messages)
    except TypeError as e:                                   # the SDK fails this way when choices is null
        raise RuntimeError("The provider returned no reply for the supervisor (empty choices). Run the cell again.") from e
    parsed = reply.choices[0].message.parsed if reply.choices else None
    if parsed is None:
        raise RuntimeError("The supervisor's reply could not be read as a Route. Run the cell again.")
    _count("supervisor", reply.usage, f"routes to {parsed.next}", _prompt_text(messages))
    return parsed.next


# ---- runs ---------------------------------------------------------------------------------------
SINGLE = ("Answer questions about countries. Find the figures with search, compute with the calculator. One sentence, "
          "citing the document id of each figure in square brackets.")
NO_ANSWER = "(no answer: "
RESULTS: dict[str, dict] = {}
RUNS: dict[str, list[dict]] = {}                             # stored events, per label and question, for replay()


def _answer(app, question: str) -> str:
    """The final answer of a run. A team answers through its writer only: anything else is no answer."""
    if app == "single":
        return agent_loop(SINGLE, question, ["search", "calculator"], agent="agent")
    start = {"question": question, "transcript": [], "messages": [], "hops": 0, "answer": ""}
    keys = get_type_hints(app.builder.state_schema)              # only the fields this graph's state declares
    out = app.invoke({k: v for k, v in start.items() if k in keys}, {"recursion_limit": 40})
    if out.get("answer"):
        speaker = out.get("speaker", "")
        return f"{NO_ANSWER}the run ended at the {speaker}) {out['answer']}" if speaker not in ("", "writer") \
            else out["answer"]
    written = [t for t in out.get("transcript", []) if t.startswith("[writer] ")]
    return written[-1].removeprefix("[writer] ") if written else f"{NO_ANSWER}the writer never spoke)"


def _correct(item: dict, answer: str) -> bool:
    return not answer.startswith((NO_ANSWER, "ERROR")) and bool(re.search(item["ok"], answer))


def _cites(item: dict, answer: str) -> bool:
    """Every document the answer depends on is cited by its exact id."""
    return not answer.startswith((NO_ANSWER, "ERROR")) and all(d in answer for d in item["src"])


def _scoring(question: str) -> dict:
    """The expected answer and the documents to cite for one of the four lab questions."""
    for item in QUESTIONS:
        if item["q"] == question:
            return item
    raise ValueError(f"No scoring rule for this question: {question!r}. Use the four questions of the notebook, unchanged.")


def _run_all(app, questions: list[str], label: str, sees: str = "full") -> None:
    rows, stored = [], []
    RUN["sees"] = sees
    try:
        for q in questions:
            item = {**_scoring(q)}
            RUN["meter"] = Meter()
            try:
                answer = _answer(app, item["q"])
            except Exception as e:                           # one failed question never stops the batch
                answer = f"ERROR: {e}"
            m = RUN["meter"]
            stored.append({"question": item["q"], "answer": answer, "meter": m})
            rows.append({"question": item["q"][:48], "correct": _correct(item, answer),
                         "cites sources": _cites(item, answer), "calls": m.calls, "tokens": m.prompt + m.completion,
                         "seconds": round(time.time() - m.t0, 1), "answer": answer[:260].replace("\n", " "),
                         "cost": m.cost})
            print(".", end="", flush=True)
    finally:
        RUN["sees"] = "full"
    print()
    df = pd.DataFrame(rows)
    RUNS[label] = stored
    RESULTS[label] = {"correct": int(df.correct.sum()), "cited": int(df["cites sources"].sum()),
                      "calls": int(df.calls.sum()), "tokens": int(df.tokens.sum()),
                      "seconds": float(df.seconds.sum()), "cost": float(df.cost.sum())}
    with pd.option_context("display.max_colwidth", None):
        display(df.drop(columns="cost"))


def run_single(questions: list[str]) -> None:
    _run_all("single", questions, "single agent")


def run_team(app, questions: list[str], label: str = "supervisor team", sees: str = "full") -> None:
    _run_all(app, questions, label, sees)


def _times(value: float, base: float) -> float:
    return round(value / base, 1) if base else float("nan")      # the base run may have failed: no division by zero


def compare() -> pd.DataFrame:
    """Every run next to the single agent, as a table and a chart."""
    if "single agent" not in RESULTS:
        raise RuntimeError("Run the single agent first (section 2).")
    base = RESULTS["single agent"]
    df = pd.DataFrame([{"run": k, "correct": f"{r['correct']}/{len(QUESTIONS)}",
                        "cites sources": f"{r['cited']}/{len(QUESTIONS)}", "model calls": r["calls"],
                        "tokens": r["tokens"], "cost (USD)": round(r["cost"], 5),
                        "x calls": _times(r["calls"], base["calls"]), "x tokens": _times(r["tokens"], base["tokens"]),
                        "tokens per correct answer": round(r["tokens"] / r["correct"]) if r["correct"] else None}
                       for k, r in RESULTS.items()])
    fig, ax = plt.subplots(figsize=(8, 2.8))
    x = range(len(df))
    ax.bar([i - 0.2 for i in x], df["x calls"], 0.38, label="model calls", color=COLORS["supervisor"])
    ax.bar([i + 0.2 for i in x], df["x tokens"], 0.38, label="tokens", color=COLORS["analyst"])
    ax.axhline(1, color="#52514e", lw=1, ls="--")
    ax.set_xticks(list(x), [f"{textwrap.fill(r, 18)}\n{c} correct" for r, c in zip(df["run"], df["correct"])], fontsize=8)
    ax.set_ylabel("x the single agent")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
    return df


# ---- drawing a run ------------------------------------------------------------------------------
COLORS = {"supervisor": "#6a2c91", "researcher": "#2b8fd0", "analyst": "#c05a1a", "checker": "#c05a1a",
          "writer": "#1b8a5a", "agent": "#52514e"}
LANES = ["agent", "supervisor", "researcher", "analyst", "checker", "writer"]    # top to bottom
TOKEN_AREA = 0.2                                             # marker area per token: the same scale on every chart


def _steps_html(m: Meter) -> str:
    """One row per step. Each model call can be opened to read its exact prompt."""
    rows = []
    for i, (agent, kind, detail, tokens, prompt) in enumerate(m.events, 1):
        c = COLORS.get(agent, "#52514e")
        badge = (f'<span style="background:{c};color:white;padding:2px 8px;border-radius:10px;'
                 f'font-weight:600">{agent}</span>')
        what = {"model": "model call", "tool": "tool call", "handoff": "handoff"}[kind]
        read = f"{tokens:,}" if kind == "model" else ""
        shown = ("<details><summary>show prompt</summary><pre style='white-space:pre-wrap;text-align:left;"
                 f"max-height:320px;overflow:auto;font-size:12px'>{html.escape(prompt)}</pre></details>") if prompt else ""
        rows.append(f"<tr><td>{i}</td><td>{badge}</td><td>{what}</td><td style='text-align:left'>{html.escape(detail)}"
                    f"</td><td style='text-align:right'>{read}</td><td style='text-align:left'>{shown}</td></tr>")
    return ("<table><tr><th>step</th><th>who</th><th>kind</th><th>what happened</th><th>tokens read</th>"
            "<th>prompt</th></tr>" + "".join(rows) + "</table>")


def _swimlane(m: Meter) -> None:
    """One row per agent, in step order: the star and the chain become visible. Marker size = tokens read."""
    steps = [(i, e) for i, e in enumerate(m.events, 1) if e[1] in ("model", "tool")]
    if not steps:
        return
    lanes = [a for a in LANES if any(e[0] == a for _, e in steps)]
    y = {a: -n for n, a in enumerate(lanes)}
    fig, ax = plt.subplots(figsize=(8.5, 0.62 * len(lanes) + 1.5))
    ax.plot([i for i, _ in steps], [y[e[0]] for _, e in steps], color="#b9b8b3", lw=1.2, zorder=1)
    top = {}
    for i, e in steps:
        if e[1] == "model":
            ax.scatter(i, y[e[0]], s=max(e[3] * TOKEN_AREA, 25), color=COLORS.get(e[0], "#52514e"),
                       edgecolors="white", linewidths=1.5, zorder=3)
            if e[3] >= top.get(e[0], (0, 0))[1]:
                top[e[0]] = (i, e[3])
        else:
            ax.scatter(i, y[e[0]], s=34, marker="s", facecolors="white", edgecolors=COLORS.get(e[0], "#52514e"),
                       linewidths=1.5, zorder=3)
    for agent, (i, tokens) in top.items():                   # label the largest prompt of each agent only
        ax.annotate(f"{tokens:,}", (i, y[agent]), xytext=(0, 13), textcoords="offset points", ha="center", fontsize=8,
                    color="#3a3936")
    ax.set_yticks([y[a] for a in lanes], lanes)
    ax.set_ylim(-len(lanes) + 0.4, 0.75)
    ax.set_xlim(0.2, len(m.events) + 0.8)
    ax.set_xlabel("step (the numbers of the table above)", fontsize=9)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(length=0)
    ax.grid(axis="y", color="#ecebe7", lw=1)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    key = [plt.scatter([], [], s=t * TOKEN_AREA, color="#8a8983", edgecolors="white", label=f"{t:,} tokens read")
           for t in (200, 1000, 3000)]
    key.append(plt.scatter([], [], s=34, marker="s", facecolors="white", edgecolors="#8a8983", label="tool call"))
    ax.legend(handles=key, frameon=False, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8,
              labelspacing=1, handletextpad=0.6, columnspacing=2)
    plt.tight_layout()
    plt.show()


def _show_run(question: str, answer: str, m: Meter) -> None:
    display(HTML(f"<p><b>Question:</b> {html.escape(question)}</p>" + _steps_html(m)
                 + f"<p><b>Answer:</b> {html.escape(answer[:400])}</p>"
                 f"<p>{m.calls} model calls, {m.prompt + m.completion:,} tokens in total.</p>"))
    _swimlane(m)


def trace(app, question: str, sees: str = "full") -> None:
    """Run one question (a new run, new tokens) and draw it: the steps with their prompts, then the swimlane."""
    RUN["meter"], RUN["sees"] = Meter(), sees
    try:
        answer = _answer(app, question)
    finally:
        RUN["sees"] = "full"
    _show_run(question, answer, RUN["meter"])


def replay(label: str, i: int = 0) -> None:
    """Draw a stored run again: no model call, no tokens, and the same failure. i is the question index, from 0."""
    if label not in RUNS:
        raise KeyError(f"No stored run named {label!r}. Stored runs: {', '.join(RUNS) or 'none yet'}")
    if not 0 <= i < len(RUNS[label]):
        raise IndexError(f"{label!r} has questions 0 to {len(RUNS[label]) - 1}")
    run = RUNS[label][i]
    _show_run(run["question"], run["answer"], run["meter"])


def tokens_by_agent(*labels: str) -> None:
    """Several stored runs, one bar each: the prompt tokens read over the four questions, stacked by agent."""
    missing = [k for k in labels if k not in RUNS]
    if missing:
        raise KeyError(f"No stored run named {missing}. Stored runs: {', '.join(RUNS) or 'none yet'}")
    totals = {k: {} for k in labels}
    for k in labels:
        for run in RUNS[k]:
            for agent, kind, _, tokens, _ in run["meter"].events:
                if kind == "model":
                    totals[k][agent] = totals[k].get(agent, 0) + tokens
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(labels) + 1.3))
    for n, k in enumerate(reversed(labels)):
        left = 0
        for agent in LANES:
            value = totals[k].get(agent, 0)
            if value:
                ax.barh(n, value, left=left, height=0.55, color=COLORS[agent], edgecolor="white", linewidth=2)
                left += value
        ax.annotate(f"{left:,}", (left, n), xytext=(5, 0), textcoords="offset points", va="center", fontsize=9,
                    color="#3a3936")
    ax.set_yticks(range(len(labels)), [textwrap.fill(k, 26) for k in reversed(labels)], fontsize=9)
    ax.set_xlabel("prompt tokens read, four questions", fontsize=9)
    ax.tick_params(length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    shown = [a for a in LANES if any(a in totals[k] for k in labels)]
    ax.legend(handles=[Patch(color=COLORS[a], label=a) for a in shown], frameon=False, ncol=len(shown),
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    plt.tight_layout()
    plt.show()


def show_graph(app) -> None:
    try:
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception:                                        # no network: print the diagram source instead
        print(app.get_graph().draw_mermaid())


# ---- the other design: handoffs (given) ----------------------------------------------------------
class HandoffState(TypedDict):
    question: str
    messages: Annotated[list[dict], operator.add]            # one shared history: it travels with the conversation
    hops: int
    answer: str
    speaker: str                                             # who wrote the answer


def build_handoffs(workers: dict, max_hops: int = 6, start: str = "researcher"):
    """The same workers, no supervisor: each agent can hand the conversation to another by calling a tool."""
    names = list(workers)

    def make_node(name):
        brief, tools = workers[name]
        transfers = [{"type": "function", "function": {
            "name": f"transfer_to_{o}", "description": f"Hand the conversation to the {o}.",
            "parameters": {"type": "object", "properties": {}}}} for o in names if o != name]
        system = (f"You are the {name} of a team ({', '.join(names)}). {brief} When another agent should continue, "
                  "call its transfer_to_ tool. Only the writer answers the user.")

        def node(state: HandoffState) -> Command:
            new: list[dict] = []
            for _ in range(6):
                msgs = [{"role": "system", "content": system}, {"role": "user", "content": state["question"]}]
                m = chat(msgs + state.get("messages", []) + new, name, tools=[SCHEMAS[t] for t in tools] + transfers)
                new.append(m.model_dump(exclude_none=True))
                if not m.tool_calls:                         # no tool call: this agent answers the user
                    return Command(goto=END, update={"messages": new, "answer": m.content or "", "speaker": name})
                target = None
                for c in m.tool_calls:
                    if c.function.name.startswith("transfer_to_"):
                        target = c.function.name.removeprefix("transfer_to_")
                        out = f"Transferred to the {target}."
                    else:
                        try:
                            out = run_tool(c.function.name, json.loads(c.function.arguments or "{}"), name)
                        except json.JSONDecodeError as e:     # the model wrote invalid JSON: tell it, do not crash
                            out = f"Error: invalid tool arguments ({e})"
                    new.append({"role": "tool", "tool_call_id": c.id, "content": out})
                if target in names:
                    if state.get("hops", 0) >= max_hops:
                        return Command(goto=END, update={"messages": new, "answer": f"{NO_ANSWER}hop limit reached)"})
                    _log(name, "handoff", f"hands over to {target}")
                    return Command(goto=target, update={"messages": new, "hops": state.get("hops", 0) + 1})
            return Command(goto=END, update={"messages": new, "answer": f"{NO_ANSWER}step limit reached)"})
        return node

    builder = StateGraph(HandoffState)
    for name in names:
        builder.add_node(name, make_node(name), destinations=tuple(o for o in names if o != name) + (END,))
    builder.add_edge(START, start)
    return builder.compile()


# ---- bonus: three agents agree on a wrong answer -------------------------------------------------
CHECKER = ("You are a checker. Work out the answer to the question from the documents, with your tools. "
           "Reply with the figure and the document ids.")
BONUS_WRITER = "Write the final answer in one sentence from the two reports."


def coordination_tax(question: str, runs: int = 5, checker_tools: tuple[str, ...] = ("search", "calculator")) -> None:
    """The blog says 470 people per km²; the official figures give 389.8. Does a checker catch it?

    The two teams differ in one thing only: whether the checker's task contains the researcher's claim."""
    q, RUN["docs"] = question, {**DOCS, **GUIDE}
    REPORTS: list[str] = []

    def team(show_claim: bool) -> str:
        claim = agent_loop("Find the answer in the documents with the search tool. Reply: the figure and the document "
                           "id.", q, ["search"], "researcher")
        task = f"Question: {q}\nClaim to check: {claim}" if show_claim else f"Question: {q}"
        check = agent_loop(CHECKER, task, list(checker_tools), "checker")
        REPORTS.append(check)
        return agent_loop(BONUS_WRITER, f"Question: {q}\nResearcher: {claim}\nChecker: {check}", [], "writer")

    designs = (("single agent", lambda: agent_loop(SINGLE, q, ["search", "calculator"])),
               ("3 agents, checker sees the claim", lambda: team(True)),
               ("3 agents, checker does not see the claim", lambda: team(False)))
    rows = []
    try:
        for label, fn in designs:
            right = wrong = both = calls = errors = 0
            REPORTS.clear()
            RUNS[label] = []
            for _ in range(runs):
                RUN["meter"] = Meter()
                try:
                    answer = fn()
                except Exception as e:                       # one failed run never stops the batch
                    answer, errors = f"ERROR: {e}", errors + 1
                calls += RUN["meter"].calls
                official, blog = bool(re.search(r"389\.8|\b390\b", answer)), bool(re.search(r"\b470\b", answer))
                right += official and not blog
                wrong += blog and not official
                both += official and blog
                RUNS[label].append({"question": q, "answer": answer, "meter": RUN["meter"]})
            rows.append({"design": label, "official figure only": f"{right}/{runs}", "blog figure only": f"{wrong}/{runs}",
                         "both figures": f"{both}/{runs}",
                         "checker repeats 470": f"{sum(bool(re.search(r'\b470\b', c)) for c in REPORTS)}/{runs}"
                         if label != "single agent" else "",
                         "errors": errors, "calls per run": round(calls / runs, 1)})
    finally:
        RUN["docs"] = DOCS
    with pd.option_context("display.max_colwidth", None):
        display(pd.DataFrame(rows))
    for label, _ in designs[1:]:                             # one sample chain per team: what did the checker search?
        run = RUNS[label][0]
        display(HTML(f"<h4>Sample run: {label}</h4>" + _steps_html(run["meter"])
                     + f"<p><b>Answer:</b> {html.escape(run['answer'][:400])}</p>"))


# ---- checks (given): no model, no tokens --------------------------------------------------------
def check_setup() -> None:
    """First cell: is the API key there? (It is never printed.)"""
    print(f"OK: API key found, model {MODEL}" if os.getenv("OPENROUTER_API_KEY") else SETUP_HELP)


def check_team(TeamState, supervisor, make_worker, WORKERS, MAX_HOPS) -> None:
    """Tests the pieces you wrote with a fake model: it costs nothing and names the blank to fix."""
    global ask_router, agent_loop
    real_router, real_loop = ask_router, agent_loop
    try:
        meta = getattr(get_type_hints(TeamState, include_extras=True)["transcript"], "__metadata__", (None,))
        assert meta[0] is operator.add, ("blank 1: without a reducer, each worker would overwrite the transcript; "
                                         "use operator.add to append")
        ask_router = lambda state: "analyst"                 # noqa: E731  the fake supervisor picks the analyst
        try:
            cmd = supervisor({"question": "q", "transcript": [], "hops": 0})
        except TypeError:
            raise AssertionError("blank 2: compare the hops with a number, the limit named at the top of the cell")
        assert cmd.goto == "analyst", f"blank 3: the supervisor chose 'analyst' but the graph goes to {cmd.goto!r}"
        assert cmd.update == {"hops": 1}, "keep update={'hops': state['hops'] + 1}"
        ask_router = lambda state: "writer"                  # noqa: E731  a second choice: a fixed name would fail here
        cmd = supervisor({"question": "q", "transcript": [], "hops": 0})
        assert cmd.goto == "writer", f"blank 3: the supervisor chose 'writer' but the graph goes to {cmd.goto!r}"
        ask_router = lambda state: "analyst"                 # noqa: E731
        cmd = supervisor({"question": "q", "transcript": [], "hops": MAX_HOPS})
        assert cmd.goto == END, f"blank 2: after {MAX_HOPS} hops the supervisor must stop (goto END)"
        cmd = supervisor({"question": "q", "transcript": [], "hops": MAX_HOPS - 1})
        assert cmd.goto != END, f"blank 2: with {MAX_HOPS - 1} hops done the supervisor must still be allowed to work"
        ask_router = lambda state: "FINISH"                  # noqa: E731
        assert supervisor({"question": "q", "transcript": [], "hops": 0}).goto == END, "FINISH must go to END"
        agent_loop = lambda brief, task, tools, agent="agent": "found it"   # noqa: E731
        cmd = make_worker("researcher", "brief", ["search"])({"question": "q", "transcript": [], "hops": 1})
        assert cmd.goto == "supervisor", f"blank 4: a worker reports back to the supervisor, not to {cmd.goto!r}"
        assert cmd.update == {"transcript": ["[researcher] found it"]}, "keep the transcript update as given"
        assert WORKERS["researcher"][1] == ["search"], "blank 5: the researcher finds figures: it needs 'search'"
        assert WORKERS["analyst"][1] == ["calculator"], "blank 6: the analyst computes: it needs 'calculator'"
        assert WORKERS["writer"][1] == [], "the writer needs no tool"
    except AssertionError as e:
        print(f"NOT YET: {e}")
    except Exception as e:
        print(f"NOT YET: a blank ___ is probably still there ({type(e).__name__}: {e})")
    else:
        print("OK: your supervisor, workers and state are right (tested with a fake model, no tokens)")
    finally:
        ask_router, agent_loop = real_router, real_loop
