"""Session 4 lab, given code: you do not need to read or edit this file.

A chat agent with four memory layers, as a LangGraph graph. A real model (DeepSeek through OpenRouter) answers; an
embedding model (OpenAI, through OpenRouter) powers the similarity search. Needs OPENROUTER_API_KEY in a `.env` file.

    layer 1  scratchpad    the "Thought:" line of one run (state, never saved)
    layer 2  buffer        `turns` (recent messages) and `summary` (older ones), per thread: the checkpointer
    layer 3  notes         free-text notes found by similarity: the store's semantic index
    layer 4  profile       exact facts found by key: the same store, no embedding

    recall -> respond -> write_memory                 (a normal turn)
    recall -> confirm_forget -> purge                 (a message starting with /forget; interrupt() as in Session 3)
"""
import json
import os
import re
import uuid
from types import SimpleNamespace
from typing import NotRequired, TypedDict

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command, interrupt
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(find_dotenv(usecwd=True))
MODEL = os.getenv("OPENROUTER_MODEL", "~deepseek/deepseek-v4-flash-latest")
EMBED_MODEL = os.getenv("OPENROUTER_EMBED_MODEL", "openai/text-embedding-3-small")
ALL_LAYERS = ("summary", "notes", "profile")
MIN_SCORE = 0.3                                              # notes less similar than this are not shown to the model
SENSITIVE = re.compile(r"password|iban|card number|secret|api[_ -]?key|token", re.I)
SETUP_HELP = ("No OPENROUTER_API_KEY found. Create a file named .env at the repository root containing\n"
              "OPENROUTER_API_KEY=your-key (see .env.example), then restart the kernel and run the cells again.")


def _client() -> OpenAI:
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError(SETUP_HELP)
    return OpenAI(base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                  api_key=os.getenv("OPENROUTER_API_KEY"), max_retries=3, timeout=90)


def chat(messages: list[dict]) -> str:
    r = _client().chat.completions.create(model=MODEL, messages=messages, temperature=0, max_tokens=800)
    return r.choices[0].message.content or ""


def embed(texts: list[str]) -> list[list[float]]:
    return [d.embedding for d in _client().embeddings.create(model=EMBED_MODEL, input=list(texts)).data]


def summarize(summary: str, turns: list[dict]) -> str:
    """Fold `turns` into the running `summary` (one model call)."""
    dialogue = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    return chat([{"role": "user", "content": "Update the running summary of a conversation. Keep names, dates, decisions "
                  "and open questions; drop small talk. At most 80 words.\n\n"
                  f"Current summary:\n{summary or '(empty)'}\n\nNew turns:\n{dialogue}\n\nUpdated summary:"}]).strip()


class Fact(BaseModel):
    key: str     # snake_case, e.g. birthday, city, allergy, nickname
    value: str   # copied VERBATIM from the user's words


class Extraction(BaseModel):
    facts: list[Fact] = []


def extract_facts(message: str) -> list[Fact]:
    """Structured output: the facts the USER states about themselves in one message."""
    system = ("Extract the facts the USER states about themselves or asks you to remember (birthday, city, allergy, "
              "nickname for what to call them, ...). Ignore questions and small talk. Copy values verbatim. "
              "Never extract passwords, card numbers or other secrets.")
    try:
        r = _client().chat.completions.parse(model=MODEL, response_format=Extraction, temperature=0, max_tokens=500,
                                             messages=[{"role": "system", "content": system},
                                                       {"role": "user", "content": message}])
        return r.choices[0].message.parsed.facts
    except Exception:                                        # one failed call must not crash the conversation
        return []


SYSTEM = """You are a helpful assistant with a long-term memory.
- KNOWN FACTS are exact and authoritative: use them verbatim (names, dates, spellings), never paraphrase.
- RELATED NOTES are similarity matches and may be wrong or outdated: use them only as hints.
- If the memory does not contain what you need, say you do not know. Never invent a memory.
Answer in two lines: "Thought: <one short line>" then "Answer: <short reply to the user>".

{context}"""

# ---- the conversation used in the notebook ------------------------------------------------------
BIRTHDAY_NOTES = ["My birthday is 14 March.", "My sister Léa's birthday is 4 May.", "Léa's birthday party last year was great fun.",
                  "Actually, my birthday is 15 March, I misspoke earlier."]
CONVERSATION = [
    "Hello, I am Albert. My birthday is 15 March.", "Please call me Bertie.", "I am allergic to peanuts.",
    "How was your day?", "I spent the weekend repainting my kitchen a deep green.", "Do you like jazz?",
    "Last night I cooked a mushroom risotto for friends.", "What is a good way to learn the guitar?",
    "I am training for a half marathon in the spring.", "Tell me a fun fact about octopuses.",
    "My neighbour just adopted a very loud parrot.", "Can you recommend a quiet book for a train ride?",
    "I started a small balcony garden with tomatoes and basil.", "Why is the sky blue?",
    "We watched an old black and white film about a lighthouse keeper.", "What is the capital of Portugal?",
    "I think I will take the night train to Lisbon in the spring.", "How do you make a good espresso?",
    "My bike was stolen outside the library last week.", "Is it true that bees can recognise faces?",
    "I have been learning to bake sourdough bread on Sundays.", "What does a sommelier do?",
    "The bakery on my street now sells excellent rye bread.", "Which planet has the most moons?",
]
PROBES = [("When is my birthday?", "15 March"), ("What am I allergic to?", "peanuts"), ("What should you call me?", "Bertie")]


# ---- defaults, used until you plug in your own functions ---------------------------------------
def _keep_all(turns, summary, keep, summarize):
    return turns, summary


def _overwrite(store, user_id, key, value):
    store.put(("users", user_id, "profile"), key, {"value": value, "history": []})


def _not_yet(store, user_id):
    raise NotImplementedError("Run the cell that defines forget_user, then build the agent again with it.")


class State(TypedDict):
    message: str
    user_id: str
    context: str
    scratchpad: list[str]
    turns: list[dict]
    summary: str
    reply: str
    intent: str
    report: dict | None
    canned: NotRequired[str]
    write: NotRequired[bool]


class Input(TypedDict):
    message: str
    user_id: NotRequired[str]
    canned: NotRequired[str]                                 # a ready-made reply: no model call (fast replay)
    write: NotRequired[bool]                                 # False: read-only turn (used by the probes)


def _approved(decision) -> bool:
    """The human's answer to /forget. Studio may send it as text ('{"approve": true}', 'true', 'yes'), the notebook as a dict."""
    if isinstance(decision, str):
        try:
            decision = json.loads(decision)
        except ValueError:
            decision = decision.strip().lower() in ("yes", "y", "approve")
    if isinstance(decision, dict):
        decision = decision.get("approve")
    return decision is True or (isinstance(decision, str) and decision.strip().lower() in ("true", "yes"))


def build_agent(consolidate_buffer=_keep_all, upsert=_overwrite, forget_user=_not_yet, layers=None, keep=6, studio=False):
    """A fresh agent with its own memory. `layers` is the set of layers the agent READS (default: all three)."""
    store = InMemoryStore(index={"embed": embed, "dims": 1536, "fields": ["text"]})
    saver = InMemorySaver()
    agent = SimpleNamespace(store=store, saver=saver, keep=keep, layers=set(ALL_LAYERS if layers is None else layers))
    notes_ns = lambda u: ("users", u, "notes")
    profile_ns = lambda u: ("users", u, "profile")

    def recall(state: State, *, store: BaseStore) -> dict:
        user = state.get("user_id") or "anon"
        facts = "\n".join(f"- {i.key}: {i.value['value']}" for i in store.search(profile_ns(user), limit=100)
                          ) if "profile" in agent.layers else ""
        notes = [n for n in store.search(notes_ns(user), query=state["message"], limit=3)
                 if n.score >= MIN_SCORE] if "notes" in agent.layers else []
        related = "\n".join(f"- ({n.score:.2f}) {n.value['text']}" for n in notes)
        earlier = state.get("summary", "") if "summary" in agent.layers else ""
        context = (f"KNOWN FACTS (exact, authoritative):\n{facts or '(none)'}\n\n"
                   f"RELATED NOTES (similarity search, may be inaccurate or outdated):\n{related or '(none)'}\n\n"
                   f"EARLIER IN THIS CONVERSATION (summary):\n{earlier or '(none)'}")
        return {"user_id": user, "context": context, "scratchpad": [], "report": None,
                "intent": "forget" if state["message"].strip().lower().startswith("/forget") else "chat"}

    def respond(state: State) -> dict:
        if state.get("canned"):                              # (these two inputs are reset by every call of `say`)
            return {"reply": state["canned"]}
        messages = [{"role": "system", "content": SYSTEM.format(context=state["context"])}]
        messages += state.get("turns", []) + [{"role": "user", "content": state["message"]}]
        text = chat(messages)
        thought, _, answer = text.partition("Answer:")
        return {"reply": (answer or text).strip(), "scratchpad": [thought.replace("Thought:", "").strip()]}

    def write_memory(state: State, *, store: BaseStore) -> dict:
        if state.get("write") is False:
            return {}
        user, message = state["user_id"], state["message"]
        if SENSITIVE.search(message):                        # write guard: a secret is stored in no layer
            message = "[redacted: sensitive content]"
        else:
            for fact in extract_facts(message):              # layer 4: exact facts
                upsert(store, user, fact.key, fact.value)
            if "?" not in message and len(message.split()) >= 6:       # layer 3: statements, not questions
                store.put(notes_ns(user), str(uuid.uuid4()), {"text": message})
        turns = state.get("turns", []) + [{"role": "user", "content": message},
                                          {"role": "assistant", "content": state["reply"]}]
        turns, summary = consolidate_buffer(turns, state.get("summary", ""), agent.keep, summarize)   # layer 2
        return {"turns": turns, "summary": summary}

    def confirm_forget(state: State, *, store: BaseStore) -> Command:
        user = state["user_id"]
        decision = interrupt({"action": "forget_user", "user_id": user, "will_delete": {
            "notes": len(store.search(notes_ns(user), limit=1000)), "facts": len(store.search(profile_ns(user), limit=1000))}})
        approve = _approved(decision)
        return Command(goto="purge" if approve else END,
                       update={} if approve else {"reply": "Cancelled: nothing was deleted."})

    def purge(state: State, *, store: BaseStore) -> dict:
        report = forget_user(store, state["user_id"])
        return {"report": report, "turns": [], "summary": "", "reply": f"Done. Deleted {report}."}

    g = StateGraph(State, input_schema=Input)
    g.add_node("recall", recall)
    g.add_node("respond", respond)
    g.add_node("write_memory", write_memory)
    g.add_node("confirm_forget", confirm_forget, destinations=("purge", END))
    g.add_node("purge", purge)
    g.add_edge(START, "recall")
    g.add_conditional_edges("recall", lambda s: "confirm_forget" if s["intent"] == "forget" else "respond",
                            {"respond": "respond", "confirm_forget": "confirm_forget"})
    g.add_edge("respond", "write_memory")
    g.add_edge("write_memory", END)
    g.add_edge("purge", END)
    # Studio (`langgraph dev`) provides the checkpointer AND the store (configured in langgraph.json): the nodes receive it
    # as the `store` argument, so what the agent writes is visible in Studio and at /store/items/search.
    agent.app = g.compile() if studio else g.compile(checkpointer=saver, store=store)
    return agent


# ---- helpers for the notebook ---------------------------------------------------------------------
def show_graph(app):
    from IPython.display import Image, display
    try:
        display(Image(app.get_graph().draw_mermaid_png()))
    except Exception:                                       # no network: print the diagram source instead
        print(app.get_graph().draw_mermaid())


def say(agent, text, thread, user="albert", show_context=False, **inputs):
    """One turn. Prints the user's message and the agent's reply; returns the final state."""
    config = {"configurable": {"thread_id": thread}}
    out = agent.app.invoke({"message": text, "user_id": user, "canned": "", "write": True, **inputs}, config)
    if out.get("__interrupt__"):
        print(f"user: {text}\nPAUSED, waiting for a human: {out['__interrupt__'][0].value}")
    else:
        print(f"user : {text}\nagent: {out['reply']}")
    if show_context:
        print("\n--- what the model saw ---\n" + out["context"])
    return out


def resume(agent, thread, approve):
    out = agent.app.invoke(Command(resume={"approve": approve}), {"configurable": {"thread_id": thread}})
    print("agent:", out["reply"])
    return out


def feed(agent, messages, thread, user="albert"):
    """Replay a conversation quickly: the agent's replies are canned, but every message is really stored."""
    for text in messages:
        agent.app.invoke({"message": text, "user_id": user, "canned": "Thanks, I will keep that in mind.", "write": True}, {"configurable": {"thread_id": thread}})
        print(".", end="", flush=True)
    print(f" {len(messages)} messages stored")


def show_memory(agent, user="albert", thread=None):
    facts = agent.store.search(("users", user, "profile"), limit=100)
    print(f"layer 4, facts : {{{', '.join(f'{i.key}: {i.value['value']}' for i in facts)}}}")
    print(f"layer 3, notes : {len(agent.store.search(('users', user, 'notes'), limit=1000))}")
    if thread:
        st = agent.app.get_state({"configurable": {"thread_id": thread}}).values
        print(f"layer 2, buffer: {len(st.get('turns', []))} messages kept, summary: {st.get('summary', '')!r}")


CONFIGS = [("all layers", ALL_LAYERS, "next-day"), ("only profile (layer 4)", ("profile",), "next-day"),
           ("only notes (layer 3)", ("notes",), "next-day"), ("only summary, new thread", ("summary",), "next-day"),
           ("only summary, same thread", ("summary",), "long"), ("no memory", (), "next-day")]


def which_layer_answers(agent) -> pd.DataFrame:
    """The three probe questions with only some layers read. True: the reply contains the right answer."""
    rows = []
    for name, layers, thread in CONFIGS:
        agent.layers = set(layers)
        row = {"layers read": name}
        for question, expected in PROBES:
            row[question] = expected.lower() in say(agent, question, thread, write=False)["reply"].lower()
        rows.append(row)
    agent.layers = set(ALL_LAYERS)
    return pd.DataFrame(rows)


# ---- checks (given): no model, no tokens ------------------------------------------------------
def _run(name, test):
    """Run a test; a blank that is still `___` or a failed assertion becomes a readable message."""
    try:
        test()
    except AssertionError as e:
        print(f"NOT YET, {name}: {e}")
    except Exception as e:                                   # a leftover blank raises many kinds of errors
        print(f"NOT YET, {name}: a blank `___` is probably still there, or is wrong ({type(e).__name__}: {e})")
    else:
        print(f"OK: {name} is right, tested with no model and no tokens")


def check_setup() -> None:
    """First cell: is the API key there? (It is never printed.)"""
    print(f"OK: API key found, models {MODEL} and {EMBED_MODEL}" if os.getenv("OPENROUTER_API_KEY") else SETUP_HELP)


def check_consolidate(consolidate_buffer) -> None:
    def test():
        calls = []

        def fake_summarize(summary, old):
            calls.append(old)
            return summary + "|" + "|".join(t["content"] for t in old)
        turns = [{"role": "user", "content": f"m{i}"} for i in range(12)]
        recent, _ = consolidate_buffer(turns, "", 4, fake_summarize)
        assert [t["content"] for t in recent] == ["m8", "m9", "m10", "m11"], "blank 1: keep the LAST `keep` turns"
        assert len(calls) == 1, "blank 2: call the summarizer exactly once"
        assert [t["content"] for t in calls[0]] == [f"m{i}" for i in range(8)], "blank 2: summarize the OLD turns"
        assert consolidate_buffer(turns[:3], "S", 4, fake_summarize) == (turns[:3], "S") and len(calls) == 1, \
            "with at most `keep` turns nothing changes and nothing is called"
    _run("consolidate_buffer", test)


def check_upsert(upsert) -> None:
    def test():
        s = InMemoryStore()
        upsert(s, "albert", "birthday", "14 March")
        upsert(s, "albert", "birthday", "15 March")
        upsert(s, "albert", "birthday", "15 March")
        rec = s.get(("users", "albert", "profile"), "birthday").value
        assert rec["value"] == "15 March", "blank 5: the record must be stored under `key` and hold the new value"
        assert rec["history"] == ["14 March"], "blanks 3 and 4: the OLD value goes to history, only when it differs"
        assert len(s.search(("users", "albert", "profile"))) == 1, "blank 5: one record per key"
    _run("upsert", test)


def check_forget(forget_user) -> None:
    def test():
        s = InMemoryStore()
        for u in ("albert", "bob"):
            s.put(("users", u, "profile"), "birthday", {"value": "1 May", "history": []})
            s.put(("users", u, "notes"), "n1", {"text": f"{u} likes jazz"})
        assert forget_user(s, "albert") == {"notes": 1, "facts": 1}, "blank 8: the report counts what was removed"
        assert s.search(("users", "albert", "profile")) == [] and s.search(("users", "albert", "notes")) == [], \
            "blank 7: delete every item"
        assert len(s.search(("users", "bob", "profile"))) == 1 and len(s.search(("users", "bob", "notes"))) == 1, \
            "blank 6: Bob's data must not change, use the user_id in the namespace"
    _run("forget_user", test)
