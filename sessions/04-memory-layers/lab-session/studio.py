"""LangGraph Studio demo: the memory agent in the browser.

Do the notebook first (sections 2, 3 and 8). This file holds finished versions of the three functions you wrote
(`consolidate_buffer`, `upsert`, `forget_user`), so open it after you have written your own.

Run from this folder:    uv run langgraph dev
Then open the Studio URL printed in the terminal (Chrome, Firefox or Edge).
Studio provides the checkpointer, so the graph is built without one.
"""
import lab


def consolidate_buffer(turns, summary, keep, summarize):
    if len(turns) <= keep:
        return turns, summary
    old, recent = turns[:-keep], turns[-keep:]
    return recent, summarize(summary, old)


def upsert(store, user_id, key, value):
    ns = ("users", user_id, "profile")
    item = store.get(ns, key)
    history = list(item.value["history"]) if item else []
    if item and item.value["value"] != value:
        history.append(item.value["value"])
    store.put(ns, key, {"value": value, "history": history})


def forget_user(store, user_id):
    report = {}
    for kind, name in (("notes", "notes"), ("profile", "facts")):
        ns = ("users", user_id, kind)
        items = store.search(ns, limit=1000)
        for item in items:
            store.delete(ns, item.key)
        report[name] = len(items)
    return report


memory_agent = lab.build_agent(consolidate_buffer, upsert, forget_user, studio=True).app
