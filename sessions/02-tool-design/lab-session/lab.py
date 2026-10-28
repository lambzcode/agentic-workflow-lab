"""Session 2 lab, given code: you do not need to read or edit this file.

It runs an agent (OpenAI Agents SDK, DeepSeek through OpenRouter) against your MCP server, with the network
misbehaving, and puts YOUR dispatcher between the agent and the server. Needs OPENROUTER_API_KEY in a `.env` file.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.mcp import MCPServerStdio
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

load_dotenv(find_dotenv(usecwd=True))
set_tracing_disabled(True)

SERVER, LEDGER = "my_mcp_server.py", Path("ledger.jsonl")   # the server appends every real change to the ledger
TASK = "Add the Lumen tracks 'Night Bus' and 'Glass Harbour' to my Focus playlist, and tell me the tempo (BPM) of each."
MODEL = OpenAIChatCompletionsModel(
    model=os.getenv("OPENROUTER_MODEL", "~deepseek/deepseek-v4-flash-latest"),
    openai_client=AsyncOpenAI(base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                              api_key=os.environ["OPENROUTER_API_KEY"]))


async def plain_call(call, tool, args, run_id, keyed, read_only):
    """v0.1: one attempt, no key."""
    return await call(args)


def typed_error(result):
    """The JSON error raised by the server (code, retryable...), or None if the call worked or the error is opaque."""
    if not result.is_error:
        return None
    text = result.content[0].text
    try:
        return json.loads(text[text.index("{"):])
    except ValueError:
        return None


class HarnessServer(MCPServerStdio):
    """The MCP server, with a dispatcher between the agent and the server. The agent never sees `idempotency_key`."""

    def __init__(self, server_file, dispatcher, run_id):
        env = {**os.environ, "LAB_CHAOS": "1", "LAB_LEDGER": str(LEDGER)}          # chaos on
        super().__init__({"command": sys.executable, "args": [server_file], "env": env})
        self.dispatcher, self.run_id = dispatcher, run_id

    async def list_tools(self, *args, **kwargs):
        tools = await super().list_tools(*args, **kwargs)
        self.keyed = {t.name for t in tools if "idempotency_key" in t.input_schema.get("properties", {})}
        self.read_only = {t.name for t in tools if t.annotations and t.annotations.read_only_hint}
        for t in tools:
            t.input_schema.get("properties", {}).pop("idempotency_key", None)
        return tools

    async def call_tool(self, tool_name, arguments=None, meta=None):
        call = lambda args: super(HarnessServer, self).call_tool(tool_name, args)
        return await self.dispatcher(call, tool_name, arguments or {}, self.run_id,
                                     tool_name in self.keyed, tool_name in self.read_only)


def ledger():
    return [json.loads(line) for line in LEDGER.read_text().splitlines()] if LEDGER.exists() else []


async def run(dispatcher, server_file, run_id):
    """One agent run on a fresh server; compare what the server really did with what the agent says."""
    LEDGER.unlink(missing_ok=True)
    async with HarnessServer(server_file, dispatcher, run_id) as server:
        agent = Agent(name="music", model=MODEL, mcp_servers=[server], model_settings=ModelSettings(max_tokens=2000),
                      instructions="You are a music assistant. Use the tools. If a tool fails, decide whether to try again.")
        result = await Runner.run(agent, TASK)
    count = Counter((e["playlist"], e["track_id"]) for e in ledger())
    night, glass = count[("Focus", "T003")], count[("Focus", "T002")]
    return {"Night Bus written (want 1)": night, "Glass Harbour written (want 1)": glass,
            "other writes": sum(count.values()) - night - glass, "agent says": result.final_output[:100]}


async def trials(dispatcher=plain_call, n=3, server_file=SERVER):
    """n independent runs of the agent. The model is not deterministic: runs differ."""
    return pd.DataFrame([await run(dispatcher, server_file, f"run{i}") for i in range(n)]).rename_axis("run")


async def check(dispatcher, server_file=SERVER):
    """Unit test, no model: the same logical write twice, with chaos on, must leave exactly one change."""
    LEDGER.unlink(missing_ok=True)
    args = {"playlist": "Focus", "track_id": "T003"}
    async with HarnessServer(server_file, dispatcher, "t1") as server:
        await server.list_tools()
        await server.call_tool("add_to_playlist", args)
        await server.call_tool("add_to_playlist", args)
    assert len(ledger()) == 1, f"{len(ledger())} changes in the ledger: {ledger()}"
    print("OK: one change")


async def check_server(server_file=SERVER):
    """Unit test, no model: are your annotations and argument types the ones the lab asks for?"""
    async with HarnessServer(server_file, plain_call, "schema") as server:
        tools = {t.name: t for t in await server.list_tools()}
    problems = []

    def hint(name, field):
        a = tools[name].annotations
        return getattr(a, field, None) if a else None

    for name in ("search_tracks", "get_track"):
        if hint(name, "read_only_hint") is not True:
            problems.append(f"{name}: readOnlyHint must be True")
        if hint(name, "idempotent_hint") is not True:
            problems.append(f"{name}: repeating a read is harmless, so idempotentHint must be True")
    if hint("add_to_playlist", "read_only_hint") is not False:
        problems.append("add_to_playlist: it changes the playlist, so readOnlyHint must be False")
    if hint("add_to_playlist", "idempotent_hint") is not True:
        problems.append("add_to_playlist: repeating it with the same key changes nothing, so idempotentHint must be True")
    for name in tools:
        if hint(name, "open_world_hint") is not False:
            problems.append(f"{name}: a closed catalogue, so openWorldHint must be False")
    for name in ("get_track", "add_to_playlist"):
        if "pattern" not in tools[name].input_schema["properties"]["track_id"]:
            problems.append(f"{name}: track_id needs a pattern (T followed by three digits)")
    if "enum" not in tools["add_to_playlist"].input_schema["properties"]["playlist"]:
        problems.append("add_to_playlist: playlist must be a Literal of the three names")
    if "minLength" not in tools["search_tracks"].input_schema["properties"]["query"]:
        problems.append("search_tracks: query needs min_length=1")
    assert not problems, "\n".join(problems)
    print("OK: annotations and argument types are as expected")
