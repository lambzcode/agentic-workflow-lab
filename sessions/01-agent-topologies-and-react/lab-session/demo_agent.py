"""Session 1 live demo (teacher only): an agent that uses your MCP server over HTTP.

    uv run python my_server.py                              # terminal 1
    uv run python demo_agent.py "Which licence does the 2002 presidential election dataset use?"
    MAX_TURNS=1 uv run python demo_agent.py "..."           # MaxTurnsExceeded

Needs OPENROUTER_API_KEY in a `.env` file at the repository root.
"""
import asyncio
import os
import sys

from agents import Agent, OpenAIChatCompletionsModel, Runner, set_tracing_disabled
from agents.mcp import MCPServerStreamableHttp
from dotenv import find_dotenv, load_dotenv
from openai import AsyncOpenAI

load_dotenv(find_dotenv(usecwd=True))
set_tracing_disabled(True)

MODEL = OpenAIChatCompletionsModel(
    model=os.getenv("OPENROUTER_MODEL", "~deepseek/deepseek-v4-flash-latest"),
    openai_client=AsyncOpenAI(base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
                              api_key=os.environ["OPENROUTER_API_KEY"]))


async def main(question: str) -> None:
    async with MCPServerStreamableHttp(params={"url": "http://127.0.0.1:8000/mcp"}) as server:
        agent = Agent(name="assistant", model=MODEL, mcp_servers=[server],
                      instructions="Answer only from your tools.")
        result = await Runner.run(agent, question, max_turns=int(os.getenv("MAX_TURNS", "8")))
    for item in result.new_items:
        if item.type == "tool_call_item":
            print(f"tool call:   {item.raw_item.name}({item.raw_item.arguments})")
        elif item.type == "tool_call_output_item":
            print(f"observation: {str(item.output)[:120]}")
    print(f"model calls: {len(result.raw_responses)}")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
