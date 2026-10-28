# Session 2 lab: harden your MCP server

A real agent (DeepSeek through OpenRouter, with the OpenAI Agents SDK) uses a music MCP server while the network misbehaves. You watch it fail, harden the server and the dispatcher, and watch it again.

Put `OPENROUTER_API_KEY=...` in a `.env` file at the repository root (see `.env.example`; never commit it). Model: `OPENROUTER_MODEL`, default `~deepseek/deepseek-pro-latest`.

```bash
uv sync                                  # from the repository root
uv run jupyter lab 02_agent_in_the_loop.ipynb
```

**Files:** `my_mcp_server.py` (the only file you edit; the block marked "simulated service" must stay as is) and `02_agent_in_the_loop.ipynb`. `lab.py` is given code (the agent, the server wrapper, the run loop): no need to read it. Start the server by hand with `uv run my_mcp_server.py`; the notebook starts it for you.

**Chaos** (`LAB_CHAOS=1`, set by the notebook): every 4th read raises a 429; the 1st, 4th, 7th... write is executed, then raises `TimeoutError`. The provider replays the first result for a repeated `idempotency_key`.

**You write:** the hardened tools in `my_mcp_server.py` (4 TODOs), then `idempotency_key()` and `call_with_retries()` in the notebook (use `tenacity` for the retries).
