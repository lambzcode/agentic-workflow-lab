# Session 2 lab: harden your MCP server

A real agent (DeepSeek through OpenRouter, with the OpenAI Agents SDK) uses a music MCP server while the network misbehaves. You watch it fail, harden the server and the dispatcher, and watch it again.

Put `OPENROUTER_API_KEY=...` in a `.env` file at the repository root (see `.env.example`; never commit it). Model: `OPENROUTER_MODEL`, default `~deepseek/deepseek-v4-flash-latest`.

```bash
uv sync                                  # from the repository root
uv run jupyter lab 02_agent_in_the_loop.ipynb
```

**Files:** `my_mcp_server.py` (the only file you edit) and `02_agent_in_the_loop.ipynb`. Given: `music_service.py` (the simulated service), `naive_server.py` (v0.1), `lab.py` (the agent, the server wrapper, the run loop and the checks): no need to read them. Start your server by hand with `uv run my_mcp_server.py`; the notebook starts it for you.

**Chaos** (`LAB_CHAOS=1`, set by the notebook): every 4th read raises a 429; the 1st, 4th, 7th... write is executed, then raises `TimeoutError`. The provider replays the first result for a repeated `idempotency_key`.

**You write:** five type annotations in `my_mcp_server.py` (the lines marked TODO): the `READ` and `WRITE` annotations, and the types `TrackId`, `Playlist` and `Query`. The typed errors, the `idempotency_key` argument, the tool bodies and the retry dispatcher are given: read them. `check_server()` in the notebook tells you what is still wrong.
