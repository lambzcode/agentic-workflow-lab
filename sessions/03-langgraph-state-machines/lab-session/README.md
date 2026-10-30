# Session 3 lab: put a guard on an agent

A real agent (DeepSeek through OpenRouter) runs a music app and gets a `delete_playlist` tool. You watch it delete a playlist without asking, then build the guard with LangGraph: a **router** that sends dangerous tools to a human, and an **approval step** (`interrupt()`) that pauses the run, even across a process restart.

Put `OPENROUTER_API_KEY=...` in a `.env` file at the repository root (see `.env.example`; never commit it). Model: `OPENROUTER_MODEL`, default `~deepseek/deepseek-v4-flash-latest`.

```bash
uv sync                                  # from the repository root
uv run jupyter lab 03_langgraph_agent.ipynb
```

**Files:** `03_langgraph_agent.ipynb` is the only thing you work in. `lab.py` is given (the music app, the model, the nodes, the restart demo): no need to read it.

**You write:** seven blanks, written `___`: four in the router (section 2) and three in the approval step (section 3). The cell under each one tests your answer without calling the model.
