# Session 5 lab: how a multi-agent framework works

You write the core of a supervisor framework in LangGraph (a shared state, a supervisor node, worker nodes), run it on four questions about real country figures, and watch every step it takes. Then you compare it with one agent alone and with the handoff design. A real model (DeepSeek through OpenRouter) does every step.

Put `OPENROUTER_API_KEY=...` in a `.env` file at the repository root or in this folder (see `.env.example`; never commit it). Model: `OPENROUTER_MODEL`, default `~deepseek/deepseek-v4-flash-latest`.

```bash
uv sync                                  # from the repository root
uv run jupyter lab 05_supervisor.ipynb   # from this folder
uv run langgraph dev                     # section 4: the same graphs in LangGraph Studio
```

**Files:** `05_supervisor.ipynb` is the only file you work in. `lab.py` is given (the figures, the tools, the drawings, the handoff team). `studio.py` and `langgraph.json` serve the finished graphs to Studio. `figures/` holds the diagram of section 3.

**You write:** six blanks, written `___`, in one cell (section 3). The cell under it tests your answer with a fake model and names the blank to fix.

**Data:** population on 1 January 2025 (Eurostat) and total area (UN Statistics Division, Demographic Yearbook).

**Optional:** `herdr/README.md` has three short demos that run the same designs side by side in [herdr](https://herdr.dev).
