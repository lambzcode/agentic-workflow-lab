# Session 6 lab: evaluate an agent you should not trust yet

A real agent (DeepSeek through OpenRouter) manages music playlists. You score it on 20 scenarios, three times each, and find out what its success rate hides: you score the final state, look at what the agent did on the way, grade its words with a real judge, and check that judge against your own labels. Nothing is staged: every failure really happened.

Put `OPENROUTER_API_KEY=...` in a `.env` file at the repository root or in this folder (see `.env.example`; never commit it). Model: `OPENROUTER_MODEL`, default `~deepseek/deepseek-v4-flash-latest`. The whole lab costs about USD 0.10.

```bash
uv sync                                  # from the repository root
uv run jupyter lab 06_agent_evaluation.ipynb
```

**No key, or the network fails?** The lab loads real runs recorded for this course (`data/recorded_runs.json`), and each cell says so.

**Files:** `06_agent_evaluation.ipynb` is the only file you work in. `lab.py` is given (the music backend, 20 scenarios, the agent, the judge, the drawings and the checks): no need to read it.

**You write:** nine blanks, written `___`, each tested with no model call; your 20 labels; a fix to the prompt; a short report.

**Deliverable:** the executed notebook, with the report and the four questions answered. Sections 1 to 6 and 10 are enough for the report; sections 7 to 9 if time.
