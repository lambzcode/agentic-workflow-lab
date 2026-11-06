# Session 4 lab: four memory layers for an agent

A real agent (DeepSeek and an embedding model, through OpenRouter) gets four memory layers: a scratchpad, a conversation buffer with a summary, similarity-searched notes, and exact facts. You fill in three small functions, compare the layers on the same question, replay a long conversation, and finally delete everything about a user.

Put `OPENROUTER_API_KEY=...` in a `.env` file at the repository root (see `.env.example`; never commit it). Models: `OPENROUTER_MODEL` (default `~deepseek/deepseek-v4-flash-latest`) and `OPENROUTER_EMBED_MODEL` (default `openai/text-embedding-3-small`).

```bash
uv sync                                  # from the repository root
uv run jupyter lab 04_memory_layers.ipynb
```

**Files:** `04_memory_layers.ipynb` is the only thing you work in. `lab.py` (the agent, the model calls, the helpers, the tests) is given: no need to read it.

**You write:** eight blanks, written `___`, in three small functions (`consolidate_buffer`, `upsert`, `forget_user`). The cell under each one tests your answer without calling the model and says which blank is wrong.

**Studio (optional, after the notebook):** `uv run langgraph dev` from this folder, then open the Studio URL it prints (Chrome, Firefox or Edge; a free LangSmith login is asked). Graph `memory_agent`: send `{"message": "My birthday is 14 March.", "user_id": "albert"}`, then a new thread with `{"message": "When is my birthday?", "user_id": "albert"}`; look at the `context` of the `recall` node, and at the **Store** tab (namespaces `users / albert / profile` and `notes`). `/forget` pauses on `confirm_forget`: resume with `{"approve": true}`, then look at the Store again. `studio.py` contains finished versions of your three functions, so read it after your own.
