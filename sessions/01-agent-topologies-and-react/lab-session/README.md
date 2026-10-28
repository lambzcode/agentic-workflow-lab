# Session 1 lab: build an MCP server, use it from any client

You write one small Python file that serves real open data (data.gouv.fr and French Wikipedia, no API key). Then you plug the same server into several tools and see that they all speak the same protocol.

```bash
uv sync                                  # from the repository root
cp server_template.py my_server.py       # in this folder
uv run python my_server.py               # leave it running: http://127.0.0.1:8000/mcp
```

## Write the missing pieces

`search_datasets` is given. In `my_server.py`, follow the numbered TODOs in order:

1. `get_dataset`: the code is given, write its docstring and fill in `resources`
2. `wiki_summary`: the code is given, write its docstring and fill in `page`
3. `search_wiki(query)`: a tool that finds the exact page title
4. a `limit` argument on `search_datasets`, with validation
5. `dataset_card`: a **resource template** (`opendata://dataset/{dataset_id}`)
6. your own **prompt**

The function name is the tool name, the docstring is the description a model reads, the type hints are the input schema.

The server already offers the two other things an MCP server can expose: a **resource** (`opendata://sources`, data a client can read) and three **prompts** (`describe_dataset`, `compare_datasets`, `fact_check`: message templates the user picks and fills in). Tools are chosen by the model, resources are attached by the app or user, prompts are picked by the user.

Restart the server after each change, and re-run `raw_mcp.sh` to see the new schema.

## Look at the endpoint

```bash
bash raw_mcp.sh        # in a second terminal: initialize, tools/list, resources/list, prompts/list, tools/call
```

This is exactly what every client does. Note the `mcp-session-id` header and the schemas in `tools/list`.

## Plug it into clients

In each client: find the tools, the resource and the prompt (they sit in different places: try them all), then ask something that needs them, for example "Which licence does the data.gouv.fr dataset about the 2002 presidential election use?" Watch which tool the client calls and with which arguments.

## Break it on purpose

0. Try each prompt from a client, for example `fact_check` with "Albert School has a campus in Geneva". Which client lets you fill in the arguments?
1. Replace the description of `search_datasets` with `"Search."`
2. Call `wiki_summary` with a title that does not exist
3. Ask "Who founded Albert School and which open datasets mention it?": count the tool calls, and which order the client chose.

## Optional: run an agent on your server

With `OPENROUTER_API_KEY` in your `.env`, with the server running:

```bash
uv run python demo_agent.py "Which licence does the 2002 presidential election dataset use?"
```

It prints each tool call and observation. `MAX_TURNS=1` shows what happens when the step bound is hit.
