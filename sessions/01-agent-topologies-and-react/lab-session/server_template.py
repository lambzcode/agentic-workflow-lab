"""Your first MCP server: open data from data.gouv.fr and Wikipedia, served over HTTP.

Copy to `my_server.py`, fill in the TODOs, then:   uv run python my_server.py
The MCP endpoint is:                                   http://127.0.0.1:8000/mcp
Any MCP client can use it: Inspector, Claude Code, Codex, Cursor...

Both APIs are public and need no key.
"""

import requests
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

DATA_GOUV = "https://www.data.gouv.fr/api/1"
WIKI = "https://fr.wikipedia.org/api/rest_v1"
HEADERS = {"User-Agent": "agentic-workflows-course/1.0"}

mcp = MCPServer("opendata-server")


def get_json(url: str, **params) -> dict:
    """GET a JSON API. Any failure becomes a ToolError: the model only reads those messages."""
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=10)
    except requests.RequestException as e:
        raise ToolError(f"The service is unreachable ({type(e).__name__}). Try again in a moment.")
    if r.status_code == 404:
        raise ToolError("Not found. Check the id or the exact title.")
    if not r.ok:
        raise ToolError(f"The service answered HTTP {r.status_code}.")
    return r.json()


# The function name becomes the tool name; the docstring is the description the model reads.
# Say what the tool does, WHEN to use it, and what the arguments look like.
@mcp.tool()
def search_datasets(query: str) -> list[dict]:
    """Search the data.gouv.fr catalogue (French open data) by keywords.

    Use it first to find a dataset and its id. Returns up to 5 datasets with id, title,
    publishing organisation, licence, update frequency and last update date.
    """
    data = get_json(f"{DATA_GOUV}/datasets/", q=query, page_size=5)["data"]
    return [{
        "id": d["id"],
        "title": d["title"],
        "organization": (d.get("organization") or {}).get("name"),
        "license": d.get("license"),
        "frequency": d.get("frequency"),
        "last_update": (d.get("last_update") or "")[:10],
    } for d in data]


# TODO 1: finish get_dataset. The code is given; you fill in two things:
#   (a) the docstring: what the tool does, WHEN to use it, what a dataset_id looks like
#   (b) "resources": the first 5 items of d["resources"], keeping title, format and url
@mcp.tool()
def get_dataset(dataset_id: str) -> dict:
    """TODO (a): describe this tool for a model."""
    d = get_json(f"{DATA_GOUV}/datasets/{dataset_id}/")
    return {
        "id": d["id"],
        "title": d["title"],
        "organization": (d.get("organization") or {}).get("name"),
        "license": d.get("license"),
        "frequency": d.get("frequency"),
        "description": (d.get("description_short") or d.get("description") or "")[:300],
        "page": d.get("page"),
        "resources": [],  # TODO (b)
    }


# TODO 2: finish wiki_summary. The code is given; you fill in two things:
#   (a) the docstring: titles are exact ("Albert_School", not "albert school"): say so
#   (b) "page": the page URL, found at d["content_urls"]["desktop"]["page"]
@mcp.tool()
def wiki_summary(title: str) -> dict:
    """TODO (a): describe this tool for a model."""
    d = get_json(f"{WIKI}/page/summary/{title.replace(' ', '_')}")
    return {"title": d["title"], "extract": d["extract"], "page": ""}  # TODO (b)

# TODO 3: search_wiki(query: str) -> list[dict]
#   Titles are exact, so a model needs a way to find them. GET "https://fr.wikipedia.org/w/api.php"
#   with action="query", list="search", srsearch=query, srlimit=3, format="json".
#   The hits are in ["query"]["search"]; return title and wordcount for each.
#   Then ask a client: "Who founded Albert School?" and watch search_wiki then wiki_summary.

# TODO 4: add a limit to search_datasets (above)
#   Add `limit: int = 5`, pass it as page_size, and raise ToolError if it is not between 1 and 20.
#   Check the new schema in tools/list: what does the client now show for `limit`?


@mcp.resource("opendata://sources")
def sources() -> str:
    """The two public APIs this server wraps (a resource: data the client can read, no arguments)."""
    return ("data.gouv.fr: French open data catalogue, https://www.data.gouv.fr/api/1/\n"
            "French Wikipedia: page summaries, https://fr.wikipedia.org/api/rest_v1/")


# TODO 5: a resource template: dataset_card(dataset_id: str) -> str
#   Use @mcp.resource("opendata://dataset/{dataset_id}"): the {dataset_id} in the URI becomes the argument.
#   Build a short text card (title, publisher, licence, frequency, page URL) by calling your get_dataset.
#   Check `resources/templates/list` in the Inspector: how is it different from `opendata://sources`?


@mcp.prompt()
def describe_dataset(topic: str) -> str:
    """A reusable prompt the user picks from the client (a prompt: a message template with arguments)."""
    return (f"Search data.gouv.fr for datasets about '{topic}'. Pick the most relevant one, then tell me "
            "who publishes it, its licence, how often it is updated, and one thing it could be used for.")


@mcp.prompt()
def compare_datasets(topic_a: str, topic_b: str) -> str:
    """Compare two topics: two arguments, so the client shows a small form."""
    return (f"Find one data.gouv.fr dataset about '{topic_a}' and one about '{topic_b}'. "
            "Compare them in a table: publisher, licence, update frequency, last update.")


@mcp.prompt()
def fact_check(claim: str) -> str:
    """Ask the model to check a claim against both sources, and say when neither can."""
    return (f"Check this claim using Wikipedia and data.gouv.fr: {claim}\n"
            "Say which source supports it, quote the evidence, and answer 'cannot verify' if neither does.")


# TODO 6: write your own prompt, e.g. explain_for_a_student(topic: str) -> str
#   A prompt returns the text of a message. Make one you would really reuse, then run it from a client.


if __name__ == "__main__":
    mcp.run("streamable-http")       # 127.0.0.1:8000/mcp
