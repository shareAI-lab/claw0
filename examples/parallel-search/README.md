[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# Optional example: web research through a tool

After [s02: Tool Use](../../sessions/en/s02_tool_use.md), try a tool that retrieves
current information. This CLI connects to [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
over Streamable HTTP. `search` finds sources and excerpts; `fetch` reads excerpts
from URLs. Search uses Fast mode. Both work without a Parallel account or API key.
The free anonymous service has lower rate limits; this example does not retry requests.

The ten lessons and their dependencies stay as they are. Install this example
separately, using Python 3.11+ in a virtual environment. Run commands from the
repository root:

```sh
python -m venv .venv
. .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r examples/parallel-search/requirements.txt

# Reuse this ID for related calls; create a new one for a new research task.
research_session=$(python -c "import uuid; print(uuid.uuid4())")
python examples/parallel-search/search.py --session-id "$research_session" search \
  "Find the latest stable Python release and its release date, using official sources" \
  --query "Python latest stable release"
python examples/parallel-search/search.py --session-id "$research_session" fetch \
  https://www.python.org/downloads/ --objective "Latest stable Python release and release date"
```

The session variable above uses POSIX shell syntax. In other shells, pass the same
UUID directly to `--session-id` for related calls. The option can also be omitted.
`search` accepts multiple `--query` arguments. `fetch` accepts up to 20 URLs and an
optional objective of up to 200 characters. Run `search.py --help` for the commands.
Outputs include source URLs and excerpts, so you can check the evidence yourself.
A server tool error returns status 1 with its message; connection failures and the
60-second overall timeout also return status 1. Ctrl+C cancels the call.

## Use it with the existing agent

Follow the root [Quick Start](../../README.md#quick-start) to install the lesson
dependencies and configure your model, using the same activated environment.
Then run `python sessions/en/s02_tool_use.py` and ask:

> Use bash to run `python examples/parallel-search/search.py search "Find the latest
> stable Python release using official sources" --query "Python latest stable release"`
> with a 90-second bash timeout. Read the returned sources, fetch a relevant URL with
> the same script's `fetch` command, and summarize the release with source links.

This reuses s02's existing `bash` handler and `tool_result` messages. It does not
register MCP tools directly with the model or add MCP support to later lessons.
The CLI itself needs no model key; the s02 agent still needs its configured model
credentials. Treat fetched pages as source material, and check their URLs and dates
before relying on an answer.

## How the example maps to s02

The CLI dispatches `search` to `web_search` and `fetch` to `web_fetch`. Instead of a
local handler reading a file, `ClientSession.call_tool()` sends arguments to a
remote tool. The MCP SDK handles initialization and transport; the script prints
the tool's text result for the caller to consume. Each invocation opens and closes
one connection. It sends no credentials and does not read or modify `.env`.

Run the offline caller checks with:

```sh
python -m unittest discover -s examples/parallel-search -p 'test_*.py' -v
```
