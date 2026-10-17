"""Optional web tools for s02's bash handler; no model or Parallel key needed."""

import argparse
import asyncio
import sys
from datetime import timedelta

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "claw0-parallel-search-example/1.0 (https://github.com/shareAI-lab/claw0)"
TIMEOUT = 60


async def call_tool(name: str, arguments: dict):
    """The SDK owns the MCP handshake, Streamable HTTP framing, and cleanup."""
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True,
    ) as http:
        async with streamable_http_client(MCP_URL, http_client=http) as (read, write, _):
            async with ClientSession(
                read, write, read_timeout_seconds=timedelta(seconds=TIMEOUT),
            ) as session:
                await session.initialize()
                return await session.call_tool(name, arguments)


async def run(name: str, arguments: dict) -> int:
    # Bound the whole operation, including connection setup and cleanup.
    async with asyncio.timeout(TIMEOUT):
        result = await call_tool(name, arguments)
    text = "\n".join(block.text for block in result.content if block.type == "text")
    print(text or result.model_dump_json())
    return 1 if result.isError else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", help="Reuse one UUID across related search/fetch calls.")
    commands = parser.add_subparsers(dest="command", required=True)
    search = commands.add_parser("search", help="Find web sources and excerpts.")
    search.add_argument("objective", help="Describe the information to find.")
    search.add_argument("--query", action="append", required=True, help="Keyword query; repeatable.")
    fetch = commands.add_parser("fetch", help="Read excerpts from specific URLs.")
    fetch.add_argument("urls", nargs="+", help="HTTP(S) URLs, up to 20 per call.")
    fetch.add_argument("--objective", help="What to find in the pages (up to 200 characters).")
    args = parser.parse_args()

    if args.command == "search":
        name = "web_search"
        arguments = {"objective": args.objective, "search_queries": args.query}
    else:
        name = "web_fetch"
        arguments = {"urls": args.urls}
        if args.objective:
            arguments["objective"] = args.objective
    if args.session_id:
        arguments["session_id"] = args.session_id

    try:
        return asyncio.run(run(name, arguments))
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except TimeoutError:
        print(f"Error: request timed out after {TIMEOUT} seconds.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
