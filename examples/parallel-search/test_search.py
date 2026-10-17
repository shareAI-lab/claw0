"""Exercise the CLI through the real MCP SDK and an in-memory HTTP server."""

import asyncio
import contextlib
import io
import json
import unittest
from unittest.mock import patch

import httpx
import search

AsyncClient = httpx.AsyncClient


class CallerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.clients = []
        self.result = {"content": [{"type": "text", "text": 'Source: https://www.python.org/\nPython release details'}]}
        self.failure = None
        self.waiting = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def handle(self, request):
        payload = json.loads(request.content)
        self.requests.append((request, payload))
        method = payload["method"]
        if method == "initialize":
            result = {"protocolVersion": payload["params"]["protocolVersion"],
                      "capabilities": {}, "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "notifications/initialized":
            return httpx.Response(202)
        elif method == "tools/list":
            result = {"tools": [{"name": name, "inputSchema": {"type": "object"}}
                                for name in ("web_search", "web_fetch")]}
        elif method == "tools/call":
            if self.failure == "http":
                return httpx.Response(503, text="Service unavailable")
            if self.failure == "wait":
                self.waiting.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled.set()
            result = self.result
        else:
            raise AssertionError(f"Unexpected MCP method: {method}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    def client(self, **kwargs):
        client = AsyncClient(**kwargs, transport=httpx.MockTransport(self.handle))
        self.clients.append(client)
        return client

    async def invoke(self, name, arguments):
        output = io.StringIO()
        with patch.object(search.httpx, "AsyncClient", self.client), contextlib.redirect_stdout(output):
            code = await search.run(name, arguments)
        return code, output.getvalue()

    def assert_requests(self, name, arguments):
        self.assertEqual([p["method"] for _, p in self.requests],
                         ["initialize", "notifications/initialized", "tools/call", "tools/list"])
        for request, _ in self.requests:
            self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
            self.assertEqual(request.headers["User-Agent"], search.USER_AGENT)
            self.assertNotIn("authorization", request.headers)
        self.assertEqual(self.requests[2][1]["params"]["name"], name)
        self.assertEqual(self.requests[2][1]["params"]["arguments"], arguments)
        self.assertTrue(all(c.is_closed for c in self.clients))

    async def test_search_output_and_anonymous_requests(self):
        arguments = {"objective": "Find Python release", "search_queries": ["Python latest release"],
                     "session_id": "b2258771-1401-4ba9-b3e1-d899ee06d23d"}
        code, output = await self.invoke("web_search", arguments)
        self.assertEqual(code, 0)
        self.assertIn("https://www.python.org/", output)
        self.assertIn("Python release details", output)
        self.assert_requests("web_search", arguments)

    async def test_fetch_output_and_arguments(self):
        arguments = {"urls": ["https://www.python.org/"], "objective": "Release details"}
        code, output = await self.invoke("web_fetch", arguments)
        self.assertEqual(code, 0)
        self.assertIn("Python release details", output)
        self.assert_requests("web_fetch", arguments)

    async def test_tool_error_is_visible_and_not_retried(self):
        self.result = {"isError": True, "content": [{"type": "text", "text": "Quota exceeded"}]}
        code, output = await self.invoke("web_search", {"objective": "x", "search_queries": ["x"]})
        self.assertEqual(code, 1)
        self.assertIn("Quota exceeded", output)
        self.assertEqual(sum(p["method"] == "tools/call" for _, p in self.requests), 1)

    async def test_http_error_closes_client(self):
        self.failure = "http"
        with self.assertRaises(Exception):
            await self.invoke("web_fetch", {"urls": ["https://www.python.org/"]})
        self.assertTrue(all(c.is_closed for c in self.clients))

    async def test_overall_timeout_cancels_request(self):
        self.failure = "wait"
        with patch.object(search, "TIMEOUT", 0.1), self.assertRaises(TimeoutError):
            await self.invoke("web_search", {"objective": "x", "search_queries": ["x"]})
        self.assertTrue(self.cancelled.is_set())
        self.assertTrue(all(c.is_closed for c in self.clients))

    async def test_cancellation_propagates_and_closes_client(self):
        self.failure = "wait"
        task = asyncio.create_task(self.invoke("web_fetch", {"urls": ["https://www.python.org/"]}))
        await asyncio.wait_for(self.waiting.wait(), timeout=2)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.cancelled.is_set())
        self.assertTrue(all(c.is_closed for c in self.clients))


class CommandTests(unittest.TestCase):
    def test_cli_selects_tools_and_arguments(self):
        cases = [
            (["--session-id", "task-id", "search", "Find releases", "--query", "Python release",
              "--query", "Python release dates"], "web_search",
             {"objective": "Find releases", "search_queries": ["Python release", "Python release dates"],
              "session_id": "task-id"}),
            (["fetch", "https://www.python.org/", "--objective", "Releases"], "web_fetch",
             {"urls": ["https://www.python.org/"], "objective": "Releases"}),
            (["fetch", "https://www.python.org/"], "web_fetch", {"urls": ["https://www.python.org/"]}),
        ]
        for argv, name, arguments in cases:
            with self.subTest(argv=argv), patch("sys.argv", ["search.py"] + argv), patch.object(search, "run", return_value=0) as run:
                self.assertEqual(search.main(), 0)
                run.assert_awaited_once_with(name, arguments)

    def test_cli_reports_timeout_and_interrupt(self):
        for error, code, message in [(TimeoutError(), 1, "timed out"), (KeyboardInterrupt(), 130, "Cancelled")]:
            with self.subTest(error=error), patch("sys.argv", ["search.py", "fetch", "https://www.python.org/"]), patch.object(search, "run", side_effect=error), contextlib.redirect_stderr(io.StringIO()) as err:
                self.assertEqual(search.main(), code)
                self.assertIn(message, err.getvalue())


if __name__ == "__main__":
    unittest.main()
