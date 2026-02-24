# s05: Gateway Server

> "The switchboard" -- A WebSocket service that unifies all communication through JSON-RPC 2.0.

## At a Glance

```
Browser     Mobile     CLI Client
   |           |           |
   v           v           v
+--------- WebSocket ---------+
|        GatewayServer         |
| +---------------------------+|
| | JSON-RPC 2.0 Router       ||
| |  health      -> status    ||
| |  chat.send   -> run_agent ||
| |  chat.history-> history   ||
| |  channels.status -> list  ||
| +---------------------------+|
|       |             |        |
| SessionStore   broadcast()   |
+------------------------------+
```

- **What we build**: A WebSocket gateway server that uses the JSON-RPC 2.0 protocol to let any client interact with the agent.
- **Core mechanism**: WebSocket full-duplex communication + JSON-RPC method routing table + event broadcasting.
- **Design pattern**: Connect -> authenticate -> message loop -> method dispatch -> result/event push.

## The Problem

1. **Only the terminal can use it.** The agents in s01-s04 interact through `input()`. Browsers, mobile apps, and remote CLI clients cannot connect.

2. **No structured protocol.** Messages are plain text with no type distinction, request IDs, or error codes. Clients cannot tell "reply" from "event" and cannot match requests to responses.

3. **No server-initiated push.** HTTP is request-response: the server cannot proactively notify a client that it is "thinking" or that another client sent a message. AI conversations need typing events, streaming output, and multi-client synchronization.

## How It Works

### 1. JSON-RPC 2.0 Protocol

All WebSocket communication follows JSON-RPC 2.0. Three message types:

```python
# Request -- client sends, includes id
{"jsonrpc": "2.0", "id": "req-1", "method": "chat.send", "params": {"text": "hello"}}

# Response -- server replies, id matches request
{"jsonrpc": "2.0", "id": "req-1", "result": {"text": "...", "session_key": "..."}}

# Event -- server pushes proactively, no id
{"jsonrpc": "2.0", "method": "event", "params": {"type": "chat.typing", "session_key": "..."}}
```

Identification rules: has `id` + `method` = request, has `id` + `result`/`error` = response, has `method` but no `id` = event.

**Three helper functions construct these three message types:**

```python
def make_result(req_id, result):
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})

def make_error(req_id, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": err})

def make_event(event_type, payload):
    return json.dumps({"jsonrpc": "2.0", "method": "event",
                        "params": {"type": event_type, **payload}})
```

### 2. Connection Lifecycle

After connecting, a client goes through: authenticate -> register -> welcome event -> message loop -> disconnect cleanup.

```python
async def _handle_connection(self, ws: ServerConnection) -> None:
    client_id = str(uuid.uuid4())[:8]

    authenticated = self._authenticate(ws.request.headers if ws.request else {})
    if not authenticated:
        await ws.send(make_error(None, AUTH_ERROR, "Authentication failed"))
        await ws.close(4001, "Unauthorized")
        return

    client = ConnectedClient(ws=ws, client_id=client_id, authenticated=True)
    self.clients[client_id] = client
    await ws.send(make_event("connect.welcome", {
        "client_id": client_id, "server_time": time.time(),
    }))

    try:
        async for raw_message in ws:
            await self._dispatch(client, raw_message)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        del self.clients[client_id]
```

**Authentication failure closes the connection immediately, without entering the message loop.**

### 3. Message Dispatch

Each message goes through four steps: parse JSON -> validate format -> look up route -> execute and return.

```python
async def _dispatch(self, client: ConnectedClient, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await client.ws.send(make_error(None, PARSE_ERROR, "invalid JSON"))
        return

    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        await client.ws.send(make_error(msg.get("id"), INVALID_REQUEST, "Invalid request"))
        return

    handler = self._methods.get(msg.get("method", ""))
    if handler is None:
        await client.ws.send(make_error(msg.get("id"), METHOD_NOT_FOUND, "Not found"))
        return

    try:
        result = await handler(client, msg.get("params", {}))
        await client.ws.send(make_result(msg.get("id"), result))
    except Exception as exc:
        await client.ws.send(make_error(msg.get("id"), INTERNAL_ERROR, str(exc)))
```

**Every failure step has a corresponding JSON-RPC error code: -32700 (parse), -32600 (format), -32601 (method), -32603 (internal).**

### 4. chat.send -- The Core RPC Method

Sends a message to the agent, pushes a typing event in between, and broadcasts a done event upon completion:

```python
async def _handle_chat_send(self, client: ConnectedClient, params: dict) -> dict:
    text = params.get("text", "").strip()
    session_key = params.get("session_key", "default")

    await client.ws.send(make_event("chat.typing", {"session_key": session_key}))

    session = self.sessions.get_or_create(session_key)
    assistant_text = run_agent(session, text)

    await self._broadcast(make_event("chat.done", {
        "session_key": session_key, "text": assistant_text,
    }))

    return {
        "text": assistant_text,
        "session_key": session_key,
        "message_count": len(session.messages),
    }
```

**Sequence: receive text -> typing event -> call LLM -> done event (broadcast) -> result response.**

### 5. Method Routing Table and Broadcasting

The gateway is essentially an RPC dispatcher plus the ability to broadcast to all connected clients:

```python
self._methods = {
    "health": self._handle_health,
    "chat.send": self._handle_chat_send,
    "chat.history": self._handle_chat_history,
    "channels.status": self._handle_channels_status,
}

async def _broadcast(self, message: str) -> None:
    tasks = [c.ws.send(message) for c in self.clients.values()]
    await asyncio.gather(*tasks, return_exceptions=True)
```

**asyncio.gather sends concurrently. A failure in one client does not affect the others.**

## What Changed from s04

| Component | s04 | s05 |
|-----------|-----|-----|
| Entry point | `input()` reads terminal | WebSocket network connections |
| Protocol | Plain text | JSON-RPC 2.0 (request/response/event) |
| Clients | Single terminal | Multiple concurrent clients (ConnectedClient) |
| Authentication | None | Bearer Token |
| Message push | None | broadcast + typing/done events |
| Error handling | Python exceptions | JSON-RPC error codes |

**Key shift**: From "in-process function calls" to "network protocol communication". The agent no longer interacts with users directly; instead it communicates with any client through the gateway using a structured RPC protocol.

## Design Decisions

**Why WebSocket instead of HTTP?**

HTTP is request-response: the server cannot push proactively. AI conversations need typing events, streaming output, and multi-client synchronization -- all requiring the server to send messages to clients on its own initiative. WebSocket provides full-duplex communication.

**Why JSON-RPC 2.0?**

The spec is minimal (three message types: request, response, notification). It has a standardized error code system, native support for batch requests, and any language can easily implement a client.

**In production OpenClaw:** Uses `wss://` (TLS encryption), supports device pairing (QR code / pairing code), filters broadcast targets by session_key and client role, and negotiates protocol versions between client and server. The method routing table scales from 4 methods in this teaching version to nearly 100.

## Try It

Start the gateway:

```sh
cd claw0
python agents/s05_gateway.py
```

In another terminal, run the test client:

```sh
python agents/s05_gateway.py --test-client
```

The test client will sequentially: receive the welcome event -> call health -> call chat.send (observe typing/done events) -> call chat.history -> call channels.status -> send an unknown method (observe the error).

You can also test manually with wscat:

```sh
wscat -c ws://127.0.0.1:18789
> {"jsonrpc":"2.0","id":"1","method":"health","params":{}}
```

If `GATEWAY_TOKEN` is set, the connection requires an authentication header:

```sh
wscat -c ws://127.0.0.1:18789 -H "Authorization: Bearer your-token"
```
