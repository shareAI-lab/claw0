"""
Section 05 -- Gateway Server
"The switchboard"

OpenClaw 8 节课从零构建教程 -- 第 5 节: 网关服务器
============================================================

本节实现 OpenClaw 架构的核心组件: Gateway.
Gateway 是所有客户端与 Agent 之间的桥梁, 负责:
- 接受 WebSocket 连接 (实时双向通信)
- 处理 HTTP Webhook (外部系统回调)
- 使用 JSON-RPC 2.0 协议进行结构化消息传递
- 管理会话 (session) 和消息历史

真实的 OpenClaw 网关 (src/gateway/server.impl.ts) 支持数十种 RPC 方法
(health, chat.send, chat.history, channels.status, sessions.list, config.get 等),
并且通过 TLS, Bearer Token, 设备身份绑定等手段保护通信安全.

本节教学版实现 4 个核心方法, 演示网关的本质逻辑.

架构图:

  Browser    Mobile    CLI Client    Webhook
     |          |          |            |
     v          v          v            v
  +------------ WebSocket / HTTP -----------+
  |            GatewayServer                 |
  |  +-------------------------------------+ |
  |  | JSON-RPC 2.0 Method Router          | |
  |  |  chat.send   -> run_agent()         | |
  |  |  chat.history -> load_history()     | |
  |  |  channels.status -> get_channels()  | |
  |  |  health       -> ok                 | |
  |  +-------------------------------------+ |
  |          |                               |
  |    SessionStore  +  Agent Loop           |
  +------------------------------------------+

JSON-RPC 2.0 协议:
  请求:  {"jsonrpc":"2.0", "id":"req-1", "method":"chat.send", "params":{"text":"hello"}}
  响应:  {"jsonrpc":"2.0", "id":"req-1", "result":{"text":"...", "session_key":"..."}}
  事件:  {"jsonrpc":"2.0", "method":"event", "params":{"type":"chat.delta", "text":"h"}}

依赖: pip install anthropic python-dotenv websockets
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Any

import websockets
from websockets.asyncio.server import ServerConnection
from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 环境与配置
# ---------------------------------------------------------------------------

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")

# 网关配置
GATEWAY_HOST = os.getenv("GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT", "18789"))
# 简单 Bearer Token 认证, 留空则跳过认证
GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN", "")

SYSTEM_PROMPT = "You are a helpful assistant running inside an OpenClaw gateway."

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("gateway")

# ---------------------------------------------------------------------------
# Anthropic Client
# ---------------------------------------------------------------------------

_client_kwargs: dict[str, Any] = {"api_key": ANTHROPIC_API_KEY}
if ANTHROPIC_BASE_URL:
    _client_kwargs["base_url"] = ANTHROPIC_BASE_URL
client = Anthropic(**_client_kwargs)

# ---------------------------------------------------------------------------
# Session Store -- 内存会话存储
# ---------------------------------------------------------------------------
# 真实 OpenClaw 使用文件系统 (~/.openclaw/agents/<id>/sessions/*.jsonl)
# 这里用 dict 模拟, 演示 session key -> 消息列表 的映射关系


@dataclass
class SessionEntry:
    """一个会话的元数据和消息历史."""
    session_key: str
    messages: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class SessionStore:
    """内存会话存储, 按 session_key 索引."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionEntry] = {}

    def get_or_create(self, session_key: str) -> SessionEntry:
        if session_key not in self._sessions:
            self._sessions[session_key] = SessionEntry(session_key=session_key)
            log.info("session created: %s", session_key)
        return self._sessions[session_key]

    def get_history(self, session_key: str) -> list[dict[str, str]]:
        entry = self._sessions.get(session_key)
        if entry is None:
            return []
        return list(entry.messages)

    def list_sessions(self) -> list[str]:
        return list(self._sessions.keys())


# ---------------------------------------------------------------------------
# Agent Runner -- 调用 LLM 生成回复
# ---------------------------------------------------------------------------

def run_agent(session: SessionEntry, user_text: str) -> str:
    """
    调用 Anthropic API 生成回复.
    将用户消息和助手回复都追加到 session 中, 实现多轮对话.
    """
    session.messages.append({"role": "user", "content": user_text})
    session.last_active = time.time()

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=session.messages,
    )

    # 提取文本回复
    assistant_text = ""
    for block in response.content:
        if block.type == "text":
            assistant_text += block.text

    session.messages.append({"role": "assistant", "content": assistant_text})
    return assistant_text


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 Protocol Helpers
# ---------------------------------------------------------------------------
# OpenClaw 网关使用 JSON-RPC 2.0 作为所有 WebSocket 通信的协议层.
# 每条消息都是一个 JSON 对象, 包含 jsonrpc, id (可选), method, params/result/error.

JSONRPC_VERSION = "2.0"


def make_result(req_id: str | int | None, result: Any) -> str:
    """构造 JSON-RPC 成功响应."""
    return json.dumps({
        "jsonrpc": JSONRPC_VERSION,
        "id": req_id,
        "result": result,
    })


def make_error(req_id: str | int | None, code: int, message: str, data: Any = None) -> str:
    """构造 JSON-RPC 错误响应."""
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return json.dumps({
        "jsonrpc": JSONRPC_VERSION,
        "id": req_id,
        "error": err,
    })


def make_event(event_type: str, payload: dict[str, Any]) -> str:
    """构造 JSON-RPC 事件通知 (无 id, 服务端主动推送)."""
    return json.dumps({
        "jsonrpc": JSONRPC_VERSION,
        "method": "event",
        "params": {"type": event_type, **payload},
    })


# JSON-RPC 错误码 (遵循规范)
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
AUTH_ERROR = -32000  # 自定义: 认证失败

# ---------------------------------------------------------------------------
# Gateway Server
# ---------------------------------------------------------------------------
# 真实 OpenClaw 的网关 (src/gateway/server.impl.ts) 使用 Node.js ws 库,
# 支持 TLS, 设备配对, 多客户端广播, protocol version 协商等.
# 这里简化为核心的 WebSocket 服务 + 方法路由.


@dataclass
class ConnectedClient:
    """跟踪一个已连接的 WebSocket 客户端."""
    ws: ServerConnection
    client_id: str
    connected_at: float = field(default_factory=time.time)
    authenticated: bool = False


class GatewayServer:
    """
    WebSocket 网关服务器.
    接受连接, 解析 JSON-RPC 请求, 路由到对应的处理方法, 返回结果.
    """

    def __init__(self, host: str, port: int, token: str = "") -> None:
        self.host = host
        self.port = port
        self.token = token  # 空字符串 = 不需要认证
        self.sessions = SessionStore()
        self.clients: dict[str, ConnectedClient] = {}
        self._start_time = time.time()

        # 方法路由表: method name -> handler function
        # 真实 OpenClaw 在 server-methods-list.ts 中注册了近百个方法
        self._methods: dict[str, Any] = {
            "health": self._handle_health,
            "chat.send": self._handle_chat_send,
            "chat.history": self._handle_chat_history,
            "channels.status": self._handle_channels_status,
        }

    # -- 认证 ----------------------------------------------------------------

    def _authenticate(self, headers: Any) -> bool:
        """
        简单 Bearer Token 认证.
        真实 OpenClaw 支持 token, password, 设备身份, TLS 指纹等多种认证方式
        (参见 src/gateway/auth.ts).
        """
        if not self.token:
            # 未配置 token, 跳过认证
            return True

        auth_header = headers.get("Authorization", "")
        if not auth_header:
            return False

        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False

        return parts[1].strip() == self.token

    # -- WebSocket 处理 -------------------------------------------------------

    async def _handle_connection(self, ws: ServerConnection) -> None:
        """处理单个 WebSocket 连接的生命周期."""
        client_id = str(uuid.uuid4())[:8]

        # 认证检查
        authenticated = self._authenticate(ws.request.headers if ws.request else {})
        if not authenticated:
            error_msg = make_error(None, AUTH_ERROR, "Authentication failed")
            await ws.send(error_msg)
            await ws.close(4001, "Unauthorized")
            log.warning("client %s: auth failed, connection rejected", client_id)
            return

        client = ConnectedClient(ws=ws, client_id=client_id, authenticated=True)
        self.clients[client_id] = client
        log.info("client %s: connected (total: %d)", client_id, len(self.clients))

        # 发送欢迎事件 (类似 OpenClaw 的 connect.challenge 事件)
        welcome = make_event("connect.welcome", {
            "client_id": client_id,
            "server_time": time.time(),
        })
        await ws.send(welcome)

        try:
            async for raw_message in ws:
                if isinstance(raw_message, bytes):
                    raw_message = raw_message.decode("utf-8")
                await self._dispatch(client, raw_message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            del self.clients[client_id]
            log.info("client %s: disconnected (total: %d)", client_id, len(self.clients))

    async def _dispatch(self, client: ConnectedClient, raw: str) -> None:
        """
        解析 JSON-RPC 请求并路由到对应处理方法.
        这是网关的核心分发逻辑.
        """
        # 1. 解析 JSON
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            resp = make_error(None, PARSE_ERROR, "Parse error: invalid JSON")
            await client.ws.send(resp)
            return

        # 2. 校验 JSON-RPC 格式
        if not isinstance(msg, dict) or msg.get("jsonrpc") != JSONRPC_VERSION:
            resp = make_error(msg.get("id"), INVALID_REQUEST, "Invalid JSON-RPC request")
            await client.ws.send(resp)
            return

        req_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        log.info("client %s: -> %s (id=%s)", client.client_id, method, req_id)

        # 3. 查找方法处理器
        handler = self._methods.get(method)
        if handler is None:
            resp = make_error(req_id, METHOD_NOT_FOUND, f"Method not found: {method}")
            await client.ws.send(resp)
            return

        # 4. 执行方法, 发送结果
        try:
            result = await handler(client, params)
            resp = make_result(req_id, result)
        except Exception as exc:
            log.exception("client %s: method %s raised error", client.client_id, method)
            resp = make_error(req_id, INTERNAL_ERROR, str(exc))

        await client.ws.send(resp)

    # -- RPC 方法实现 ----------------------------------------------------------

    async def _handle_health(self, client: ConnectedClient, params: dict) -> dict:
        """
        health -- 服务器健康检查.
        真实 OpenClaw 返回更详细的信息: 版本号, 运行时间, 活跃会话数等.
        """
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "connected_clients": len(self.clients),
            "active_sessions": len(self.sessions.list_sessions()),
        }

    async def _handle_chat_send(self, client: ConnectedClient, params: dict) -> dict:
        """
        chat.send -- 发送消息给 Agent 并获取回复.
        这是最核心的方法, 对应 OpenClaw 中 src/gateway/server-chat.ts.
        """
        text = params.get("text", "").strip()
        if not text:
            raise ValueError("Parameter 'text' is required and must be non-empty")

        session_key = params.get("session_key", "default")

        # 在调用 LLM 之前, 发送 "typing" 事件通知客户端
        typing_event = make_event("chat.typing", {
            "session_key": session_key,
        })
        await client.ws.send(typing_event)

        # 调用 Agent
        session = self.sessions.get_or_create(session_key)
        assistant_text = run_agent(session, text)

        # 广播给所有连接到同一个 session 的客户端
        done_event = make_event("chat.done", {
            "session_key": session_key,
            "text": assistant_text,
        })
        await self._broadcast(done_event)

        return {
            "text": assistant_text,
            "session_key": session_key,
            "message_count": len(session.messages),
        }

    async def _handle_chat_history(self, client: ConnectedClient, params: dict) -> dict:
        """
        chat.history -- 获取会话的消息历史.
        对应 OpenClaw 中 sessions.preview 方法.
        """
        session_key = params.get("session_key", "default")
        limit = params.get("limit", 50)

        messages = self.sessions.get_history(session_key)
        # 截取最近的消息
        if len(messages) > limit:
            messages = messages[-limit:]

        return {
            "session_key": session_key,
            "messages": messages,
            "total": len(self.sessions.get_history(session_key)),
        }

    async def _handle_channels_status(self, client: ConnectedClient, params: dict) -> dict:
        """
        channels.status -- 返回各通道的状态.
        真实 OpenClaw 会探测每个通道 (Telegram, Discord, Slack...) 的连接状态.
        这里返回模拟数据.
        """
        return {
            "channels": [
                {"id": "websocket", "status": "connected", "clients": len(self.clients)},
                {"id": "http_webhook", "status": "listening"},
            ]
        }

    # -- 广播 -----------------------------------------------------------------

    async def _broadcast(self, message: str) -> None:
        """
        向所有已连接客户端广播消息.
        真实 OpenClaw 在 src/gateway/server-broadcast.ts 中实现,
        支持按 session_key / 客户端角色过滤广播目标.
        """
        if not self.clients:
            return

        tasks = []
        for c in self.clients.values():
            tasks.append(c.ws.send(message))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.warning("broadcast: failed to send to a client: %s", result)

    # -- HTTP Webhook 处理 ----------------------------------------------------

    async def _handle_http(self, path: str, body: dict) -> dict:
        """
        处理 HTTP Webhook 请求.
        真实 OpenClaw 的 HTTP 层 (src/gateway/server-http.ts) 处理:
        - /hook/wake: 唤醒心跳
        - /hook/agent: 外部触发 Agent 运行
        - /api/v1/chat/completions: OpenAI 兼容 API
        - Slack Events API 等
        """
        if path == "/health":
            return {"status": "ok"}

        if path == "/hook/agent":
            text = body.get("text", "")
            session_key = body.get("session_key", "webhook")
            if not text:
                return {"error": "Missing 'text' field"}

            session = self.sessions.get_or_create(session_key)
            reply = run_agent(session, text)
            return {"text": reply, "session_key": session_key}

        return {"error": f"Unknown path: {path}"}

    # -- 启动 -----------------------------------------------------------------

    async def start(self) -> None:
        """启动 WebSocket 服务器."""
        log.info("Gateway starting on ws://%s:%d", self.host, self.port)
        if self.token:
            log.info("Authentication: Bearer token required")
        else:
            log.info("Authentication: disabled (no GATEWAY_TOKEN set)")

        async with websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
        ):
            log.info("Gateway ready. Waiting for connections...")
            # 保持服务器运行
            await asyncio.Future()


# ---------------------------------------------------------------------------
# 测试客户端 -- 用于验证网关工作
# ---------------------------------------------------------------------------

async def test_client() -> None:
    """
    简易测试客户端: 连接到网关, 发送几个 JSON-RPC 请求, 打印结果.
    运行方式: 先启动网关 (python s05_gateway.py), 再在另一个终端运行测试客户端:
      python s05_gateway.py --test-client
    """
    uri = f"ws://{GATEWAY_HOST}:{GATEWAY_PORT}"
    headers = {}
    if GATEWAY_TOKEN:
        headers["Authorization"] = f"Bearer {GATEWAY_TOKEN}"

    print(f"[test-client] connecting to {uri} ...")

    async with websockets.connect(uri, additional_headers=headers) as ws:
        # 接收欢迎事件
        welcome = json.loads(await ws.recv())
        print(f"[test-client] welcome: {json.dumps(welcome, indent=2)}")

        # -- health ---
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": "h-1",
            "method": "health",
            "params": {},
        }))
        health_resp = json.loads(await ws.recv())
        print(f"[test-client] health: {json.dumps(health_resp, indent=2)}")

        # -- chat.send ---
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": "c-1",
            "method": "chat.send",
            "params": {"text": "What is a gateway in software architecture?", "session_key": "test"},
        }))

        # 可能收到 typing 事件 + done 事件 + result 响应, 按顺序读取
        while True:
            raw = await ws.recv()
            msg = json.loads(raw)
            # 事件没有 id, 而 result 响应有 id
            if msg.get("id") == "c-1":
                text = msg.get("result", {}).get("text", "")
                print(f"[test-client] chat.send result: {text[:200]}...")
                break
            else:
                event_type = msg.get("params", {}).get("type", "unknown")
                print(f"[test-client] event: {event_type}")

        # -- chat.history ---
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": "h-2",
            "method": "chat.history",
            "params": {"session_key": "test"},
        }))
        history_resp = json.loads(await ws.recv())
        msg_count = history_resp.get("result", {}).get("total", 0)
        print(f"[test-client] chat.history: {msg_count} messages in session")

        # -- channels.status ---
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": "s-1",
            "method": "channels.status",
            "params": {},
        }))
        status_resp = json.loads(await ws.recv())
        print(f"[test-client] channels.status: {json.dumps(status_resp.get('result', {}), indent=2)}")

        # -- unknown method (测试错误处理) ---
        await ws.send(json.dumps({
            "jsonrpc": "2.0",
            "id": "e-1",
            "method": "no.such.method",
            "params": {},
        }))
        err_resp = json.loads(await ws.recv())
        print(f"[test-client] error test: {json.dumps(err_resp.get('error', {}))}")

    print("[test-client] done")


# ---------------------------------------------------------------------------
# Main -- 启动网关或测试客户端
# ---------------------------------------------------------------------------

def main() -> None:
    """
    默认启动网关服务器.
    传入 --test-client 参数时启动测试客户端.
    """
    import sys
    if "--test-client" in sys.argv:
        asyncio.run(test_client())
    else:
        print("=" * 60)
        print("  OpenClaw Mini -- Section 05: Gateway Server")
        print("  The switchboard")
        print("=" * 60)
        print(f"  Host:  {GATEWAY_HOST}")
        print(f"  Port:  {GATEWAY_PORT}")
        print(f"  Model: {MODEL_ID}")
        print(f"  Auth:  {'Bearer token' if GATEWAY_TOKEN else 'disabled'}")
        print()
        print("  Test:  python s05_gateway.py --test-client")
        print("=" * 60)
        gateway = GatewayServer(
            host=GATEWAY_HOST,
            port=GATEWAY_PORT,
            token=GATEWAY_TOKEN,
        )
        asyncio.run(gateway.start())


if __name__ == "__main__":
    main()
