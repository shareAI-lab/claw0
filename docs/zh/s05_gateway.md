# s05: Gateway Server (网关服务器)

> 网关是所有客户端与 Agent 之间的总线 -- 一个 WebSocket 服务, 用 JSON-RPC 2.0 统一所有通信。

## 问题

在前面的章节中, Agent 只能在终端里运行, 通过 `input()` 接收用户输入。这意味着:

1. **只有一个客户端能用**: 谁打开终端, 谁才能和 Agent 对话。
2. **没有网络协议**: 浏览器、手机 App、CLI 工具无法连接。
3. **没有结构化通信**: 消息是纯文本, 没有类型、ID、错误码等元数据。
4. **没有会话管理**: 关掉终端, 所有对话历史丢失。

一个真正可用的 AI 助手需要一个网络入口, 让任意客户端通过统一协议与 Agent 交互。这就是 Gateway (网关) 的作用。

## 解决方案

```
  Browser    Mobile    CLI Client    Webhook
     |          |          |            |
     v          v          v            v
  +------------ WebSocket / HTTP -----------+
  |            GatewayServer                 |
  |  +-------------------------------------+ |
  |  | JSON-RPC 2.0 Method Router          | |
  |  |  health       -> ok                 | |
  |  |  chat.send    -> run_agent()        | |
  |  |  chat.history -> load_history()     | |
  |  |  channels.status -> get_channels()  | |
  |  +-------------------------------------+ |
  |          |                               |
  |    SessionStore  +  Agent Loop           |
  +------------------------------------------+
```

核心思路: 用 WebSocket 提供实时双向通信, 用 JSON-RPC 2.0 定义请求/响应/事件三种消息类型, 用方法路由表把不同请求分发到对应处理函数。

## 工作原理

### 1. JSON-RPC 2.0 协议

所有 WebSocket 通信都遵循 JSON-RPC 2.0 规范, 有三种消息类型:

**请求** -- 客户端发起, 带 `id` (用于匹配响应):

```python
{"jsonrpc": "2.0", "id": "req-1", "method": "chat.send", "params": {"text": "hello"}}
```

**响应** -- 服务端返回, `id` 与请求匹配:

```python
{"jsonrpc": "2.0", "id": "req-1", "result": {"text": "...", "session_key": "..."}}
```

**事件** -- 服务端主动推送, 没有 `id`:

```python
{"jsonrpc": "2.0", "method": "event", "params": {"type": "chat.typing", "session_key": "..."}}
```

这三种消息类型的区分规则很简单: 有 `id` + `method` 是请求, 有 `id` + `result`/`error` 是响应, 只有 `method` 没有 `id` 是事件。

### 2. 连接生命周期

客户端通过 WebSocket 连接到网关后, 经历以下流程:

```python
async def _handle_connection(self, ws: ServerConnection) -> None:
    client_id = str(uuid.uuid4())[:8]

    # 认证检查
    authenticated = self._authenticate(ws.request.headers if ws.request else {})
    if not authenticated:
        error_msg = make_error(None, AUTH_ERROR, "Authentication failed")
        await ws.send(error_msg)
        await ws.close(4001, "Unauthorized")
        return

    client = ConnectedClient(ws=ws, client_id=client_id, authenticated=True)
    self.clients[client_id] = client

    # 发送欢迎事件
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
```

流程: 连接 -> 认证 -> 注册客户端 -> 发送欢迎事件 -> 进入消息循环 -> 断开时清理。

### 3. 消息分发

每条收到的消息都经过 `_dispatch` 方法处理:

```python
async def _dispatch(self, client: ConnectedClient, raw: str) -> None:
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
        resp = make_error(req_id, INTERNAL_ERROR, str(exc))

    await client.ws.send(resp)
```

四步: 解析 JSON -> 校验格式 -> 查路由表 -> 执行并返回。每一步失败都有对应的 JSON-RPC 错误码。

### 4. chat.send 流程

最核心的 RPC 方法 -- 发送消息给 Agent:

```python
async def _handle_chat_send(self, client: ConnectedClient, params: dict) -> dict:
    text = params.get("text", "").strip()
    if not text:
        raise ValueError("Parameter 'text' is required and must be non-empty")

    session_key = params.get("session_key", "default")

    # 在调用 LLM 之前, 发送 "typing" 事件通知客户端
    typing_event = make_event("chat.typing", {"session_key": session_key})
    await client.ws.send(typing_event)

    # 调用 Agent
    session = self.sessions.get_or_create(session_key)
    assistant_text = run_agent(session, text)

    # 广播给所有连接的客户端
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
```

时序: 收到文本 -> 发 typing 事件 -> 调用 LLM -> 发 done 事件 (广播) -> 返回 result 响应。

### 5. 会话存储

`SessionStore` 用 dict 管理所有会话, 按 `session_key` 索引:

```python
@dataclass
class SessionEntry:
    session_key: str
    messages: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionEntry] = {}

    def get_or_create(self, session_key: str) -> SessionEntry:
        if session_key not in self._sessions:
            self._sessions[session_key] = SessionEntry(session_key=session_key)
        return self._sessions[session_key]
```

### 6. 广播

网关可以向所有已连接客户端广播消息:

```python
async def _broadcast(self, message: str) -> None:
    if not self.clients:
        return
    tasks = []
    for c in self.clients.values():
        tasks.append(c.ws.send(message))
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

使用 `asyncio.gather` 并发发送, 单个客户端失败不影响其他客户端。

## 核心代码

方法路由表 -- 网关的本质就是一个 RPC 方法分发器:

```python
class GatewayServer:
    def __init__(self, host: str, port: int, token: str = "") -> None:
        self.host = host
        self.port = port
        self.token = token
        self.sessions = SessionStore()
        self.clients: dict[str, ConnectedClient] = {}

        # 方法路由表: method name -> handler function
        self._methods: dict[str, Any] = {
            "health": self._handle_health,
            "chat.send": self._handle_chat_send,
            "chat.history": self._handle_chat_history,
            "channels.status": self._handle_channels_status,
        }
```

JSON-RPC 消息构造三件套:

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

## 和上一节的区别

| 组件 | s04 | s05 |
|------|-----|-----|
| 入口 | `input()` 直接读终端 | WebSocket 网络连接 |
| 协议 | 纯文本, 无结构 | JSON-RPC 2.0 (请求/响应/事件) |
| 客户端 | 单一终端进程 | 多客户端并发 (ConnectedClient 跟踪) |
| 认证 | 无 | Bearer Token |
| 消息推送 | 无 | 广播 (broadcast) + 事件通知 |
| 会话管理 | 单一消息列表 | SessionStore (session_key -> 消息列表) |
| 错误处理 | Python 异常 | JSON-RPC 错误码 (PARSE_ERROR / METHOD_NOT_FOUND 等) |

关键转变: 从 "进程内函数调用" 变成 "网络协议通信"。Agent 不再直接和用户交互, 而是通过网关这个中间层, 以结构化的 RPC 协议与任意客户端通信。

## 设计解析

**为什么选 WebSocket 而不是 HTTP?**

HTTP 是请求-响应模式, 服务端不能主动推送。而 AI 对话需要:
- typing 事件 (告诉客户端 "正在思考")
- 流式输出 (token 逐个推送)
- 广播 (多客户端同步)

WebSocket 提供全双工通信, 服务端可以随时向客户端推送消息。

**为什么选 JSON-RPC 2.0?**

- 规范简单 (只有请求/响应/通知三种类型)
- 天然支持批量请求
- 有标准化的错误码体系
- 容易在任何语言中实现客户端

**OpenClaw 生产版的不同之处:**

- TLS 加密: 生产环境使用 `wss://` 而不是 `ws://`
- 设备配对: 客户端通过 QR 码或配对码绑定设备身份
- 多客户端广播: 按 session_key 和客户端角色过滤广播目标
- Protocol version 协商: 客户端和服务端协商支持的协议版本
- 近百个 RPC 方法: 本节实现 4 个, 生产版在 `src/gateway/server-methods-list.ts` 中注册了约 100 个
- HTTP 层: 额外支持 Webhook (`/hook/wake`, `/hook/agent`), OpenAI 兼容 API, Slack Events API 等

## 试一试

启动网关服务器:

```sh
cd mini-claw
python agents/s05_gateway.py
```

在另一个终端运行测试客户端:

```sh
python agents/s05_gateway.py --test-client
```

测试客户端会依次:

1. 接收 `connect.welcome` 事件
2. 调用 `health` 查看服务器状态
3. 调用 `chat.send` 发送消息并观察 typing/done 事件流
4. 调用 `chat.history` 查看会话历史
5. 调用 `channels.status` 查看通道状态
6. 发送一个不存在的方法, 观察错误响应

你也可以用 `wscat` 手动测试:

```sh
# 安装 wscat
npm install -g wscat

# 连接并发送 JSON-RPC 请求
wscat -c ws://127.0.0.1:18789
> {"jsonrpc":"2.0","id":"1","method":"health","params":{}}
```

如果设置了 `GATEWAY_TOKEN` 环境变量, 连接时需要加上认证头:

```sh
wscat -c ws://127.0.0.1:18789 -H "Authorization: Bearer your-token"
```
