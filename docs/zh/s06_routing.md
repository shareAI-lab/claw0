# s06: Message Routing & Bindings (消息路由与绑定)

> 每条消息都能找到它的归属 -- 路由器根据来源信息, 决定哪个 Agent 来处理, 用哪个会话来隔离上下文。

## 问题

s05 的网关只有一个 Agent。但现实场景中, 你可能需要:

1. **多个不同性格的 Agent**: 一个擅长创意写作, 一个擅长技术问题, 一个负责日常对话。
2. **按来源分配**: Telegram 上的消息给 Agent A, Discord 上的给 Agent B, 特定用户给专属 Agent。
3. **会话隔离**: 不同用户的对话互不干扰, 同一用户在不同通道的对话可以独立或共享。

没有路由系统, 你需要为每个 Agent 部署一个独立的网关 -- 维护成本随 Agent 数量线性增长。路由系统让一个网关同时服务多个 Agent, 并自动决定每条消息的归属。

## 解决方案

```
  Inbound Message
  {channel:"telegram", sender:"user123", kind:"direct"}
       |
       v
  +--- MessageRouter.resolve() ----------------+
  |                                             |
  |  Bindings (priority order):                 |
  |  [P4] peer_id:user123           -> alice    |
  |  [P3] guild_id:server1          -> bob      |
  |  [P2] account_id:bot-account    -> main     |
  |  [P1] channel:telegram          -> main     |
  |  [P0] default                   -> main     |
  |                                             |
  +---------------------------------------------+
       |
       v
  AgentConfig(id="alice") + session_key
       |
       v
  run_agent(config, session_key, message)
```

核心思路: 用 Binding (绑定规则) 定义 "什么条件匹配什么 Agent", 按 priority 从高到低尝试匹配, 第一个命中的就是目标 Agent。同时根据 dm_scope 配置自动构建 session key, 控制会话隔离粒度。

## 工作原理

### 1. AgentConfig -- Agent 配置

每个 Agent 有独立的 model、system_prompt 和 tools 配置:

```python
@dataclass
class AgentConfig:
    id: str
    model: str
    system_prompt: str
    tools: list[dict] = field(default_factory=list)
```

通过不同的 system_prompt, 同一个底层模型可以表现出完全不同的 "性格":

```python
DEFAULT_CONFIG = {
    "agents": [
        {
            "id": "main",
            "model": MODEL_ID,
            "system_prompt": "You are a helpful assistant.",
        },
        {
            "id": "alice",
            "model": MODEL_ID,
            "system_prompt": (
                "You are Alice, a creative writing assistant. "
                "You speak in a literary, poetic style and help with creative writing tasks."
            ),
        },
        {
            "id": "bob",
            "model": MODEL_ID,
            "system_prompt": (
                "You are Bob, a technical assistant. "
                "You are precise and methodical, focusing on code and engineering topics."
            ),
        },
    ],
    # ...
}
```

### 2. Binding -- 路由绑定规则

Binding 是路由系统的核心数据结构。每个 Binding 定义一组匹配条件和目标 Agent:

```python
@dataclass
class Binding:
    channel: str | None = None       # 通道: telegram, discord, slack...
    account_id: str | None = None    # 账号 ID
    peer_id: str | None = None       # 发送者 ID
    peer_kind: str | None = None     # "direct" 或 "group"
    guild_id: str | None = None      # 群组/服务器 ID
    agent_id: str = "main"           # 目标 Agent
    priority: int = 0                # 优先级, 越高越优先
```

字段为 `None` 表示 "不关心此维度" (通配)。优先级体现了匹配的具体程度:

- **P4 (peer)**: 最具体 -- 精确到某个用户
- **P3 (guild)**: 群组级别
- **P2 (account)**: 账号级别
- **P1 (channel)**: 通道级别
- **P0 (default)**: 兜底默认

### 3. MessageRouter -- 路由解析

路由器按 priority 降序尝试匹配每个 Binding:

```python
class MessageRouter:
    def __init__(self, agents, bindings, default_agent="main", dm_scope="per-peer"):
        self.agents = agents
        # 按 priority 降序排列, 高优先级先匹配
        self.bindings = sorted(bindings, key=lambda b: b.priority, reverse=True)
        self.default_agent = default_agent
        self.dm_scope = dm_scope

    def resolve(self, channel, sender, peer_kind="direct",
                guild_id=None, account_id=None) -> tuple[AgentConfig, str]:
        matched_agent_id = self.default_agent

        for binding in self.bindings:
            if self._matches(binding, channel, sender, peer_kind, guild_id, account_id):
                matched_agent_id = binding.agent_id
                break

        agent = self.agents.get(matched_agent_id)
        if agent is None:
            agent = self.agents[self.default_agent]

        session_key = build_session_key(
            agent_id=agent.id, channel=channel,
            account_id=account_id or "default",
            peer_kind=peer_kind,
            peer_id=sender if peer_kind == "direct" else (guild_id or sender),
            dm_scope=self.dm_scope,
        )
        return agent, session_key
```

匹配逻辑: 每个非空条件都必须匹配, 全部条件满足才算命中:

```python
def _matches(self, binding, channel, sender, peer_kind, guild_id, account_id) -> bool:
    if binding.channel and binding.channel.lower() != channel.lower():
        return False
    if binding.account_id and binding.account_id.lower() != (account_id or "").lower():
        return False
    if binding.guild_id and binding.guild_id.lower() != (guild_id or "").lower():
        return False
    if binding.peer_id and binding.peer_id.lower() != sender.lower():
        return False
    if binding.peer_kind and binding.peer_kind.lower() != peer_kind.lower():
        return False
    return True
```

### 4. Session Key 构建

session key 决定了会话隔离的粒度。`dm_scope` 控制 DM (直接消息) 的隔离方式:

```python
def build_session_key(agent_id, channel, account_id, peer_kind, peer_id,
                      dm_scope="per-peer") -> str:
    agent_id = agent_id.strip().lower()
    channel = channel.strip().lower()
    peer_id = peer_id.strip().lower()
    peer_kind = peer_kind.strip().lower() or "direct"

    # 群组消息总是按 channel + kind + peerId 隔离
    if peer_kind != "direct":
        return f"agent:{agent_id}:{channel}:{peer_kind}:{peer_id}"

    # DM 会话根据 scope 决定隔离粒度
    if dm_scope == "main":
        return f"agent:{agent_id}:main"
    elif dm_scope == "per-peer":
        return f"agent:{agent_id}:direct:{peer_id}"
    elif dm_scope == "per-channel-peer":
        return f"agent:{agent_id}:{channel}:direct:{peer_id}"
    else:
        return f"agent:{agent_id}:direct:{peer_id}"
```

三种 DM scope 对应不同场景:

| dm_scope | session key 示例 | 适用场景 |
|----------|-----------------|---------|
| `main` | `agent:alice:main` | 个人助手, 所有 DM 共享一个会话 |
| `per-peer` | `agent:alice:direct:user123` | 多用户机器人, 每个用户独立 |
| `per-channel-peer` | `agent:alice:telegram:direct:user123` | 最细粒度, 同一用户在不同通道独立 |

### 5. RoutingGateway -- 带路由的网关

在 s05 GatewayServer 基础上, 增加了路由相关的 RPC 方法:

```python
class RoutingGateway:
    def __init__(self, host, port, router, sessions, token=""):
        # ...
        self._methods = {
            "health": self._handle_health,
            "chat.send": self._handle_chat_send,
            "chat.history": self._handle_chat_history,
            "routing.resolve": self._handle_routing_resolve,   # 路由诊断
            "routing.bindings": self._handle_routing_bindings, # 查看绑定
            "sessions.list": self._handle_sessions_list,       # 列出会话
            "identify": self._handle_identify,                 # 客户端身份声明
        }
```

`identify` 方法让客户端声明自己的通道和身份信息:

```python
async def _handle_identify(self, client, params) -> dict:
    client.channel = params.get("channel", "websocket")
    client.sender = params.get("sender", client.client_id)
    client.peer_kind = params.get("peer_kind", "direct")
    client.guild_id = params.get("guild_id", "")
    client.account_id = params.get("account_id", "")
    return {"identified": True, "channel": client.channel, "sender": client.sender}
```

`chat.send` 使用客户端身份 (或 params 覆盖) 进行路由:

```python
async def _handle_chat_send(self, client, params) -> dict:
    text = params.get("text", "").strip()
    channel = params.get("channel", client.channel)
    sender = params.get("sender", client.sender)
    # ...

    # 路由解析
    agent_config, session_key = self.router.resolve(
        channel=channel, sender=sender,
        peer_kind=peer_kind, guild_id=guild_id, account_id=account_id,
    )

    # 调用对应的 Agent
    session = self.sessions.get_or_create(session_key, agent_id=agent_config.id)
    assistant_text = run_agent(agent_config, session, text)
    # ...
```

### 6. 配置加载

路由配置可以从 JSON 文件加载, 也可以使用内置默认值:

```python
def load_routing_config(config_path=None):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = DEFAULT_CONFIG

    agents = {}
    for a in raw.get("agents", []):
        cfg = AgentConfig(id=a["id"], model=a.get("model", MODEL_ID),
                          system_prompt=a.get("system_prompt", "..."))
        agents[cfg.id] = cfg

    bindings = []
    for b in raw.get("bindings", []):
        binding = Binding(channel=b.get("channel"), peer_id=b.get("peer_id"),
                          guild_id=b.get("guild_id"), agent_id=b.get("agent_id", "main"),
                          priority=b.get("priority", 0))
        bindings.append(binding)

    return agents, bindings, raw.get("default_agent", "main"), raw.get("dm_scope", "per-peer")
```

## 核心代码

路由解析的完整流程 -- 从入站消息到 Agent 调用:

```python
# 1. 初始化路由器
agents, bindings, default_agent, dm_scope = load_routing_config()
router = MessageRouter(agents, bindings, default_agent, dm_scope)

# 2. 解析消息归属
agent_config, session_key = router.resolve(
    channel="telegram",
    sender="user-alice-fan",
    peer_kind="direct",
)
# -> agent_config.id = "alice" (命中 peer_id 绑定)
# -> session_key = "agent:alice:direct:user-alice-fan"

# 3. 调用对应 Agent
session = sessions.get_or_create(session_key, agent_id=agent_config.id)
reply = run_agent(agent_config, session, user_text)
```

`run_agent` 根据 AgentConfig 使用不同的 model 和 system_prompt:

```python
def run_agent(agent: AgentConfig, session: SessionEntry, user_text: str) -> str:
    session.messages.append({"role": "user", "content": user_text})
    session.last_active = time.time()

    response = client.messages.create(
        model=agent.model,
        max_tokens=2048,
        system=agent.system_prompt,
        messages=session.messages,
    )

    assistant_text = ""
    for block in response.content:
        if block.type == "text":
            assistant_text += block.text

    session.messages.append({"role": "assistant", "content": assistant_text})
    return assistant_text
```

## 和上一节的区别

| 组件 | s05 | s06 |
|------|-----|-----|
| Agent | 单一 Agent, 固定 system_prompt | 多 Agent, 每个有独立配置 |
| 路由 | 无, 所有消息给同一个 Agent | Binding 优先级匹配, 自动分配 |
| 会话 key | 客户端手动指定 | 根据 dm_scope 自动构建 |
| 客户端身份 | 无 | `identify` 方法声明通道/发送者 |
| RPC 方法 | 4 个 (health/chat.send/chat.history/channels.status) | 7 个 (移除 channels.status, 增加 routing.resolve/routing.bindings/sessions.list/identify) |
| 配置 | 硬编码 | JSON 文件或默认配置, 支持运行时加载 |
| 运行模式 | 网关 / 测试客户端 | 网关 / 测试客户端 / REPL (本地路由调试) |

关键转变: 从 "一个网关一个 Agent" 变成 "一个网关多个 Agent, 路由自动分配"。消息不再 "找人", 而是 "路由器替你找人"。

## 设计解析

**为什么用优先级而不是规则链/决策树?**

优先级模型简单直观: 最具体的规则优先级最高, 兜底规则优先级最低。不需要复杂的条件组合逻辑, 运维人员一眼就能看懂哪条规则会生效。OpenClaw 生产版也采用同样的优先级排序方式。

**为什么 session key 不直接用 sender ID?**

同一个用户可能:
- 在 Telegram 上聊天, 也在 Discord 上聊天
- 在同一通道上和不同 Agent 对话
- 在不同群组中有不同上下文

session key 需要编码 agent_id + channel + peer 信息, 才能正确隔离这些场景。`dm_scope` 让管理员根据实际需求选择隔离粒度。

**OpenClaw 生产版的不同之处:**

- Binding 定义在 `src/routing/bindings.ts`, session key 构建在 `src/routing/session-key.ts`
- 支持 identity links: 把不同通道的同一用户关联起来 (Telegram 的 user123 和 Discord 的 user456 是同一个人)
- 支持团队级绑定: 一个绑定可以指向一组 Agent
- 路由解析在 `src/auto-reply/reply/route-reply.ts`, 考虑更多维度 (允许列表、静音列表、命令门控等)
- DM scope 还有 `per-account-channel-peer` 选项, 在多账号场景下使用

## 试一试

启动网关服务器:

```sh
cd mini-claw
python agents/s06_routing.py
```

在另一个终端运行测试客户端, 观察不同来源的消息被路由到不同 Agent:

```sh
python agents/s06_routing.py --test-client
```

或者用 REPL 模式本地调试路由逻辑 (不需要启动网关, 不调用 LLM):

```sh
python agents/s06_routing.py --repl
```

REPL 中可以输入:

```sh
route> telegram user-alice-fan
# -> Agent: alice, Session Key: agent:alice:direct:user-alice-fan

route> discord dev-person group dev-server
# -> Agent: bob, Session Key: agent:bob:discord:group:dev-server

route> slack someone
# -> Agent: main (default), Session Key: agent:main:direct:someone

route> bindings
# 列出所有绑定规则
```

使用自定义配置文件:

```sh
python agents/s06_routing.py --config my_routing.json
```
