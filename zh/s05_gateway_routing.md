# 第 05 节: 网关与路由

> 一张绑定表将 (channel, peer) 映射到 agent_id. 最具体的匹配优先.

## 架构

```
    Inbound Message (channel, account_id, peer_id, text)
           |
    +------v------+     +----------+
    |   Gateway    | <-- | WS/REPL  |  JSON-RPC 2.0
    +------+------+     +----------+
           |
    +------v------+
    | BindingTable |  5-tier resolution:
    +------+------+    T1: peer_id     (most specific)
           |           T2: guild_id
           |           T3: account_id
           |           T4: channel
           |           T5: default     (least specific)
           |
     (agent_id, binding)
           |
    +------v---------+
    | build_session_key() |  dm_scope controls isolation
    +------+---------+
           |
    +------v------+
    | AgentManager |  per-agent config / personality / sessions
    +------+------+
           |
        LLM API
```

## 本节要点

- **BindingTable**: 排序的路由绑定列表. 从 tier 1 到 tier 5 遍历, 首次匹配即返回.
- **build_session_key()**: `dm_scope` 控制会话隔离 (每用户、每通道等).
- **AgentManager**: 多 agent 注册中心 -- 每个 agent 有自己的性格和模型.
- **GatewayServer**: 可选的 WebSocket 服务器, 使用 JSON-RPC 2.0 协议.
- **共享事件循环**: daemon 线程中的 asyncio 循环, 信号量限制并发数为 4.

## 核心代码走读

### 1. BindingTable.resolve() -- 路由核心

绑定按 `(tier, -priority)` 排序. 解析时线性遍历, 首次匹配即返回.

```python
@dataclass
class Binding:
    agent_id: str
    tier: int           # 1-5, 越小越具体
    match_key: str      # "peer_id" | "guild_id" | "account_id" | "channel" | "default"
    match_value: str    # e.g. "telegram:12345", "discord", "*"
    priority: int = 0   # 同一 tier 内, 越高越优先

class BindingTable:
    def resolve(self, channel="", account_id="",
                guild_id="", peer_id="") -> tuple[str | None, Binding | None]:
        for b in self._bindings:
            if b.tier == 1 and b.match_key == "peer_id":
                if ":" in b.match_value:
                    if b.match_value == f"{channel}:{peer_id}":
                        return b.agent_id, b
                elif b.match_value == peer_id:
                    return b.agent_id, b
            elif b.tier == 2 and b.match_key == "guild_id" and b.match_value == guild_id:
                return b.agent_id, b
            elif b.tier == 3 and b.match_key == "account_id" and b.match_value == account_id:
                return b.agent_id, b
            elif b.tier == 4 and b.match_key == "channel" and b.match_value == channel:
                return b.agent_id, b
            elif b.tier == 5 and b.match_key == "default":
                return b.agent_id, b
        return None, None
```

给定以下示例绑定:

```python
bt.add(Binding(agent_id="luna", tier=5, match_key="default", match_value="*"))
bt.add(Binding(agent_id="sage", tier=4, match_key="channel", match_value="telegram"))
bt.add(Binding(agent_id="sage", tier=1, match_key="peer_id",
               match_value="discord:admin-001", priority=10))
```

| 输入                              | Tier | Agent |
|-----------------------------------|------|-------|
| `channel=cli, peer=user1`         | 5    | Luna  |
| `channel=telegram, peer=user2`    | 4    | Sage  |
| `channel=discord, peer=admin-001` | 1    | Sage  |
| `channel=discord, peer=user3`     | 5    | Luna  |

### 2. 带 dm_scope 的会话 key

agent 解析完成后, agent 配置上的 `dm_scope` 控制会话隔离方式:

```python
def build_session_key(agent_id, channel="", account_id="",
                      peer_id="", dm_scope="per-peer"):
    aid = normalize_agent_id(agent_id)
    if dm_scope == "per-account-channel-peer" and peer_id:
        return f"agent:{aid}:{channel}:{account_id}:direct:{peer_id}"
    if dm_scope == "per-channel-peer" and peer_id:
        return f"agent:{aid}:{channel}:direct:{peer_id}"
    if dm_scope == "per-peer" and peer_id:
        return f"agent:{aid}:direct:{peer_id}"
    return f"agent:{aid}:main"
```

| dm_scope                   | Key 格式                                 | 效果                      |
|----------------------------|------------------------------------------|---------------------------|
| `main`                     | `agent:{id}:main`                        | 所有人共享一个会话        |
| `per-peer`                 | `agent:{id}:direct:{peer}`               | 每个用户隔离              |
| `per-channel-peer`         | `agent:{id}:{ch}:direct:{peer}`          | 每个平台的不同会话        |
| `per-account-channel-peer` | `agent:{id}:{ch}:{acc}:direct:{peer}`    | 最大隔离度                |

### 3. AgentConfig -- 每个 agent 的性格

每个 agent 携带自己的配置. 系统提示词从配置生成:

```python
@dataclass
class AgentConfig:
    id: str
    name: str
    personality: str = ""
    model: str = ""              # 空 = 使用全局 MODEL_ID
    dm_scope: str = "per-peer"

    def system_prompt(self) -> str:
        parts = [f"You are {self.name}."]
        if self.personality:
            parts.append(f"Your personality: {self.personality}")
        parts.append("Answer questions helpfully and stay in character.")
        return " ".join(parts)
```

## 试一试

```sh
python zh/s05_gateway_routing.py

# 测试路由
# You > /bindings                      (查看所有路由绑定)
# You > /route cli user1               (通过 default 解析到 Luna)
# You > /route telegram user2           (通过 channel 绑定解析到 Sage)

# 强制指定 agent
# You > /switch sage
# You > Hello!                          (无论路由结果如何都和 Sage 对话)
# You > /switch off                     (恢复正常路由)

# 启动 WebSocket 网关
# You > /gateway
# Gateway running on ws://localhost:8765
```

## OpenClaw 中的对应实现

| 方面             | claw0 (本文件)                 | OpenClaw 生产代码                      |
|------------------|--------------------------------|----------------------------------------|
| 路由解析         | 5 层线性扫描                   | 相同的层级系统 + 配置文件              |
| 会话 key         | `dm_scope` 参数                | 相同的 dm_scope + 持久化会话           |
| 多 agent         | 内存中的 AgentConfig           | 每个 agent 独立的工作区目录            |
| 网关             | WebSocket + JSON-RPC 2.0       | 相同协议 + HTTP API                    |
| 并发控制         | `asyncio.Semaphore(4)`         | 相同的信号量模式                       |
