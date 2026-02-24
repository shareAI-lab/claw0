# s03: Sessions (会话持久化)

> "Conversations that survive restarts" -- 重启不丢上下文, 才是真正可用的 agent.

## At a Glance

```
                sessions.json (metadata index)
                     |
User --> agent_loop() --> SessionStore --> transcripts/
              |                             session_abc.jsonl
         load_session()                     session_def.jsonl
         save_turn()
              |
         session_key = agent:channel:peer
```

- **What we build**: JSONL 持久化层, 让对话历史在进程重启后完整恢复.
- **Core mechanism**: append-only 的 JSONL transcript 是 source of truth, `_rebuild_history()` 从中重建 API 格式的 messages.
- **Design pattern**: agent_loop 从自包含函数变为接收外部状态的纯函数 -- 加载历史、处理消息、保存结果都通过 SessionStore 完成.

## The Problem

1. **退出即失忆.** s01/s02 的所有对话历史只在内存中. 进程退出, 一切归零. 调试了半小时的 agent, 重启后什么都不记得.

2. **无法区分不同对话.** 一个 agent 需要同时维护多个独立会话 -- 不同用户、不同话题, 每个都应该有自己的历史, 互不干扰.

3. **工具调用结果丢失.** 如果模型之前读过一个文件, 重启后它不知道自己读过什么, 可能重复读取或给出矛盾的回答.

## How It Works

### 1. JSONL Transcript 格式

每个会话对应一个 `.jsonl` 文件, 每行一个 JSON 对象:

```
{"type":"session","id":"abc123","key":"main:cli:user","created":"2025-01-01T00:00:00Z"}
{"type":"user","content":"hello","ts":"2025-01-01T00:00:01Z"}
{"type":"assistant","content":"Hi there!","ts":"2025-01-01T00:00:02Z"}
{"type":"tool_use","name":"read_file","tool_use_id":"tu_001","input":{"path":"config.json"},"ts":"..."}
{"type":"tool_result","tool_use_id":"tu_001","output":"{\"key\": \"value\"}","ts":"..."}
```

JSONL 是 append-only: 只追加, 不修改. 进程崩溃最多丢最后一行, 可以被 `tail -f` 实时监控, 不需要加锁.

**JSONL 是 source of truth, sessions.json 只是索引.**

### 2. SessionStore 创建和加载会话

创建会话时生成唯一 session_id 和对应的 JSONL 文件; 加载时从 JSONL 重建 messages:

```python
def create_session(self, session_key: str) -> dict:
    session_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    transcript_file = f"{session_key.replace(':', '_')}_{session_id}.jsonl"
    metadata = {
        "session_key": session_key, "session_id": session_id,
        "created_at": now, "updated_at": now,
        "message_count": 0, "transcript_file": transcript_file,
    }
    self._index[session_key] = metadata
    self._save_index()
    self.append_transcript(session_key, {
        "type": "session", "id": session_id, "key": session_key, "created": now,
    })
    return metadata

def load_session(self, session_key: str) -> dict:
    if session_key not in self._index:
        metadata = self.create_session(session_key)
        return {"metadata": metadata, "history": []}
    metadata = self._index[session_key]
    history = self._rebuild_history(metadata["transcript_file"])
    return {"metadata": metadata, "history": history}
```

**load_session 自动创建不存在的会话, 调用者不需要区分 "新建" 和 "恢复".**

### 3. 核心: _rebuild_history()

从 JSONL 重建 Anthropic API 格式的 messages 列表. 难点在于 tool_use/tool_result 的重组 -- JSONL 中它们是独立的行, 但 API 要求 tool_use 在 assistant 消息中, tool_result 在 user 消息中:

```python
def _rebuild_history(self, transcript_file: str) -> list[dict]:
    messages: list[dict] = []
    pending_tool_uses: list[dict] = []

    for line in filepath.read_text(encoding="utf-8").strip().splitlines():
        entry = json.loads(line)
        entry_type = entry.get("type")

        if entry_type == "session":
            continue

        if entry_type == "user":
            if pending_tool_uses:
                messages.append({"role": "assistant", "content": pending_tool_uses})
                pending_tool_uses = []
            messages.append({"role": "user", "content": entry.get("content", "")})

        elif entry_type == "tool_use":
            pending_tool_uses.append({
                "type": "tool_use", "id": entry.get("tool_use_id", ""),
                "name": entry.get("name", ""), "input": entry.get("input", {}),
            })

        elif entry_type == "tool_result":
            if pending_tool_uses:
                messages.append({"role": "assistant", "content": pending_tool_uses})
                pending_tool_uses = []
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": entry.get("tool_use_id", ""),
                "content": entry.get("output", ""),
            }]})

    return messages
```

**pending_tool_uses 缓冲区是关键: 它把连续的 tool_use 行合并为一个 assistant 消息的 content 数组.**

### 4. agent_loop 变为纯函数

agent_loop 不再自己管理 messages, 而是从 SessionStore 加载, 处理完后保存:

```python
def agent_loop(user_input: str, session_key: str,
               session_store: SessionStore, client: Anthropic) -> str:
    session_data = session_store.load_session(session_key)
    messages = session_data["history"]
    messages.append({"role": "user", "content": user_input})

    all_assistant_blocks: list = []
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=4096,
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        all_assistant_blocks.extend(response.content)
        # ... 工具循环逻辑同 s02 ...

    session_store.save_turn(session_key, user_input, all_assistant_blocks)
    return final_text
```

**agent_loop 从 "拥有状态" 变为 "接收状态" -- 这是为多通道架构做准备.**

### 5. Session Key 格式

`<agent_id>:<channel>:<peer_id>`, 例如 `main:cli:user`. 三段式设计:

- **agent**: 支持多个 agent 实例
- **channel**: 同一用户在不同通道有不同会话
- **peer**: 同一通道的不同用户有不同会话

**这个格式在 s04 多通道架构中直接使用.**

## What Changed from s02

| Component | s02 | s03 |
|-----------|-----|-----|
| 历史存储 | 仅内存 (退出即丢) | JSONL 文件 (持久化) |
| agent_loop 签名 | 无参数, 内部管理 messages | 接收 session_key + session_store |
| 工具结果 | 只在内存 | 也写入 transcript |
| 多会话 | 不支持 | 通过 session_key 区分 |
| 会话命令 | 无 | /new, /sessions, /switch, /history, /delete |

**Key shift**: agent_loop 从自包含函数变为纯函数, 状态管理交给 SessionStore.

## Design Decisions

**为什么用 JSONL 而不是 SQLite?**

Append-only 写入天然适合消息日志, 崩溃不会损坏已有数据; 人类可读, 直接 `cat` 就能看; 无依赖, 不需要数据库驱动; 可以逐行处理, 不需要一次加载全部.

**为什么 sessions.json 和 JSONL 分开?**

sessions.json 是索引, JSONL 是内容. 列出所有会话只需读索引 (快), 恢复具体会话才读 JSONL (按需). 类似数据库的索引和数据分离.

**In production OpenClaw:** session key 格式为 `agent:<agentId>:<channel>:<peerKind>:<peerId>` (多了 peerKind 区分 direct/group/thread), JSONL 存储在 `~/.openclaw/agents/<agentId>/sessions/`, 恢复历史时根据模型上下文窗口大小自动截断老消息, 会话可以设置 TTL 自动过期.

## Try It

```sh
cd claw0
python agents/s03_sessions.py
```

试试这些操作:

- 和 agent 聊几句, 然后 `/quit` 退出
- 重新启动 -- 观察 "Restored: N previous turns" 提示, 上下文已恢复
- `/new` 创建新会话, `/sessions` 查看所有会话
- `/switch <key>` 切换回之前的会话, 验证上下文正确
- 查看 `workspace/.sessions/transcripts/` 下的 JSONL 文件
