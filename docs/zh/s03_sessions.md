# s03: Sessions (会话持久化)

> "Conversations that survive restarts" -- 重启不丢上下文, 才是真正可用的 agent.

## 问题

前两节的 agent 有一个致命缺陷: 所有对话历史只存在于内存中. 一旦进程退出, 一切归零. 下次启动, 模型不记得你之前说过什么.

这在开发环境中是不可接受的. 一个编程 agent 可能帮你调试了半小时, 有了丰富的上下文 -- 你去吃个饭回来, 一切都得重头开始.

更关键的是, 一个 agent 需要同时维护多个独立会话. 不同用户、不同通道、不同话题, 每个都应该有自己的对话历史, 互不干扰.

这就需要一个 **结构化的持久化层**: 将对话历史序列化到磁盘, 重启后能完整恢复.

## 解决方案

```
                  sessions.json (metadata index)
                       |
User --> Agent Loop --> SessionStore --> transcripts/
                       |                  session_abc.jsonl
                  load/save               session_def.jsonl
                       |
                 session_key = agent:channel:peer
```

持久化层由两部分组成:

1. **sessions.json** -- 元数据索引, 记录所有会话的摘要信息
2. **transcripts/** -- 每个会话一个 JSONL 文件, 是消息的完整记录 (source of truth)

Session key 的格式: `<agent_id>:<channel>:<peer_id>`, 例如 `main:cli:user`. 这个格式在下一节多通道架构中会发挥关键作用.

## 工作原理

### JSONL Transcript 格式

每个 `.jsonl` 文件是一个会话的完整记录, 每行一个 JSON 对象:

```md
{"type":"session","id":"abc123","key":"main:cli:user","created":"2025-01-01T00:00:00Z"}
{"type":"user","content":"hello","ts":"2025-01-01T00:00:01Z"}
{"type":"assistant","content":"Hi there! How can I help?","ts":"2025-01-01T00:00:02Z"}
{"type":"tool_use","name":"read_file","tool_use_id":"tu_001","input":{"path":"config.json"},"ts":"..."}
{"type":"tool_result","tool_use_id":"tu_001","output":"{\"key\": \"value\"}","ts":"..."}
```

关键设计: JSONL 是 **append-only** (只追加, 不修改). 这意味着:
- 写入操作是安全的 -- 即使进程崩溃, 最多丢最后一行
- 文件可以被外部工具 (如 `tail -f`) 实时监控
- 不需要加锁就能并发读写 (只要写入是原子的)

### SessionStore 初始化

创建 SessionStore 时, 加载 sessions.json 索引到内存:

```python
class SessionStore:
    def __init__(
        self,
        store_path: Path | None = None,
        transcript_dir: Path | None = None,
    ):
        self.store_path = store_path or SESSIONS_INDEX
        self.transcript_dir = transcript_dir or TRANSCRIPTS_DIR

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        self._index: dict[str, dict] = self._load_index()
```

### 创建会话

每个新会话生成唯一的 session_id 和对应的 JSONL 文件:

```python
def create_session(self, session_key: str) -> dict:
    session_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    transcript_file = f"{session_key.replace(':', '_')}_{session_id}.jsonl"

    metadata = {
        "session_key": session_key,
        "session_id": session_id,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
        "transcript_file": transcript_file,
    }

    self._index[session_key] = metadata
    self._save_index()

    # JSONL 的第一行是会话元数据
    self.append_transcript(session_key, {
        "type": "session",
        "id": session_id,
        "key": session_key,
        "created": now,
    })
    return metadata
```

### 追加消息到 Transcript

所有消息通过 `append_transcript()` 写入 JSONL, 每次只追加一行:

```python
def append_transcript(self, session_key: str, entry: dict) -> None:
    metadata = self._index.get(session_key)
    if not metadata:
        return
    filepath = self.transcript_dir / metadata["transcript_file"]
    line = json.dumps(entry, ensure_ascii=False)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line + "\n")
```

### 保存一轮对话

`save_turn()` 将用户消息和 assistant 回复的所有 content block 逐条写入 transcript:

```python
def save_turn(self, session_key: str, user_msg: str, assistant_blocks: list) -> None:
    now = datetime.now(timezone.utc).isoformat()

    self.append_transcript(session_key, {
        "type": "user",
        "content": user_msg,
        "ts": now,
    })

    for block in assistant_blocks:
        block_type = block.type if hasattr(block, "type") else block.get("type", "unknown")
        if block_type == "text":
            text_content = block.text if hasattr(block, "text") else block.get("text", "")
            self.append_transcript(session_key, {
                "type": "assistant",
                "content": text_content,
                "ts": now,
            })
        elif block_type == "tool_use":
            # ... 记录工具调用详情
```

### 核心恢复逻辑: _rebuild_history()

这是整个 SessionStore 最关键的方法. 从 JSONL 文件重建 Anthropic API 格式的 messages 列表:

```python
def _rebuild_history(self, transcript_file: str) -> list[dict]:
    filepath = self.transcript_dir / transcript_file
    if not filepath.exists():
        return []

    messages: list[dict] = []
    pending_tool_uses: list[dict] = []

    for line in filepath.read_text(encoding="utf-8").strip().splitlines():
        entry = json.loads(line)
        entry_type = entry.get("type")

        if entry_type == "session":
            continue  # 元数据行, 跳过

        if entry_type == "user":
            if pending_tool_uses:
                messages.append({"role": "assistant", "content": pending_tool_uses})
                pending_tool_uses = []
            messages.append({"role": "user", "content": entry.get("content", "")})

        elif entry_type == "assistant":
            if pending_tool_uses:
                messages.append({"role": "assistant", "content": pending_tool_uses})
                pending_tool_uses = []
            messages.append({"role": "assistant", "content": entry.get("content", "")})

        elif entry_type == "tool_use":
            pending_tool_uses.append({
                "type": "tool_use",
                "id": entry.get("tool_use_id", ""),
                "name": entry.get("name", ""),
                "input": entry.get("input", {}),
            })

        elif entry_type == "tool_result":
            if pending_tool_uses:
                messages.append({"role": "assistant", "content": pending_tool_uses})
                pending_tool_uses = []
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": entry.get("tool_use_id", ""),
                    "content": entry.get("output", ""),
                }],
            })

    if pending_tool_uses:
        messages.append({"role": "assistant", "content": pending_tool_uses})

    return messages
```

这里的难点在于 tool_use 和 tool_result 的重组. JSONL 中它们是独立的行, 但 Anthropic API 要求:
- tool_use block 必须在 assistant 消息的 content 数组中
- tool_result block 必须在 user 消息的 content 数组中

所以需要 `pending_tool_uses` 缓冲区来正确分组.

### Agent Loop 集成

agent_loop 函数现在接收 session_key 和 session_store 作为参数:

```python
def agent_loop(
    user_input: str,
    session_key: str,
    session_store: SessionStore,
    client: Anthropic,
) -> str:
    # 加载会话历史
    session_data = session_store.load_session(session_key)
    messages = session_data["history"]

    # 追加用户消息
    messages.append({"role": "user", "content": user_input})

    all_assistant_blocks: list = []

    # 工具调用循环 (和 s02 相同)
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=4096,
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        all_assistant_blocks.extend(response.content)
        # ... 工具处理逻辑同 s02 ...

    # 持久化本轮对话
    session_store.save_turn(session_key, user_input, all_assistant_blocks)
    return final_text
```

### 会话管理命令

REPL 主循环增加了会话管理命令:

| 命令 | 功能 |
|------|------|
| `/new` | 创建新会话 (用时间戳区分) |
| `/sessions` | 列出所有会话及其状态 |
| `/switch <key>` | 切换到指定会话 |
| `/history` | 显示当前会话的消息历史 |
| `/delete <key>` | 删除指定会话 |
| `/quit` | 退出 |

## 核心代码

整个持久化层的核心是 `_rebuild_history()` 方法 (来自 `agents/s03_sessions.py`, 第 373-465 行). 它是从磁盘恢复上下文的关键:

```python
def _rebuild_history(self, transcript_file: str) -> list[dict]:
    """从 JSONL 重建 Anthropic API 格式的 messages 列表.

    JSONL 是 source of truth, 不是 sessions.json.
    """
    messages: list[dict] = []
    pending_tool_uses: list[dict] = []

    for line in filepath.read_text(encoding="utf-8").strip().splitlines():
        entry = json.loads(line)
        entry_type = entry.get("type")

        if entry_type == "user":
            # 刷出 pending tool_use, 然后追加 user 消息
            ...
        elif entry_type == "tool_use":
            # 缓冲, 等待对应的 tool_result
            pending_tool_uses.append({...})
        elif entry_type == "tool_result":
            # 先刷出 tool_use 作为 assistant, 再追加 tool_result 作为 user
            ...

    return messages
```

核心思想: **JSONL 是平坦的逐行记录, _rebuild_history 负责将其重组为 API 要求的 user/assistant 交替格式**.

## 和上一节的区别

| 组件 | s02 | s03 |
|------|-----|-----|
| 历史存储 | 仅内存 (退出即丢) | JSONL 文件 (持久化) |
| agent_loop 签名 | 无参数, 内部管理 messages | 接收 session_key + session_store |
| 工具结果 | 只在内存 | 也写入 transcript |
| 多会话 | 不支持 | 通过 session_key 区分 |
| 会话命令 | 无 | /new, /sessions, /switch 等 |
| 元数据索引 | 无 | sessions.json |

核心改动: agent_loop 从 "自包含函数" 变为 "接收外部状态的纯函数" -- 加载历史、处理消息、保存结果都通过 SessionStore 完成.

## 设计解析

### 为什么用 JSONL 而不是 SQLite?

1. **Append-only 写入模式** -- JSONL 只追加, 天然适合消息日志. 即使进程崩溃也不会损坏已有数据
2. **人类可读** -- 直接用 `cat` 或 `tail -f` 就能查看, 调试友好
3. **无依赖** -- 不需要额外的数据库驱动
4. **流式友好** -- 可以逐行处理, 不需要一次加载全部内容

### 为什么 sessions.json 和 JSONL 分开?

sessions.json 是索引, JSONL 是内容. 类似于数据库的索引和数据分离:
- 列出所有会话时只需读 sessions.json (快)
- 恢复具体会话时才读对应的 JSONL (按需)

### Session key 格式的设计

`agent:channel:peer` 这个三段式格式有深意:
- **agent**: 支持多个 agent 实例, 各自独立的会话空间
- **channel**: 同一个用户在不同通道 (CLI / Telegram / Discord) 有不同会话
- **peer**: 同一个通道的不同用户有不同会话

这个格式在下一节多通道架构中会直接使用.

### OpenClaw 生产版本做了什么不同?

- **完整 session key**: 格式为 `agent:<agentId>:<channel>:<peerKind>:<peerId>`, 比教学版多了 peerKind (区分 direct/group/thread)
- **JSONL 路径**: 存储在 `~/.openclaw/agents/<agentId>/sessions/*.jsonl`
- **Token 感知**: 恢复历史时会根据模型上下文窗口大小自动截断老消息
- **会话标题**: 自动从对话内容生成会话标题
- **TTL 管理**: 会话可以设置过期时间, 自动清理

## 试一试

```sh
cd mini-claw
python agents/s03_sessions.py
```

可以尝试的操作:

1. 和 agent 聊几句, 然后用 `/quit` 退出
2. 重新启动 -- 观察 "Restored: N previous turns" 提示, 之前的上下文已恢复
3. `/new` 创建一个新会话, 聊不同的话题
4. `/sessions` 查看所有会话
5. `/switch <key>` 切换回之前的会话, 验证上下文是否正确
6. `/history` 查看当前会话的完整消息历史
7. 查看 `workspace/.sessions/transcripts/` 目录下的 JSONL 文件, 了解持久化格式
