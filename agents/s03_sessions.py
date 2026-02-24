"""
s03_sessions.py  --  Session Persistence
"Conversations that survive restarts"

=== Part 3 of 8: Build OpenClaw from Scratch ===

    s01: agent loop (REPL + LLM)
    s02: tool use  (function calling)
 >> s03: session persistence (stateful across restarts) <<
    s04: multi-channel abstraction
    s05: ...

[KEY INSIGHT]
  OpenClaw 的会话 (session) 不是简单的 "聊天记录".
  它是一个 **结构化的持久化层**, 包含:
  - session key: 唯一标识一个会话 (格式: agent:channel:peer)
  - JSONL transcript: 只追加 (append-only) 的完整消息日志
  - session store: JSON 文件, 记录所有会话的元数据索引

  这意味着:
  1. 重启后, agent 能恢复之前的上下文
  2. 同一个 agent 可以同时维护多个独立会话
  3. 每条消息都被持久化, 可以回溯审计

[ARCHITECTURE]

                  sessions.json (metadata index)
                       |
User --> Agent Loop --> SessionStore --> transcripts/
                       |                  session_abc.jsonl
                  load/save               session_def.jsonl

[JSONL TRANSCRIPT FORMAT]
  每个 .jsonl 文件是一个会话的完整记录, 每行一个 JSON 对象:

  {"type":"session","id":"abc123","created":"2025-01-01T00:00:00Z"}
  {"type":"user","content":"hello","ts":"2025-01-01T00:00:01Z"}
  {"type":"assistant","content":"hi there","ts":"2025-01-01T00:00:02Z"}
  {"type":"tool_use","name":"read_file","input":{...},"ts":"..."}
  {"type":"tool_result","tool_use_id":"...","output":"...","ts":"..."}

[SESSION KEY FORMAT]
  在真正的 OpenClaw 中, session key 的格式是:
    agent:<agentId>:<channel>:<peerKind>:<peerId>
  例如: agent:main:telegram:direct:123456

  在这个教学版本中, 我们简化为:
    <agent_id>:<channel>:<peer_id>
  例如: main:cli:user

[COMMANDS]
  /new           -- 创建新会话
  /sessions      -- 列出所有会话
  /switch <key>  -- 切换到指定会话
  /history       -- 显示当前会话历史
  /quit          -- 退出

[TOOLS]
  本节的工具: read_file, list_directory, get_current_time
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.getenv("ANTHROPIC_BASE_URL", None)
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """\
You are a helpful assistant with access to local tools.
You can read files and list directories on the user's machine.
Keep answers concise. When using tools, explain what you found."""

# 会话数据存储在 workspace 目录下
WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
SESSIONS_DIR = WORKSPACE_DIR / ".sessions"
SESSIONS_INDEX = SESSIONS_DIR / "sessions.json"
TRANSCRIPTS_DIR = SESSIONS_DIR / "transcripts"

# 允许工具读取的目录 (安全边界)
ALLOWED_ROOT = WORKSPACE_DIR

# ---------------------------------------------------------------------------
# 工具定义 (继承自 s02)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file. "
            "Path must be within the workspace directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to workspace",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_directory",
        "description": (
            "List files and directories at the given path. "
            "Path must be within the workspace directory."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path relative to workspace (default: '.')",
                },
            },
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time in ISO format.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------


def _resolve_safe_path(relative: str) -> Path:
    """将相对路径解析为绝对路径, 并检查是否在允许范围内."""
    target = (ALLOWED_ROOT / relative).resolve()
    if not str(target).startswith(str(ALLOWED_ROOT.resolve())):
        raise PermissionError(f"Access denied: {relative} is outside workspace")
    return target


def execute_tool(name: str, tool_input: dict) -> str:
    """执行工具调用, 返回字符串结果."""
    if name == "read_file":
        path = _resolve_safe_path(tool_input["path"])
        if not path.is_file():
            return f"Error: file not found: {tool_input['path']}"
        content = path.read_text(encoding="utf-8", errors="replace")
        # 限制返回大小, 避免消耗过多 token
        if len(content) > 10000:
            return content[:10000] + "\n... [truncated]"
        return content

    if name == "list_directory":
        path = _resolve_safe_path(tool_input.get("path", "."))
        if not path.is_dir():
            return f"Error: not a directory: {tool_input.get('path', '.')}"
        entries = []
        for entry in sorted(path.iterdir()):
            # 跳过隐藏的会话目录
            if entry.name.startswith("."):
                continue
            kind = "dir" if entry.is_dir() else "file"
            size = entry.stat().st_size if entry.is_file() else 0
            entries.append(f"  [{kind}] {entry.name}" + (f"  ({size} bytes)" if size else ""))
        if not entries:
            return "(empty directory)"
        return "\n".join(entries)

    if name == "get_current_time":
        return datetime.now(timezone.utc).isoformat()

    return f"Error: unknown tool '{name}'"


# ---------------------------------------------------------------------------
# SessionStore -- 会话持久化的核心
# ---------------------------------------------------------------------------
#
# 设计要点 (对应真正的 OpenClaw):
#
# 1. sessions.json 是一个索引文件, 记录所有会话的元数据:
#    - session_key: 唯一标识
#    - created_at: 创建时间
#    - updated_at: 最后更新时间
#    - message_count: 消息计数
#    - transcript_file: 对应的 JSONL 文件名
#
# 2. transcripts/ 目录下, 每个会话一个 .jsonl 文件
#    - 只追加 (append-only), 不修改已有行
#    - 第一行是 session 元数据
#    - 后续每行是一条消息
#
# 3. 加载会话时, 从 JSONL 重建 messages 数组 (Anthropic API 格式)
#    - 这是关键: JSONL 是 source of truth, 不是 sessions.json


class SessionStore:
    """管理会话的持久化存储."""

    def __init__(
        self,
        store_path: Path | None = None,
        transcript_dir: Path | None = None,
    ):
        self.store_path = store_path or SESSIONS_INDEX
        self.transcript_dir = transcript_dir or TRANSCRIPTS_DIR

        # 确保目录存在
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        # 内存中的索引缓存
        self._index: dict[str, dict] = self._load_index()

    # -- 索引文件操作 --

    def _load_index(self) -> dict[str, dict]:
        """从 sessions.json 加载索引."""
        if not self.store_path.exists():
            return {}
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return {}

    def _save_index(self) -> None:
        """将索引写回 sessions.json."""
        self.store_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # -- 会话生命周期 --

    def create_session(self, session_key: str) -> dict:
        """创建一个新会话, 返回其元数据."""
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

        # 写入 JSONL 的第一行: 会话元数据
        self.append_transcript(session_key, {
            "type": "session",
            "id": session_id,
            "key": session_key,
            "created": now,
        })

        return metadata

    def load_session(self, session_key: str) -> dict:
        """加载会话, 返回 {metadata, history}.

        history 是 Anthropic API 格式的 messages 列表.
        如果会话不存在, 自动创建.
        """
        if session_key not in self._index:
            metadata = self.create_session(session_key)
            return {"metadata": metadata, "history": []}

        metadata = self._index[session_key]
        history = self._rebuild_history(metadata["transcript_file"])

        return {"metadata": metadata, "history": history}

    def save_turn(self, session_key: str, user_msg: str, assistant_blocks: list) -> None:
        """保存一轮对话 (用户消息 + 助手回复的所有 content block).

        assistant_blocks 是从 API 响应中提取的 content block 列表.
        一轮对话可能包含多个 block (text, tool_use, tool_result 等).
        """
        if session_key not in self._index:
            self.create_session(session_key)

        now = datetime.now(timezone.utc).isoformat()

        # 记录用户消息
        self.append_transcript(session_key, {
            "type": "user",
            "content": user_msg,
            "ts": now,
        })

        # 记录助手回复的每个 block
        for block in assistant_blocks:
            if hasattr(block, "type"):
                block_type = block.type
            elif isinstance(block, dict):
                block_type = block.get("type", "unknown")
            else:
                block_type = "unknown"

            if block_type == "text":
                text_content = block.text if hasattr(block, "text") else block.get("text", "")
                self.append_transcript(session_key, {
                    "type": "assistant",
                    "content": text_content,
                    "ts": now,
                })
            elif block_type == "tool_use":
                tool_name = block.name if hasattr(block, "name") else block.get("name", "")
                tool_input = block.input if hasattr(block, "input") else block.get("input", {})
                tool_id = block.id if hasattr(block, "id") else block.get("id", "")
                self.append_transcript(session_key, {
                    "type": "tool_use",
                    "name": tool_name,
                    "tool_use_id": tool_id,
                    "input": tool_input,
                    "ts": now,
                })

        # 更新索引元数据
        metadata = self._index[session_key]
        metadata["updated_at"] = now
        metadata["message_count"] = metadata.get("message_count", 0) + 1
        self._save_index()

    def save_tool_result(self, session_key: str, tool_use_id: str, output: str) -> None:
        """保存工具调用结果到 transcript."""
        now = datetime.now(timezone.utc).isoformat()
        self.append_transcript(session_key, {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "output": output,
            "ts": now,
        })

    # -- Transcript (JSONL) 操作 --

    def append_transcript(self, session_key: str, entry: dict) -> None:
        """向会话的 JSONL 文件追加一行."""
        metadata = self._index.get(session_key)
        if not metadata:
            return
        filepath = self.transcript_dir / metadata["transcript_file"]
        line = json.dumps(entry, ensure_ascii=False)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _rebuild_history(self, transcript_file: str) -> list[dict]:
        """从 JSONL 文件重建 Anthropic API 格式的 messages 列表.

        这是核心恢复逻辑:
        - type=user     -> {"role": "user", "content": "..."}
        - type=assistant -> {"role": "assistant", "content": "..."}
        - type=tool_use  -> 合并到上一个 assistant 消息的 content 中
        - type=tool_result -> {"role": "user", "content": [tool_result_block]}

        注意: Anthropic API 要求 tool_result 在 user 角色消息中返回.
        """
        filepath = self.transcript_dir / transcript_file
        if not filepath.exists():
            return []

        messages: list[dict] = []
        pending_tool_uses: list[dict] = []

        for line in filepath.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")

            if entry_type == "session":
                # 会话元数据行, 跳过
                continue

            if entry_type == "user":
                # 如果有未处理的 tool_use, 先刷出 assistant 消息
                if pending_tool_uses:
                    messages.append({
                        "role": "assistant",
                        "content": pending_tool_uses,
                    })
                    pending_tool_uses = []
                messages.append({
                    "role": "user",
                    "content": entry.get("content", ""),
                })

            elif entry_type == "assistant":
                # 如果有未处理的 tool_use, 先刷出
                if pending_tool_uses:
                    messages.append({
                        "role": "assistant",
                        "content": pending_tool_uses,
                    })
                    pending_tool_uses = []
                messages.append({
                    "role": "assistant",
                    "content": entry.get("content", ""),
                })

            elif entry_type == "tool_use":
                pending_tool_uses.append({
                    "type": "tool_use",
                    "id": entry.get("tool_use_id", ""),
                    "name": entry.get("name", ""),
                    "input": entry.get("input", {}),
                })

            elif entry_type == "tool_result":
                # tool_result 需要先刷出 pending tool_use 作为 assistant 消息
                if pending_tool_uses:
                    messages.append({
                        "role": "assistant",
                        "content": pending_tool_uses,
                    })
                    pending_tool_uses = []
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": entry.get("tool_use_id", ""),
                            "content": entry.get("output", ""),
                        }
                    ],
                })

        # 刷出最后的 pending tool_use
        if pending_tool_uses:
            messages.append({
                "role": "assistant",
                "content": pending_tool_uses,
            })

        return messages

    # -- 列举 / 查询 --

    def list_sessions(self) -> list[dict]:
        """返回所有会话的摘要列表, 按最后更新时间倒序."""
        sessions = list(self._index.values())
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def session_exists(self, session_key: str) -> bool:
        return session_key in self._index

    def delete_session(self, session_key: str) -> bool:
        """删除一个会话 (包括 transcript 文件)."""
        if session_key not in self._index:
            return False
        metadata = self._index.pop(session_key)
        self._save_index()
        filepath = self.transcript_dir / metadata["transcript_file"]
        if filepath.exists():
            filepath.unlink()
        return True


# ---------------------------------------------------------------------------
# Agent Loop (带会话持久化)
# ---------------------------------------------------------------------------
#
# 和 s02 的区别:
#   1. 每轮对话前, 从 SessionStore 加载历史
#   2. 每轮对话后, 保存到 SessionStore
#   3. 支持多会话切换
#   4. 工具调用结果也被持久化


def agent_loop(
    user_input: str,
    session_key: str,
    session_store: SessionStore,
    client: Anthropic,
) -> str:
    """处理一轮用户输入, 调用 LLM (含工具循环), 返回最终文本回复.

    完整流程:
    1. 从 session_store 加载历史 messages
    2. 追加本轮 user message
    3. 调用 LLM, 处理工具调用循环
    4. 保存本轮所有消息到 session_store
    5. 返回最终文本
    """
    # 加载会话历史
    session_data = session_store.load_session(session_key)
    messages = session_data["history"]

    # 追加用户消息
    messages.append({"role": "user", "content": user_input})

    # 本轮所有 assistant content blocks (用于持久化)
    all_assistant_blocks: list = []

    # 工具调用循环: LLM 可能多次请求工具
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # 收集本次响应的所有 content blocks
        all_assistant_blocks.extend(response.content)

        # 检查是否有工具调用
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "tool_use" and tool_use_blocks:
            # 将 assistant 消息加入 messages (供下一轮 API 调用)
            messages.append({"role": "assistant", "content": response.content})

            # 执行每个工具调用
            tool_results = []
            for tool_block in tool_use_blocks:
                print(f"  [tool] {tool_block.name}({json.dumps(tool_block.input, ensure_ascii=False)})")
                result = execute_tool(tool_block.name, tool_block.input)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result,
                })

                # 持久化工具结果
                session_store.save_tool_result(session_key, tool_block.id, result)

            # 将工具结果加入 messages
            messages.append({"role": "user", "content": tool_results})
            continue

        # 没有更多工具调用, 提取最终文本
        final_text = ""
        for block in response.content:
            if block.type == "text":
                final_text += block.text

        break

    # 持久化本轮对话
    session_store.save_turn(session_key, user_input, all_assistant_blocks)

    return final_text


# ---------------------------------------------------------------------------
# REPL 主循环
# ---------------------------------------------------------------------------
#
# 新增的会话管理命令:
#   /new           -- 创建并切换到新会话
#   /sessions      -- 列出所有会话
#   /switch <key>  -- 切换到指定会话
#   /history       -- 显示当前会话的消息历史
#   /delete <key>  -- 删除指定会话
#   /quit          -- 退出


def generate_session_key(agent_id: str = "main", channel: str = "cli", peer: str = "user") -> str:
    """生成 session key.

    格式: <agent_id>:<channel>:<peer>
    对应 OpenClaw 真实格式的简化版.
    """
    return f"{agent_id}:{channel}:{peer}"


def format_session_summary(meta: dict) -> str:
    """格式化会话摘要用于显示."""
    key = meta.get("session_key", "?")
    updated = meta.get("updated_at", "?")
    count = meta.get("message_count", 0)
    # 截取时间的日期部分用于紧凑显示
    if len(updated) > 19:
        updated = updated[:19]
    return f"  {key}  ({count} msgs, last: {updated})"


def print_session_history(session_store: SessionStore, session_key: str) -> None:
    """打印会话历史的可读版本."""
    session_data = session_store.load_session(session_key)
    messages = session_data["history"]
    if not messages:
        print("  (empty session)")
        return
    for msg in messages:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            # 截断长内容
            display = content[:200] + "..." if len(content) > 200 else content
            print(f"  [{role}] {display}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "?")
                    if btype == "tool_use":
                        print(f"  [{role}:tool_use] {block.get('name', '?')}(...)")
                    elif btype == "tool_result":
                        output = block.get("content", "")
                        display = output[:100] + "..." if len(output) > 100 else output
                        print(f"  [{role}:tool_result] {display}")
                    elif btype == "text":
                        text = block.get("text", "")
                        display = text[:200] + "..." if len(text) > 200 else text
                        print(f"  [{role}] {display}")


def main() -> None:
    if not API_KEY:
        print("Error: ANTHROPIC_API_KEY not set.")
        print("Copy .env.example to .env and fill in your key.")
        sys.exit(1)

    # 初始化 Anthropic 客户端
    client_kwargs = {"api_key": API_KEY}
    if BASE_URL:
        client_kwargs["base_url"] = BASE_URL
    client = Anthropic(**client_kwargs)

    # 初始化 SessionStore
    session_store = SessionStore()

    # 默认会话 key
    current_key = generate_session_key()

    # 如果已有这个会话, 恢复; 否则创建
    session_data = session_store.load_session(current_key)
    msg_count = session_data["metadata"].get("message_count", 0)

    print("=" * 60)
    print("  mini-claw s03: Session Persistence")
    print("  Model:", MODEL)
    print("  Session:", current_key)
    if msg_count > 0:
        print(f"  Restored: {msg_count} previous turns")
    print()
    print("  Commands:")
    print("    /new              Create a new session")
    print("    /sessions         List all sessions")
    print("    /switch <key>     Switch to session")
    print("    /history          Show current session history")
    print("    /delete <key>     Delete a session")
    print("    /quit             Exit")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input(f"[{current_key}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        # -- 命令处理 --

        if user_input == "/quit":
            print("Bye.")
            break

        if user_input == "/new":
            # 创建新会话, 用时间戳区分
            ts_suffix = datetime.now(timezone.utc).strftime("%H%M%S")
            current_key = generate_session_key(peer=f"user_{ts_suffix}")
            session_store.create_session(current_key)
            print(f"  New session: {current_key}")
            continue

        if user_input == "/sessions":
            sessions = session_store.list_sessions()
            if not sessions:
                print("  (no sessions)")
            else:
                print(f"  {len(sessions)} session(s):")
                for meta in sessions:
                    marker = " *" if meta["session_key"] == current_key else ""
                    print(format_session_summary(meta) + marker)
            continue

        if user_input.startswith("/switch"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("  Usage: /switch <session_key>")
                continue
            target_key = parts[1].strip()
            if not session_store.session_exists(target_key):
                print(f"  Session not found: {target_key}")
                print("  Use /sessions to list available sessions.")
                continue
            current_key = target_key
            meta = session_store.load_session(current_key)["metadata"]
            print(f"  Switched to: {current_key} ({meta.get('message_count', 0)} msgs)")
            continue

        if user_input == "/history":
            print(f"  Session: {current_key}")
            print_session_history(session_store, current_key)
            continue

        if user_input.startswith("/delete"):
            parts = user_input.split(maxsplit=1)
            if len(parts) < 2:
                print("  Usage: /delete <session_key>")
                continue
            target_key = parts[1].strip()
            if target_key == current_key:
                print("  Cannot delete the current session. Switch first.")
                continue
            if session_store.delete_session(target_key):
                print(f"  Deleted: {target_key}")
            else:
                print(f"  Session not found: {target_key}")
            continue

        if user_input.startswith("/"):
            print(f"  Unknown command: {user_input}")
            continue

        # -- 调用 agent loop --

        try:
            response = agent_loop(user_input, current_key, session_store, client)
            print()
            print(response)
            print()
        except Exception as exc:
            print(f"\n  Error: {exc}\n")


if __name__ == "__main__":
    main()
