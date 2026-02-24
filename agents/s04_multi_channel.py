"""
s04_multi_channel.py  --  Multi-Channel Abstraction
"Same brain, many mouths"

=== Part 4 of 8: Build OpenClaw from Scratch ===

    s01: agent loop (REPL + LLM)
    s02: tool use  (function calling)
    s03: session persistence
 >> s04: multi-channel abstraction <<
    s05: ...

[KEY INSIGHT]
  OpenClaw 和 Claude Code 最大的区别在于:
  Claude Code = 单通道 CLI 工具
  OpenClaw   = 多通道 AI 网关

  OpenClaw 的 gateway 同时连接 Telegram, Discord, Slack, WhatsApp, Signal,
  iMessage, IRC, Google Chat 等. 每个通道都是一个 "Channel plugin",
  实现统一的收发接口.

  这意味着:
  1. Agent 逻辑只写一次, 但可以通过任意通道交互
  2. 每个通道有自己的消息格式/长度限制/媒体能力
  3. 消息在进入 agent 前被标准化为统一的 InboundMessage
  4. 回复在发出前根据通道限制自动分块

[ARCHITECTURE]

  Telegram     Discord      CLI       File (webhook sim)
     |            |           |            |
     v            v           v            v
  +------------ ChannelRegistry -----------+
  |          receive() / send()            |
  +----------------------------------------+
                    |
              InboundMessage
                    |
           Agent Loop + SessionStore
                    |
              response text
                    |
  +----------------------------------------+
  |     channel.send(chunked text)         |
  +------------ ChannelRegistry -----------+
     |            |           |            |
     v            v           v            v
  Telegram     Discord      CLI       File

[CHANNEL PLUGIN INTERFACE]
  每个 Channel 必须实现:
  - id:              通道唯一标识 (如 "telegram", "cli")
  - max_text_length: 单条消息最大字符数
  - receive():       非阻塞轮询, 返回 InboundMessage 或 None
  - send():          发送文本 (自动分块)
  - chunk_text():    按通道限制拆分长文本

[MESSAGE NORMALIZATION]
  不同通道的消息格式各异, 但进入 agent 前统一为:
  InboundMessage(channel, sender, text, media_urls, thread_id, timestamp)

  这种标准化是 OpenClaw 能用同一个 agent 服务多个通道的关键.

[COMMANDS]
  /channels        -- 列出已注册通道
  /poll            -- 手动触发一次全通道轮询
  /send <ch> <msg> -- 通过指定通道发送消息 (测试用)
  /new             -- 创建新会话
  /sessions        -- 列出所有会话
  /history         -- 显示当前会话历史
  /quit            -- 退出
"""

import json
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

BASE_URL = os.getenv("ANTHROPIC_BASE_URL", None)
MODEL = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = """\
You are a helpful assistant with access to local tools.
You can read files and list directories on the user's machine.
Keep answers concise. When using tools, explain what you found."""

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
SESSIONS_DIR = WORKSPACE_DIR / ".sessions"
SESSIONS_INDEX = SESSIONS_DIR / "sessions.json"
TRANSCRIPTS_DIR = SESSIONS_DIR / "transcripts"
ALLOWED_ROOT = WORKSPACE_DIR

# FileChannel 的监视文件 (模拟 webhook 接收)
FILE_CHANNEL_INBOX = WORKSPACE_DIR / ".channels" / "file_inbox.txt"
FILE_CHANNEL_OUTBOX = WORKSPACE_DIR / ".channels" / "file_outbox.txt"

# MockTelegramChannel 的监视文件
MOCK_TELEGRAM_INBOX = WORKSPACE_DIR / ".channels" / "telegram_inbox.txt"
MOCK_TELEGRAM_OUTBOX = WORKSPACE_DIR / ".channels" / "telegram_outbox.txt"

# ---------------------------------------------------------------------------
# 工具定义 (继承自 s02/s03)
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
        if len(content) > 10000:
            return content[:10000] + "\n... [truncated]"
        return content

    if name == "list_directory":
        path = _resolve_safe_path(tool_input.get("path", "."))
        if not path.is_dir():
            return f"Error: not a directory: {tool_input.get('path', '.')}"
        entries = []
        for entry in sorted(path.iterdir()):
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
# SessionStore (从 s03 继承, 完整复制以保持 self-contained)
# ---------------------------------------------------------------------------


class SessionStore:
    """管理会话的持久化存储. 完整实现见 s03_sessions.py 的注释."""

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

    def _load_index(self) -> dict[str, dict]:
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
        self.store_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
        self.append_transcript(session_key, {
            "type": "session",
            "id": session_id,
            "key": session_key,
            "created": now,
        })
        return metadata

    def load_session(self, session_key: str) -> dict:
        if session_key not in self._index:
            metadata = self.create_session(session_key)
            return {"metadata": metadata, "history": []}
        metadata = self._index[session_key]
        history = self._rebuild_history(metadata["transcript_file"])
        return {"metadata": metadata, "history": history}

    def save_turn(self, session_key: str, user_msg: str, assistant_blocks: list) -> None:
        if session_key not in self._index:
            self.create_session(session_key)
        now = datetime.now(timezone.utc).isoformat()
        self.append_transcript(session_key, {
            "type": "user",
            "content": user_msg,
            "ts": now,
        })
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
                tool_input_data = block.input if hasattr(block, "input") else block.get("input", {})
                tool_id = block.id if hasattr(block, "id") else block.get("id", "")
                self.append_transcript(session_key, {
                    "type": "tool_use",
                    "name": tool_name,
                    "tool_use_id": tool_id,
                    "input": tool_input_data,
                    "ts": now,
                })
        metadata = self._index[session_key]
        metadata["updated_at"] = now
        metadata["message_count"] = metadata.get("message_count", 0) + 1
        self._save_index()

    def save_tool_result(self, session_key: str, tool_use_id: str, output: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.append_transcript(session_key, {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "output": output,
            "ts": now,
        })

    def append_transcript(self, session_key: str, entry: dict) -> None:
        metadata = self._index.get(session_key)
        if not metadata:
            return
        filepath = self.transcript_dir / metadata["transcript_file"]
        line = json.dumps(entry, ensure_ascii=False)
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _rebuild_history(self, transcript_file: str) -> list[dict]:
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
                continue
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

    def list_sessions(self) -> list[dict]:
        sessions = list(self._index.values())
        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions

    def session_exists(self, session_key: str) -> bool:
        return session_key in self._index


# ---------------------------------------------------------------------------
# InboundMessage -- 统一的入站消息格式
# ---------------------------------------------------------------------------
#
# 无论消息来自哪个通道, 进入 agent 之前都标准化为 InboundMessage.
# 这是 OpenClaw 多通道架构的核心抽象.
#
# 在真正的 OpenClaw 中, InboundMessage 还包含:
#   - quoted_text (引用回复)
#   - reactions
#   - metadata (通道特定的元数据)
# 这里简化为核心字段.


@dataclass
class InboundMessage:
    """标准化的入站消息."""
    channel: str           # 来源通道 ID (如 "cli", "telegram")
    sender: str            # 发送者标识 (如用户名, 用户 ID)
    text: str              # 消息文本
    media_urls: list[str] = field(default_factory=list)   # 附件 URL 列表
    thread_id: str | None = None                          # 线程/话题 ID
    timestamp: float = field(default_factory=time.time)   # Unix 时间戳


# ---------------------------------------------------------------------------
# Channel -- 通道抽象基类
# ---------------------------------------------------------------------------
#
# 每个通道插件必须实现这个接口.
# 在真正的 OpenClaw 中, 通道插件还需要:
#   - start() / stop() 生命周期方法
#   - 媒体上传/下载支持
#   - webhook / long-polling / WebSocket 连接管理
#   - 速率限制和重试
#
# 这里简化为最核心的 receive/send 接口.


class Channel(ABC):
    """通道插件的抽象基类."""

    @property
    @abstractmethod
    def id(self) -> str:
        """通道唯一标识, 如 'telegram', 'discord', 'cli'."""
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """人类可读的通道名称."""
        ...

    @property
    @abstractmethod
    def max_text_length(self) -> int:
        """单条消息的最大字符数. 超过此限制需要分块."""
        ...

    @abstractmethod
    def receive(self) -> InboundMessage | None:
        """非阻塞轮询: 如果有新消息, 返回 InboundMessage; 否则返回 None.

        这是一个关键设计: receive() 必须是非阻塞的,
        因为 gateway 需要在一个循环里轮询所有通道.
        阻塞任何一个通道会饿死其他通道.
        """
        ...

    @abstractmethod
    def send(self, text: str, media: list | None = None) -> None:
        """发送消息到通道. 长文本会被自动分块."""
        ...

    def chunk_text(self, text: str) -> list[str]:
        """将长文本按通道限制拆分为多段.

        默认策略: 按段落拆分, 尽量不在句子中间断开.
        通道可以覆盖此方法实现自定义拆分 (如 Telegram 的 Markdown 安全拆分).
        """
        max_len = self.max_text_length
        if len(text) <= max_len:
            return [text]

        chunks = []
        remaining = text

        while remaining:
            if len(remaining) <= max_len:
                chunks.append(remaining)
                break

            # 尝试在段落边界拆分
            cut = remaining[:max_len]
            split_pos = cut.rfind("\n\n")

            # 退而求其次: 在换行处拆分
            if split_pos < max_len // 4:
                split_pos = cut.rfind("\n")

            # 最后手段: 在空格处拆分
            if split_pos < max_len // 4:
                split_pos = cut.rfind(" ")

            # 实在找不到分割点, 硬切
            if split_pos < max_len // 4:
                split_pos = max_len

            chunks.append(remaining[:split_pos].rstrip())
            remaining = remaining[split_pos:].lstrip()

        return [c for c in chunks if c]  # 过滤空块


# ---------------------------------------------------------------------------
# CLIChannel -- 命令行通道
# ---------------------------------------------------------------------------
#
# 最简单的通道实现: 从 stdin 读取, 向 stdout 输出.
# 这就是 s01/s02/s03 一直在用的交互方式, 现在封装为一个 Channel plugin.
#
# 特殊之处: CLIChannel 的 receive() 会阻塞 (等待用户输入),
# 但在 gateway 模式下它只在被主动调用时才读取.


class CLIChannel(Channel):
    """命令行交互通道."""

    def __init__(self):
        self._pending: InboundMessage | None = None

    @property
    def id(self) -> str:
        return "cli"

    @property
    def label(self) -> str:
        return "CLI (stdin/stdout)"

    @property
    def max_text_length(self) -> int:
        # 终端没有硬性限制, 但太长的输出可读性差
        return 8000

    def enqueue(self, text: str, sender: str = "user") -> None:
        """从外部将消息放入队列 (供 gateway 循环使用).

        在纯 CLI 模式下, 由 REPL 主循环调用此方法将用户输入放入通道.
        """
        self._pending = InboundMessage(
            channel=self.id,
            sender=sender,
            text=text,
        )

    def receive(self) -> InboundMessage | None:
        """返回队列中的消息, 然后清空."""
        msg = self._pending
        self._pending = None
        return msg

    def send(self, text: str, media: list | None = None) -> None:
        """输出到 stdout."""
        chunks = self.chunk_text(text)
        for i, chunk in enumerate(chunks):
            if i > 0:
                print("---")  # 分块分隔符
            print(chunk)


# ---------------------------------------------------------------------------
# FileChannel -- 文件监视通道 (模拟 webhook)
# ---------------------------------------------------------------------------
#
# 监视一个文本文件的变化. 当有新内容被追加时, 将其作为入站消息处理.
# 回复写入另一个文件.
#
# 这模拟了 webhook 式通道的工作方式:
# - 外部系统向 inbox 文件写入消息
# - FileChannel 轮询文件变化
# - 回复写入 outbox 文件
#
# 测试方法: 在另一个终端 echo "hello" >> workspace/.channels/file_inbox.txt


class FileChannel(Channel):
    """基于文件监视的通道, 模拟 webhook 行为."""

    def __init__(
        self,
        inbox_path: Path | None = None,
        outbox_path: Path | None = None,
    ):
        self._inbox = inbox_path or FILE_CHANNEL_INBOX
        self._outbox = outbox_path or FILE_CHANNEL_OUTBOX

        # 确保文件和目录存在
        self._inbox.parent.mkdir(parents=True, exist_ok=True)
        if not self._inbox.exists():
            self._inbox.write_text("", encoding="utf-8")
        if not self._outbox.exists():
            self._outbox.write_text("", encoding="utf-8")

        # 记录已读取的位置 (字节偏移), 避免重复处理
        self._read_offset: int = self._inbox.stat().st_size

    @property
    def id(self) -> str:
        return "file"

    @property
    def label(self) -> str:
        return f"File ({self._inbox.name})"

    @property
    def max_text_length(self) -> int:
        return 4000

    def receive(self) -> InboundMessage | None:
        """检查 inbox 文件是否有新内容."""
        current_size = self._inbox.stat().st_size
        if current_size <= self._read_offset:
            return None

        # 读取新增部分
        with open(self._inbox, "r", encoding="utf-8") as f:
            f.seek(self._read_offset)
            new_content = f.read()
        self._read_offset = current_size

        # 取最后一条非空行作为消息
        lines = [line.strip() for line in new_content.strip().splitlines() if line.strip()]
        if not lines:
            return None

        text = lines[-1]
        return InboundMessage(
            channel=self.id,
            sender="file_user",
            text=text,
        )

    def send(self, text: str, media: list | None = None) -> None:
        """将回复追加到 outbox 文件."""
        chunks = self.chunk_text(text)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(self._outbox, "a", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(f"[{now}] {chunk}\n")
                f.write("---\n")


# ---------------------------------------------------------------------------
# MockTelegramChannel -- 模拟 Telegram Bot
# ---------------------------------------------------------------------------
#
# 模拟 Telegram 的行为特征:
#   1. 消息长度限制: 4096 字符
#   2. 支持 Markdown 格式化
#   3. 基于 long-polling 的消息接收 (这里用文件模拟)
#
# 在真正的 OpenClaw 中, Telegram 通道通过 Bot API 的 getUpdates
# 或 Webhook 接收消息, 通过 sendMessage 发送回复.


class MockTelegramChannel(Channel):
    """模拟 Telegram Bot 行为的通道."""

    def __init__(
        self,
        inbox_path: Path | None = None,
        outbox_path: Path | None = None,
    ):
        self._inbox = inbox_path or MOCK_TELEGRAM_INBOX
        self._outbox = outbox_path or MOCK_TELEGRAM_OUTBOX

        self._inbox.parent.mkdir(parents=True, exist_ok=True)
        if not self._inbox.exists():
            self._inbox.write_text("", encoding="utf-8")
        if not self._outbox.exists():
            self._outbox.write_text("", encoding="utf-8")

        self._read_offset: int = self._inbox.stat().st_size
        # 模拟 Telegram 的 update offset (防止重复处理)
        self._update_id: int = 0

    @property
    def id(self) -> str:
        return "telegram"

    @property
    def label(self) -> str:
        return "Telegram (Mock)"

    @property
    def max_text_length(self) -> int:
        # Telegram Bot API 实际限制是 4096 字符
        return 4096

    def _format_telegram_markdown(self, text: str) -> str:
        """简单的 Telegram MarkdownV2 格式化.

        真正的 OpenClaw 有完整的 Markdown -> Telegram MarkdownV2 转换器,
        处理转义、嵌套格式等复杂情况. 这里只做示意.
        """
        # Telegram MarkdownV2 需要转义这些字符
        # 但这里我们只是模拟, 不做完整转义
        return text

    def receive(self) -> InboundMessage | None:
        """从 inbox 文件读取新消息, 模拟 getUpdates."""
        current_size = self._inbox.stat().st_size
        if current_size <= self._read_offset:
            return None

        with open(self._inbox, "r", encoding="utf-8") as f:
            f.seek(self._read_offset)
            new_content = f.read()
        self._read_offset = current_size

        lines = [line.strip() for line in new_content.strip().splitlines() if line.strip()]
        if not lines:
            return None

        text = lines[-1]
        self._update_id += 1

        # 模拟 Telegram 消息结构
        # 真正的 Telegram 消息包含 chat_id, message_id, from, date 等
        return InboundMessage(
            channel=self.id,
            sender=f"tg_user_{self._update_id}",
            text=text,
            thread_id=None,
        )

    def send(self, text: str, media: list | None = None) -> None:
        """模拟 sendMessage, 输出到 outbox 文件."""
        chunks = self.chunk_text(text)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(self._outbox, "a", encoding="utf-8") as f:
            for chunk in chunks:
                formatted = self._format_telegram_markdown(chunk)
                # 模拟 Telegram 消息信封
                envelope = {
                    "ok": True,
                    "result": {
                        "message_id": self._update_id,
                        "date": now,
                        "text": formatted,
                    },
                }
                f.write(json.dumps(envelope, ensure_ascii=False) + "\n")

    def chunk_text(self, text: str) -> list[str]:
        """Telegram 特定的分块: 避免在 Markdown 标记中间断开.

        真正的 OpenClaw 会解析 Markdown AST 来找安全的分割点.
        这里使用基类的简单实现.
        """
        return super().chunk_text(text)


# ---------------------------------------------------------------------------
# ChannelRegistry -- 通道注册表
# ---------------------------------------------------------------------------
#
# 管理所有已注册的通道插件.
# 在真正的 OpenClaw 中, ChannelRegistry 还负责:
#   - 通道的动态加载/卸载 (插件热加载)
#   - 通道别名解析 (如 "imsg" -> "imessage")
#   - 通道状态监控 (connected/disconnected)
#   - 通道间的消息路由


class ChannelRegistry:
    """通道注册表: 管理所有通道插件的注册和查找."""

    def __init__(self):
        self._channels: dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        """注册一个通道."""
        if channel.id in self._channels:
            raise ValueError(f"Channel already registered: {channel.id}")
        self._channels[channel.id] = channel

    def get(self, channel_id: str) -> Channel | None:
        """根据 ID 获取通道."""
        return self._channels.get(channel_id)

    def list_channels(self) -> list[Channel]:
        """返回所有已注册通道."""
        return list(self._channels.values())

    @property
    def channels(self) -> list[Channel]:
        """所有已注册通道 (属性访问)."""
        return list(self._channels.values())

    def poll_all(self) -> list[InboundMessage]:
        """轮询所有通道, 收集新消息.

        这是 gateway 主循环的核心:
        每次循环调用 poll_all(), 收集所有通道的新消息,
        然后逐个分发给 agent 处理.
        """
        messages = []
        for channel in self._channels.values():
            msg = channel.receive()
            if msg is not None:
                messages.append(msg)
        return messages


# ---------------------------------------------------------------------------
# Agent Loop (带多通道支持)
# ---------------------------------------------------------------------------
#
# 和 s03 的区别:
#   1. session key 的 channel 部分来自消息的通道 ID
#   2. 回复通过对应通道发送
#   3. 支持从任意通道接收消息


def build_session_key(channel_id: str, sender: str, agent_id: str = "main") -> str:
    """根据通道和发送者构建 session key.

    格式: <agent_id>:<channel>:<sender>
    这样每个通道的每个用户都有独立的会话.
    """
    # 清理特殊字符, 保留 session key 的可用性
    safe_sender = sender.replace(":", "_").replace("/", "_")
    return f"{agent_id}:{channel_id}:{safe_sender}"


def agent_loop(
    user_input: str,
    session_key: str,
    session_store: SessionStore,
    client: Anthropic,
) -> str:
    """处理一轮用户输入, 返回最终文本回复. (逻辑同 s03)"""
    session_data = session_store.load_session(session_key)
    messages = session_data["history"]
    messages.append({"role": "user", "content": user_input})

    all_assistant_blocks: list = []

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        all_assistant_blocks.extend(response.content)
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if response.stop_reason == "tool_use" and tool_use_blocks:
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for tool_block in tool_use_blocks:
                print(f"  [tool] {tool_block.name}({json.dumps(tool_block.input, ensure_ascii=False)})")
                result = execute_tool(tool_block.name, tool_block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": result,
                })
                session_store.save_tool_result(session_key, tool_block.id, result)
            messages.append({"role": "user", "content": tool_results})
            continue

        final_text = ""
        for block in response.content:
            if block.type == "text":
                final_text += block.text
        break

    session_store.save_turn(session_key, user_input, all_assistant_blocks)
    return final_text


# ---------------------------------------------------------------------------
# Gateway -- 多通道消息分发
# ---------------------------------------------------------------------------
#
# Gateway 是 OpenClaw 的核心运行时. 它做三件事:
#   1. 轮询所有通道, 收集新消息
#   2. 为每条消息确定 session key, 调用 agent loop
#   3. 将回复发回对应通道
#
# 在真正的 OpenClaw 中, gateway 还需要:
#   - 并发处理 (多消息并行)
#   - 消息队列 (避免丢消息)
#   - 重试和错误恢复
#   - 速率限制
#   - 优雅关机


def gateway_poll_once(
    registry: ChannelRegistry,
    session_store: SessionStore,
    client: Anthropic,
) -> int:
    """执行一次全通道轮询, 返回处理的消息数.

    这是 gateway 主循环的单次迭代.
    """
    messages = registry.poll_all()
    processed = 0

    for msg in messages:
        channel = registry.get(msg.channel)
        if not channel:
            print(f"  [gateway] Unknown channel: {msg.channel}, skipping")
            continue

        session_key = build_session_key(msg.channel, msg.sender)
        print(f"  [gateway] {msg.channel}:{msg.sender} -> session {session_key}")
        print(f"  [gateway] message: {msg.text[:80]}{'...' if len(msg.text) > 80 else ''}")

        try:
            response = agent_loop(msg.text, session_key, session_store, client)
            channel.send(response)
            processed += 1
            print(f"  [gateway] replied via {msg.channel}")
        except Exception as exc:
            error_msg = f"Error processing message: {exc}"
            print(f"  [gateway] {error_msg}")
            channel.send(f"[Error] {error_msg}")

    return processed


# ---------------------------------------------------------------------------
# REPL 主循环
# ---------------------------------------------------------------------------
#
# 这个版本的 REPL 运行为一个迷你 gateway:
# - CLI 通道由用户直接输入
# - File 和 MockTelegram 通道在每次 CLI 输入后也会被轮询
# - 可以通过 /poll 命令手动触发一次全通道轮询
#
# 这演示了 OpenClaw gateway 的核心模式:
# 多个通道共享同一个 agent, 各自独立的会话.


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

    # 初始化通道注册表
    registry = ChannelRegistry()

    # 注册通道
    cli_channel = CLIChannel()
    registry.register(cli_channel)
    registry.register(FileChannel())
    registry.register(MockTelegramChannel())

    # CLI 通道的默认 session key
    cli_session_key = build_session_key("cli", "user")

    print("=" * 60)
    print("  claw0 s04: Multi-Channel Gateway")
    print("  Model:", MODEL)
    print("  Session:", cli_session_key)
    print()
    print("  Registered channels:")
    for ch in registry.list_channels():
        print(f"    [{ch.id}] {ch.label} (max {ch.max_text_length} chars)")
    print()
    print("  Commands:")
    print("    /channels         List registered channels")
    print("    /poll             Poll all channels for messages")
    print("    /send <ch> <msg>  Inject a test message into a channel")
    print("    /new              Create a new CLI session")
    print("    /sessions         List all sessions")
    print("    /history          Show current session history")
    print("    /quit             Exit")
    print()
    print("  File channel inbox:", FILE_CHANNEL_INBOX)
    print("  Telegram inbox:    ", MOCK_TELEGRAM_INBOX)
    print("  (echo messages into these files from another terminal)")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input(f"[{cli_session_key}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue

        # -- 命令处理 --

        if user_input == "/quit":
            print("Bye.")
            break

        if user_input == "/channels":
            print("  Registered channels:")
            for ch in registry.list_channels():
                print(f"    [{ch.id}] {ch.label}")
                print(f"      max_text_length: {ch.max_text_length}")
            continue

        if user_input == "/poll":
            print("  Polling all channels...")
            count = gateway_poll_once(registry, session_store, client)
            print(f"  Processed {count} message(s) from non-CLI channels.")
            continue

        if user_input.startswith("/send"):
            # /send <channel_id> <message>
            parts = user_input.split(maxsplit=2)
            if len(parts) < 3:
                print("  Usage: /send <channel_id> <message>")
                continue
            target_ch_id = parts[1]
            inject_text = parts[2]
            target_ch = registry.get(target_ch_id)
            if not target_ch:
                print(f"  Unknown channel: {target_ch_id}")
                print(f"  Available: {', '.join(ch.id for ch in registry.list_channels())}")
                continue
            if target_ch_id == "file":
                # 向 file channel 的 inbox 写入消息
                with open(FILE_CHANNEL_INBOX, "a", encoding="utf-8") as f:
                    f.write(inject_text + "\n")
                print(f"  Injected into {target_ch_id} inbox. Use /poll to process.")
            elif target_ch_id == "telegram":
                # 向 mock telegram 的 inbox 写入消息
                with open(MOCK_TELEGRAM_INBOX, "a", encoding="utf-8") as f:
                    f.write(inject_text + "\n")
                print(f"  Injected into {target_ch_id} inbox. Use /poll to process.")
            else:
                print(f"  Direct injection not supported for channel: {target_ch_id}")
            continue

        if user_input == "/new":
            ts_suffix = datetime.now(timezone.utc).strftime("%H%M%S")
            cli_session_key = build_session_key("cli", f"user_{ts_suffix}")
            session_store.create_session(cli_session_key)
            print(f"  New session: {cli_session_key}")
            continue

        if user_input == "/sessions":
            sessions = session_store.list_sessions()
            if not sessions:
                print("  (no sessions)")
            else:
                print(f"  {len(sessions)} session(s):")
                for meta in sessions:
                    key = meta.get("session_key", "?")
                    updated = meta.get("updated_at", "?")[:19]
                    count = meta.get("message_count", 0)
                    marker = " *" if key == cli_session_key else ""
                    print(f"    {key}  ({count} msgs, last: {updated}){marker}")
            continue

        if user_input == "/history":
            session_data = session_store.load_session(cli_session_key)
            messages = session_data["history"]
            if not messages:
                print("  (empty session)")
            else:
                for msg in messages:
                    role = msg.get("role", "?")
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        display = content[:200] + "..." if len(content) > 200 else content
                        print(f"  [{role}] {display}")
                    elif isinstance(content, list):
                        for block in content:
                            if isinstance(block, dict):
                                btype = block.get("type", "?")
                                if btype == "tool_use":
                                    print(f"  [{role}:tool_use] {block.get('name', '?')}(...)")
                                elif btype == "tool_result":
                                    out = block.get("content", "")
                                    display = out[:100] + "..." if len(out) > 100 else out
                                    print(f"  [{role}:tool_result] {display}")
            continue

        if user_input.startswith("/"):
            print(f"  Unknown command: {user_input}")
            continue

        # -- CLI 通道: 直接处理用户输入 --

        try:
            response = agent_loop(user_input, cli_session_key, session_store, client)
            print()
            cli_channel.send(response)
            print()
        except Exception as exc:
            print(f"\n  Error: {exc}\n")

        # -- 顺便轮询其他通道 --
        # 在每次 CLI 交互后, 也检查一下其他通道是否有新消息
        # 这是一种简化的 "搭便车" 轮询方式
        # 真正的 gateway 会用独立的事件循环

        other_count = 0
        for ch in registry.list_channels():
            if ch.id == "cli":
                continue
            msg = ch.receive()
            if msg is not None:
                other_count += 1
                sk = build_session_key(msg.channel, msg.sender)
                print(f"  [auto-poll] New message from {msg.channel}:{msg.sender}")
                try:
                    resp = agent_loop(msg.text, sk, session_store, client)
                    ch.send(resp)
                    print(f"  [auto-poll] Replied via {msg.channel}")
                except Exception as exc:
                    print(f"  [auto-poll] Error: {exc}")

        if other_count > 0:
            print(f"  [auto-poll] Processed {other_count} message(s) from other channels.")


if __name__ == "__main__":
    main()
