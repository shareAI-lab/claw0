"""
第04节: 通道 -- "同一大脑, 多个嘴巴"

Channel 封装了平台差异, 使 agent 循环只看到统一的 InboundMessage。
添加新平台 = 实现 receive() + send(); 循环不需要改动。

    Telegram ----.                          .---- sendMessage API
    Feishu -------+-- InboundMessage ---+---- im/v1/messages
    CLI (stdin) --'    Agent Loop        '---- print(stdout)

运行方法:  cd claw0 && python zh/s04_channels.py

需要在 .env 中配置:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
    # 可选: TELEGRAM_BOT_TOKEN, FEISHU_APP_ID, FEISHU_APP_SECRET, FEISHU_ENCRYPT_KEY
"""

import json, os, sys, time, threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR = WORKSPACE_DIR / ".state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant connected to multiple messaging channels.\n"
    "You can save and search notes using the provided tools.\n"
    "When responding, be concise and helpful."
)

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN, GREEN, YELLOW, DIM, RESET = "\033[36m", "\033[32m", "\033[33m", "\033[2m", "\033[0m"
BOLD, RED, BLUE = "\033[1m", "\033[31m", "\033[34m"


def print_assistant(text: str, ch: str = "cli") -> None:
    prefix = f"[{ch}] " if ch != "cli" else ""
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {prefix}{text}\n")

def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")

def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")

def print_channel(text: str) -> None:
    print(f"{BLUE}{text}{RESET}")

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class InboundMessage:
    """所有通道都规范化为此结构。Agent 循环只看到 InboundMessage。"""
    text: str
    sender_id: str
    channel: str = ""
    account_id: str = ""
    peer_id: str = ""
    is_group: bool = False
    media: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

@dataclass
class ChannelAccount:
    """每个 bot 的配置。同一通道类型可以运行多个 bot。"""
    channel: str
    account_id: str
    token: str = ""
    config: dict = field(default_factory=dict)

# ---------------------------------------------------------------------------
# 会话键
# ---------------------------------------------------------------------------

def build_session_key(channel: str, account_id: str, peer_id: str) -> str:
    return f"agent:main:direct:{channel}:{peer_id}"

# ---------------------------------------------------------------------------
# Channel 抽象基类
# ---------------------------------------------------------------------------

class Channel(ABC):
    name: str = "unknown"

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool: ...

    def close(self) -> None:
        pass

# ---------------------------------------------------------------------------
# CLIChannel
# ---------------------------------------------------------------------------

class CLIChannel(Channel):
    name = "cli"

    def __init__(self) -> None:
        self.account_id = "cli-local"

    def receive(self) -> InboundMessage | None:
        try:
            text = input(f"{CYAN}{BOLD}You > {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            return None
        if not text:
            return None
        return InboundMessage(
            text=text, sender_id="cli-user", channel="cli",
            account_id=self.account_id, peer_id="cli-user",
        )

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        print_assistant(text)
        return True

# ---------------------------------------------------------------------------
# 偏移量持久化 -- 两个简单函数
# ---------------------------------------------------------------------------

def save_offset(path: Path, offset: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(offset))

def load_offset(path: Path) -> int:
    try:
        return int(path.read_text().strip())
    except Exception:
        return 0

# ---------------------------------------------------------------------------
# TelegramChannel -- Bot API 长轮询
# ---------------------------------------------------------------------------

class TelegramChannel(Channel):
    name = "telegram"
    MAX_MSG_LEN = 4096

    def __init__(self, account: ChannelAccount) -> None:
        if not HAS_HTTPX:
            raise RuntimeError("TelegramChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        self.base_url = f"https://api.telegram.org/bot{account.token}"
        self._http = httpx.Client(timeout=35.0)
        raw = account.config.get("allowed_chats", "")
        self.allowed_chats = {c.strip() for c in raw.split(",") if c.strip()} if raw else set()

        self._offset_path = STATE_DIR / "telegram" / f"offset-{self.account_id}.txt"
        self._offset = load_offset(self._offset_path)
        self._seen: set[int] = set()
        self._media_groups: dict[str, dict] = {}
        self._text_buf: dict[tuple[str, str], dict] = {}

    def _api(self, method: str, **params: Any) -> dict:
        filtered = {k: v for k, v in params.items() if v is not None}
        try:
            resp = self._http.post(f"{self.base_url}/{method}", json=filtered)
            data = resp.json()
            if not data.get("ok"):
                print(f"  {RED}[telegram] {method}: {data.get('description', '?')}{RESET}")
                return {}
            return data.get("result", {})
        except Exception as exc:
            print(f"  {RED}[telegram] {method}: {exc}{RESET}")
            return {}

    def send_typing(self, chat_id: str) -> None:
        self._api("sendChatAction", chat_id=chat_id, action="typing")

    def poll(self) -> list[InboundMessage]:
        result = self._api("getUpdates", offset=self._offset, timeout=30,
                           allowed_updates=["message"])
        if not result or not isinstance(result, list):
            return self._flush_all()

        for update in result:
            uid = update.get("update_id", 0)
            if uid >= self._offset:
                self._offset = uid + 1
                save_offset(self._offset_path, self._offset)
            if uid in self._seen:
                continue
            self._seen.add(uid)
            if len(self._seen) > 5000:
                self._seen.clear()

            msg = update.get("message")
            if not msg:
                continue
            if msg.get("media_group_id"):
                self._buf_media(msg, update)
                continue
            inbound = self._parse(msg, update)
            if not inbound:
                continue
            if self.allowed_chats and inbound.peer_id not in self.allowed_chats:
                continue
            self._buf_text(inbound)

        return self._flush_all()

    def _flush_all(self) -> list[InboundMessage]:
        ready = self._flush_media()
        ready.extend(self._flush_text())
        return ready

    # -- 媒体组缓冲 (500ms 窗口) --

    def _buf_media(self, msg: dict, update: dict) -> None:
        mgid = msg["media_group_id"]
        if mgid not in self._media_groups:
            self._media_groups[mgid] = {"ts": time.monotonic(), "entries": []}
        self._media_groups[mgid]["entries"].append((msg, update))

    def _flush_media(self) -> list[InboundMessage]:
        now = time.monotonic()
        ready: list[InboundMessage] = []
        expired = [k for k, g in self._media_groups.items() if (now - g["ts"]) >= 0.5]
        for mgid in expired:
            entries = self._media_groups.pop(mgid)["entries"]
            captions, media_items = [], []
            for m, _ in entries:
                if m.get("caption"):
                    captions.append(m["caption"])
                for mt in ("photo", "video", "document", "audio"):
                    if mt in m:
                        raw_m = m[mt]
                        fid = raw_m[-1]["file_id"] if isinstance(raw_m, list) and raw_m else raw_m.get("file_id", "") if isinstance(raw_m, dict) else ""
                        media_items.append({"type": mt, "file_id": fid})
            inbound = self._parse(entries[0][0], entries[0][1])
            if inbound:
                inbound.text = "\n".join(captions) if captions else "[media group]"
                inbound.media = media_items
                if not self.allowed_chats or inbound.peer_id in self.allowed_chats:
                    ready.append(inbound)
        return ready

    # -- 文本合并 (1s 窗口) --
    # Telegram 会将长粘贴拆分成多个片段; 缓冲后在 1s 静默后发出。

    def _buf_text(self, inbound: InboundMessage) -> None:
        key = (inbound.peer_id, inbound.sender_id)
        now = time.monotonic()
        if key in self._text_buf:
            self._text_buf[key]["text"] += "\n" + inbound.text
            self._text_buf[key]["ts"] = now
        else:
            self._text_buf[key] = {"text": inbound.text, "msg": inbound, "ts": now}

    def _flush_text(self) -> list[InboundMessage]:
        now = time.monotonic()
        ready: list[InboundMessage] = []
        expired = [k for k, b in self._text_buf.items() if (now - b["ts"]) >= 1.0]
        for key in expired:
            buf = self._text_buf.pop(key)
            buf["msg"].text = buf["text"]
            ready.append(buf["msg"])
        return ready

    # -- 消息解析 --

    def _parse(self, msg: dict, raw_update: dict) -> InboundMessage | None:
        chat = msg.get("chat", {})
        chat_type = chat.get("type", "")
        chat_id = str(chat.get("id", ""))
        user_id = str(msg.get("from", {}).get("id", ""))
        text = msg.get("text", "") or msg.get("caption", "")
        if not text:
            return None

        thread_id = msg.get("message_thread_id")
        is_forum = chat.get("is_forum", False)
        is_group = chat_type in ("group", "supergroup")

        if chat_type == "private":
            peer_id = user_id
        elif is_group and is_forum and thread_id is not None:
            peer_id = f"{chat_id}:topic:{thread_id}"
        else:
            peer_id = chat_id

        return InboundMessage(
            text=text, sender_id=user_id, channel="telegram",
            account_id=self.account_id, peer_id=peer_id,
            is_group=is_group, raw=raw_update,
        )

    def receive(self) -> InboundMessage | None:
        msgs = self.poll()
        return msgs[0] if msgs else None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        chat_id, thread_id = to, None
        if ":topic:" in to:
            parts = to.split(":topic:")
            chat_id, thread_id = parts[0], int(parts[1]) if len(parts) > 1 else None
        ok = True
        for chunk in self._chunk(text):
            if not self._api("sendMessage", chat_id=chat_id, text=chunk,
                             message_thread_id=thread_id):
                ok = False
        return ok

    def _chunk(self, text: str) -> list[str]:
        if len(text) <= self.MAX_MSG_LEN:
            return [text]
        chunks = []
        while text:
            if len(text) <= self.MAX_MSG_LEN:
                chunks.append(text); break
            cut = text.rfind("\n", 0, self.MAX_MSG_LEN)
            if cut <= 0:
                cut = self.MAX_MSG_LEN
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    def close(self) -> None:
        self._http.close()

# ---------------------------------------------------------------------------
# FeishuChannel -- 基于 webhook (飞书/Lark)
# ---------------------------------------------------------------------------

class FeishuChannel(Channel):
    name = "feishu"

    def __init__(self, account: ChannelAccount) -> None:
        if not HAS_HTTPX:
            raise RuntimeError("FeishuChannel requires httpx: pip install httpx")
        self.account_id = account.account_id
        self.app_id = account.config.get("app_id", "")
        self.app_secret = account.config.get("app_secret", "")
        self._encrypt_key = account.config.get("encrypt_key", "")
        self._bot_open_id = account.config.get("bot_open_id", "")
        is_lark = account.config.get("is_lark", False)
        self.api_base = ("https://open.larksuite.com/open-apis" if is_lark
                         else "https://open.feishu.cn/open-apis")
        self._tenant_token: str = ""
        self._token_expires_at: float = 0.0
        self._http = httpx.Client(timeout=15.0)

    def _refresh_token(self) -> str:
        if self._tenant_token and time.time() < self._token_expires_at:
            return self._tenant_token
        try:
            resp = self._http.post(
                f"{self.api_base}/auth/v3/tenant_access_token/internal",
                json={"app_id": self.app_id, "app_secret": self.app_secret},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"  {RED}[feishu] Token error: {data.get('msg', '?')}{RESET}")
                return ""
            self._tenant_token = data.get("tenant_access_token", "")
            self._token_expires_at = time.time() + data.get("expire", 7200) - 300
            return self._tenant_token
        except Exception as exc:
            print(f"  {RED}[feishu] Token error: {exc}{RESET}")
            return ""

    def _bot_mentioned(self, event: dict) -> bool:
        for m in event.get("message", {}).get("mentions", []):
            mid = m.get("id", {})
            if isinstance(mid, dict) and mid.get("open_id") == self._bot_open_id:
                return True
            if isinstance(mid, str) and mid == self._bot_open_id:
                return True
            if m.get("key") == self._bot_open_id:
                return True
        return False

    def _parse_content(self, message: dict) -> tuple[str, list]:
        msg_type = message.get("msg_type", "text")
        raw = message.get("content", "{}")
        try:
            content = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return "", []

        media: list[dict] = []
        if msg_type == "text":
            return content.get("text", ""), media
        if msg_type == "post":
            texts: list[str] = []
            for lc in content.values():
                if not isinstance(lc, dict):
                    continue
                title = lc.get("title", "")
                if title:
                    texts.append(title)
                for para in lc.get("content", []):
                    for node in para:
                        tag = node.get("tag")
                        if tag == "text":
                            texts.append(node.get("text", ""))
                        elif tag == "a":
                            texts.append(node.get("text", "") + " " + node.get("href", ""))
            return "\n".join(texts), media
        if msg_type == "image":
            key = content.get("image_key", "")
            if key:
                media.append({"type": "image", "key": key})
            return "[image]", media
        return "", media

    def parse_event(self, payload: dict, token: str = "") -> InboundMessage | None:
        """解析飞书事件回调。使用简单的 token 校验进行验证。"""
        if self._encrypt_key and token and token != self._encrypt_key:
            print(f"  {RED}[feishu] Token verification failed{RESET}")
            return None
        if "challenge" in payload:
            print_info(f"[feishu] Challenge: {payload['challenge']}")
            return None

        event = payload.get("event", {})
        message = event.get("message", {})
        sender = event.get("sender", {}).get("sender_id", {})
        user_id = sender.get("open_id", sender.get("user_id", ""))
        chat_id = message.get("chat_id", "")
        chat_type = message.get("chat_type", "")
        is_group = chat_type == "group"

        if is_group and self._bot_open_id and not self._bot_mentioned(event):
            return None

        text, media = self._parse_content(message)
        if not text:
            return None

        return InboundMessage(
            text=text, sender_id=user_id, channel="feishu",
            account_id=self.account_id,
            peer_id=user_id if chat_type == "p2p" else chat_id,
            media=media, is_group=is_group, raw=payload,
        )

    def receive(self) -> InboundMessage | None:
        return None

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        token = self._refresh_token()
        if not token:
            return False
        try:
            resp = self._http.post(
                f"{self.api_base}/im/v1/messages",
                params={"receive_id_type": "chat_id"},
                headers={"Authorization": f"Bearer {token}"},
                json={"receive_id": to, "msg_type": "text",
                      "content": json.dumps({"text": text})},
            )
            data = resp.json()
            if data.get("code") != 0:
                print(f"  {RED}[feishu] Send: {data.get('msg', '?')}{RESET}")
                return False
            return True
        except Exception as exc:
            print(f"  {RED}[feishu] Send: {exc}{RESET}")
            return False

    def close(self) -> None:
        self._http.close()

# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
MEMORY_FILE = WORKSPACE_DIR / "MEMORY.md"

def tool_memory_write(content: str) -> str:
    print_tool("memory_write", f"{len(content)} chars")
    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n- {content}\n")
        return f"Written to memory: {content[:80]}..."
    except Exception as exc:
        return f"Error: {exc}"

def tool_memory_search(query: str) -> str:
    print_tool("memory_search", query)
    if not MEMORY_FILE.exists():
        return "Memory file is empty."
    try:
        lines = MEMORY_FILE.read_text(encoding="utf-8").splitlines()
        matches = [l for l in lines if query.lower() in l.lower()]
        return "\n".join(matches[:20]) if matches else f"No matches for '{query}'."
    except Exception as exc:
        return f"Error: {exc}"

TOOLS = [
    {"name": "memory_write", "description": "Save a note to long-term memory.",
     "input_schema": {"type": "object", "required": ["content"],
                      "properties": {"content": {"type": "string",
                                                  "description": "The text to remember."}}}},
    {"name": "memory_search", "description": "Search through saved memory notes.",
     "input_schema": {"type": "object", "required": ["query"],
                      "properties": {"query": {"type": "string",
                                               "description": "Search keyword."}}}},
]

TOOL_HANDLERS: dict[str, Any] = {
    "memory_write": tool_memory_write,
    "memory_search": tool_memory_search,
}

def process_tool_call(tool_name: str, tool_input: dict) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"

# ---------------------------------------------------------------------------
# ChannelManager
# ---------------------------------------------------------------------------

class ChannelManager:
    def __init__(self) -> None:
        self.channels: dict[str, Channel] = {}
        self.accounts: list[ChannelAccount] = []

    def register(self, channel: Channel) -> None:
        self.channels[channel.name] = channel
        print_channel(f"  [+] Channel registered: {channel.name}")

    def list_channels(self) -> list[str]:
        return list(self.channels.keys())

    def get(self, name: str) -> Channel | None:
        return self.channels.get(name)

    def close_all(self) -> None:
        for ch in self.channels.values():
            ch.close()

# ---------------------------------------------------------------------------
# Telegram 后台轮询线程
# ---------------------------------------------------------------------------

def telegram_poll_loop(
    tg: TelegramChannel, queue: list, lock: threading.Lock, stop: threading.Event,
) -> None:
    print_channel(f"  [telegram] Polling started for {tg.account_id}")
    while not stop.is_set():
        try:
            msgs = tg.poll()
            if msgs:
                with lock:
                    queue.extend(msgs)
        except Exception as exc:
            print(f"  {RED}[telegram] Poll error: {exc}{RESET}")
            stop.wait(5.0)

# ---------------------------------------------------------------------------
# REPL 命令
# ---------------------------------------------------------------------------

def handle_repl_command(cmd: str, mgr: ChannelManager) -> bool:
    cmd = cmd.strip().lower()
    if cmd == "/channels":
        for name in mgr.list_channels():
            print_channel(f"  - {name}")
        return True
    if cmd == "/accounts":
        for acc in mgr.accounts:
            masked = acc.token[:8] + "..." if len(acc.token) > 8 else "(none)"
            print_channel(f"  - {acc.channel}/{acc.account_id}  token={masked}")
        return True
    if cmd in ("/help", "/h"):
        print_info("  /channels  /accounts  /help  quit/exit")
        return True
    return False

# ---------------------------------------------------------------------------
# Agent 回合
# ---------------------------------------------------------------------------

def run_agent_turn(
    inbound: InboundMessage,
    conversations: dict[str, list[dict]],
    mgr: ChannelManager,
) -> None:
    sk = build_session_key(inbound.channel, inbound.account_id, inbound.peer_id)
    if sk not in conversations:
        conversations[sk] = []
    messages = conversations[sk]
    messages.append({"role": "user", "content": inbound.text})

    if inbound.channel == "telegram":
        tg = mgr.get("telegram")
        if isinstance(tg, TelegramChannel):
            tg.send_typing(inbound.peer_id.split(":topic:")[0])

    while True:
        try:
            response = client.messages.create(
                model=MODEL_ID, max_tokens=8096,
                system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
            )
        except Exception as exc:
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()
            return

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            if text:
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, text)
                else:
                    print_assistant(text, inbound.channel)
            break
        elif response.stop_reason == "tool_use":
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    results.append({
                        "type": "tool_result", "tool_use_id": block.id,
                        "content": process_tool_call(block.name, block.input),
                    })
            messages.append({"role": "user", "content": results})
        else:
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            if text:
                ch = mgr.get(inbound.channel)
                if ch:
                    ch.send(inbound.peer_id, text)
            break

# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def agent_loop() -> None:
    mgr = ChannelManager()
    cli = CLIChannel()
    mgr.register(cli)

    tg_channel: TelegramChannel | None = None
    stop_event = threading.Event()
    msg_queue: list[InboundMessage] = []
    q_lock = threading.Lock()
    tg_thread: threading.Thread | None = None

    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if tg_token and HAS_HTTPX:
        tg_acc = ChannelAccount(
            channel="telegram", account_id="tg-primary", token=tg_token,
            config={"allowed_chats": os.getenv("TELEGRAM_ALLOWED_CHATS", "")},
        )
        mgr.accounts.append(tg_acc)
        tg_channel = TelegramChannel(tg_acc)
        mgr.register(tg_channel)
        tg_thread = threading.Thread(
            target=telegram_poll_loop, daemon=True,
            args=(tg_channel, msg_queue, q_lock, stop_event),
        )
        tg_thread.start()

    fs_id = os.getenv("FEISHU_APP_ID", "").strip()
    fs_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
    if fs_id and fs_secret and HAS_HTTPX:
        fs_acc = ChannelAccount(
            channel="feishu", account_id="feishu-primary",
            config={
                "app_id": fs_id, "app_secret": fs_secret,
                "encrypt_key": os.getenv("FEISHU_ENCRYPT_KEY", ""),
                "bot_open_id": os.getenv("FEISHU_BOT_OPEN_ID", ""),
                "is_lark": os.getenv("FEISHU_IS_LARK", "").lower() in ("1", "true"),
            },
        )
        mgr.accounts.append(fs_acc)
        mgr.register(FeishuChannel(fs_acc))

    print_info("=" * 60)
    print_info("  claw0  |  Section 04: Channels")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Channels: {', '.join(mgr.list_channels())}")
    print_info("  Commands: /channels /accounts /help  |  quit/exit")
    print_info("=" * 60)
    print()

    conversations: dict[str, list[dict]] = {}

    while True:
        # 排空 Telegram 队列
        with q_lock:
            tg_msgs = msg_queue[:]
            msg_queue.clear()
        for m in tg_msgs:
            print_channel(f"\n  [telegram] {m.sender_id}: {m.text[:80]}")
            run_agent_turn(m, conversations, mgr)

        # CLI 输入 (当 Telegram 活跃时使用非阻塞模式)
        if tg_channel:
            import select
            if not select.select([sys.stdin], [], [], 0.5)[0]:
                continue
            try:
                user_input = sys.stdin.readline().strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input:
                continue
        else:
            msg = cli.receive()
            if msg is None:
                break
            user_input = msg.text

        if user_input.lower() in ("quit", "exit"):
            break
        if user_input.startswith("/") and handle_repl_command(user_input, mgr):
            continue

        run_agent_turn(
            InboundMessage(text=user_input, sender_id="cli-user",
                           channel="cli", account_id="cli-local", peer_id="cli-user"),
            conversations, mgr,
        )

    print(f"{DIM}Goodbye.{RESET}")
    stop_event.set()
    if tg_thread and tg_thread.is_alive():
        tg_thread.join(timeout=3.0)
    mgr.close_all()

# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)
    agent_loop()

if __name__ == "__main__":
    main()
