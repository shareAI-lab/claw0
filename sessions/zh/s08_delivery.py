"""
Section 08: Delivery
"Write to disk first, then try to send"

所有出站消息都经过可靠的投递队列.
如果发送失败, 使用退避策略重试. 如果进程崩溃, 重启时扫描磁盘恢复.

    Agent Reply / Heartbeat / Cron
              |
        chunk_message()       -- 按平台限制分片
              |
        DeliveryQueue.enqueue()  -- 写入磁盘 (预写日志)
              |
        DeliveryRunner (后台线程)
              |
         deliver_fn(channel, to, text)
            /     \
         success    failure
           |           |
         ack()      fail() + backoff
           |           |
        delete      retry or move_to_failed/

    指数退避: [5s, 25s, 2min, 10min]
    最大重试次数: 5

用法:
    cd claw0
    python zh/s08_delivery.py

.env 中需要:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
import json
import os
import random
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from anthropic import Anthropic

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
QUEUE_DIR = WORKSPACE_DIR / "delivery-queue"
FAILED_DIR = QUEUE_DIR / "failed"

BACKOFF_MS = [5_000, 25_000, 120_000, 600_000]  # [5s, 25s, 2min, 10min]
MAX_RETRIES = 5

SYSTEM_PROMPT = (
    "You are Luna, a warm and curious AI companion. "
    "Keep replies concise and helpful. "
    "Use memory_write to save important facts. "
    "Use memory_search to recall past context."
)

# ---------------------------------------------------------------------------
# ANSI 颜色
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"
ORANGE = "\033[38;5;208m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def print_delivery(text: str) -> None:
    print(f"  {BLUE}[delivery]{RESET} {text}")


def print_warn(text: str) -> None:
    print(f"  {YELLOW}[warn]{RESET} {text}")


def print_error(text: str) -> None:
    print(f"  {RED}[error]{RESET} {text}")


# ---------------------------------------------------------------------------
# 1. QueuedDelivery -- 队列条目数据结构
# ---------------------------------------------------------------------------

@dataclass
class QueuedDelivery:
    id: str
    channel: str
    to: str
    text: str
    retry_count: int = 0
    last_error: str | None = None
    enqueued_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "channel": self.channel,
            "to": self.to,
            "text": self.text,
            "retry_count": self.retry_count,
            "last_error": self.last_error,
            "enqueued_at": self.enqueued_at,
            "next_retry_at": self.next_retry_at,
        }

    @staticmethod
    def from_dict(data: dict) -> "QueuedDelivery":
        return QueuedDelivery(
            id=data["id"],
            channel=data["channel"],
            to=data["to"],
            text=data["text"],
            retry_count=data.get("retry_count", 0),
            last_error=data.get("last_error"),
            enqueued_at=data.get("enqueued_at", 0.0),
            next_retry_at=data.get("next_retry_at", 0.0),
        )


def compute_backoff_ms(retry_count: int) -> int:
    """指数退避, 加 +/- 20% 抖动以避免惊群效应."""
    if retry_count <= 0:
        return 0
    idx = min(retry_count - 1, len(BACKOFF_MS) - 1)
    base = BACKOFF_MS[idx]
    jitter = random.randint(-base // 5, base // 5)
    return max(0, base + jitter)


# ---------------------------------------------------------------------------
# 2. DeliveryQueue -- 磁盘持久化可靠投递队列
# ---------------------------------------------------------------------------
# 预写日志: 先写入磁盘, 再尝试投递.
# 原子写入: 临时文件 + os.replace(), 崩溃安全.


class DeliveryQueue:
    def __init__(self, queue_dir: Path | None = None):
        self.queue_dir = queue_dir or QUEUE_DIR
        self.failed_dir = self.queue_dir / "failed"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def enqueue(self, channel: str, to: str, text: str) -> str:
        """创建队列条目并原子写入磁盘. 返回 delivery_id."""
        delivery_id = uuid.uuid4().hex[:12]
        entry = QueuedDelivery(
            id=delivery_id,
            channel=channel,
            to=to,
            text=text,
            enqueued_at=time.time(),
            next_retry_at=0.0,
        )
        self._write_entry(entry)
        return delivery_id

    def _write_entry(self, entry: QueuedDelivery) -> None:
        """通过 tmp + os.replace() 实现原子写入."""
        final_path = self.queue_dir / f"{entry.id}.json"
        tmp_path = self.queue_dir / f".tmp.{os.getpid()}.{entry.id}.json"
        data = json.dumps(entry.to_dict(), indent=2, ensure_ascii=False)
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp_path), str(final_path))

    def _read_entry(self, delivery_id: str) -> QueuedDelivery | None:
        file_path = self.queue_dir / f"{delivery_id}.json"
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return QueuedDelivery.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def ack(self, delivery_id: str) -> None:
        """投递成功 -- 删除队列文件."""
        file_path = self.queue_dir / f"{delivery_id}.json"
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass

    def fail(self, delivery_id: str, error: str) -> None:
        """递增重试计数, 计算下次重试时间. 重试耗尽时移入 failed/ 目录."""
        entry = self._read_entry(delivery_id)
        if entry is None:
            return
        entry.retry_count += 1
        entry.last_error = error
        if entry.retry_count >= MAX_RETRIES:
            self.move_to_failed(delivery_id)
            return
        backoff_ms = compute_backoff_ms(entry.retry_count)
        entry.next_retry_at = time.time() + backoff_ms / 1000.0
        self._write_entry(entry)

    def move_to_failed(self, delivery_id: str) -> None:
        src = self.queue_dir / f"{delivery_id}.json"
        dst = self.failed_dir / f"{delivery_id}.json"
        try:
            os.replace(str(src), str(dst))
        except FileNotFoundError:
            pass

    def load_pending(self) -> list[QueuedDelivery]:
        """扫描队列目录, 加载所有待处理条目, 按入队时间排序."""
        entries: list[QueuedDelivery] = []
        if not self.queue_dir.exists():
            return entries
        for file_path in self.queue_dir.glob("*.json"):
            if not file_path.is_file():
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append(QueuedDelivery.from_dict(data))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        entries.sort(key=lambda e: e.enqueued_at)
        return entries

    def load_failed(self) -> list[QueuedDelivery]:
        entries: list[QueuedDelivery] = []
        if not self.failed_dir.exists():
            return entries
        for file_path in self.failed_dir.glob("*.json"):
            if not file_path.is_file():
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entries.append(QueuedDelivery.from_dict(data))
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        entries.sort(key=lambda e: e.enqueued_at)
        return entries

    def retry_failed(self) -> int:
        """将所有 failed/ 条目移回队列, 重置重试计数."""
        count = 0
        if not self.failed_dir.exists():
            return count
        for file_path in self.failed_dir.glob("*.json"):
            if not file_path.is_file():
                continue
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                entry = QueuedDelivery.from_dict(data)
                entry.retry_count = 0
                entry.last_error = None
                entry.next_retry_at = 0.0
                self._write_entry(entry)
                file_path.unlink()
                count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return count


# ---------------------------------------------------------------------------
# 3. 渠道感知的消息分片
# ---------------------------------------------------------------------------

CHANNEL_LIMITS: dict[str, int] = {
    "telegram": 4096,
    "telegram_caption": 1024,
    "discord": 2000,
    "whatsapp": 4096,
    "default": 4096,
}


def chunk_message(text: str, channel: str = "default") -> list[str]:
    """将消息按平台限制分片. 两级拆分: 段落, 然后硬切."""
    if not text:
        return []
    limit = CHANNEL_LIMITS.get(channel, CHANNEL_LIMITS["default"])
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    for para in text.split("\n\n"):
        if chunks and len(chunks[-1]) + len(para) + 2 <= limit:
            chunks[-1] += "\n\n" + para
        else:
            while len(para) > limit:
                chunks.append(para[:limit])
                para = para[limit:]
            if para:
                chunks.append(para)
    return chunks or [text[:limit]]


# ---------------------------------------------------------------------------
# 4. DeliveryRunner -- 后台投递线程
# ---------------------------------------------------------------------------

class DeliveryRunner:
    def __init__(
        self,
        queue: DeliveryQueue,
        deliver_fn: Callable[[str, str, str], None],
    ):
        self.queue = queue
        self.deliver_fn = deliver_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.total_attempted = 0
        self.total_succeeded = 0
        self.total_failed = 0

    def start(self) -> None:
        """运行恢复扫描, 然后启动后台投递线程."""
        self._recovery_scan()
        self._thread = threading.Thread(
            target=self._background_loop,
            daemon=True,
            name="delivery-runner",
        )
        self._thread.start()

    def _recovery_scan(self) -> None:
        """启动时统计待处理和失败条目."""
        pending = self.queue.load_pending()
        failed = self.queue.load_failed()
        parts = []
        if pending:
            parts.append(f"{len(pending)} pending")
        if failed:
            parts.append(f"{len(failed)} failed")
        if parts:
            print_delivery(f"Recovery: {', '.join(parts)}")
        else:
            print_delivery("Recovery: queue is clean")

    def _background_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._process_pending()
            except Exception as exc:
                print_error(f"Delivery loop error: {exc}")
            self._stop_event.wait(timeout=1.0)

    def _process_pending(self) -> None:
        """处理所有 next_retry_at <= now 的待处理条目."""
        pending = self.queue.load_pending()
        now = time.time()

        for entry in pending:
            if self._stop_event.is_set():
                break
            if entry.next_retry_at > now:
                continue

            self.total_attempted += 1
            try:
                self.deliver_fn(entry.channel, entry.to, entry.text)
                self.queue.ack(entry.id)
                self.total_succeeded += 1
            except Exception as exc:
                error_msg = str(exc)
                self.queue.fail(entry.id, error_msg)
                self.total_failed += 1
                retry_info = f"retry {entry.retry_count + 1}/{MAX_RETRIES}"
                if entry.retry_count + 1 >= MAX_RETRIES:
                    print_warn(
                        f"Delivery {entry.id[:8]}... -> failed/ ({retry_info}): {error_msg}"
                    )
                else:
                    backoff = compute_backoff_ms(entry.retry_count + 1)
                    print_warn(
                        f"Delivery {entry.id[:8]}... failed ({retry_info}), "
                        f"next retry in {backoff / 1000:.0f}s: {error_msg}"
                    )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def get_stats(self) -> dict:
        pending = self.queue.load_pending()
        failed = self.queue.load_failed()
        return {
            "pending": len(pending),
            "failed": len(failed),
            "total_attempted": self.total_attempted,
            "total_succeeded": self.total_succeeded,
            "total_failed": self.total_failed,
        }


# ---------------------------------------------------------------------------
# 5. MockDeliveryChannel -- 模拟投递渠道
# ---------------------------------------------------------------------------

class MockDeliveryChannel:
    def __init__(self, name: str, fail_rate: float = 0.0):
        self.name = name
        self.fail_rate = fail_rate
        self.sent: list[dict] = []

    def send(self, to: str, text: str) -> None:
        """模拟发送. 按配置的失败率抛出 ConnectionError."""
        if random.random() < self.fail_rate:
            raise ConnectionError(
                f"[{self.name}] Simulated delivery failure to {to}"
            )
        self.sent.append({"to": to, "text": text, "time": time.time()})
        preview = text[:60].replace("\n", " ")
        print_delivery(f"[{self.name}] -> {to}: {preview}...")

    def set_fail_rate(self, rate: float) -> None:
        self.fail_rate = max(0.0, min(1.0, rate))


# ---------------------------------------------------------------------------
# 6. Soul + Memory (简化版, 带工具集成)
# ---------------------------------------------------------------------------


class SoulSystem:
    def __init__(self):
        soul_path = WORKSPACE_DIR / "SOUL.md"
        if soul_path.exists():
            self.personality = soul_path.read_text(encoding="utf-8")
        else:
            self.personality = ""

    def get_system_prompt(self) -> str:
        base = SYSTEM_PROMPT
        if self.personality:
            base = f"{self.personality}\n\n{base}"
        return base


class MemoryStore:
    def __init__(self):
        self.memory_file = WORKSPACE_DIR / "memory.jsonl"
        if not self.memory_file.exists():
            self.memory_file.touch()

    def write(self, content: str) -> str:
        entry = {"content": content, "time": time.time()}
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        return f"Saved: {content[:50]}"

    def search(self, query: str) -> str:
        if not self.memory_file.exists():
            return "No memories found."
        query_lower = query.lower()
        results: list[str] = []
        with open(self.memory_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    if query_lower in entry.get("content", "").lower():
                        results.append(entry["content"])
                except json.JSONDecodeError:
                    continue
        if not results:
            return "No memories found."
        return "\n".join(f"- {r}" for r in results[-5:])


TOOLS = [
    {
        "name": "memory_write",
        "description": "Save an important fact or preference to long-term memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The fact or preference to remember.",
                },
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search long-term memory for relevant facts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
            },
            "required": ["query"],
        },
    },
]


# ---------------------------------------------------------------------------
# 7. HeartbeatRunner -- 通过 DeliveryQueue 入队的定时器
# ---------------------------------------------------------------------------


class HeartbeatRunner:
    def __init__(
        self,
        queue: DeliveryQueue,
        channel: str,
        to: str,
        interval: float = 60.0,
    ):
        self.queue = queue
        self.channel = channel
        self.to = to
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lane_lock = threading.Lock()
        self.last_run: float = 0.0
        self.run_count: int = 0
        self.enabled: bool = False

    def start(self) -> None:
        self.enabled = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="heartbeat-runner",
        )
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.interval)
            if self._stop_event.is_set():
                break
            if not self.enabled:
                continue
            self.trigger()

    def trigger(self) -> None:
        """生成 heartbeat 文本并入队投递."""
        with self._lane_lock:
            self.last_run = time.time()
            self.run_count += 1
            heartbeat_text = (
                f"[Heartbeat #{self.run_count}] "
                f"System check at {time.strftime('%H:%M:%S')} -- all OK."
            )
            chunks = chunk_message(heartbeat_text, self.channel)
            for chunk in chunks:
                self.queue.enqueue(self.channel, self.to, chunk)
            print_info(f"  {MAGENTA}[heartbeat]{RESET} triggered #{self.run_count}")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "interval": self.interval,
            "run_count": self.run_count,
            "last_run": time.strftime(
                "%H:%M:%S", time.localtime(self.last_run)
            ) if self.last_run else "never",
        }


# ---------------------------------------------------------------------------
# 8. Agent 循环 + REPL
# ---------------------------------------------------------------------------


def process_tool_call(
    tool_name: str,
    tool_input: dict,
    memory: MemoryStore,
) -> str:
    if tool_name == "memory_write":
        return memory.write(tool_input["content"])
    elif tool_name == "memory_search":
        return memory.search(tool_input["query"])
    return f"Error: Unknown tool '{tool_name}'"


def handle_repl_command(
    cmd: str,
    queue: DeliveryQueue,
    runner: DeliveryRunner,
    heartbeat: HeartbeatRunner,
    mock_channel: MockDeliveryChannel,
) -> bool:
    """处理 REPL 命令. 返回 True 表示命令已处理."""
    if cmd == "/queue":
        pending = queue.load_pending()
        if not pending:
            print_info("  Queue is empty.")
            return True
        print_info(f"  Pending deliveries ({len(pending)}):")
        now = time.time()
        for entry in pending:
            wait = ""
            if entry.next_retry_at > now:
                remaining = entry.next_retry_at - now
                wait = f", wait {remaining:.0f}s"
            preview = entry.text[:40].replace("\n", " ")
            print_info(
                f"    {entry.id[:8]}... "
                f"retry={entry.retry_count}{wait} "
                f'"{preview}"'
            )
        return True

    if cmd == "/failed":
        failed = queue.load_failed()
        if not failed:
            print_info("  No failed deliveries.")
            return True
        print_info(f"  Failed deliveries ({len(failed)}):")
        for entry in failed:
            preview = entry.text[:40].replace("\n", " ")
            err = entry.last_error or "unknown"
            print_info(
                f"    {entry.id[:8]}... "
                f"retries={entry.retry_count} "
                f'error="{err[:30]}" '
                f'"{preview}"'
            )
        return True

    if cmd == "/retry":
        count = queue.retry_failed()
        print_info(f"  Moved {count} entries back to queue.")
        return True

    if cmd == "/simulate-failure":
        if mock_channel.fail_rate > 0:
            mock_channel.set_fail_rate(0.0)
            print_info(f"  {mock_channel.name} fail rate -> 0% (reliable)")
        else:
            mock_channel.set_fail_rate(0.5)
            print_info(f"  {mock_channel.name} fail rate -> 50% (unreliable)")
        return True

    if cmd == "/heartbeat":
        status = heartbeat.get_status()
        print_info(f"  Heartbeat: enabled={status['enabled']}, "
                   f"interval={status['interval']}s, "
                   f"runs={status['run_count']}, "
                   f"last={status['last_run']}")
        return True

    if cmd == "/trigger":
        heartbeat.trigger()
        return True

    if cmd == "/stats":
        stats = runner.get_stats()
        print_info(f"  Delivery stats: "
                   f"pending={stats['pending']}, "
                   f"failed={stats['failed']}, "
                   f"attempted={stats['total_attempted']}, "
                   f"succeeded={stats['total_succeeded']}, "
                   f"errors={stats['total_failed']}")
        return True

    return False


def agent_loop() -> None:
    soul = SoulSystem()
    memory = MemoryStore()
    system_prompt = soul.get_system_prompt()

    mock_channel = MockDeliveryChannel("console", fail_rate=0.0)
    default_channel = "console"
    default_to = "user"

    queue = DeliveryQueue()

    def deliver_fn(channel: str, to: str, text: str) -> None:
        mock_channel.send(to, text)

    runner = DeliveryRunner(queue, deliver_fn)
    runner.start()

    heartbeat = HeartbeatRunner(
        queue=queue,
        channel=default_channel,
        to=default_to,
        interval=120.0,
    )
    heartbeat.start()

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 08: Delivery")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Queue: {QUEUE_DIR}")
    print_info("  Commands:")
    print_info("    /queue             - show pending deliveries")
    print_info("    /failed            - show failed deliveries")
    print_info("    /retry             - retry all failed")
    print_info("    /simulate-failure  - toggle 50% failure rate")
    print_info("    /heartbeat         - heartbeat status")
    print_info("    /trigger           - manually trigger heartbeat")
    print_info("    /stats             - delivery statistics")
    print_info("  Type 'quit' or 'exit' to leave.")
    print_info("=" * 60)
    print()

    while True:
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        if user_input.startswith("/"):
            if handle_repl_command(
                user_input, queue, runner, heartbeat, mock_channel
            ):
                continue
            print_info(f"  Unknown command: {user_input}")
            continue

        messages.append({"role": "user", "content": user_input})

        # Agent 内循环 (工具调用)
        while True:
            try:
                response = client.messages.create(
                    model=MODEL_ID,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            if response.stop_reason == "end_turn":
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                    chunks = chunk_message(assistant_text, default_channel)
                    for chunk in chunks:
                        queue.enqueue(default_channel, default_to, chunk)
                break

            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = process_tool_call(block.name, block.input, memory)
                    print_info(f"  {DIM}[tool: {block.name}]{RESET}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
                messages.append({"role": "user", "content": tool_results})
                continue

            else:
                print_info(f"[stop_reason={response.stop_reason}]")
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                    chunks = chunk_message(assistant_text, default_channel)
                    for chunk in chunks:
                        queue.enqueue(default_channel, default_to, chunk)
                break

    heartbeat.stop()
    runner.stop()
    print_info("Delivery runner stopped. Queue state preserved on disk.")


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
