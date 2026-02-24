"""
Section 10: Delivery Queue & Reliable Messaging
"Write before you send, retry until you succeed"

OpenClaw 的消息投递采用 write-ahead queue 模式:
先把消息持久化到磁盘, 再尝试发送. 成功则删除记录 (ack), 失败则按指数退避重试.
即使网关进程崩溃, 重启后也能从磁盘恢复未送达的消息.

在 OpenClaw 中 (src/infra/outbound/delivery-queue.ts):
  - enqueue: 原子写入磁盘 (先写 .tmp 再 rename)
  - ack: 发送成功后删除文件
  - fail: 发送失败后更新重试计数和退避时间
  - move_to_failed: 超过 MAX_RETRIES 后移入 failed/
  - recoverPendingDeliveries: 网关启动时扫描队列恢复
  - computeBackoffMs: 退避时间表 [5s, 25s, 2m, 10m]

架构图:

  Agent Response
       |
       v
  enqueue (disk write)   <-- 先写后发, 保证不丢
       |
       v
  attempt delivery
       |
       +---- success ----> ack (delete .json)
       |
       +---- failure ----> fail (update retry_count + backoff)
                |
                +-- retry_count <= MAX_RETRIES --> wait backoff, retry
                |     backoff: [5s, 25s, 2m, 10m]
                |
                +-- retry_count > MAX_RETRIES ---> move to failed/

  Gateway restart --> recovery_scan() --> resume pending entries

典型场景:

  "小明的 Telegram API 断了 2 分钟"
    -> enqueue 写入磁盘 -> 5s 后重试失败 -> 25s 后重试成功 -> ack

  "小明的网关进程崩溃重启"
    -> recovery_scan 发现 pending 文件 -> 按时间排序依次重投 -> 全部送达

  "消息发送连续失败 5 次"
    -> move_to_failed 移入 failed/ -> 管理员 /retry 手动恢复

运行方式:
    cd mini-claw
    python agents/s10_delivery.py

需要在 .env 中配置:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
import json
import math
import os
import random
import re
import sys
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "60"))
HEARTBEAT_ACTIVE_START = int(os.getenv("HEARTBEAT_ACTIVE_START", "9"))
HEARTBEAT_ACTIVE_END = int(os.getenv("HEARTBEAT_ACTIVE_END", "22"))
HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"

# 投递队列退避时间表 (毫秒), 与 OpenClaw delivery-queue.ts 一致
BACKOFF_MS = [5_000, 25_000, 120_000, 600_000]
MAX_RETRIES = 5

BASE_SYSTEM_PROMPT = (
    "You are a helpful AI assistant running on the mini-claw framework.\n"
    "Current date and time: {datetime}\n"
    "You have access to memory tools to store and recall information.\n"
    "Your responses will be delivered through a reliable delivery queue."
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


def print_tool(name: str, detail: str) -> None:
    print(f"  {MAGENTA}[tool:{name}]{RESET} {DIM}{detail}{RESET}")


def print_heartbeat(text: str) -> None:
    print(f"\n{BLUE}{BOLD}[Heartbeat]{RESET} {text}\n")


def print_delivery(text: str) -> None:
    """投递状态用橙色标记."""
    print(f"  {ORANGE}[delivery]{RESET} {DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# QueuedDelivery 数据结构
# ---------------------------------------------------------------------------

@dataclass
class QueuedDelivery:
    """一条待投递的消息记录, 对应 OpenClaw 的 QueuedDelivery 接口."""
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
            "id": self.id, "channel": self.channel, "to": self.to,
            "text": self.text, "retry_count": self.retry_count,
            "last_error": self.last_error, "enqueued_at": self.enqueued_at,
            "next_retry_at": self.next_retry_at,
        }

    @staticmethod
    def from_dict(data: dict) -> "QueuedDelivery":
        return QueuedDelivery(
            id=data["id"], channel=data["channel"], to=data["to"],
            text=data["text"], retry_count=data.get("retry_count", 0),
            last_error=data.get("last_error"),
            enqueued_at=data.get("enqueued_at", 0.0),
            next_retry_at=data.get("next_retry_at", 0.0),
        )


# ---------------------------------------------------------------------------
# DeliveryQueue -- 磁盘持久化投递队列
# ---------------------------------------------------------------------------
# 对应 OpenClaw src/infra/outbound/delivery-queue.ts.
# 设计要点: 原子写入 (tmp + rename), ack 删除, fail 更新退避, move_to_failed.
# ---------------------------------------------------------------------------

class DeliveryQueue:
    """磁盘持久化的消息投递队列.

    目录结构:
      delivery-queue/
        {uuid}.json          -- 待投递
        failed/
          {uuid}.json        -- 超过最大重试次数
    """

    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.failed_dir = queue_dir / "failed"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, channel: str, to: str, text: str) -> str:
        """将消息写入队列, 返回投递 ID.

        对应 enqueueDelivery(): 生成 UUID -> 构建记录 -> 原子写入.
        """
        delivery_id = uuid.uuid4().hex[:16]
        entry = QueuedDelivery(
            id=delivery_id, channel=channel, to=to, text=text,
            enqueued_at=time.time(), next_retry_at=0.0,
        )
        self._write_entry(entry)
        return delivery_id

    def _entry_path(self, delivery_id: str) -> Path:
        return self.queue_dir / f"{delivery_id}.json"

    def _read_entry(self, delivery_id: str) -> QueuedDelivery | None:
        try:
            raw = self._entry_path(delivery_id).read_text(encoding="utf-8")
            return QueuedDelivery.from_dict(json.loads(raw))
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _write_entry(self, entry: QueuedDelivery) -> None:
        """原子写入: 先写 .tmp.{pid}, 再 os.replace. 对应 OpenClaw 的写入模式."""
        file_path = self._entry_path(entry.id)
        tmp_path = file_path.parent / f".tmp.{os.getpid()}.{entry.id}.json"
        tmp_path.write_text(
            json.dumps(entry.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(file_path))

    def ack(self, delivery_id: str) -> None:
        """确认投递成功, 删除队列文件. 对应 ackDelivery()."""
        try:
            self._entry_path(delivery_id).unlink()
        except FileNotFoundError:
            pass

    def fail(self, delivery_id: str, error: str) -> None:
        """记录投递失败. 对应 failDelivery().

        更新 retry_count, last_error, next_retry_at.
        超过 MAX_RETRIES 则 move_to_failed.
        """
        entry = self._read_entry(delivery_id)
        if entry is None:
            return
        entry.retry_count += 1
        entry.last_error = error
        if entry.retry_count > MAX_RETRIES:
            self.move_to_failed(delivery_id)
            return
        backoff_ms = self.compute_backoff_ms(entry.retry_count)
        entry.next_retry_at = time.time() + (backoff_ms / 1000.0)
        self._write_entry(entry)

    def move_to_failed(self, delivery_id: str) -> None:
        """移入 failed/ 目录. 对应 moveToFailed()."""
        try:
            os.replace(
                str(self._entry_path(delivery_id)),
                str(self.failed_dir / f"{delivery_id}.json"),
            )
        except FileNotFoundError:
            pass

    def _scan_dir(self, directory: Path) -> list[QueuedDelivery]:
        """扫描目录下所有 *.json, 返回按 enqueued_at 排序的记录."""
        entries: list[QueuedDelivery] = []
        if not directory.exists():
            return entries
        for fp in directory.iterdir():
            if not fp.is_file() or not fp.name.endswith(".json"):
                continue
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                entries.append(QueuedDelivery.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue
        entries.sort(key=lambda e: e.enqueued_at)
        return entries

    def load_pending(self) -> list[QueuedDelivery]:
        """加载所有待投递记录. 对应 loadPendingDeliveries()."""
        return self._scan_dir(self.queue_dir)

    def load_failed(self) -> list[QueuedDelivery]:
        """加载 failed/ 目录下所有记录."""
        return self._scan_dir(self.failed_dir)

    def retry_failed(self) -> int:
        """将 failed/ 下所有记录移回队列, 重置重试计数. 返回移回数量."""
        count = 0
        for fp in list(self.failed_dir.iterdir()):
            if not fp.is_file() or not fp.name.endswith(".json"):
                continue
            try:
                entry = QueuedDelivery.from_dict(
                    json.loads(fp.read_text(encoding="utf-8"))
                )
                entry.retry_count = 0
                entry.next_retry_at = 0.0
                entry.last_error = None
                self._write_entry(entry)
                fp.unlink()
                count += 1
            except (json.JSONDecodeError, KeyError, OSError):
                continue
        return count

    @staticmethod
    def compute_backoff_ms(retry_count: int) -> int:
        """计算退避时间 (ms). 对应 computeBackoffMs().

        退避表: retry1=5s, retry2=25s, retry3=2m, retry4+=10m.
        """
        if retry_count <= 0:
            return 0
        return BACKOFF_MS[min(retry_count - 1, len(BACKOFF_MS) - 1)]


# ---------------------------------------------------------------------------
# DeliveryRunner -- 后台投递线程
# ---------------------------------------------------------------------------

class DeliveryRunner:
    """后台投递线程. deliver_fn: (channel, to, text) -> None, 失败则抛异常."""

    def __init__(self, queue: DeliveryQueue, deliver_fn: "callable"):
        self.queue = queue
        self.deliver_fn = deliver_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._total_attempted = 0
        self._total_succeeded = 0
        self._total_failed = 0

    def _attempt_delivery(self, entry: QueuedDelivery) -> bool:
        try:
            self.deliver_fn(entry.channel, entry.to, entry.text)
            return True
        except Exception:
            return False

    def _process_pending(self) -> None:
        """处理所有到期的待投递条目."""
        now = time.time()
        for entry in self.queue.load_pending():
            if self._stop_event.is_set():
                break
            if entry.next_retry_at > now:
                continue
            self._total_attempted += 1
            if self._attempt_delivery(entry):
                self.queue.ack(entry.id)
                self._total_succeeded += 1
                print_delivery(f"delivered {entry.id[:8]}.. to {entry.channel}:{entry.to}")
            else:
                error_msg = "delivery failed"
                try:
                    self.deliver_fn(entry.channel, entry.to, entry.text)
                except Exception as exc:
                    error_msg = str(exc)
                self._total_failed += 1
                self.queue.fail(entry.id, error_msg)
                retry_n = entry.retry_count + 1
                if retry_n > MAX_RETRIES:
                    print_delivery(f"FAILED {entry.id[:8]}.. -> moved to failed/")
                else:
                    bk = DeliveryQueue.compute_backoff_ms(retry_n)
                    print_delivery(
                        f"failed {entry.id[:8]}.. "
                        f"(retry {retry_n}/{MAX_RETRIES}, next in {bk/1000:.0f}s)"
                    )

    def _recovery_scan(self) -> None:
        """启动时扫描队列, 报告待恢复的条目数."""
        pending = self.queue.load_pending()
        failed = self.queue.load_failed()
        if pending:
            print_delivery(f"recovery: {len(pending)} pending entries, resuming")
        if failed:
            print_delivery(f"recovery: {len(failed)} entries in failed/")
        if not pending and not failed:
            print_delivery("recovery: queue empty")

    def _background_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                with self._lock:
                    self._process_pending()
            except Exception as exc:
                print_delivery(f"runner error: {exc}")
            self._stop_event.wait(1.0)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._recovery_scan()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_loop, daemon=True, name="delivery-runner",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def get_stats(self) -> dict:
        return {
            "pending": len(self.queue.load_pending()),
            "failed": len(self.queue.load_failed()),
            "total_attempted": self._total_attempted,
            "total_succeeded": self._total_succeeded,
            "total_failed": self._total_failed,
        }


# ---------------------------------------------------------------------------
# MockDeliveryChannel -- 模拟投递渠道
# ---------------------------------------------------------------------------

class MockDeliveryChannel:
    """模拟投递渠道, 支持可配置的失败率."""

    def __init__(self, name: str, fail_rate: float = 0.0):
        self.name = name
        self.fail_rate = fail_rate

    def send(self, to: str, text: str) -> None:
        if random.random() < self.fail_rate:
            raise ConnectionError(
                f"channel={self.name}: connection refused (simulated)"
            )
        preview = text[:80].replace("\n", " ")
        if len(text) > 80:
            preview += "..."
        print(f"\n{GREEN}[{self.name} -> {to}]{RESET} {preview}\n")

    def set_fail_rate(self, rate: float) -> None:
        self.fail_rate = max(0.0, min(1.0, rate))


# ---------------------------------------------------------------------------
# Soul System (简化复用)
# ---------------------------------------------------------------------------

class SoulSystem:
    def __init__(self, soul_dir: Path):
        self.soul_path = soul_dir / "SOUL.md"

    def load_soul(self) -> str:
        if self.soul_path.exists():
            return self.soul_path.read_text(encoding="utf-8").strip()
        return ""

    def build_system_prompt(self, base_prompt: str) -> str:
        soul = self.load_soul()
        return f"{soul}\n\n---\n\n{base_prompt}" if soul else base_prompt


# ---------------------------------------------------------------------------
# Memory System (简化复用)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower()) if len(t) > 1]


class MemoryStore:
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.evergreen_path = memory_dir / "MEMORY.md"
        self.daily_dir = memory_dir / "memory"
        self.daily_dir.mkdir(parents=True, exist_ok=True)

    def write_memory(self, content: str, category: str = "general") -> str:
        today = date.today().isoformat()
        path = self.daily_dir / f"{today}.md"
        entry = f"\n## [{datetime.now().strftime('%H:%M:%S')}] {category}\n\n{content}\n"
        if not path.exists():
            path.write_text(f"# Memory Log: {today}\n", encoding="utf-8")
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)
        return f"memory/{today}.md"

    def load_evergreen(self) -> str:
        if self.evergreen_path.exists():
            return self.evergreen_path.read_text(encoding="utf-8").strip()
        return ""

    def get_recent_memories(self, days: int = 3) -> list[dict]:
        results = []
        today = date.today()
        for i in range(days):
            d = today - timedelta(days=i)
            path = self.daily_dir / f"{d.isoformat()}.md"
            if path.exists():
                results.append({
                    "date": d.isoformat(),
                    "content": path.read_text(encoding="utf-8").strip(),
                })
        return results

    def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
        chunks = self._load_all_chunks()
        if not chunks:
            return []
        doc_freq: Counter = Counter()
        tok_lists = []
        for ch in chunks:
            toks = _tokenize(ch["text"])
            for t in set(toks):
                doc_freq[t] += 1
            tok_lists.append(toks)
        n = len(chunks)
        q_toks = _tokenize(query)
        q_tf = Counter(q_toks)
        q_vec = {t: (c / max(len(q_toks), 1)) * (math.log(n / doc_freq[t]) if doc_freq[t] else 0)
                 for t, c in q_tf.items()}
        scored = []
        for i, ch in enumerate(chunks):
            toks = tok_lists[i]
            if not toks:
                continue
            tf = Counter(toks)
            c_vec = {t: (c / len(toks)) * (math.log(n / doc_freq[t]) if doc_freq[t] else 0)
                     for t, c in tf.items()}
            common = set(q_vec) & set(c_vec)
            if not common:
                continue
            dot = sum(q_vec[k] * c_vec[k] for k in common)
            na = math.sqrt(sum(v * v for v in q_vec.values()))
            nb = math.sqrt(sum(v * v for v in c_vec.values()))
            if na == 0 or nb == 0:
                continue
            s = dot / (na * nb)
            if s > 0.01:
                scored.append({"path": ch["path"], "score": round(s, 4), "snippet": ch["text"][:300]})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _load_all_chunks(self) -> list[dict]:
        chunks: list[dict] = []
        if self.evergreen_path.exists():
            chunks.extend(self._split(self.evergreen_path.read_text(encoding="utf-8"), "MEMORY.md"))
        for f in sorted(self.daily_dir.glob("*.md"), reverse=True):
            chunks.extend(self._split(f.read_text(encoding="utf-8"), f"memory/{f.name}"))
        return chunks

    @staticmethod
    def _split(content: str, path: str) -> list[dict]:
        chunks: list[dict] = []
        buf: list[str] = []
        for line in content.split("\n"):
            if line.startswith("#") and buf:
                t = "\n".join(buf).strip()
                if t:
                    chunks.append({"path": path, "text": t})
                buf = [line]
            else:
                buf.append(line)
        if buf:
            t = "\n".join(buf).strip()
            if t:
                chunks.append({"path": path, "text": t})
        return chunks


# ---------------------------------------------------------------------------
# HeartbeatRunner (简化版, 完整版见 s08)
# ---------------------------------------------------------------------------

class HeartbeatRunner:
    def __init__(self, interval_seconds: int = 1800,
                 active_hours: tuple[int, int] = (9, 22),
                 heartbeat_path: Path | None = None):
        self.interval = interval_seconds
        self.active_start, self.active_end = active_hours
        self.heartbeat_path = heartbeat_path or (WORKSPACE_DIR / "HEARTBEAT.md")
        self.last_run: float = 0.0
        self.running = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._output_queue: list[str] = []
        self._output_lock = threading.Lock()

    def _is_enabled(self) -> bool:
        return self.heartbeat_path.exists()

    def _interval_elapsed(self) -> bool:
        return (time.time() - self.last_run) >= self.interval

    def _is_active_hours(self) -> bool:
        h = datetime.now().hour
        if self.active_start <= self.active_end:
            return self.active_start <= h < self.active_end
        return h >= self.active_start or h < self.active_end

    def should_run(self) -> tuple[bool, str]:
        if not self._is_enabled():
            return False, "disabled"
        if not self._interval_elapsed():
            return False, "interval not elapsed"
        if not self._is_active_hours():
            return False, "outside active hours"
        if self.running:
            return False, "already running"
        return True, "ok"

    def run_heartbeat_turn(self, agent_fn) -> str | None:
        if not self.heartbeat_path.exists():
            return None
        content = self.heartbeat_path.read_text(encoding="utf-8").strip()
        prompt = (
            "This is a scheduled heartbeat check. "
            "Follow the HEARTBEAT.md instructions strictly.\n"
            "If nothing needs attention, respond with exactly: HEARTBEAT_OK\n\n"
            f"--- HEARTBEAT.md ---\n{content}\n--- end ---"
        )
        resp = agent_fn(prompt)
        if not resp:
            return None
        without = resp.strip().replace(HEARTBEAT_OK_TOKEN, "").strip()
        if not without or len(without) <= 5:
            return None
        return without if HEARTBEAT_OK_TOKEN in resp else resp.strip()

    def _background_loop(self, agent_fn) -> None:
        while not self._stop_event.is_set():
            should, _ = self.should_run()
            if should:
                acquired = self._lock.acquire(blocking=False)
                if not acquired:
                    self._stop_event.wait(1.0)
                    continue
                try:
                    self.running = True
                    self.last_run = time.time()
                    result = self.run_heartbeat_turn(agent_fn)
                    if result:
                        with self._output_lock:
                            self._output_queue.append(result)
                except Exception as exc:
                    with self._output_lock:
                        self._output_queue.append(f"[heartbeat error: {exc}]")
                finally:
                    self.running = False
                    self._lock.release()
            self._stop_event.wait(1.0)

    def start(self, agent_fn) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_loop, args=(agent_fn,), daemon=True,
            name="heartbeat-runner",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def drain_output(self) -> list[str]:
        with self._output_lock:
            msgs = self._output_queue[:]
            self._output_queue.clear()
            return msgs


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------

memory_store = MemoryStore(WORKSPACE_DIR)

TOOLS = [
    {
        "name": "memory_write",
        "description": "Write a memory to persistent storage.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The information to remember."},
                "category": {"type": "string", "description": "Category: preference, fact, todo."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search through stored memories.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {"type": "integer", "description": "Max results. Default 5."},
            },
            "required": ["query"],
        },
    },
]

TOOL_HANDLERS = {
    "memory_write": lambda p: json.dumps({
        "status": "saved",
        "path": memory_store.write_memory(p.get("content", ""), p.get("category", "general")),
    }),
    "memory_search": lambda p: json.dumps({
        "results": memory_store.search_memory(p.get("query", ""), p.get("top_k", 5)),
    }),
}

# ---------------------------------------------------------------------------
# System Prompt + Agent 执行
# ---------------------------------------------------------------------------

soul_system = SoulSystem(WORKSPACE_DIR)


def build_system_prompt() -> str:
    base = BASE_SYSTEM_PROMPT.format(datetime=datetime.now().strftime("%Y-%m-%d %H:%M"))
    prompt = soul_system.build_system_prompt(base)
    evergreen = memory_store.load_evergreen()
    if evergreen:
        prompt += f"\n\n---\n\n## Evergreen Memory\n\n{evergreen}"
    for entry in memory_store.get_recent_memories(days=3):
        prompt += f"\n\n### {entry['date']}\n{entry['content'][:500]}"
    return prompt


def run_agent_single_turn(prompt: str) -> str:
    """单轮 agent 调用 (无工具), 用于心跳."""
    try:
        resp = client.messages.create(
            model=MODEL_ID, max_tokens=1024,
            system=build_system_prompt(),
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text")).strip()
    except Exception as exc:
        return f"[agent error: {exc}]"


# ---------------------------------------------------------------------------
# Agent 循环 + Delivery Queue + Heartbeat
# ---------------------------------------------------------------------------
# 核心整合: agent 回复不直接输出, 而是 enqueue 到投递队列 (write-ahead).
# DeliveryRunner 后台线程从队列取出, 通过 MockDeliveryChannel 投递.
# ---------------------------------------------------------------------------

def agent_loop(
    heartbeat: HeartbeatRunner,
    delivery_runner: DeliveryRunner,
    delivery_queue: DeliveryQueue,
    mock_channel: MockDeliveryChannel,
) -> None:
    messages: list[dict] = []

    print_info("=" * 64)
    print_info("  Mini-Claw  |  Section 10: Delivery Queue & Reliable Messaging")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Queue: {delivery_queue.queue_dir}")
    print_info("-" * 64)
    print_info("  /queue  /failed  /retry  /simulate-failure")
    print_info("  /heartbeat  /trigger  quit/exit")
    print_info("=" * 64)
    print()

    while True:
        # 心跳产生的内容通过投递队列发送
        for msg in heartbeat.drain_output():
            did = delivery_queue.enqueue(mock_channel.name, "user", msg)
            print_heartbeat(f"enqueued heartbeat message (id={did[:8]}..)")

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

        # -- /queue --
        if user_input == "/queue":
            pending = delivery_queue.load_pending()
            stats = delivery_runner.get_stats()
            print(f"\n{ORANGE}--- Delivery Queue ---{RESET}")
            print(f"  Pending: {len(pending)}  "
                  f"Attempted: {stats['total_attempted']}  "
                  f"OK: {stats['total_succeeded']}  "
                  f"Fail: {stats['total_failed']}")
            now = time.time()
            for e in pending:
                retry_s = f"{e.retry_count}/{MAX_RETRIES}"
                wait = f"in {e.next_retry_at - now:.0f}s" if e.next_retry_at > now else "ready"
                print(f"  {e.id[:12]:<12} retry={retry_s:<6} {wait:<12} {e.text[:40]}")
            print(f"{ORANGE}---{RESET}\n")
            continue

        # -- /failed --
        if user_input == "/failed":
            failed = delivery_queue.load_failed()
            print(f"\n{RED}--- Failed ({len(failed)}) ---{RESET}")
            for e in failed:
                err = (e.last_error or "")[:30]
                print(f"  {e.id[:12]} retries={e.retry_count} err={err}")
            print(f"  Use /retry to move back.{RESET}\n")
            continue

        # -- /retry --
        if user_input == "/retry":
            n = delivery_queue.retry_failed()
            print_info(f"Moved {n} entries from failed/ back to queue.\n" if n else "No failed entries.\n")
            continue

        # -- /simulate-failure --
        if user_input == "/simulate-failure":
            if mock_channel.fail_rate < 0.01:
                mock_channel.set_fail_rate(0.5)
                print(f"\n{YELLOW}Fail rate -> 50%{RESET}\n")
            else:
                mock_channel.set_fail_rate(0.0)
                print(f"\n{GREEN}Fail rate -> 0%{RESET}\n")
            continue

        # -- /heartbeat --
        if user_input == "/heartbeat":
            should, reason = heartbeat.should_run()
            elapsed = time.time() - heartbeat.last_run
            print(f"\n{BLUE}--- Heartbeat ---{RESET}")
            print(f"  Enabled: {heartbeat._is_enabled()}  Should: {should} ({reason})")
            print(f"  Last: {elapsed:.0f}s ago  Next: ~{max(0, heartbeat.interval - elapsed):.0f}s")
            print(f"{BLUE}---{RESET}\n")
            continue

        # -- /trigger --
        if user_input == "/trigger":
            print_info("Triggering heartbeat...")
            result = heartbeat.run_heartbeat_turn(run_agent_single_turn)
            if result:
                did = delivery_queue.enqueue(mock_channel.name, "user", result)
                print_heartbeat(f"enqueued heartbeat message (id={did[:8]}..)")
            else:
                print_info("Heartbeat: HEARTBEAT_OK (nothing to report).\n")
            heartbeat.last_run = time.time()
            continue

        # -- 处理用户消息: agent 回复走投递队列 --
        heartbeat._lock.acquire()
        try:
            messages.append({"role": "user", "content": user_input})
            system_prompt = build_system_prompt()

            while True:
                try:
                    response = client.messages.create(
                        model=MODEL_ID, max_tokens=8096,
                        system=system_prompt, messages=messages, tools=TOOLS,
                    )
                except Exception as exc:
                    print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                    messages.pop()
                    break

                if response.stop_reason == "tool_use":
                    messages.append({"role": "assistant", "content": response.content})
                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            handler = TOOL_HANDLERS.get(block.name)
                            if handler:
                                print_tool(block.name, json.dumps(block.input, ensure_ascii=False)[:120])
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": handler(block.input),
                                })
                            else:
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": json.dumps({"error": f"Unknown tool: {block.name}"}),
                                    "is_error": True,
                                })
                    messages.append({"role": "user", "content": tool_results})
                else:
                    text = "".join(b.text for b in response.content if hasattr(b, "text"))
                    if text:
                        did = delivery_queue.enqueue(mock_channel.name, "user", text)
                        print_info(f"  enqueued -> delivery queue (id={did[:8]}..)")
                    messages.append({"role": "assistant", "content": response.content})
                    break
        finally:
            heartbeat._lock.release()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)

    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    soul_path = WORKSPACE_DIR / "SOUL.md"
    if not soul_path.exists():
        soul_path.write_text(
            "# Soul\n\nYou are Koda, a thoughtful AI assistant.\n\n"
            "## Personality\n- Warm but concise\n- Direct and clear\n",
            encoding="utf-8",
        )
        print_info(f"Created sample SOUL.md")

    heartbeat_path = WORKSPACE_DIR / "HEARTBEAT.md"
    if not heartbeat_path.exists():
        heartbeat_path.write_text(
            "# Heartbeat Instructions\n\n"
            "Check the following and report ONLY if action is needed:\n\n"
            "1. Review today's memory log for any unfinished tasks.\n"
            "2. If the user mentioned a deadline, check if it is approaching.\n\n"
            "If nothing needs attention, respond with exactly: HEARTBEAT_OK\n",
            encoding="utf-8",
        )
        print_info(f"Created sample HEARTBEAT.md")

    # 投递队列
    queue_dir = WORKSPACE_DIR / "delivery-queue"
    delivery_queue = DeliveryQueue(queue_dir)
    print_info(f"Delivery queue at {queue_dir}")

    # 模拟渠道
    mock_channel = MockDeliveryChannel(name="telegram", fail_rate=0.0)

    # 投递运行器
    delivery_runner = DeliveryRunner(
        queue=delivery_queue,
        deliver_fn=lambda ch, to, text: mock_channel.send(to, text),
    )

    # 心跳运行器
    heartbeat = HeartbeatRunner(
        interval_seconds=HEARTBEAT_INTERVAL,
        active_hours=(HEARTBEAT_ACTIVE_START, HEARTBEAT_ACTIVE_END),
        heartbeat_path=heartbeat_path,
    )

    delivery_runner.start()
    print_info("Delivery runner started")
    heartbeat.start(run_agent_single_turn)
    print_info(f"Heartbeat started (interval={HEARTBEAT_INTERVAL}s)\n")

    try:
        agent_loop(heartbeat, delivery_runner, delivery_queue, mock_channel)
    finally:
        heartbeat.stop()
        delivery_runner.stop()
        stats = delivery_runner.get_stats()
        print_info(
            f"Final: {stats['total_succeeded']} delivered, "
            f"{stats['total_failed']} failed, "
            f"{stats['pending']} pending, "
            f"{stats['failed']} in failed/"
        )


if __name__ == "__main__":
    main()
