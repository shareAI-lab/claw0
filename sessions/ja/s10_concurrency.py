"""
Section 10: 並行処理
「名前付きレーンが混沌を直列化する」

Section 07 の単一 threading.Lock を、適切な名前付きレーンシステムに置き換える。
各レーンは設定可能な max_concurrency を持つ FIFO キュー。タスクはコーラブルとして
エンキューされ、専用スレッドで実行され、concurrent.futures.Future を通じて結果を返す。

    Incoming Work
        |
    CommandQueue.enqueue(lane, fn)
        |
    +--------+    +--------+    +-----------+
    | main   |    |  cron  |    | heartbeat |
    | max=1  |    | max=1  |    |   max=1   |
    | FIFO   |    | FIFO   |    |   FIFO    |
    +---+----+    +---+----+    +-----+-----+
        |             |               |
    [active]      [active]        [active]
        |             |               |
    _task_done    _task_done      _task_done
        |             |               |
    _pump()       _pump()         _pump()
    (dequeue      (dequeue        (dequeue
     next if       next if         next if
     active<max)   active<max)     active<max)

使い方:
    cd claw0
    python ja/s10_concurrency.py

必要な設定: ANTHROPIC_API_KEY, MODEL_ID (.env で設定)
ワークスペースファイル: SOUL.md, MEMORY.md, HEARTBEAT.md, CRON.json
"""

import json
import os
import sys
import threading
import time
import concurrent.futures
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from anthropic import Anthropic
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)
WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"

# ---------------------------------------------------------------------------
# ANSI カラー
# ---------------------------------------------------------------------------
CYAN, GREEN, YELLOW, DIM, RESET, BOLD = "\033[36m", "\033[32m", "\033[33m", "\033[2m", "\033[0m", "\033[1m"
MAGENTA, RED, BLUE, ORANGE = "\033[35m", "\033[31m", "\033[34m", "\033[38;5;208m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def print_lane(lane_name: str, text: str) -> None:
    color = {
        "main": CYAN, "cron": MAGENTA, "heartbeat": BLUE,
    }.get(lane_name, YELLOW)
    print(f"{color}{BOLD}[{lane_name}]{RESET} {text}")


# ---------------------------------------------------------------------------
# 標準レーン名
# ---------------------------------------------------------------------------
LANE_MAIN = "main"
LANE_CRON = "cron"
LANE_HEARTBEAT = "heartbeat"

# ---------------------------------------------------------------------------
# LaneQueue -- 並行性制御付きの単一名前付き FIFO レーン
# ---------------------------------------------------------------------------


class LaneQueue:
    """最大 max_concurrency 個のタスクを並列実行する名前付き FIFO キュー。

    エンキューされた各コーラブルは専用スレッドで実行される。結果は
    concurrent.futures.Future を通じて返される。世代カウンターが
    リスタートリカバリをサポート: 世代がインクリメントされると、
    古い世代の完了タスクはキューを再ポンプしない。
    """

    def __init__(self, name: str, max_concurrency: int = 1) -> None:
        self.name = name
        self.max_concurrency = max(1, max_concurrency)
        self._deque: deque[tuple[Callable, concurrent.futures.Future, int]] = deque()
        self._condition = threading.Condition()
        self._active_count = 0
        self._generation = 0

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    @generation.setter
    def generation(self, value: int) -> None:
        with self._condition:
            self._generation = value
            self._condition.notify_all()

    def enqueue(self, fn: Callable[[], Any], generation: int | None = None) -> concurrent.futures.Future:
        """コーラブルをキューに追加する。結果の Future を返す。

        generation が None の場合、現在のレーン世代が使用される。
        """
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._condition:
            gen = generation if generation is not None else self._generation
            self._deque.append((fn, future, gen))
            self._pump()
        return future

    def _pump(self) -> None:
        """active < max_concurrency の間、デキューからタスクをポップして実行する。

        self._condition を保持した状態で呼び出す必要がある。
        """
        while self._active_count < self.max_concurrency and self._deque:
            fn, future, gen = self._deque.popleft()
            self._active_count += 1
            t = threading.Thread(
                target=self._run_task,
                args=(fn, future, gen),
                daemon=True,
                name=f"lane-{self.name}",
            )
            t.start()

    def _run_task(
        self,
        fn: Callable[[], Any],
        future: concurrent.futures.Future,
        gen: int,
    ) -> None:
        """fn を実行し、Future に結果を設定してから _task_done を呼ぶ。"""
        try:
            result = fn()
            future.set_result(result)
        except Exception as exc:
            future.set_exception(exc)
        finally:
            self._task_done(gen)

    def _task_done(self, gen: int) -> None:
        """アクティブカウントをデクリメント。世代が一致する場合のみ再ポンプ。"""
        with self._condition:
            self._active_count -= 1
            if gen == self._generation:
                self._pump()
            self._condition.notify_all()

    def wait_for_idle(self, timeout: float | None = None) -> bool:
        """active_count == 0 かつデキューが空になるまでブロックする。

        アイドルに達した場合は True、タイムアウトの場合は False を返す。
        """
        deadline = (time.monotonic() + timeout) if timeout is not None else None
        with self._condition:
            while self._active_count > 0 or len(self._deque) > 0:
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                self._condition.wait(timeout=remaining)
            return True

    def stats(self) -> dict[str, Any]:
        with self._condition:
            return {
                "name": self.name,
                "queue_depth": len(self._deque),
                "active": self._active_count,
                "max_concurrency": self.max_concurrency,
                "generation": self._generation,
            }


# ---------------------------------------------------------------------------
# CommandQueue -- 名前付きレーンに作業をルーティング
# ---------------------------------------------------------------------------


class CommandQueue:
    """コーラブルを名前付き LaneQueue にルーティングする中央ディスパッチャー。

    レーンは最初の使用時に遅延作成される。reset_all() は全ての世代
    カウンターをインクリメントし、前回のライフサイクルからの古いタスクが
    キューを再ポンプしないようにする。
    """

    def __init__(self) -> None:
        self._lanes: dict[str, LaneQueue] = {}
        self._lock = threading.Lock()

    def get_or_create_lane(self, name: str, max_concurrency: int = 1) -> LaneQueue:
        """既存のレーンを取得するか、新しいレーンを作成する。"""
        with self._lock:
            if name not in self._lanes:
                self._lanes[name] = LaneQueue(name, max_concurrency)
            return self._lanes[name]

    def enqueue(self, lane_name: str, fn: Callable[[], Any]) -> concurrent.futures.Future:
        """コーラブルを指定レーンにルーティングする。Future を返す。"""
        lane = self.get_or_create_lane(lane_name)
        return lane.enqueue(fn)

    def reset_all(self) -> dict[str, int]:
        """全レーンの世代をインクリメントする。リスタートリカバリに使用。

        lane_name -> new_generation の辞書を返す。
        """
        result: dict[str, int] = {}
        with self._lock:
            for name, lane in self._lanes.items():
                with lane._condition:
                    lane._generation += 1
                    result[name] = lane._generation
        return result

    def wait_for_all(self, timeout: float = 10.0) -> bool:
        """全レーンがアイドルになるまで待機する。全てアイドルなら True を返す。"""
        deadline = time.monotonic() + timeout
        with self._lock:
            lanes = list(self._lanes.values())
        for lane in lanes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            if not lane.wait_for_idle(timeout=remaining):
                return False
        return True

    def stats(self) -> dict[str, dict[str, Any]]:
        """全レーンの統計を集約する。"""
        with self._lock:
            return {name: lane.stats() for name, lane in self._lanes.items()}

    def lane_names(self) -> list[str]:
        with self._lock:
            return list(self._lanes.keys())


# ---------------------------------------------------------------------------
# ソウル + メモリ (簡易版)
# ---------------------------------------------------------------------------


class SoulSystem:
    def __init__(self, workspace: Path) -> None:
        self.soul_path = workspace / "SOUL.md"

    def load(self) -> str:
        if self.soul_path.exists():
            return self.soul_path.read_text(encoding="utf-8").strip()
        return "You are a helpful AI assistant."

    def build_system_prompt(self, extra: str = "") -> str:
        parts = [self.load()]
        if extra:
            parts.append(extra)
        return "\n\n".join(parts)


class MemoryStore:
    def __init__(self, workspace: Path) -> None:
        self.memory_path = workspace / "MEMORY.md"

    def load_evergreen(self) -> str:
        if self.memory_path.exists():
            return self.memory_path.read_text(encoding="utf-8").strip()
        return ""

    def write_memory(self, content: str) -> str:
        existing = self.load_evergreen()
        updated = existing + "\n\n" + content.strip() if existing else content.strip()
        self.memory_path.write_text(updated, encoding="utf-8")
        return f"Memory saved ({len(content)} chars)"

    def search_memory(self, query: str) -> str:
        text = self.load_evergreen()
        if not text:
            return "No memories found."
        matches = [l for l in text.split("\n") if query.lower() in l.lower()]
        return "\n".join(matches[:10]) if matches else f"No memories matching '{query}'."


MEMORY_TOOLS = [
    {"name": "memory_write",
     "description": "Save an important fact or preference to long-term memory.",
     "input_schema": {"type": "object", "properties": {
         "content": {"type": "string", "description": "The fact or preference to remember."}},
         "required": ["content"]}},
    {"name": "memory_search",
     "description": "Search long-term memory for relevant information.",
     "input_schema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "Search query."}},
         "required": ["query"]}},
]


# ---------------------------------------------------------------------------
# エージェントヘルパー -- シングルターン LLM 呼び出し (ハートビートと cron で共有)
# ---------------------------------------------------------------------------


def run_agent_single_turn(prompt: str, system_prompt: str | None = None) -> str:
    """シングルターン LLM 呼び出し、ツールなし、プレーンテキストを返す。"""
    sys_prompt = system_prompt or "You are a helpful assistant performing a background check."
    try:
        response = client.messages.create(
            model=MODEL_ID, max_tokens=2048, system=sys_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in response.content if hasattr(b, "text")).strip()
    except Exception as exc:
        return f"[agent error: {exc}]"


# ---------------------------------------------------------------------------
# HeartbeatRunner -- CommandQueue を通じてエンキュー
# ---------------------------------------------------------------------------


class HeartbeatRunner:
    """ハートビートレーンに作業をエンキューするバックグラウンドハートビート。

    各ティックで前提条件をチェックする。ハートビートレーンに既に
    アクティブな作業がある場合、ティックはスキップされる (ノンブロッキング)。
    Section 07 の Lock.acquire(blocking=False) パターンをレーン対応の
    チェックに置き換える。
    """

    def __init__(
        self,
        workspace: Path,
        command_queue: CommandQueue,
        interval: float = 1800.0,
        active_hours: tuple[int, int] = (9, 22),
    ) -> None:
        self.workspace = workspace
        self.heartbeat_path = workspace / "HEARTBEAT.md"
        self.command_queue = command_queue
        self.interval = interval
        self.active_hours = active_hours
        self.last_run_at: float = 0.0
        self._stopped: bool = False
        self._thread: threading.Thread | None = None
        self._output_queue: list[str] = []
        self._queue_lock = threading.Lock()
        self._last_output: str = ""
        self._soul = SoulSystem(workspace)
        self._memory = MemoryStore(workspace)

    def should_run(self) -> tuple[bool, str]:
        """ハートビート試行前の前提条件チェック。"""
        if not self.heartbeat_path.exists():
            return False, "HEARTBEAT.md not found"
        if not self.heartbeat_path.read_text(encoding="utf-8").strip():
            return False, "HEARTBEAT.md is empty"
        now = time.time()
        elapsed = now - self.last_run_at
        if elapsed < self.interval:
            return False, f"interval not elapsed ({self.interval - elapsed:.0f}s remaining)"
        hour = datetime.now().hour
        s, e = self.active_hours
        in_hours = (s <= hour < e) if s <= e else not (e <= hour < s)
        if not in_hours:
            return False, f"outside active hours ({s}:00-{e}:00)"
        return True, "all checks passed"

    def _build_heartbeat_prompt(self) -> tuple[str, str]:
        instructions = self.heartbeat_path.read_text(encoding="utf-8").strip()
        mem = self._memory.load_evergreen()
        extra = ""
        if mem:
            extra = f"## Known Context\n\n{mem}\n\n"
        extra += f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return instructions, self._soul.build_system_prompt(extra)

    def _parse_response(self, response: str) -> str | None:
        if "HEARTBEAT_OK" in response:
            stripped = response.replace("HEARTBEAT_OK", "").strip()
            return stripped if len(stripped) > 5 else None
        return response.strip() or None

    def heartbeat_tick(self) -> None:
        """1回のハートビートティック。レーンがビジーでない場合のみ作業をエンキュー。"""
        ok, reason = self.should_run()
        if not ok:
            return

        lane_stats = self.command_queue.get_or_create_lane(LANE_HEARTBEAT).stats()
        if lane_stats["active"] > 0:
            return

        def _do_heartbeat() -> str | None:
            instructions, sys_prompt = self._build_heartbeat_prompt()
            if not instructions:
                return None
            response = run_agent_single_turn(instructions, sys_prompt)
            return self._parse_response(response)

        future = self.command_queue.enqueue(LANE_HEARTBEAT, _do_heartbeat)

        def _on_done(f: concurrent.futures.Future) -> None:
            self.last_run_at = time.time()
            try:
                meaningful = f.result()
                if meaningful is None:
                    return
                if meaningful.strip() == self._last_output:
                    return
                self._last_output = meaningful.strip()
                with self._queue_lock:
                    self._output_queue.append(meaningful)
                print_lane(LANE_HEARTBEAT, f"output queued ({len(meaningful)} chars)")
            except Exception as exc:
                with self._queue_lock:
                    self._output_queue.append(f"[heartbeat error: {exc}]")

        future.add_done_callback(_on_done)

    def _loop(self) -> None:
        while not self._stopped:
            try:
                self.heartbeat_tick()
            except Exception:
                pass
            time.sleep(1.0)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="heartbeat-timer")
        self._thread.start()

    def stop(self) -> None:
        self._stopped = True
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def drain_output(self) -> list[str]:
        with self._queue_lock:
            items = list(self._output_queue)
            self._output_queue.clear()
            return items

    def status(self) -> dict[str, Any]:
        now = time.time()
        elapsed = now - self.last_run_at if self.last_run_at > 0 else None
        next_in = max(0.0, self.interval - elapsed) if elapsed is not None else self.interval
        ok, reason = self.should_run()
        with self._queue_lock:
            qsize = len(self._output_queue)
        return {
            "enabled": self.heartbeat_path.exists(),
            "should_run": ok, "reason": reason,
            "last_run": datetime.fromtimestamp(self.last_run_at).isoformat() if self.last_run_at > 0 else "never",
            "next_in": f"{round(next_in)}s",
            "interval": f"{self.interval}s",
            "active_hours": f"{self.active_hours[0]}:00-{self.active_hours[1]}:00",
            "queue_size": qsize,
        }


# ---------------------------------------------------------------------------
# CronService -- CommandQueue を通じてエンキュー
# ---------------------------------------------------------------------------


class CronService:
    """cron レーンにジョブをエンキューする簡易 cron サービス。"""

    def __init__(self, cron_file: Path, command_queue: CommandQueue) -> None:
        self.cron_file = cron_file
        self.command_queue = command_queue
        self.jobs: list[dict[str, Any]] = []
        self._output_queue: list[str] = []
        self._queue_lock = threading.Lock()
        self.load_jobs()

    def load_jobs(self) -> None:
        self.jobs.clear()
        if not self.cron_file.exists():
            return
        try:
            raw = json.loads(self.cron_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        now = time.time()
        for jd in raw.get("jobs", []):
            sched = jd.get("schedule", {})
            every = sched.get("every_seconds", 0)
            if every <= 0:
                continue
            job = {
                "id": jd.get("id", ""),
                "name": jd.get("name", ""),
                "enabled": jd.get("enabled", True),
                "every_seconds": every,
                "payload": jd.get("payload", {}),
                "last_run_at": 0.0,
                "next_run_at": now + every,
                "consecutive_errors": 0,
            }
            self.jobs.append(job)

    def cron_tick(self) -> None:
        """毎秒呼び出される。期限到来のジョブを cron レーンにエンキュー。"""
        now = time.time()
        for job in self.jobs:
            if not job["enabled"]:
                continue
            if now < job["next_run_at"]:
                continue
            self._enqueue_job(job, now)

    def _enqueue_job(self, job: dict[str, Any], now: float) -> None:
        payload = job["payload"]
        message = payload.get("message", "")
        job_name = job["name"]

        if not message:
            job["next_run_at"] = now + job["every_seconds"]
            return

        def _do_cron() -> str:
            sys_prompt = (
                "You are performing a scheduled background task. Be concise. "
                f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return run_agent_single_turn(message, sys_prompt)

        future = self.command_queue.enqueue(LANE_CRON, _do_cron)

        def _on_done(f: concurrent.futures.Future, j: dict = job, n: str = job_name) -> None:
            j["last_run_at"] = time.time()
            j["next_run_at"] = time.time() + j["every_seconds"]
            try:
                result = f.result()
                j["consecutive_errors"] = 0
                if result:
                    with self._queue_lock:
                        self._output_queue.append(f"[{n}] {result}")
                    print_lane(LANE_CRON, f"job '{n}' completed")
            except Exception as exc:
                j["consecutive_errors"] += 1
                with self._queue_lock:
                    self._output_queue.append(f"[{n}] error: {exc}")
                if j["consecutive_errors"] >= 5:
                    j["enabled"] = False
                    print_lane(LANE_CRON, f"job '{n}' auto-disabled after 5 consecutive errors")

        future.add_done_callback(_on_done)
        job["next_run_at"] = now + job["every_seconds"]

    def drain_output(self) -> list[str]:
        with self._queue_lock:
            items = list(self._output_queue)
            self._output_queue.clear()
            return items

    def list_jobs(self) -> list[dict[str, Any]]:
        now = time.time()
        result = []
        for j in self.jobs:
            nxt = max(0.0, j["next_run_at"] - now) if j["next_run_at"] > 0 else None
            result.append({
                "id": j["id"], "name": j["name"], "enabled": j["enabled"],
                "every_seconds": j["every_seconds"],
                "errors": j["consecutive_errors"],
                "last_run": datetime.fromtimestamp(j["last_run_at"]).isoformat() if j["last_run_at"] > 0 else "never",
                "next_in": round(nxt) if nxt is not None else None,
            })
        return result


# ---------------------------------------------------------------------------
# REPL + エージェントループ
# ---------------------------------------------------------------------------


def print_repl_help() -> None:
    print_info("REPL commands:")
    print_info("  /lanes                    -- show all lanes with stats")
    print_info("  /queue                    -- show pending items per lane")
    print_info("  /enqueue <lane> <message> -- manually enqueue work into a named lane")
    print_info("  /concurrency <lane> <N>   -- change max_concurrency for a lane")
    print_info("  /generation               -- show generation counters")
    print_info("  /reset                    -- simulate restart (reset_all)")
    print_info("  /heartbeat                -- heartbeat status")
    print_info("  /trigger                  -- force heartbeat now")
    print_info("  /cron                     -- list cron jobs")
    print_info("  /help                     -- this help")
    print_info("  quit / exit               -- exit")


def agent_loop() -> None:
    cmd_queue = CommandQueue()
    cmd_queue.get_or_create_lane(LANE_MAIN, max_concurrency=1)
    cmd_queue.get_or_create_lane(LANE_CRON, max_concurrency=1)
    cmd_queue.get_or_create_lane(LANE_HEARTBEAT, max_concurrency=1)

    soul = SoulSystem(WORKSPACE_DIR)
    memory = MemoryStore(WORKSPACE_DIR)

    heartbeat = HeartbeatRunner(
        workspace=WORKSPACE_DIR,
        command_queue=cmd_queue,
        interval=float(os.getenv("HEARTBEAT_INTERVAL", "1800")),
        active_hours=(int(os.getenv("HEARTBEAT_ACTIVE_START", "9")),
                      int(os.getenv("HEARTBEAT_ACTIVE_END", "22"))),
    )
    cron_svc = CronService(WORKSPACE_DIR / "CRON.json", cmd_queue)
    heartbeat.start()

    cron_stop = threading.Event()

    def cron_loop() -> None:
        while not cron_stop.is_set():
            try:
                cron_svc.cron_tick()
            except Exception:
                pass
            cron_stop.wait(timeout=1.0)

    threading.Thread(target=cron_loop, daemon=True, name="cron-tick").start()

    messages: list[dict] = []
    mem_text = memory.load_evergreen()
    extra = f"## Long-term Memory\n\n{mem_text}" if mem_text else ""
    system_prompt = soul.build_system_prompt(extra)

    def handle_tool(name: str, inp: dict) -> str:
        if name == "memory_write":
            return memory.write_memory(inp.get("content", ""))
        if name == "memory_search":
            return memory.search_memory(inp.get("query", ""))
        return f"Unknown tool: {name}"

    lane_stats = cmd_queue.stats()
    print_info("=" * 60)
    print_info("  claw0  |  Section 10: Concurrency")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Lanes: {', '.join(lane_stats.keys())}")
    hb_st = heartbeat.status()
    print_info(f"  Heartbeat: {'on' if hb_st['enabled'] else 'off'} ({heartbeat.interval}s)")
    print_info(f"  Cron jobs: {len(cron_svc.jobs)}")
    print_info("  /help for commands. quit to exit.")
    print_info("=" * 60)
    print()

    while True:
        for msg in heartbeat.drain_output():
            print_lane(LANE_HEARTBEAT, msg)
        for msg in cron_svc.drain_output():
            print_lane(LANE_CRON, msg)

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

        # REPL commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=2)
            cmd = parts[0].lower()

            if cmd == "/help":
                print_repl_help()

            elif cmd == "/lanes":
                all_stats = cmd_queue.stats()
                if not all_stats:
                    print_info("  No lanes.")
                for name, st in all_stats.items():
                    active_bar = "*" * st["active"] + "." * (st["max_concurrency"] - st["active"])
                    print_info(
                        f"  {name:12s}  active=[{active_bar}]  "
                        f"queued={st['queue_depth']}  max={st['max_concurrency']}  "
                        f"gen={st['generation']}"
                    )

            elif cmd == "/queue":
                all_stats = cmd_queue.stats()
                total = sum(st["queue_depth"] for st in all_stats.values())
                if total == 0:
                    print_info("  All lanes empty.")
                else:
                    for name, st in all_stats.items():
                        if st["queue_depth"] > 0 or st["active"] > 0:
                            print_info(
                                f"  {name}: {st['queue_depth']} queued, "
                                f"{st['active']} active"
                            )

            elif cmd == "/enqueue":
                if len(parts) < 3:
                    print(f"{YELLOW}Usage: /enqueue <lane> <message>{RESET}")
                else:
                    lane_name = parts[1]
                    message = parts[2]
                    print_info(f"  Enqueueing into '{lane_name}': {message[:60]}...")

                    def _make_enqueued_fn(msg: str = message) -> Callable[[], str]:
                        def _fn() -> str:
                            return run_agent_single_turn(msg)
                        return _fn

                    future = cmd_queue.enqueue(lane_name, _make_enqueued_fn())

                    def _on_enqueue_done(
                        f: concurrent.futures.Future,
                        ln: str = lane_name,
                    ) -> None:
                        try:
                            result = f.result()
                            print_lane(ln, f"result: {result[:200]}")
                        except Exception as exc:
                            print_lane(ln, f"error: {exc}")

                    future.add_done_callback(_on_enqueue_done)

            elif cmd == "/concurrency":
                if len(parts) < 3:
                    print(f"{YELLOW}Usage: /concurrency <lane> <N>{RESET}")
                else:
                    lane_name = parts[1]
                    try:
                        new_max = max(1, int(parts[2]))
                    except ValueError:
                        print(f"{YELLOW}N must be an integer.{RESET}")
                        continue
                    lane = cmd_queue.get_or_create_lane(lane_name)
                    old_max = lane.max_concurrency
                    lane.max_concurrency = new_max
                    print_info(f"  {lane_name}: max_concurrency {old_max} -> {new_max}")
                    with lane._condition:
                        lane._pump()

            elif cmd == "/generation":
                all_stats = cmd_queue.stats()
                for name, st in all_stats.items():
                    print_info(f"  {name}: generation={st['generation']}")

            elif cmd == "/reset":
                result = cmd_queue.reset_all()
                print_info("  Generation incremented on all lanes:")
                for name, gen in result.items():
                    print_info(f"    {name}: generation -> {gen}")
                print_info("  Stale tasks from the old generation will be ignored.")

            elif cmd == "/heartbeat":
                for k, v in heartbeat.status().items():
                    print_info(f"  {k}: {v}")

            elif cmd == "/trigger":
                heartbeat.heartbeat_tick()
                print_info("  Heartbeat tick triggered.")
                time.sleep(0.5)
                for m in heartbeat.drain_output():
                    print_lane(LANE_HEARTBEAT, m)

            elif cmd == "/cron":
                jobs = cron_svc.list_jobs()
                if not jobs:
                    print_info("  No cron jobs.")
                for j in jobs:
                    tag = f"{GREEN}ON{RESET}" if j["enabled"] else f"{RED}OFF{RESET}"
                    err = f" {YELLOW}err:{j['errors']}{RESET}" if j["errors"] else ""
                    nxt = f" in {j['next_in']}s" if j["next_in"] is not None else ""
                    print(f"  [{tag}] {j['id']} - {j['name']}{err}{nxt}")

            else:
                print(f"{YELLOW}Unknown: {cmd}. /help for commands.{RESET}")
            continue

        def _make_user_turn(
            user_msg: str,
            msgs: list[dict],
            sys_prompt: str,
            tool_handler: Callable,
        ) -> Callable[[], str]:
            """ユーザーメッセージに対する完全なエージェントターンを実行するコーラブルを作成する。"""

            def _turn() -> str:
                msgs.append({"role": "user", "content": user_msg})
                final_text = ""
                while True:
                    try:
                        response = client.messages.create(
                            model=MODEL_ID, max_tokens=8096, system=sys_prompt,
                            tools=MEMORY_TOOLS, messages=msgs,
                        )
                    except Exception as exc:
                        while msgs and msgs[-1]["role"] != "user":
                            msgs.pop()
                        if msgs:
                            msgs.pop()
                        return f"[API Error: {exc}]"

                    msgs.append({"role": "assistant", "content": response.content})

                    if response.stop_reason == "end_turn":
                        final_text = "".join(
                            b.text for b in response.content if hasattr(b, "text")
                        )
                        break
                    elif response.stop_reason == "tool_use":
                        results = []
                        for block in response.content:
                            if block.type != "tool_use":
                                continue
                            print_info(f"  [tool: {block.name}]")
                            results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": tool_handler(block.name, block.input),
                            })
                        msgs.append({"role": "user", "content": results})
                    else:
                        final_text = "".join(
                            b.text for b in response.content if hasattr(b, "text")
                        )
                        break
                return final_text

            return _turn

        print_lane(LANE_MAIN, "processing...")
        future = cmd_queue.enqueue(
            LANE_MAIN,
            _make_user_turn(user_input, messages, system_prompt, handle_tool),
        )

        try:
            result_text = future.result(timeout=120)
            if result_text:
                print_assistant(result_text)
        except concurrent.futures.TimeoutError:
            print(f"\n{YELLOW}Request timed out.{RESET}\n")
        except Exception as exc:
            print(f"\n{YELLOW}Error: {exc}{RESET}\n")

    heartbeat.stop()
    cron_stop.set()
    cmd_queue.wait_for_all(timeout=3.0)


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)
    agent_loop()


if __name__ == "__main__":
    main()
