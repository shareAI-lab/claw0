"""
Section 07: ハートビートと Cron
「リアクティブだけでなく、プロアクティブに」

タイマースレッドが「実行すべきか?」をチェックし、ユーザーメッセージと同じ
パイプラインに作業をキューイングする。レーンの排他制御によりユーザーメッセージが優先される。

    メインレーン:      ユーザー入力 --> lock.acquire() -------> LLM --> 表示
    ハートビートレーン: タイマーティック --> lock.acquire(False) -+
                                                        |
                                   取得成功? --いいえ--> スキップ (ユーザーが優先)
                                      |はい
                                  エージェント実行 --> 重複排除 --> キュー
    Cron サービス:   CRON.json --> tick() --> 実行時刻? --> run_agent --> ログ

使い方:
    cd claw0
    python ja/s07_heartbeat_cron.py

必要な設定: ANTHROPIC_API_KEY, MODEL_ID (.env で設定)
ワークスペースファイル: HEARTBEAT.md, SOUL.md, MEMORY.md, CRON.json
"""

import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from anthropic import Anthropic
from croniter import croniter
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)
WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"
CRON_DIR = WORKSPACE_DIR / "cron"

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

def print_heartbeat(text: str) -> None:
    print(f"{BLUE}{BOLD}[heartbeat]{RESET} {text}")

def print_cron(text: str) -> None:
    print(f"{MAGENTA}{BOLD}[cron]{RESET} {text}")

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
    """シングルターンの LLM 呼び出し。ツールなし、プレーンテキストを返す。"""
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
# HeartbeatRunner
# ---------------------------------------------------------------------------

class HeartbeatRunner:
    def __init__(
        self, workspace: Path, lane_lock: threading.Lock,
        interval: float = 1800.0, active_hours: tuple[int, int] = (9, 22),
        max_queue_size: int = 10,
    ) -> None:
        self.workspace = workspace
        self.heartbeat_path = workspace / "HEARTBEAT.md"
        self.lane_lock = lane_lock
        self.interval = interval
        self.active_hours = active_hours
        self.max_queue_size = max_queue_size
        self.last_run_at: float = 0.0
        self.running: bool = False
        self._stopped: bool = False
        self._thread: threading.Thread | None = None
        self._output_queue: list[str] = []
        self._queue_lock = threading.Lock()
        self._last_output: str = ""
        self._soul = SoulSystem(workspace)
        self._memory = MemoryStore(workspace)

    def should_run(self) -> tuple[bool, str]:
        """4つの前提条件チェック。ロックは _execute() 内で別途テストする。"""
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
        if self.running:
            return False, "already running"
        return True, "all checks passed"

    def _parse_response(self, response: str) -> str | None:
        """HEARTBEAT_OK は報告事項なしを意味する。"""
        if "HEARTBEAT_OK" in response:
            stripped = response.replace("HEARTBEAT_OK", "").strip()
            return stripped if len(stripped) > 5 else None
        return response.strip() or None

    def _build_heartbeat_prompt(self) -> tuple[str, str]:
        instructions = self.heartbeat_path.read_text(encoding="utf-8").strip()
        mem = self._memory.load_evergreen()
        extra = ""
        if mem:
            extra = f"## Known Context\n\n{mem}\n\n"
        extra += f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        return instructions, self._soul.build_system_prompt(extra)

    def _execute(self) -> None:
        """ハートビートを1回実行する。ノンブロッキングのロック取得。ビジーの場合は譲る。"""
        acquired = self.lane_lock.acquire(blocking=False)
        if not acquired:
            return
        self.running = True
        try:
            instructions, sys_prompt = self._build_heartbeat_prompt()
            if not instructions:
                return
            response = run_agent_single_turn(instructions, sys_prompt)
            meaningful = self._parse_response(response)
            if meaningful is None:
                return
            if meaningful.strip() == self._last_output:
                return
            self._last_output = meaningful.strip()
            with self._queue_lock:
                self._output_queue.append(meaningful)
        except Exception as exc:
            with self._queue_lock:
                self._output_queue.append(f"[heartbeat error: {exc}]")
        finally:
            self.running = False
            self.last_run_at = time.time()
            self.lane_lock.release()

    def _loop(self) -> None:
        while not self._stopped:
            try:
                ok, _ = self.should_run()
                if ok:
                    self._execute()
            except Exception:
                pass
            time.sleep(1.0)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stopped = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name="heartbeat")
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

    def trigger(self) -> str:
        """手動でハートビートをトリガーする。インターバルチェックをバイパスする。"""
        acquired = self.lane_lock.acquire(blocking=False)
        if not acquired:
            return "main lane occupied, cannot trigger"
        self.running = True
        try:
            instructions, sys_prompt = self._build_heartbeat_prompt()
            if not instructions:
                return "HEARTBEAT.md is empty"
            response = run_agent_single_turn(instructions, sys_prompt)
            meaningful = self._parse_response(response)
            if meaningful is None:
                return "HEARTBEAT_OK (nothing to report)"
            if meaningful.strip() == self._last_output:
                return "duplicate content (skipped)"
            self._last_output = meaningful.strip()
            with self._queue_lock:
                self._output_queue.append(meaningful)
            return f"triggered, output queued ({len(meaningful)} chars)"
        except Exception as exc:
            return f"trigger failed: {exc}"
        finally:
            self.running = False
            self.last_run_at = time.time()
            self.lane_lock.release()

    def status(self) -> dict[str, Any]:
        now = time.time()
        elapsed = now - self.last_run_at if self.last_run_at > 0 else None
        next_in = max(0.0, self.interval - elapsed) if elapsed is not None else self.interval
        ok, reason = self.should_run()
        with self._queue_lock:
            qsize = len(self._output_queue)
        return {
            "enabled": self.heartbeat_path.exists(),
            "running": self.running,
            "should_run": ok, "reason": reason,
            "last_run": datetime.fromtimestamp(self.last_run_at).isoformat() if self.last_run_at > 0 else "never",
            "next_in": f"{round(next_in)}s",
            "interval": f"{self.interval}s",
            "active_hours": f"{self.active_hours[0]}:00-{self.active_hours[1]}:00",
            "queue_size": qsize,
        }

# ---------------------------------------------------------------------------
# CronJob + CronService
# ---------------------------------------------------------------------------
# スケジュール種別: at (1回) | every (固定間隔) | cron (5フィールド式)
# N 回連続エラーで自動無効化。実行ログ -> cron-runs.jsonl

CRON_AUTO_DISABLE_THRESHOLD = 5

@dataclass
class CronJob:
    id: str
    name: str
    enabled: bool
    schedule_kind: str       # "at" | "every" | "cron"
    schedule_config: dict
    payload: dict
    delete_after_run: bool = False
    consecutive_errors: int = 0
    last_run_at: float = 0.0
    next_run_at: float = 0.0


class CronService:
    def __init__(self, cron_file: Path) -> None:
        self.cron_file = cron_file
        self.jobs: list[CronJob] = []
        self._soul = SoulSystem(WORKSPACE_DIR)
        self._output_queue: list[str] = []
        self._queue_lock = threading.Lock()
        CRON_DIR.mkdir(parents=True, exist_ok=True)
        self._run_log = CRON_DIR / "cron-runs.jsonl"
        self.load_jobs()

    def load_jobs(self) -> None:
        self.jobs.clear()
        if not self.cron_file.exists():
            return
        try:
            raw = json.loads(self.cron_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(f"{YELLOW}CRON.json load error: {exc}{RESET}")
            return
        now = time.time()
        for jd in raw.get("jobs", []):
            sched = jd.get("schedule", {})
            kind = sched.get("kind", "")
            if kind not in ("at", "every", "cron"):
                continue
            job = CronJob(
                id=jd.get("id", ""), name=jd.get("name", ""),
                enabled=jd.get("enabled", True), schedule_kind=kind,
                schedule_config=sched, payload=jd.get("payload", {}),
                delete_after_run=jd.get("delete_after_run", False),
            )
            job.next_run_at = self._compute_next(job, now)
            self.jobs.append(job)

    def _compute_next(self, job: CronJob, now: float) -> float:
        """次の実行タイムスタンプを計算する。スケジューリングなしの場合は 0.0 を返す。"""
        cfg = job.schedule_config
        if job.schedule_kind == "at":
            try:
                ts = datetime.fromisoformat(cfg.get("at", "")).timestamp()
                return ts if ts > now else 0.0
            except (ValueError, OSError):
                return 0.0
        if job.schedule_kind == "every":
            every = cfg.get("every_seconds", 3600)
            try:
                anchor = datetime.fromisoformat(cfg.get("anchor", "")).timestamp()
            except (ValueError, OSError, TypeError):
                anchor = now
            if now < anchor:
                return anchor
            steps = int((now - anchor) / every) + 1
            return anchor + steps * every
        if job.schedule_kind == "cron":
            expr = cfg.get("expr", "")
            if not expr:
                return 0.0
            try:
                return croniter(expr, datetime.fromtimestamp(now)).get_next(datetime).timestamp()
            except (ValueError, KeyError):
                return 0.0
        return 0.0

    def tick(self) -> None:
        """毎秒呼び出され、期限到来のジョブをチェックして実行する。"""
        now = time.time()
        remove_ids: list[str] = []
        for job in self.jobs:
            if not job.enabled or job.next_run_at <= 0 or now < job.next_run_at:
                continue
            self._run_job(job, now)
            if job.delete_after_run and job.schedule_kind == "at":
                remove_ids.append(job.id)
        if remove_ids:
            self.jobs = [j for j in self.jobs if j.id not in remove_ids]

    def _run_job(self, job: CronJob, now: float) -> None:
        payload = job.payload
        kind = payload.get("kind", "")
        output, status, error = "", "ok", ""
        try:
            if kind == "agent_turn":
                msg = payload.get("message", "")
                if not msg:
                    output, status = "[empty message]", "skipped"
                else:
                    sys_prompt = (
                        "You are performing a scheduled background task. Be concise. "
                        f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                    )
                    output = run_agent_single_turn(msg, sys_prompt)
            elif kind == "system_event":
                output = payload.get("text", "")
                if not output:
                    status = "skipped"
            else:
                output, status, error = f"[unknown kind: {kind}]", "error", f"unknown kind: {kind}"
        except Exception as exc:
            status, error, output = "error", str(exc), f"[cron error: {exc}]"

        job.last_run_at = now
        if status == "error":
            job.consecutive_errors += 1
            if job.consecutive_errors >= CRON_AUTO_DISABLE_THRESHOLD:
                job.enabled = False
                msg = (f"Job '{job.name}' auto-disabled after "
                       f"{job.consecutive_errors} consecutive errors: {error}")
                print(f"{RED}{msg}{RESET}")
                with self._queue_lock:
                    self._output_queue.append(msg)
        else:
            job.consecutive_errors = 0
        job.next_run_at = self._compute_next(job, now)
        entry = {"job_id": job.id,
                 "run_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                 "status": status, "output_preview": output[:200]}
        if error:
            entry["error"] = error
        try:
            with open(self._run_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass
        if output and status != "skipped":
            with self._queue_lock:
                self._output_queue.append(f"[{job.name}] {output}")

    def trigger_job(self, job_id: str) -> str:
        for job in self.jobs:
            if job.id == job_id:
                self._run_job(job, time.time())
                return f"'{job.name}' triggered (errors={job.consecutive_errors})"
        return f"Job '{job_id}' not found"

    def drain_output(self) -> list[str]:
        with self._queue_lock:
            items = list(self._output_queue)
            self._output_queue.clear()
            return items

    def list_jobs(self) -> list[dict[str, Any]]:
        now = time.time()
        result = []
        for j in self.jobs:
            nxt = max(0.0, j.next_run_at - now) if j.next_run_at > 0 else None
            result.append({
                "id": j.id, "name": j.name, "enabled": j.enabled,
                "kind": j.schedule_kind, "errors": j.consecutive_errors,
                "last_run": datetime.fromtimestamp(j.last_run_at).isoformat() if j.last_run_at > 0 else "never",
                "next_run": datetime.fromtimestamp(j.next_run_at).isoformat() if j.next_run_at > 0 else "n/a",
                "next_in": round(nxt) if nxt is not None else None,
            })
        return result

# ---------------------------------------------------------------------------
# REPL + エージェントループ
# ---------------------------------------------------------------------------

def print_repl_help() -> None:
    print_info("REPL commands:")
    print_info("  /heartbeat         -- heartbeat status")
    print_info("  /trigger           -- force heartbeat now")
    print_info("  /cron              -- list cron jobs")
    print_info("  /cron-trigger <id> -- trigger a cron job")
    print_info("  /lanes             -- lane lock status")
    print_info("  /help              -- this help")
    print_info("  quit / exit        -- exit")


def agent_loop() -> None:
    lane_lock = threading.Lock()
    soul = SoulSystem(WORKSPACE_DIR)
    memory = MemoryStore(WORKSPACE_DIR)

    heartbeat = HeartbeatRunner(
        workspace=WORKSPACE_DIR, lane_lock=lane_lock,
        interval=float(os.getenv("HEARTBEAT_INTERVAL", "1800")),
        active_hours=(int(os.getenv("HEARTBEAT_ACTIVE_START", "9")),
                      int(os.getenv("HEARTBEAT_ACTIVE_END", "22"))),
    )
    cron_svc = CronService(WORKSPACE_DIR / "CRON.json")
    heartbeat.start()

    cron_stop = threading.Event()
    def cron_loop() -> None:
        while not cron_stop.is_set():
            try:
                cron_svc.tick()
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

    hb_st = heartbeat.status()
    print_info("=" * 60)
    print_info("  claw0  |  Section 07: Heartbeat & Cron")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Heartbeat: {'on' if hb_st['enabled'] else 'off'} ({heartbeat.interval}s)")
    print_info(f"  Cron jobs: {len(cron_svc.jobs)}")
    print_info("  /help for commands. quit to exit.")
    print_info("=" * 60)
    print()

    while True:
        for msg in heartbeat.drain_output():
            print_heartbeat(msg)
        for msg in cron_svc.drain_output():
            print_cron(msg)

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

        # REPL コマンド
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/help":
                print_repl_help()
            elif cmd == "/heartbeat":
                for k, v in heartbeat.status().items():
                    print_info(f"  {k}: {v}")
            elif cmd == "/trigger":
                print_info(f"  {heartbeat.trigger()}")
                for m in heartbeat.drain_output():
                    print_heartbeat(m)
            elif cmd == "/cron":
                jobs = cron_svc.list_jobs()
                if not jobs:
                    print_info("No cron jobs.")
                for j in jobs:
                    tag = f"{GREEN}ON{RESET}" if j["enabled"] else f"{RED}OFF{RESET}"
                    err = f" {YELLOW}err:{j['errors']}{RESET}" if j["errors"] else ""
                    nxt = f" in {j['next_in']}s" if j["next_in"] is not None else ""
                    print(f"  [{tag}] {j['id']} - {j['name']}{err}{nxt}")
            elif cmd == "/cron-trigger":
                if not arg:
                    print(f"{YELLOW}Usage: /cron-trigger <job_id>{RESET}")
                else:
                    print_info(f"  {cron_svc.trigger_job(arg)}")
                    for m in cron_svc.drain_output():
                        print_cron(m)
            elif cmd == "/lanes":
                locked = not lane_lock.acquire(blocking=False)
                if not locked:
                    lane_lock.release()
                print_info(f"  main_locked: {locked}  heartbeat_running: {heartbeat.running}")
            else:
                print(f"{YELLOW}Unknown: {cmd}. /help for commands.{RESET}")
            continue

        # ユーザー会話: ブロッキング取得 (ユーザーが常に優先)
        lane_lock.acquire()
        try:
            messages.append({"role": "user", "content": user_input})
            while True:
                try:
                    response = client.messages.create(
                        model=MODEL_ID, max_tokens=8096, system=system_prompt,
                        tools=MEMORY_TOOLS, messages=messages,
                    )
                except Exception as exc:
                    print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                    while messages and messages[-1]["role"] != "user":
                        messages.pop()
                    if messages:
                        messages.pop()
                    break

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    text = "".join(b.text for b in response.content if hasattr(b, "text"))
                    if text:
                        print_assistant(text)
                    break
                elif response.stop_reason == "tool_use":
                    results = []
                    for block in response.content:
                        if block.type != "tool_use":
                            continue
                        print_info(f"  [tool: {block.name}]")
                        results.append({"type": "tool_result", "tool_use_id": block.id,
                                        "content": handle_tool(block.name, block.input)})
                    messages.append({"role": "user", "content": results})
                else:
                    print_info(f"[stop_reason={response.stop_reason}]")
                    text = "".join(b.text for b in response.content if hasattr(b, "text"))
                    if text:
                        print_assistant(text)
                    break
        finally:
            lane_lock.release()

    heartbeat.stop()
    cron_stop.set()

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
