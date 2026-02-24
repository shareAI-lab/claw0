# s09: Cron Scheduler

> "When the agent masters time" -- from fixed heartbeat to precise calendar.

## At a Glance

```
  +--- CronService (background thread, 1s poll) ---+
  |                                                  |
  |  for each enabled job:                           |
  |    compute_next_run_at(schedule, now)             |
  |    if next_run_at <= now:                         |
  |      execute_job(job)                             |
  |                                                  |
  +--------------------------------------------------+
        |
        v
  +-----+-----+-----+
  |     |     |     |
  v     v     v     |
 [at]  [every] [cron]     Schedule Types
  |     |     |     |
  v     v     v     |
  agent_fn(payload.message)
        |
        +-- success --> update state, log to JSONL
        |                 if delete_after_run: disable
        |
        +-- error ----> consecutive_errors++
                          if >= 5: auto-disable

  +--- CronStore ---+     +--- CronRunLog ---+
  | jobs.json       |     | run-log.jsonl     |
  | (atomic write)  |     | (append + prune)  |
  +-----------------+     +------------------+
```

- **What we build**: A scheduled-task system supporting three schedule types, with persistent storage, run logs, and automatic fault handling
- **Core mechanism**: compute_next_run_at calculates the next trigger per schedule type; at returns None when expired, every uses an anchor formula to prevent drift, cron uses croniter to parse expressions
- **Design pattern**: 1-second polling scans all jobs; 5 consecutive failures auto-disable

## The Problem

1. **Fixed intervals only**: The s08 heartbeat can only "check every N seconds." It cannot express "every Monday at 9am."
2. **No one-shot tasks**: The user says "remind me about the meeting tomorrow at 3pm" -- the heartbeat has no semantics for "execute once at a specific point in time."
3. **No multi-task management**: The heartbeat has a single global HEARTBEAT.md. There is no way to run multiple independent scheduled tasks and track their states separately.

## How It Works

### 1. Three Schedule Types

The scheduling system uses a `kind` field to distinguish three types, each with different time semantics:

```python
# at: one-shot absolute time
{"kind": "at", "at_time": "2026-02-25T15:00:00"}

# every: anchor-based equal interval
{"kind": "every", "every_seconds": 3600, "anchor": "2026-02-24T10:00:00+08:00"}

# cron: standard cron expression
{"kind": "cron", "expr": "0 9 * * 1"}
```

| Type | User says | schedule | Behavior |
|------|-----------|----------|----------|
| at | "Remind me tomorrow at 3pm" | `at_time: "2026-02-25T15:00:00"` | Auto-disables after execution |
| every | "Check every hour" | `every_seconds: 3600` | Anchor-aligned, no drift |
| cron | "Every Monday at 9am" | `expr: "0 9 * * 1"` | Standard 5-field cron |

**Three types cover the vast majority of scheduling needs: reminders, polling, and periodic tasks.**

### 2. compute_next_run_at -- The Core Scheduling Algorithm

Calculates the next trigger time based on schedule type. This is the single most critical function in the entire cron system:

```python
def compute_next_run_at(schedule: dict, now_ts: float) -> float | None:
    kind = schedule.get("kind")

    if kind == "at":
        at_time_str = schedule.get("at_time", "")
        try:
            dt = datetime.fromisoformat(at_time_str)
            if dt.tzinfo is None:
                dt = dt.astimezone()
            at_ts = dt.timestamp()
        except (ValueError, OSError):
            return None
        return at_ts if at_ts > now_ts else None  # expired returns None

    if kind == "every":
        every_seconds = max(1, int(schedule.get("every_seconds", 60)))
        # parse anchor
        anchor_str = schedule.get("anchor")
        anchor_ts = now_ts
        if anchor_str:
            try:
                anchor_dt = datetime.fromisoformat(anchor_str)
                if anchor_dt.tzinfo is None:
                    anchor_dt = anchor_dt.astimezone()
                anchor_ts = anchor_dt.timestamp()
            except (ValueError, OSError):
                pass
        if now_ts < anchor_ts:
            return anchor_ts
        # anchor formula: anchor + ceil((now - anchor) / interval) * interval
        elapsed = now_ts - anchor_ts
        steps = max(1, math.ceil(elapsed / every_seconds))
        return anchor_ts + steps * every_seconds

    if kind == "cron":
        expr = schedule.get("expr", "").strip()
        if not expr or croniter is None:
            return None
        base_dt = datetime.fromtimestamp(now_ts).astimezone()
        cron = croniter(expr, base_dt)
        next_dt = cron.get_next(datetime)
        next_ts = next_dt.timestamp()
        if next_ts > now_ts:
            return next_ts
        # prevent same-second loop: retry from now+1s
        retry_dt = datetime.fromtimestamp(math.floor(now_ts) + 1.0).astimezone()
        cron2 = croniter(expr, retry_dt)
        return cron2.get_next(datetime).timestamp()

    return None
```

**The anchor formula for every** is the key design. Compare the two approaches:

```
last_run_based: 10:00 -> 11:02 -> 12:04 -> 13:06  (execution delay causes drift)
anchor_based:   10:00 -> 11:00 -> 12:00 -> 13:00  (always aligned)
```

The formula `anchor + ceil((now - anchor) / interval) * interval` guarantees that no matter how long execution takes, the next trigger aligns to an integer multiple.

**The same-second loop guard for cron**: croniter may return a time <= now (when now happens to fall exactly on a trigger second); retrying from now+1s avoids an infinite loop.

### 3. CronStore -- JSON Persistence with Atomic Writes

All job definitions and states are saved in `jobs.json`, using tmp + rename for atomic writes:

```python
class CronStore:
    def _save_unlocked(self, jobs: list[dict]) -> None:
        data = {"version": 1, "jobs": jobs}
        content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        tmp_path = self.store_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            tmp_path.write_text(content, encoding="utf-8")
            tmp_path.replace(self.store_path)
        except OSError:
            self.store_path.write_text(content, encoding="utf-8")
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
```

`os.replace()` is an atomic operation on POSIX systems. Even if the process crashes mid-write, either the old file remains intact or the new file is complete -- never a half-written state.

**Atomic write guarantee: after a crash and restart, the job list is either pre-update or post-update, never corrupted.**

### 4. CronRunLog -- JSONL Append Log

Each job execution result is appended to a JSONL file. When the file exceeds 2MB, it is pruned to keep only the most recent half:

```python
class CronRunLog:
    MAX_SIZE_BYTES = 2 * 1024 * 1024
    MAX_LINES = 2000

    def append(self, entry: dict) -> None:
        with self._lock:
            line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
            self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        if self.log_path.stat().st_size <= self.MAX_SIZE_BYTES:
            return
        raw = self.log_path.read_text(encoding="utf-8")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        keep_count = self.MAX_LINES // 2
        kept = lines[-keep_count:] if len(lines) > keep_count else lines
        self.log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
```

**JSONL advantages: append writes are O(1) -- no need to read the entire file first. Pruning keeps only the tail, so old logs are naturally retired.**

### 5. CronService -- Background Scheduling Engine

A background thread scans all jobs every second. When a job is due, it executes:

```python
class CronService:
    AUTO_DISABLE_THRESHOLD = 5

    def _background_loop(self) -> None:
        while not self._stop_event.is_set():
            self._compute_all_next_runs()
            due_jobs = self._find_due_jobs(time.time())
            for job in due_jobs:
                result = self._execute_job(job)
                # output to queue for main thread to display
                if result["status"] == "ok" and result["response"]:
                    self._output_queue.append(f"Job '{job['name']}': {result['response']}")
                self._compute_all_next_runs()  # recompute immediately after execution
            self._stop_event.wait(1.0)

    def _execute_job(self, job: dict) -> dict:
        start_ts = time.time()
        try:
            response_text = agent_fn(payload_message)
            # success: reset consecutive_errors
            state_patch["consecutive_errors"] = 0
        except Exception as exc:
            # failure: increment consecutive_errors
            state_patch["consecutive_errors"] = prev_errors + 1

        # delete_after_run: disable after one-shot execution
        if job.get("delete_after_run") and result["status"] == "ok":
            self.store.update_job(job_id, {"enabled": False})

        # auto-disable after 5 consecutive failures
        if state_patch["consecutive_errors"] >= self.AUTO_DISABLE_THRESHOLD:
            self.store.update_job(job_id, {"enabled": False})
```

**5 consecutive failures trigger auto-disable: allows occasional network-related failures (1-2), but 5 in a row almost certainly indicates a configuration problem.**

### 6. Agent Tools: cron_create / cron_list / cron_delete

The agent manages scheduled tasks autonomously through tools:

```python
TOOLS = [
    {
        "name": "cron_create",
        "description": "Create a new scheduled cron job. "
                       "schedule_type: 'at' | 'every' | 'cron'...",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "schedule_type": {"type": "string"},
                "schedule_value": {"type": "string"},
                "message": {"type": "string"},
                "delete_after_run": {"type": "boolean"},
            },
            "required": ["name", "schedule_type", "schedule_value", "message"],
        },
    },
    {"name": "cron_list", ...},
    {"name": "cron_delete", ...},
]
```

When the user says "check server status every hour," the agent calls `cron_create(schedule_type="every", schedule_value="3600", ...)` to create the task automatically.

**The agent creates and manages scheduled tasks on its own -- the user just describes requirements in natural language.**

## What Changed from s08

| Component | s08 | s09 |
|-----------|-----|-----|
| Timing capability | Fixed-interval heartbeat (every N seconds) | Three modes: at / every / cron |
| Task management | Single HEARTBEAT.md | CronStore manages multiple independent jobs |
| Run log | No persistence | CronRunLog JSONL with auto-pruning |
| Error handling | Simple try/except | consecutive_errors count + auto-disable |
| Persistence | In-memory state, lost on restart | JSON atomic writes, survives restart |
| Agent tools | None | cron_create / cron_list / cron_delete |

**Key shift**: From "single heartbeat metronome" to "multi-task scheduling engine." s08 answered "can the agent act proactively"; s09 answers "can the agent do the right thing at the right time."

## Design Decisions

**Why anchor-based interval instead of last-run-based?**

last_run + interval suffers from cumulative drift caused by execution time. Anchor-based scheduling aligns from the creation time at fixed intervals, guaranteeing that "check every hour on the hour" stays on the hour regardless of how long the previous execution took.

**Why auto-disable after 5 consecutive errors?**

A cron job with a broken prompt failing every minute burns API quota and log space continuously. The threshold of 5 allows for occasional network failures but 5 in a row almost certainly means a configuration error. After disabling, the user can inspect and fix it via `/cron`.

**Why JSONL instead of JSON array for the run log?**

A JSON array requires reading the entire file before appending -- O(N) write complexity. JSONL appends a single line -- O(1) writes. For a high-frequency write log, this difference matters.

**In production OpenClaw:** The cron system is implemented in `src/cron/`, with compute_next_run_at in `src/cron/schedule.ts`. The production version supports a timezone field (cross-timezone scheduling), `*/5` step values and `1-5` ranges in advanced cron syntax, and dynamic task management via API. The scheduling service shares lane mutual exclusion with the heartbeat to prevent simultaneous execution. The run log is stored in SQLite.

## Try It

```sh
cd claw0
python agents/s09_cron.py
```

On the first run, two sample jobs are created: `demo-cron` (every minute, enabled) and `demo-every` (every 90 seconds, disabled by default).

Create scheduled tasks in natural language:

```
You > Check the current time every 30 seconds and report it.
  [tool:cron_create] {"name":"time-check","schedule_type":"every","schedule_value":"30",...}

You > Remind me to drink water in 2 minutes.
  [tool:cron_create] {"name":"drink-water","schedule_type":"at",...}
```

View jobs and logs:

```
You > /cron
--- Cron Jobs ---
  [ON] demo-cron | demo-every-minute
       schedule: cron '* * * * *'
       next_run: 2026-02-24 14:31:00 | last: ok | errors: 0
  [OFF] demo-every | demo-every-90s
       schedule: every 90s
       next_run: N/A | last: - | errors: 0
--- end (2 jobs) ---

You > /cron-log
--- Recent Cron Runs ---
  14:30:00 ok demo-every-minute (1200ms) Cron check at 14:30...
--- end (1 entries) ---
```
