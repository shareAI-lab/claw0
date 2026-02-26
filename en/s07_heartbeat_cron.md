# Section 07: Heartbeat & Cron

> A timer thread checks "should I run?" and queues work alongside user messages.

## Architecture

```
    Main Lane (user input):
        User Input --> lane_lock.acquire() -------> LLM --> Print
                       (blocking: always wins)

    Heartbeat Lane (background thread, 1s poll):
        should_run()?
            |no --> sleep 1s
            |yes
        _execute():
            lane_lock.acquire(blocking=False)
                |fail --> yield (user has priority)
                |success
            build prompt from HEARTBEAT.md + SOUL.md + MEMORY.md
                |
            run_agent_single_turn()
                |
            parse: "HEARTBEAT_OK"? --> suppress
                   meaningful text? --> duplicate? --> suppress
                                           |no
                                       output_queue.append()

    Cron Service (background thread, 1s tick):
        CRON.json --> load jobs --> tick() every 1s
            |
        for each job: enabled? --> due? --> _run_job()
            |
        error? --> consecutive_errors++ --> >=5? --> auto-disable
            |ok
        consecutive_errors = 0 --> log to cron-runs.jsonl
```

## What's New

- **Lane mutual exclusion**: `threading.Lock` shared between user and heartbeat. User always wins (blocking acquire); heartbeat yields (non-blocking).
- **should_run()**: 4 precondition checks before each heartbeat attempt.
- **HEARTBEAT_OK**: convention for the agent to signal "nothing to report."
- **CronService**: 3 schedule types (`at`, `every`, `cron`), auto-disable after 5 consecutive errors.
- **Output queues**: background results drain into the REPL via thread-safe lists.

## Key Code Walkthrough

### 1. Lane mutual exclusion

The most important design principle: user messages always win.

```python
lane_lock = threading.Lock()

# Main lane: blocking acquire. User ALWAYS gets in.
lane_lock.acquire()
try:
    # handle user message, call LLM
finally:
    lane_lock.release()

# Heartbeat lane: non-blocking acquire. Yields if user is active.
def _execute(self) -> None:
    acquired = self.lane_lock.acquire(blocking=False)
    if not acquired:
        return   # user has the lock, skip this heartbeat
    self.running = True
    try:
        instructions, sys_prompt = self._build_heartbeat_prompt()
        response = run_agent_single_turn(instructions, sys_prompt)
        meaningful = self._parse_response(response)
        if meaningful and meaningful.strip() != self._last_output:
            self._last_output = meaningful.strip()
            with self._queue_lock:
                self._output_queue.append(meaningful)
    finally:
        self.running = False
        self.last_run_at = time.time()
        self.lane_lock.release()
```

### 2. should_run() -- the precondition chain

Four checks must all pass. The lock is tested separately in `_execute()`
to avoid TOCTOU races.

```python
def should_run(self) -> tuple[bool, str]:
    if not self.heartbeat_path.exists():
        return False, "HEARTBEAT.md not found"
    if not self.heartbeat_path.read_text(encoding="utf-8").strip():
        return False, "HEARTBEAT.md is empty"

    elapsed = time.time() - self.last_run_at
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
```

### 3. CronService -- 3 schedule types

Jobs are defined in `CRON.json`. Each has a `schedule.kind` and a `payload`:

```python
@dataclass
class CronJob:
    id: str
    name: str
    enabled: bool
    schedule_kind: str       # "at" | "every" | "cron"
    schedule_config: dict
    payload: dict            # {"kind": "agent_turn", "message": "..."}
    consecutive_errors: int = 0

def _compute_next(self, job, now):
    if job.schedule_kind == "at":
        ts = datetime.fromisoformat(cfg.get("at", "")).timestamp()
        return ts if ts > now else 0.0
    if job.schedule_kind == "every":
        every = cfg.get("every_seconds", 3600)
        # align to anchor for predictable firing times
        steps = int((now - anchor) / every) + 1
        return anchor + steps * every
    if job.schedule_kind == "cron":
        return croniter(expr, datetime.fromtimestamp(now)).get_next(datetime).timestamp()
```

Auto-disable after 5 consecutive errors:

```python
if status == "error":
    job.consecutive_errors += 1
    if job.consecutive_errors >= 5:
        job.enabled = False
else:
    job.consecutive_errors = 0
```

## Try It

```sh
python en/s07_heartbeat_cron.py

# Create workspace/HEARTBEAT.md with instructions:
# "Check if there are any unread reminders. Reply HEARTBEAT_OK if nothing to report."

# Check heartbeat status
# You > /heartbeat

# Force a heartbeat run
# You > /trigger

# List cron jobs (requires workspace/CRON.json)
# You > /cron

# Check lane lock status
# You > /lanes
# main_locked: False  heartbeat_running: False
```

Example `CRON.json`:

```json
{
  "jobs": [
    {
      "id": "daily-check",
      "name": "Daily Check",
      "enabled": true,
      "schedule": {"kind": "cron", "expr": "0 9 * * *"},
      "payload": {"kind": "agent_turn", "message": "Generate a daily summary."}
    }
  ]
}
```

## How OpenClaw Does It

| Aspect           | claw0 (this file)             | OpenClaw production                     |
|------------------|-------------------------------|-----------------------------------------|
| Lane exclusion   | `threading.Lock`, non-blocking| Same lock pattern                       |
| Heartbeat config | `HEARTBEAT.md` in workspace   | Same file + env var overrides           |
| Cron schedules   | `CRON.json`, 3 kinds          | Same format + webhook triggers          |
| Auto-disable     | 5 consecutive errors          | Same threshold, configurable            |
| Output delivery  | In-memory queue, drain to REPL| Delivery queue (Section 08)             |
