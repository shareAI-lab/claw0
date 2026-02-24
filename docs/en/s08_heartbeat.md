# s08: Heartbeat & Proactive Behavior

> "Not just reactive -- proactive" -- from passive chat to active monitoring.

## At a Glance

```
  +--- HeartbeatRunner (background thread) -------+
  |  loop every 1s:                                |
  |    should_run()?                                |
  |      [1] enabled?        (HEARTBEAT.md exists)  |
  |      [2] interval?       (>= N seconds elapsed) |
  |      [3] active hours?   (09:00-22:00)          |
  |      [4] has content?    (skip empty headings)   |
  |      [5] main lane idle? (lock.acquire nowait)   |
  |      [6] not running?    (self.running == False)  |
  |    all pass -> acquire lock -> run agent          |
  +---------------------------------------------------+
            |                         |
            v                    (mutual exclusion)
     Agent(HEARTBEAT.md)              |
            |                         v
            v                    User Message
     +------+------+            (holds same lock)
     |             |
  HEARTBEAT_OK   Content
  (suppress)       |
                   v
              SHA-256 dedup
              (24h window)
                   |
                   v
              Output to user
```

- **What we build**: A background thread that lets the agent periodically check whether something needs proactive reporting
- **Core mechanism**: 6-step should_run check chain + HEARTBEAT_OK silent signal + SHA-256 deduplication
- **Design pattern**: Heartbeat yields to user messages (mutual exclusion lock), runs only at safe moments

## The Problem

1. **Completely passive**: The s07 agent must wait for user input before it can respond. If the user says "submit the report by 3pm," the agent will not proactively remind them at 2:50.
2. **Middle-of-the-night interruptions**: Without a time-window control, background tasks could message the user at 3am.
3. **Duplicate notifications**: The agent might repeatedly report the same thing -- "you have a pending task" every 30 seconds until the user is overwhelmed.

## How It Works

### 1. HEARTBEAT.md -- The Heartbeat Instruction File

Like SOUL.md, HEARTBEAT.md is a Markdown file defining what the agent should check during heartbeats:

```md
# Heartbeat Instructions

Check the following and report ONLY if action is needed:

1. Review today's memory log for any unfinished tasks or pending items.
2. If the user mentioned a deadline or reminder, check if it is approaching.
3. If there are new daily memories, summarize any actionable items.

If nothing needs attention, respond with exactly: HEARTBEAT_OK
```

Whether the file exists determines whether heartbeat is enabled. Delete the file to disable heartbeat; create it to enable -- no code changes needed.

**File existence = on/off switch. Non-programmers can control the heartbeat.**

### 2. 6-Step should_run Check Chain

Checked once per second; all 6 conditions must pass before a heartbeat fires:

```python
def should_run(self) -> tuple[bool, str]:
    if not self._is_enabled():
        return False, "disabled (no HEARTBEAT.md)"
    if not self._interval_elapsed():
        return False, "interval not elapsed"
    if not self._is_active_hours():
        return False, "outside active hours"
    if not self._heartbeat_has_content():
        return False, "HEARTBEAT.md has no actionable content"
    if not self._main_lane_idle():
        return False, "main lane busy (user message in progress)"
    if self.running:
        return False, "heartbeat already running"
    return True, "ok"
```

`_heartbeat_has_content()` skips pure headings and empty checkboxes, ensuring the file contains real instructions:

```python
def _heartbeat_has_content(self) -> bool:
    if not self.heartbeat_path.exists():
        return False
    content = self.heartbeat_path.read_text(encoding="utf-8")
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#+(\s|$)", stripped):
            continue
        if re.match(r"^[-*+]\s*(\[[\sXx]?\]\s*)?$", stripped):
            continue
        return True
    return False
```

**Every failed step returns a specific reason, making debugging easy -- no guessing why the heartbeat did not fire.**

### 3. Lane Mutual Exclusion -- Heartbeat Yields to User

The heartbeat and user messages share a single `threading.Lock`. The key difference is in how each acquires it:

```python
# Heartbeat thread: non-blocking acquire, skip if fails
def _background_loop(self, agent_fn) -> None:
    while not self._stop_event.is_set():
        should, reason = self.should_run()
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
            finally:
                self.running = False
                self._lock.release()
        self._stop_event.wait(1.0)

# Main thread: blocking acquire, waits for heartbeat to finish
heartbeat._lock.acquire()
try:
    # process user message...
finally:
    heartbeat._lock.release()
```

**User first: heartbeat uses non-blocking acquire (fails = skip), user messages use blocking acquire (waits for heartbeat to finish). The user never waits for the heartbeat.**

### 4. HEARTBEAT_OK -- The Silent Signal

When the agent decides there is nothing to report, it returns `HEARTBEAT_OK` and the system suppresses output:

```python
def _strip_heartbeat_ok(self, text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return True, ""
    without_token = stripped.replace(HEARTBEAT_OK_TOKEN, "").strip()
    # nothing left after removing token, or residual <= 5 chars -> silent
    if not without_token or len(without_token) <= 5:
        return True, ""
    if HEARTBEAT_OK_TOKEN in stripped:
        return False, without_token
    return False, stripped
```

**HEARTBEAT_OK lets the agent itself judge "is there anything worth reporting," rather than bothering the user on every heartbeat cycle.**

### 5. SHA-256 Deduplication -- 24-Hour Window

Even if the agent thinks something is worth reporting, if the same content was already sent within 24 hours, it is not sent again:

```python
def _content_hash(self, content: str) -> str:
    normalized = content.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

def is_duplicate(self, content: str) -> bool:
    h = self._content_hash(content)
    now = time.time()
    expired = [k for k, v in self.dedup_cache.items()
               if now - v > DEDUP_WINDOW_SECONDS]
    for k in expired:
        del self.dedup_cache[k]
    if h in self.dedup_cache:
        return True
    self.dedup_cache[h] = now
    return False
```

**The hash is only 16 characters. Normalization (strip + lower) before hashing prevents false mismatches from casing and whitespace differences.**

### 6. Full Heartbeat Execution Flow

A complete heartbeat: load instructions -> build prompt -> call agent -> silence check -> dedup check -> output:

```python
def run_heartbeat_turn(self, agent_fn) -> str | None:
    heartbeat_content = self.heartbeat_path.read_text(encoding="utf-8").strip()
    heartbeat_prompt = (
        "This is a scheduled heartbeat check. "
        "Follow the HEARTBEAT.md instructions below strictly.\n"
        "Do NOT infer or repeat old tasks from prior context.\n"
        "If nothing needs attention, respond with exactly: HEARTBEAT_OK\n\n"
        f"--- HEARTBEAT.md ---\n{heartbeat_content}\n--- end ---"
    )
    response_text = agent_fn(heartbeat_prompt)
    if not response_text:
        return None
    is_ok, cleaned = self._strip_heartbeat_ok(response_text)
    if is_ok:
        return None
    if self.is_duplicate(cleaned):
        return None
    return cleaned
```

The heartbeat uses a single-turn, tool-free call to keep token usage low and avoid side effects like writing to memory during the heartbeat.

**Heartbeats are low-cost probes: simple question, simple answer, no tool chain triggered.**

## What Changed from s07

| Component | s07 | s08 |
|-----------|-----|-----|
| Behavior model | Passive (waits for user input) | Active + passive (background thread, periodic triggers) |
| Config files | SOUL.md + MEMORY.md | Added HEARTBEAT.md |
| Thread model | Single thread | Main thread + background heartbeat thread |
| Concurrency | None | threading.Lock mutual exclusion |
| Output filtering | None | HEARTBEAT_OK silence + SHA-256 dedup |
| Time control | None | active_hours window |

**Key shift**: The agent goes from "waits for the user to speak" to "checks on its own whether there is something to report." The heartbeat system is one of OpenClaw's most distinctive features.

## Design Decisions

**Why a 6-step chain instead of just a timer?**

A plain timer cannot handle complex constraints: it should not interrupt while the user is chatting, should not message at 3am, should not run if the file has been deleted. The 6-step chain makes every precondition explicit, and each failed step returns a concrete reason, making debugging straightforward.

**Why HEARTBEAT_OK instead of an empty string?**

LLMs do not return completely empty responses. Having the model return an explicit token is more reliable than expecting it to return nothing. It also gives the model a clear exit mechanism: finish checking, nothing to report, say HEARTBEAT_OK -- no need to fabricate meaningless content.

**Why SHA-256 content hash instead of time-based dedup?**

Time-only dedup would miss: the same issue triggered at different times has different timestamps but identical content. Content hashing ensures that as long as the output is the same (ignoring case and whitespace), it is considered a duplicate.

**In production OpenClaw:** The heartbeat is implemented via `HeartbeatRunner` in `src/heartbeat/`. The should_run check chain aligns with this section exactly. The production version adds cron-expression-based trigger timing, HEARTBEAT_OK detection that handles HTML/Markdown wrapping (`<b>HEARTBEAT_OK</b>`), and ackMaxChars allowing an OK with a small amount of text to still count as silent. Mutual exclusion uses CommandLane queue depth rather than a simple mutex, providing finer-grained control.

## Try It

```sh
cd claw0
python agents/s08_heartbeat.py
```

Default heartbeat interval is 60 seconds (adjustable via `HEARTBEAT_INTERVAL` environment variable).

First write a pending task, then observe whether the heartbeat detects it:

```
You > Remember that I need to submit the report by 3pm tomorrow.
Assistant: I've saved that to memory...

(wait for heartbeat to trigger)
[Heartbeat] Reminder: you need to submit the report by 3pm tomorrow.
```

View heartbeat status and trigger manually:

```
You > /heartbeat
--- Heartbeat Status ---
  Enabled: True
  Active hours: True
  Interval: 60s
  Last run: 45s ago
  Next in: ~15s
  Should run: False (interval not elapsed)
  Dedup cache: 1 entries

You > /trigger
[Heartbeat] Your memory log shows a pending task...
```
