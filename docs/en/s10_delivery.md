# s10: Delivery Queue & Reliable Messaging

> "Write before you send, retry until you succeed" -- never lose a message.

## At a Glance

```
  Agent Response / Heartbeat Output
       |
       v
  enqueue(channel, to, text)
       |
       v
  delivery-queue/{uuid}.json    <-- atomic: .tmp -> os.replace
       |
       v
  DeliveryRunner (background thread, 1s poll)
       |
       v
  attempt_delivery()
       |
       +--- success ---> ack() (delete .json)
       |
       +--- failure ---> fail()
                |            retry_count++
                |            compute_backoff_ms(retry_count)
                |            next_retry_at = now + backoff
                |
                +-- retry_count <= 5 --> wait, retry
                |     backoff: [5s, 25s, 2m, 10m, 10m]
                |
                +-- retry_count > 5 ---> move_to_failed/

  Gateway restart --> recovery_scan() --> resume pending
```

- **What we build**: A disk-persistent delivery queue guaranteeing at-least-once delivery
- **Core mechanism**: Write-ahead (write before send) + exponential backoff retries [5s, 25s, 2m, 10m, 10m]
- **Design pattern**: The file system is the queue -- each message is a JSON file; ack deletes, fail updates, move_to_failed relocates

## The Problem

1. **Failed send = lost message**: Telegram/Discord APIs can become temporarily unavailable due to network fluctuations, rate limiting, or service outages. Previous implementations used `try/except` to print the error, and the message was gone.
2. **Process crash = unrecoverable**: The agent generated a reply but the process crashed before sending. After restart there is no record; the message is permanently lost.
3. **No retry mechanism**: After a single send failure there is no second chance. The user might never receive that critical scheduled reminder.

## How It Works

### 1. QueuedDelivery -- The Delivery Record

Complete lifecycle information for each pending delivery:

```python
@dataclass
class QueuedDelivery:
    id: str                      # UUID, also used as filename
    channel: str                 # target channel: "telegram", "discord"
    to: str                      # recipient ID
    text: str                    # message content
    retry_count: int = 0         # number of retries so far
    last_error: str | None = None
    enqueued_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0   # earliest time for next retry

    def to_dict(self) -> dict: ...

    @staticmethod
    def from_dict(data: dict) -> "QueuedDelivery": ...
```

**Every field serves the delivery lifecycle: retry_count determines whether to keep retrying, next_retry_at controls backoff spacing, last_error aids troubleshooting.**

### 2. DeliveryQueue -- Four Core Operations on the Disk Queue

```python
class DeliveryQueue:
    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.failed_dir = queue_dir / "failed"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
```

**enqueue -- atomic write to disk**:

```python
def enqueue(self, channel: str, to: str, text: str) -> str:
    delivery_id = uuid.uuid4().hex[:16]
    entry = QueuedDelivery(
        id=delivery_id, channel=channel, to=to, text=text,
        enqueued_at=time.time(), next_retry_at=0.0,
    )
    self._write_entry(entry)
    return delivery_id

def _write_entry(self, entry: QueuedDelivery) -> None:
    file_path = self._entry_path(entry.id)
    tmp_path = file_path.parent / f".tmp.{os.getpid()}.{entry.id}.json"
    tmp_path.write_text(
        json.dumps(entry.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(str(tmp_path), str(file_path))
```

**ack -- delivery succeeded, delete the file**:

```python
def ack(self, delivery_id: str) -> None:
    try:
        self._entry_path(delivery_id).unlink()
    except FileNotFoundError:
        pass
```

**fail -- delivery failed, update backoff or move to failed/**:

```python
def fail(self, delivery_id: str, error: str) -> None:
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
```

**move_to_failed -- exceeded maximum retries**:

```python
def move_to_failed(self, delivery_id: str) -> None:
    os.replace(
        str(self._entry_path(delivery_id)),
        str(self.failed_dir / f"{delivery_id}.json"),
    )
```

**The file system is the queue: enqueue creates a file, ack deletes it, fail updates it, move_to_failed relocates it. `ls delivery-queue/` shows every pending message.**

### 3. Exponential Backoff -- The Backoff Sequence

After failure, instead of retrying immediately, the system waits progressively longer:

```python
BACKOFF_MS = [5_000, 25_000, 120_000, 600_000]
MAX_RETRIES = 5

@staticmethod
def compute_backoff_ms(retry_count: int) -> int:
    if retry_count <= 0:
        return 0
    idx = min(retry_count - 1, len(BACKOFF_MS) - 1)
    return BACKOFF_MS[idx]
```

Backoff sequence for retries 1 through 5:

| retry_count | idx | backoff |
|-------------|-----|---------|
| 1 | 0 | 5s |
| 2 | 1 | 25s |
| 3 | 2 | 2m |
| 4 | 3 | 10m |
| 5 | 3 (capped) | 10m |

When retry_count exceeds 5, `move_to_failed` is called and no further retries are attempted.

**5s handles network jitter, 25s handles brief rate limiting, 2m handles API restarts, 10m is the ceiling -- waiting any longer is not worth doing automatically.**

### 4. DeliveryRunner -- Background Delivery Thread

Scans the queue every second and attempts delivery for entries whose backoff has expired:

```python
class DeliveryRunner:
    def __init__(self, queue: DeliveryQueue, deliver_fn):
        self.queue = queue
        self.deliver_fn = deliver_fn  # (channel, to, text) -> None, raises on failure

    def _process_pending(self) -> None:
        now = time.time()
        for entry in self.queue.load_pending():
            if entry.next_retry_at > now:
                continue  # backoff not yet elapsed, skip
            if self._attempt_delivery(entry):
                self.queue.ack(entry.id)
            else:
                self.queue.fail(entry.id, "delivery failed")

    def _attempt_delivery(self, entry: QueuedDelivery) -> bool:
        try:
            self.deliver_fn(entry.channel, entry.to, entry.text)
            return True
        except Exception:
            return False

    def _background_loop(self) -> None:
        while not self._stop_event.is_set():
            self._process_pending()
            self._stop_event.wait(1.0)
```

**The logic is simple but reliable: scan -> check backoff -> attempt send -> ack or fail. Every second.**

### 5. Startup Recovery -- recovery_scan

On process restart, the queue directory is scanned to report and resume unfinished deliveries:

```python
def _recovery_scan(self) -> None:
    pending = self.queue.load_pending()
    failed = self.queue.load_failed()
    if pending:
        print_delivery(f"recovery: {len(pending)} pending entries, resuming")
    if failed:
        print_delivery(f"recovery: {len(failed)} entries in failed/")
    if not pending and not failed:
        print_delivery("recovery: queue empty")
```

This is the core value of the write-ahead pattern: the message is already on disk. After a crash and restart, `load_pending()` scans the directory and recovery happens automatically.

**No special recovery logic is needed -- the files are the state, and scanning the files is recovery.**

### 6. Agent Replies Go Through the Delivery Queue

In s10, agent replies are no longer printed directly. Instead, they are enqueued to the delivery queue:

```python
# Before (s09): direct output
print_assistant(text)

# Now (s10): enqueue first, DeliveryRunner delivers
did = delivery_queue.enqueue(mock_channel.name, "user", text)
print_info(f"  enqueued -> delivery queue (id={did[:8]}..)")
```

Heartbeat output also goes through the delivery queue:

```python
for msg in heartbeat.drain_output():
    did = delivery_queue.enqueue(mock_channel.name, "user", msg)
    print_heartbeat(f"enqueued heartbeat message (id={did[:8]}..)")
```

**All outbound messages (user replies, heartbeat notifications) go through the delivery queue, ensuring consistent reliability.**

## What Changed from s09

| Component | s09 | s10 |
|-----------|-----|-----|
| Message delivery | Direct send, no guarantees | Write-ahead queue, at-least-once |
| Failure handling | try/except prints error | Backoff retries [5s, 25s, 2m, 10m, 10m] |
| Persistence | Only cron tasks persisted | Every outbound message persisted to disk |
| Retry limit | Cron auto-disables after 5 consecutive errors | MAX_RETRIES=5 then move to failed/ |
| Crash recovery | Restart loads cron jobs | Restart scans queue directory, resumes delivery |
| Commands | /cron, /cron-log | /queue, /failed, /retry, /simulate-failure |

**Key shift**: From "best-effort send" to "guaranteed delivery." Production systems assume everything will fail (network down, API rate-limited, process crash) and design a recovery path for each failure mode.

## Design Decisions

**Why file-system queue instead of SQLite?**

The file-system queue has clear advantages for teaching: `ls delivery-queue/` shows every pending message, a text editor can inspect and modify the JSON, `os.replace` provides atomicity, and there are zero external dependencies. When queue depth exceeds a few thousand entries or complex queries are needed, SQLite is the better choice.

**Why this specific backoff sequence [5s, 25s, 2m, 10m]?**

Empirically fitted to common failure recovery times: 5s handles transient faults like DNS timeouts, 25s handles brief rate-limiting windows on platforms like Telegram (typically 10-30s), 2m handles API deployment updates and service restarts, 10m is the ceiling -- failures lasting longer than 10 minutes likely require human intervention. Pure exponential backoff (1s, 2s, 4s...) was not used because the early retries are too short (the rate limit may still be active) and the later ones are too long (the user cannot wait).

**Why MAX_RETRIES=5 and move_to_failed instead of infinite retry?**

Infinite retry masks configuration errors: an invalid API token or a nonexistent recipient will never succeed no matter how many times you retry. Five retries cover every stage of the backoff sequence, totaling about 13 minutes of wait time. Afterward, the entry moves to failed/ and an administrator can recover it manually via `/retry`.

**In production OpenClaw:** The delivery queue is implemented in `src/infra/outbound/delivery-queue.ts`. The production version uses a SQLite backend for high throughput, and the backoff strategy adds jitter (random perturbation) to prevent multiple messages from retrying at the same instant (thundering herd). It supports priority queues (direct user replies take priority over cron results), per-channel rate-limit controls, and a web UI for viewing and manually retrying failed entries.

## Try It

```sh
cd claw0
python agents/s10_delivery.py
```

Normal conversation -- observe messages being enqueued then delivered:

```
You > Hello
  enqueued -> delivery queue (id=a1b2c3d4..)
  [delivery] delivered a1b2c3d4.. to telegram:user

[telegram -> user] Hello! How can I help you today?
```

Enable failure simulation and observe backoff retries:

```
You > /simulate-failure
Fail rate -> 50%

You > Tell me a joke.
  enqueued -> delivery queue (id=e5f6g7h8..)
  [delivery] failed e5f6g7h8.. (retry 1/5, next in 5s)
  [delivery] failed e5f6g7h8.. (retry 2/5, next in 25s)
  [delivery] delivered e5f6g7h8.. to telegram:user

You > /simulate-failure
Fail rate -> 0%
```

View queue and failed status:

```
You > /queue
--- Delivery Queue ---
  Pending: 0  Attempted: 3  OK: 2  Fail: 1
---

You > /failed
--- Failed (0) ---
  Use /retry to move back.

You > /retry
Moved 0 entries from failed/ back to queue.
```
