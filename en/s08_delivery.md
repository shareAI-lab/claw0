# Section 08: Delivery

> Write to disk first, then try to send. Crash-safe.

## Architecture

```
    Agent Reply / Heartbeat / Cron
              |
        chunk_message()          split by platform limits
              |                  (telegram=4096, discord=2000, etc.)
              v
        DeliveryQueue.enqueue()
          1. Generate unique ID
          2. Write to .tmp.{pid}.{id}.json
          3. fsync()
          4. os.replace() to {id}.json    <-- WRITE-AHEAD
              |
              v
        DeliveryRunner (background thread, 1s scan)
              |
        deliver_fn(channel, to, text)
           /          \
        success      failure
          |              |
        ack()         fail()
        (delete       (retry_count++, compute backoff,
         .json)        update .json on disk)
                         |
                    retry_count >= 5?
                      |yes
                    move to failed/

    Backoff: [5s, 25s, 2min, 10min] with +/-20% jitter
```

## What's New

- **DeliveryQueue**: disk-persisted write-ahead queue. Enqueue writes to disk before attempting delivery.
- **Atomic writes**: tmp file + `os.fsync()` + `os.replace()` -- no half-written files on crash.
- **DeliveryRunner**: background thread that processes pending entries with exponential backoff.
- **chunk_message()**: splits text by platform size limits, respecting paragraph boundaries.
- **Recovery scan**: on startup, pending entries from a previous crash are automatically retried.

## Key Code Walkthrough

### 1. DeliveryQueue.enqueue() + atomic write

The fundamental rule: write to disk first, then attempt delivery. If the
process crashes between enqueue and delivery, the message survives on disk.

```python
def enqueue(self, channel: str, to: str, text: str) -> str:
    delivery_id = uuid.uuid4().hex[:12]
    entry = QueuedDelivery(
        id=delivery_id, channel=channel, to=to, text=text,
        enqueued_at=time.time(), next_retry_at=0.0,
    )
    self._write_entry(entry)
    return delivery_id

def _write_entry(self, entry: QueuedDelivery) -> None:
    final_path = self.queue_dir / f"{entry.id}.json"
    tmp_path = self.queue_dir / f".tmp.{os.getpid()}.{entry.id}.json"

    data = json.dumps(entry.to_dict(), indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())        # data is on disk

    os.replace(str(tmp_path), str(final_path))  # atomic on POSIX
```

Three-step guarantee:
- Step 1: write to `.tmp.{pid}.{id}.json` (crash = orphaned temp, harmless)
- Step 2: `fsync()` -- data is on disk
- Step 3: `os.replace()` -- atomic swap (crash = old file or new file, never partial)

### 2. ack() / fail() -- the retry lifecycle

```python
def ack(self, delivery_id: str) -> None:
    """Delivery succeeded. Delete the queue file."""
    (self.queue_dir / f"{delivery_id}.json").unlink()

def fail(self, delivery_id: str, error: str) -> None:
    """Increment retry_count, compute next retry time, or give up."""
    entry = self._read_entry(delivery_id)
    entry.retry_count += 1
    entry.last_error = error
    if entry.retry_count >= MAX_RETRIES:
        self.move_to_failed(delivery_id)
        return
    backoff_ms = compute_backoff_ms(entry.retry_count)
    entry.next_retry_at = time.time() + backoff_ms / 1000.0
    self._write_entry(entry)  # update on disk with new retry state
```

Backoff with jitter prevents thundering herd:

```python
BACKOFF_MS = [5_000, 25_000, 120_000, 600_000]
MAX_RETRIES = 5

def compute_backoff_ms(retry_count: int) -> int:
    if retry_count <= 0:
        return 0
    idx = min(retry_count - 1, len(BACKOFF_MS) - 1)
    base = BACKOFF_MS[idx]
    jitter = random.randint(-base // 5, base // 5)   # +/- 20%
    return max(0, base + jitter)
```

### 3. DeliveryRunner -- the background loop

Scans pending entries every second. Processes only those whose `next_retry_at`
has passed. On startup, runs a recovery scan for entries from prior crashes.

```python
class DeliveryRunner:
    def start(self) -> None:
        self._recovery_scan()
        self._thread = threading.Thread(
            target=self._background_loop, daemon=True)
        self._thread.start()

    def _process_pending(self) -> None:
        pending = self.queue.load_pending()
        now = time.time()
        for entry in pending:
            if entry.next_retry_at > now:
                continue
            self.total_attempted += 1
            try:
                self.deliver_fn(entry.channel, entry.to, entry.text)
                self.queue.ack(entry.id)
                self.total_succeeded += 1
            except Exception as exc:
                self.queue.fail(entry.id, str(exc))
                self.total_failed += 1
```

## Try It

```sh
python en/s08_delivery.py

# Send a message -- watch it get enqueued and delivered
# You > Hello!

# Enable 50% failure rate
# You > /simulate-failure

# Send another message -- watch retries with backoff
# You > Test message under failure

# Inspect the queue
# You > /queue
# You > /failed

# Restore reliability and watch pending entries deliver
# You > /simulate-failure

# Check statistics
# You > /stats
```

## How OpenClaw Does It

| Aspect         | claw0 (this file)               | OpenClaw production                   |
|----------------|----------------------------------|---------------------------------------|
| Queue storage  | JSON files in a directory        | Same file-per-entry pattern           |
| Atomic writes  | tmp + fsync + os.replace         | Same approach                         |
| Backoff        | [5s, 25s, 2min, 10min] + jitter | Same schedule                         |
| Message chunks | Paragraph-boundary splitting     | Same + code fence awareness           |
| Recovery       | Scan queue dir on startup        | Same scan + orphan cleanup            |

## Series Summary

Over 8 sections, the core mechanisms of an agent framework:

```
    Section 01: while True + stop_reason        (the loop)
    Section 02: TOOLS + TOOL_HANDLERS           (the hands)
    Section 03: JSONL + ContextGuard            (the memory)
    Section 04: Channel ABC + InboundMessage    (the mouths)
    Section 05: BindingTable + session key      (the router)
    Section 06: 8-layer prompt + TF-IDF         (the brain)
    Section 07: Heartbeat + Cron                (the initiative)
    Section 08: DeliveryQueue + backoff         (the guarantee)
```

The agent loop from Section 01 is still recognizable at the core of
Section 08. An AI agent is a `while True` loop with a dispatch table,
wrapped in layers of persistence, routing, intelligence, scheduling,
and reliability.
