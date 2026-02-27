# Section 10: Concurrency

> Named lanes serialize the chaos.

## Architecture

```
    Incoming Work
        |
    CommandQueue.enqueue(lane, fn)
        |
    +---v---+    +--------+    +-----------+
    | main  |    |  cron  |    | heartbeat |
    | max=1 |    | max=1  |    |   max=1   |
    | FIFO  |    | FIFO   |    |   FIFO    |
    +---+---+    +---+----+    +-----+-----+
        |            |              |
    [active]     [active]       [active]
        |            |              |
    _task_done   _task_done     _task_done
        |            |              |
    _pump()      _pump()        _pump()
    (dequeue     (dequeue       (dequeue
     next if      next if        next if
     active<max)  active<max)    active<max)
```

Each lane is a `LaneQueue`: a FIFO deque guarded by a `threading.Condition`. Tasks enter as plain callables and return results through `concurrent.futures.Future`. The `CommandQueue` dispatches work into the correct lane by name and manages the full lifecycle.

## Key Concepts

- **Named lanes**: Each lane has a name (e.g. `"main"`, `"cron"`, `"heartbeat"`) and its own independent FIFO queue. Lanes are created lazily on first use.
- **max_concurrency**: Each lane caps how many tasks run simultaneously. Default is 1 (serial execution). Increase to allow parallel work within a lane.
- **_pump() loop**: After each task completes (`_task_done`), the lane checks if more tasks can be dequeued. This self-pumping design means no external scheduler is needed.
- **Future-based results**: Every `enqueue()` returns a `concurrent.futures.Future`. Callers can block on `future.result()` or attach callbacks via `add_done_callback()`.
- **Generation tracking**: Each lane has an integer generation counter. On `reset_all()`, all generations increment. When a stale task completes (its generation does not match current), `_pump()` is not called -- preventing zombie tasks from draining the queue after a restart.
- **Condition-based synchronization**: `threading.Condition` replaces the raw `threading.Lock` from Section 07. This enables `wait_for_idle()` to sleep efficiently until notified rather than polling.
- **User priority**: User input goes into the `main` lane and blocks on the result. Background work (heartbeat, cron) goes into separate lanes and never blocks the REPL.

## Key Code Walkthrough

### 1. LaneQueue -- the core primitive

A lane is a deque + condition variable + active counter. `_pump()` is the engine:

```python
class LaneQueue:
    def __init__(self, name: str, max_concurrency: int = 1) -> None:
        self.name = name
        self.max_concurrency = max(1, max_concurrency)
        self._deque = deque()           # [(fn, future, generation), ...]
        self._condition = threading.Condition()
        self._active_count = 0
        self._generation = 0

    def enqueue(self, fn, generation=None):
        future = concurrent.futures.Future()
        with self._condition:
            gen = generation if generation is not None else self._generation
            self._deque.append((fn, future, gen))
            self._pump()
        return future

    def _pump(self):
        """Pop and start tasks while active < max_concurrency."""
        while self._active_count < self.max_concurrency and self._deque:
            fn, future, gen = self._deque.popleft()
            self._active_count += 1
            threading.Thread(
                target=self._run_task, args=(fn, future, gen), daemon=True
            ).start()

    def _task_done(self, gen):
        with self._condition:
            self._active_count -= 1
            if gen == self._generation:  # stale tasks do not re-pump
                self._pump()
            self._condition.notify_all()
```

### 2. CommandQueue -- the dispatcher

The `CommandQueue` holds a dict of lane_name to `LaneQueue`. Lanes are created lazily:

```python
class CommandQueue:
    def __init__(self):
        self._lanes: dict[str, LaneQueue] = {}
        self._lock = threading.Lock()

    def get_or_create_lane(self, name, max_concurrency=1):
        with self._lock:
            if name not in self._lanes:
                self._lanes[name] = LaneQueue(name, max_concurrency)
            return self._lanes[name]

    def enqueue(self, lane_name, fn):
        lane = self.get_or_create_lane(lane_name)
        return lane.enqueue(fn)

    def reset_all(self):
        """Increment generation on all lanes for restart recovery."""
        with self._lock:
            for lane in self._lanes.values():
                with lane._condition:
                    lane._generation += 1
```

### 3. Generation tracking -- restart recovery

The generation counter solves a subtle problem: if the system restarts while tasks are in flight, those tasks may complete and try to pump the queue with stale state. By incrementing the generation, all old callbacks become harmless no-ops:

```python
def _task_done(self, gen):
    with self._condition:
        self._active_count -= 1
        if gen == self._generation:
            self._pump()       # current generation: normal flow
        # else: stale task -- do NOT pump, let it die quietly
        self._condition.notify_all()
```

### 4. HeartbeatRunner -- lane-aware skip

Instead of `lock.acquire(blocking=False)`, the heartbeat checks the lane stats:

```python
def heartbeat_tick(self):
    ok, reason = self.should_run()
    if not ok:
        return

    lane_stats = self.command_queue.get_or_create_lane(LANE_HEARTBEAT).stats()
    if lane_stats["active"] > 0:
        return  # lane is busy, skip this tick

    future = self.command_queue.enqueue(LANE_HEARTBEAT, _do_heartbeat)
    future.add_done_callback(_on_done)
```

This is functionally equivalent to the non-blocking lock pattern but expressed in terms of the lane abstraction.

## Try It

```sh
python en/s10_concurrency.py

# Show all lanes and their current status
# You > /lanes
#   main          active=[.]  queued=0  max=1  gen=0
#   cron          active=[.]  queued=0  max=1  gen=0
#   heartbeat     active=[.]  queued=0  max=1  gen=0

# Manually enqueue work into a named lane
# You > /enqueue main What is the capital of France?

# Create a custom lane and enqueue work into it
# You > /enqueue research Summarize recent AI developments

# Change max_concurrency for a lane
# You > /concurrency research 3

# Show generation counters
# You > /generation

# Simulate a restart (increment all generations)
# You > /reset

# Show pending items per lane
# You > /queue
```

## How OpenClaw Does It

| Aspect              | claw0 (this file)                         | OpenClaw production                            |
|---------------------|-------------------------------------------|------------------------------------------------|
| Lane primitive      | `LaneQueue` with `threading.Condition`    | Same pattern, with metrics instrumentation     |
| Dispatcher          | `CommandQueue` dict of lanes              | Same lazy-creation dispatcher                  |
| Concurrency control | `max_concurrency` per lane, default 1     | Same, configurable per deployment              |
| Task execution      | `threading.Thread` per task               | Thread pool with bounded workers               |
| Result delivery     | `concurrent.futures.Future`               | Same Future-based interface                    |
| Generation tracking | Integer counter, stale tasks skip pump    | Same generation pattern for restart safety     |
| Idle detection      | `wait_for_idle()` with Condition.wait()   | Same, used for graceful shutdown               |
| Standard lanes      | main, cron, heartbeat                     | Same defaults + plugin-defined custom lanes    |
| User priority       | Main lane blocks on result                | Same blocking semantics for user input         |
