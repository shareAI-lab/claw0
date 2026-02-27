# 第 07 节: 心跳与 Cron

> 一个定时器线程检查"该不该运行", 然后将任务排入与用户消息相同的队列.

## 架构

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

## 本节要点

- **Lane 互斥**: `threading.Lock` 在用户和心跳之间共享. 用户总是赢 (阻塞获取); 心跳让步 (非阻塞获取).
- **should_run()**: 每次心跳尝试前的 4 个前置条件检查.
- **HEARTBEAT_OK**: agent 用来表示"没有需要报告的内容"的约定.
- **CronService**: 3 种调度类型 (`at`, `every`, `cron`), 连续错误 5 次后自动禁用.
- **输出队列**: 后台结果通过线程安全的列表输送到 REPL.

## 核心代码走读

### 1. Lane 互斥

最重要的设计原则: 用户消息始终优先.

```python
lane_lock = threading.Lock()

# Main lane: 阻塞获取. 用户始终能进入.
lane_lock.acquire()
try:
    # 处理用户消息, 调用 LLM
finally:
    lane_lock.release()

# Heartbeat lane: 非阻塞获取. 用户活跃时让步.
def _execute(self) -> None:
    acquired = self.lane_lock.acquire(blocking=False)
    if not acquired:
        return   # 用户持有锁, 跳过本次心跳
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

### 2. should_run() -- 前置条件链

四个检查必须全部通过. 锁的检测在 `_execute()` 中单独进行,
以避免 TOCTOU 竞态条件.

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

### 3. CronService -- 3 种调度类型

任务定义在 `CRON.json` 中. 每个任务有一个 `schedule.kind` 和一个 `payload`:

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
        # 对齐到锚点, 保证触发时间可预测
        steps = int((now - anchor) / every) + 1
        return anchor + steps * every
    if job.schedule_kind == "cron":
        return croniter(expr, datetime.fromtimestamp(now)).get_next(datetime).timestamp()
```

连续 5 次错误后自动禁用:

```python
if status == "error":
    job.consecutive_errors += 1
    if job.consecutive_errors >= 5:
        job.enabled = False
else:
    job.consecutive_errors = 0
```

## 试一试

```sh
python zh/s07_heartbeat_cron.py

# 创建 workspace/HEARTBEAT.md 写入指令:
# "Check if there are any unread reminders. Reply HEARTBEAT_OK if nothing to report."

# 检查心跳状态
# You > /heartbeat

# 手动触发心跳
# You > /trigger

# 列出 cron 任务 (需要 workspace/CRON.json)
# You > /cron

# 检查 lane 锁状态
# You > /lanes
# main_locked: False  heartbeat_running: False
```

`CRON.json` 示例:

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

## OpenClaw 中的对应实现

| 方面             | claw0 (本文件)                 | OpenClaw 生产代码                       |
|------------------|-------------------------------|-----------------------------------------|
| Lane 互斥       | `threading.Lock`, 非阻塞      | 相同的锁模式                            |
| 心跳配置         | 工作区中的 `HEARTBEAT.md`      | 相同文件 + 环境变量覆盖                 |
| Cron 调度        | `CRON.json`, 3 种类型         | 相同格式 + webhook 触发器               |
| 自动禁用         | 连续 5 次错误                  | 相同阈值, 可配置                        |
| 输出投递         | 内存队列, 排出到 REPL          | 投递队列 (第 08 节)                     |
