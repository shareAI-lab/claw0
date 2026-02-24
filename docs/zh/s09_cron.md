# s09: Cron Scheduler (定时调度)

> 让 agent 掌握时间的维度 -- 不仅能响应消息, 还能按时间表主动执行任务。

## 问题

s08 的心跳实现了后台定期检查, 但它的调度能力非常有限:

1. **只有固定间隔**: 心跳只能 "每 N 秒检查一次", 无法表达 "每周一早上 9 点" 这种基于日历的时间规则。
2. **无法处理一次性任务**: 用户说 "明天下午 3 点提醒我开会", 心跳没有 "在特定时间点执行一次然后销毁" 的语义。
3. **缺少多种调度模式**: 真实场景需要三种模式混合使用 -- 一次性任务 (at)、固定间隔 (every)、cron 表达式 (cron)。s08 的 CronScheduler 只是一个简化原型, 不具备这些能力。

本节实现一个完整的定时调度系统, 支持三种调度类型、任务持久化、运行日志和自动容错。

## 解决方案

```
  CronStore (jobs.json)
       |
  CronService (background thread)
       |
  every 1s: find_due_jobs()
       |
       v
  [at] one-shot    [every] interval    [cron] expression
  "2026-02-25T15:00"  "every 3600s"    "0 9 * * 1"
       |                  |                 |
       v                  v                 v
  run_agent() -> result -> CronRunLog (JSONL)
       |
  if delete_after_run: disable job
  if consecutive_errors >= 5: auto-disable
```

核心思路: CronStore 用 JSON 文件持久化所有任务定义和状态。CronService 后台线程每秒扫描一次到期任务, 根据调度类型计算下次运行时间。每次执行结果写入 JSONL 运行日志。支持一次性任务自动销毁和连续失败自动禁用。

## 工作原理

### 1. CronSchedule -- 三种调度类型

调度系统的核心抽象是 `CronSchedule`, 用一个 tagged union 表示三种类型:

```python
@dataclass
class CronSchedule:
    kind: str        # "at" | "every" | "cron"
    value: str       # ISO datetime | seconds | cron expression

    @staticmethod
    def at(iso_time: str) -> "CronSchedule":
        """一次性任务: 在指定时间点执行一次。"""
        return CronSchedule(kind="at", value=iso_time)

    @staticmethod
    def every(seconds: int) -> "CronSchedule":
        """固定间隔: 每 N 秒执行一次。"""
        return CronSchedule(kind="every", value=str(seconds))

    @staticmethod
    def cron(expression: str) -> "CronSchedule":
        """Cron 表达式: 标准 5 字段格式。"""
        return CronSchedule(kind="cron", value=expression)
```

三种类型覆盖了常见的定时需求:

- **at**: `"2026-02-25T15:00:00"` -- 明天下午 3 点提醒我。执行后自动禁用。
- **every**: `"3600"` -- 每小时检查一次 API 状态。基于 anchor 时间计算, 不会漂移。
- **cron**: `"0 9 * * 1"` -- 每周一早上 9 点发周报。标准 5 字段 cron 表达式 (分 时 日 月 星期)。

### 2. compute_next_run_at() -- 下次运行时间计算

这是调度系统最核心的函数, 根据不同的调度类型计算下次应该运行的时间:

```python
def compute_next_run_at(schedule: CronSchedule, anchor: float, now: float) -> float | None:
    if schedule.kind == "at":
        target = datetime.fromisoformat(schedule.value).timestamp()
        if target > now:
            return target
        return None  # 已过期

    elif schedule.kind == "every":
        interval = int(schedule.value)
        if interval <= 0:
            return None
        # anchor-based: 从创建时间开始, 按固定间隔对齐
        elapsed = now - anchor
        periods = int(elapsed / interval)
        next_run = anchor + (periods + 1) * interval
        return next_run

    elif schedule.kind == "cron":
        return _next_cron_match(schedule.value, now)

    return None
```

**anchor-based interval 计算**:

`every` 类型不使用 "上次运行时间 + 间隔" 的方式, 而是以任务创建时间 (anchor) 为基准计算。公式:

```
next_run = anchor + (periods + 1) * interval
periods  = floor((now - anchor) / interval)
```

例如: anchor = 10:00, interval = 3600s (1 小时)。不管实际运行时间是 11:02 还是 11:58, 下次运行都是 12:00。这样避免了执行延迟导致的时间漂移。

### 3. CronJob & CronJobState -- 任务定义与状态追踪

任务分为静态定义和运行时状态两部分:

```python
@dataclass
class CronJobState:
    last_run_at: float = 0.0
    next_run_at: float = 0.0
    run_count: int = 0
    consecutive_errors: int = 0
    last_error: str = ""

@dataclass
class CronJob:
    id: str
    name: str
    schedule: CronSchedule
    prompt: str                  # 执行时发送给 agent 的 prompt
    enabled: bool = True
    delete_after_run: bool = False  # at 类型默认 True
    created_at: float = 0.0
    state: CronJobState = field(default_factory=CronJobState)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "schedule": {"kind": self.schedule.kind, "value": self.schedule.value},
            "prompt": self.prompt,
            "enabled": self.enabled,
            "delete_after_run": self.delete_after_run,
            "created_at": self.created_at,
            "state": {
                "last_run_at": self.state.last_run_at,
                "next_run_at": self.state.next_run_at,
                "run_count": self.state.run_count,
                "consecutive_errors": self.state.consecutive_errors,
                "last_error": self.state.last_error,
            },
        }

    @staticmethod
    def from_dict(d: dict) -> "CronJob":
        sched = CronSchedule(kind=d["schedule"]["kind"], value=d["schedule"]["value"])
        state = CronJobState(**d.get("state", {}))
        return CronJob(
            id=d["id"], name=d["name"], schedule=sched,
            prompt=d["prompt"], enabled=d.get("enabled", True),
            delete_after_run=d.get("delete_after_run", False),
            created_at=d.get("created_at", 0.0), state=state,
        )
```

`CronJobState` 追踪运行时信息: 上次运行时间、下次计划时间、累计运行次数、连续失败次数。连续失败次数用于自动禁用。

### 4. CronStore -- JSON 持久化与原子写入

所有任务定义和状态保存在一个 JSON 文件中, 使用原子写入避免进程崩溃时数据损坏:

```python
class CronStore:
    def __init__(self, store_path: Path):
        self.store_path = store_path
        self.jobs: dict[str, CronJob] = {}
        self._load()

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        data = json.loads(self.store_path.read_text(encoding="utf-8"))
        for d in data.get("jobs", []):
            job = CronJob.from_dict(d)
            self.jobs[job.id] = job

    def _save(self) -> None:
        data = {"jobs": [job.to_dict() for job in self.jobs.values()]}
        content = json.dumps(data, indent=2, ensure_ascii=False)
        # 原子写入: 先写临时文件, 再 rename
        tmp_path = self.store_path.with_suffix(".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, self.store_path)

    def add_job(self, job: CronJob) -> None:
        self.jobs[job.id] = job
        self._save()

    def remove_job(self, job_id: str) -> bool:
        if job_id in self.jobs:
            del self.jobs[job_id]
            self._save()
            return True
        return False

    def update_state(self, job_id: str, state: CronJobState) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].state = state
            self._save()

    def disable_job(self, job_id: str) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].enabled = False
            self._save()
```

原子写入的关键: `os.replace()` 在 POSIX 系统上是原子操作。即使进程在写入途中崩溃, 要么旧文件完好, 要么新文件完整, 不会出现半写状态。

### 5. CronRunLog -- JSONL 运行日志与裁剪

每次任务执行的结果写入 JSONL (JSON Lines) 格式的日志文件, 每行一条记录:

```python
class CronRunLog:
    def __init__(self, log_path: Path, max_lines: int = 500):
        self.log_path = log_path
        self.max_lines = max_lines

    def append(self, job_id: str, job_name: str, success: bool,
               result: str, duration_ms: int) -> None:
        entry = {
            "ts": datetime.now().isoformat(),
            "job_id": job_id,
            "job_name": job_name,
            "success": success,
            "result": result[:1000],  # 截断过长的结果
            "duration_ms": duration_ms,
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._prune()

    def _prune(self) -> None:
        """保留最近 max_lines 条记录, 删除旧条目。"""
        if not self.log_path.exists():
            return
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self.max_lines:
            return
        keep = lines[-self.max_lines:]
        self.log_path.write_text("\n".join(keep) + "\n", encoding="utf-8")

    def recent(self, count: int = 20) -> list[dict]:
        if not self.log_path.exists():
            return []
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines[-count:]:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
        entries.reverse()  # 最新的在前
        return entries
```

JSONL 格式的优势: 追加写入不需要解析整个文件 (对比 JSON 数组需要先读取再追加)。裁剪时只需读取行数, 保留尾部即可。

### 6. CronService -- 后台调度服务

CronService 是后台线程, 每秒扫描一次所有任务, 找出到期的任务并执行:

```python
class CronService:
    MAX_CONSECUTIVE_ERRORS = 5

    def __init__(self, store: CronStore, run_log: CronRunLog):
        self.store = store
        self.run_log = run_log
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._agent_fn = None

    def start(self, agent_fn) -> None:
        self._agent_fn = agent_fn
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="cron-service",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._tick()
            self._stop_event.wait(1.0)

    def _tick(self) -> None:
        now = time.time()
        for job in list(self.store.jobs.values()):
            if not job.enabled:
                continue
            next_run = compute_next_run_at(job.schedule, job.created_at, now)
            if next_run is None:
                # at 类型已过期且未执行, 标记禁用
                if job.schedule.kind == "at" and job.state.run_count == 0:
                    self._execute_job(job)
                continue
            job.state.next_run_at = next_run
            if next_run <= now:
                self._execute_job(job)

    def _execute_job(self, job: CronJob) -> None:
        start_ms = int(time.time() * 1000)
        try:
            result = self._agent_fn(job.prompt)
            duration = int(time.time() * 1000) - start_ms
            job.state.last_run_at = time.time()
            job.state.run_count += 1
            job.state.consecutive_errors = 0
            job.state.last_error = ""
            self.run_log.append(job.id, job.name, True, result, duration)

            if job.delete_after_run:
                job.enabled = False

        except Exception as exc:
            duration = int(time.time() * 1000) - start_ms
            job.state.last_run_at = time.time()
            job.state.consecutive_errors += 1
            job.state.last_error = str(exc)
            self.run_log.append(job.id, job.name, False, str(exc), duration)

            # 连续失败超过阈值, 自动禁用
            if job.state.consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                job.enabled = False

        self.store.update_state(job.id, job.state)
```

关键设计:
- 每秒 tick 一次, 扫描所有任务。任务数量在合理范围内 (几十到几百), O(N) 扫描没有性能问题。
- `_execute_job` 成功时清零 `consecutive_errors`, 失败时累加。达到阈值自动禁用, 防止一个坏任务无限重试浪费资源。
- `delete_after_run` 用于 `at` 类型的一次性任务, 执行后自动标记为禁用。

## 核心代码

调度系统最核心的函数 -- `compute_next_run_at`, 它决定了每个任务何时运行:

```python
def compute_next_run_at(schedule: CronSchedule, anchor: float, now: float) -> float | None:
    """根据调度类型计算下次运行时间。

    Args:
        schedule: 调度定义 (at/every/cron)
        anchor: 任务创建时间, 作为 every 类型的对齐基准
        now: 当前时间戳

    Returns:
        下次运行的 Unix 时间戳, 或 None (at 类型已过期)
    """
    if schedule.kind == "at":
        target = datetime.fromisoformat(schedule.value).timestamp()
        if target > now:
            return target
        return None

    elif schedule.kind == "every":
        interval = int(schedule.value)
        if interval <= 0:
            return None
        elapsed = now - anchor
        periods = int(elapsed / interval)
        next_run = anchor + (periods + 1) * interval
        return next_run

    elif schedule.kind == "cron":
        return _next_cron_match(schedule.value, now)

    return None


def _next_cron_match(expression: str, now: float) -> float:
    """简化版 cron 表达式匹配, 找到下一个匹配时间。

    支持 5 字段格式: 分 时 日 月 星期
    支持 * (任意)、具体数字、逗号分隔列表。
    """
    fields = expression.split()
    if len(fields) != 5:
        raise ValueError(f"invalid cron expression: {expression}")

    def parse_field(field: str, min_val: int, max_val: int) -> set[int]:
        if field == "*":
            return set(range(min_val, max_val + 1))
        values = set()
        for part in field.split(","):
            values.add(int(part))
        return values

    minutes = parse_field(fields[0], 0, 59)
    hours = parse_field(fields[1], 0, 23)
    days = parse_field(fields[2], 1, 31)
    months = parse_field(fields[3], 1, 12)
    weekdays = parse_field(fields[4], 0, 6)  # 0=Monday in Python

    dt = datetime.fromtimestamp(now).replace(second=0, microsecond=0)
    dt += timedelta(minutes=1)  # 至少下一分钟

    # 最多搜索 366 天
    limit = dt + timedelta(days=366)
    while dt < limit:
        if (dt.month in months and dt.day in days
                and dt.weekday() in weekdays and dt.hour in hours
                and dt.minute in minutes):
            return dt.timestamp()
        dt += timedelta(minutes=1)

    return now + 86400  # fallback: 24 小时后
```

## 和上一节的区别

| 组件 | s08 | s09 |
|------|-----|-----|
| 定时能力 | 固定间隔心跳 (每 N 秒) | 三种模式: at / every / cron |
| 调度类型 | 单一: 周期性检查 | 一次性、固定间隔、cron 表达式 |
| 任务管理 | 无 (只有一个心跳) | CronStore 管理多个独立任务 |
| 运行日志 | 无持久化日志 | CronRunLog JSONL 记录每次执行 |
| 错误处理 | 简单 try/except | 连续失败计数 + 自动禁用 |
| 工具 | /heartbeat, /trigger | /cron, /cron-log, /trigger-cron |
| 持久化 | 无 (内存状态, 重启丢失) | JSON 文件持久化, 重启恢复 |

关键转变: 从 "单一心跳" 变成 "多任务调度引擎"。s08 回答了 "Agent 能不能主动行动", s09 回答了 "Agent 能不能在正确的时间做正确的事"。这是从固定节拍器到灵活日历的跨越。

## 设计解析

**为什么用 anchor-based interval 而不是 last-run-based?**

如果用 `last_run + interval` 的方式, 每次执行的延迟都会累积。假设 interval 是 1 小时, 但每次执行耗时 2 分钟:

- last-run-based: 10:00 -> 11:02 -> 12:04 -> 13:06 ... (逐步漂移)
- anchor-based: 10:00 -> 11:00 -> 12:00 -> 13:00 ... (始终对齐)

anchor-based 保证间隔的稳定性, 这对 "每小时整点检查" 这类场景至关重要。OpenClaw 的生产实现也使用 anchor-based 方式。

**为什么 cron job 要支持 auto-disable?**

一个 prompt 写得有问题的 cron job 可能每次都失败 (比如引用了不存在的工具, 或者 prompt 导致 API 报错)。如果不自动禁用, 它会每分钟失败一次, 持续消耗 API 额度和日志空间。

`MAX_CONSECUTIVE_ERRORS = 5` 是一个保守的阈值: 允许网络抖动导致的偶发失败 (1-2 次), 但连续 5 次失败几乎可以确定是配置问题。禁用后写入日志, 用户可以通过 `/cron` 查看状态并修复。

**OpenClaw 生产版的不同之处:**

- `src/cron/` 实现了完整的 cron 解析器, 支持 `*/5` (步进)、`1-5` (范围)、`L` (最后一天) 等高级语法
- 任务定义支持 `timezone` 字段, 跨时区调度
- 支持 `every` 的自然语言语法: `every 30m`, `every 2h`, `every 1d`
- 支持任务依赖: 一个任务可以在另一个任务成功后触发
- run log 存储在 SQLite 中, 支持按时间范围查询和聚合统计
- 调度服务与心跳服务共享 lane 互斥, 避免 cron 和心跳同时运行
- 支持通过 API 动态创建/修改/删除 cron job, 不需要重启 gateway

## 试一试

```sh
cd mini-claw
python agents/s09_cron.py
```

启动后, 系统会加载 `workspace/cron-jobs.json` 中的任务定义 (如果存在)。

可以尝试:

1. 创建一个每 30 秒执行的定时任务:

```
You > 帮我创建一个定时任务, 每 30 秒检查一下当前时间并报告。
```

Agent 会调用 cron 工具创建一个 `every` 类型的任务。

2. 创建一个一次性提醒:

```
You > 2 分钟后提醒我喝水。
```

Agent 会创建一个 `at` 类型的任务, 执行后自动禁用。

3. 查看所有定时任务:

```
You > /cron
```

输出类似:

```
Cron Jobs:
  [1] check-time (every 30s) -- enabled, next: 14:30:30, runs: 3
  [2] drink-water (at 2026-02-24T14:35:00) -- enabled, next: 14:35:00, runs: 0
```

4. 查看运行日志:

```
You > /cron-log
```

输出类似:

```
Recent Cron Runs:
  2026-02-24T14:30:30 | check-time | OK | 1.2s | "Current time is 14:30."
  2026-02-24T14:30:00 | check-time | OK | 0.9s | "It is 2:30 PM."
```

5. 手动触发一个任务:

```
You > /trigger-cron check-time
```

6. 创建一个 cron 表达式任务:

```
You > 每周一早上 9 点给我发一份本周待办总结。
```

Agent 会创建一个 `cron` 类型的任务, 表达式为 `0 9 * * 0`。

7. 观察自动禁用: 创建一个必定失败的任务 (比如 prompt 引用不存在的工具), 等待它连续失败 5 次后被自动禁用, 通过 `/cron` 查看状态变化。
