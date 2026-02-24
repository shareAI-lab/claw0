# s08: Heartbeat & Proactive Behavior (心跳与主动行为)

> 不只是被动回复 -- 让 Agent 自己检查、自己汇报, 从 "等人问" 变成 "主动说"。

## 问题

前面所有章节的 Agent 都是 **被动的**: 用户说一句, Agent 回一句。不说, 就不动。

但一个真正的助手应该能主动行动:

1. **定时检查**: 用户设了一个提醒, Agent 应该到时间自动提醒, 而不是等用户来问。
2. **后台监控**: 如果记忆中有待办事项快到 deadline, Agent 应该主动通知。
3. **不该打扰的时候不打扰**: 凌晨 3 点不应该发消息; 用户正在对话时心跳应该让位。
4. **不重复通知**: 同一件事已经说过了, 24 小时内不应该再说。

这就是 heartbeat (心跳) 系统: 让 Agent 定期 "检查一下" 是否有需要汇报的事情, 在正确的时机、以不重复的方式主动通知用户。

## 解决方案

```
  +--- HeartbeatRunner (background thread) ---+
  |  every Ns:                                 |
  |  [1] enabled?                              |
  |  [2] interval elapsed?                     |
  |  [3] active hours?                         |
  |  [4] HEARTBEAT.md exists?                  |
  |  [5] main lane idle? ---+                  |
  |  [6] not running?       |                  |
  +-------------------------+------------------+
           |                |
           v          (mutual exclusion)
    Run agent with              |
    HEARTBEAT.md context        |
           |                    |
           v                    v
    Response check         User Message
    /            \              |
  HEARTBEAT_OK   Content       v
  (suppress)     |         Agent Loop
                 v         (takes priority)
            Dedup check
                 |
                 v
            Output to user
```

核心思路: 后台线程每秒检查一次 6 步条件链, 全部满足时执行心跳调用。心跳结果经过 HEARTBEAT_OK 静默判断和 24 小时去重后, 才输出给用户。通过互斥锁保证心跳不干扰正在进行的用户对话。

## 工作原理

### 1. 6 步检查链 (should_run)

心跳不是 "到时间就跑", 而是每次都要通过 6 步检查:

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

逐步解析每个检查:

**[1] 是否启用**: 通过 HEARTBEAT.md 文件是否存在来判断。删除文件就等于关闭心跳。

```python
def _is_enabled(self) -> bool:
    return self.heartbeat_path.exists()
```

**[2] 间隔是否已过**: 距离上次运行是否超过了配置的间隔时间。

```python
def _interval_elapsed(self) -> bool:
    return (time.time() - self.last_run) >= self.interval
```

**[3] 是否在活跃时段**: 只在配置的时间窗口内运行, 避免凌晨打扰用户。支持跨午夜时段 (如 22:00-06:00)。

```python
def _is_active_hours(self) -> bool:
    current_hour = datetime.now().hour
    if self.active_start <= self.active_end:
        return self.active_start <= current_hour < self.active_end
    else:
        # 跨午夜
        return current_hour >= self.active_start or current_hour < self.active_end
```

**[4] HEARTBEAT.md 是否有实质内容**: 跳过纯空行、纯 heading、空 checkbox, 只有真正有指令的行才算有内容。

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

**[5] 主通道是否空闲**: 用非阻塞方式尝试获取互斥锁, 获取成功说明没有用户消息在处理。

```python
def _main_lane_idle(self) -> bool:
    acquired = self._lock.acquire(blocking=False)
    if acquired:
        self._lock.release()
        return True
    return False
```

**[6] 当前是否空闲**: 避免上一轮心跳还没跑完就启动新一轮。

### 2. HEARTBEAT_OK 静默机制

HEARTBEAT.md 告诉 Agent 检查哪些事项。如果检查后发现没什么要汇报的, Agent 返回 `HEARTBEAT_OK`, 网关将其静默处理 (不输出给用户)。

```python
HEARTBEAT_OK_TOKEN = "HEARTBEAT_OK"

def _strip_heartbeat_ok(self, text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return True, ""

    # 移除 HEARTBEAT_OK 标记
    without_token = stripped.replace(HEARTBEAT_OK_TOKEN, "").strip()

    # 移除后没有实质内容 -> 静默
    if not without_token or len(without_token) <= 5:
        return True, ""

    # 有实质内容, 返回去掉标记后的文本
    if HEARTBEAT_OK_TOKEN in stripped:
        return False, without_token
    return False, stripped
```

HEARTBEAT.md 示例:

```md
# Heartbeat Instructions

Check the following and report ONLY if action is needed:

1. Review today's memory log for any unfinished tasks or pending items.
2. If the user mentioned a deadline or reminder, check if it is approaching.
3. If there are new daily memories, summarize any actionable items.

If nothing needs attention, respond with exactly: HEARTBEAT_OK
```

### 3. 24 小时内容去重

使用 SHA-256 哈希避免同一内容在 24 小时内重复发送:

```python
DEDUP_WINDOW_SECONDS = 24 * 60 * 60

def _content_hash(self, content: str) -> str:
    normalized = content.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

def is_duplicate(self, content: str) -> bool:
    h = self._content_hash(content)
    now = time.time()

    # 清理过期条目
    expired = [k for k, v in self.dedup_cache.items()
               if now - v > DEDUP_WINDOW_SECONDS]
    for k in expired:
        del self.dedup_cache[k]

    if h in self.dedup_cache:
        return True

    self.dedup_cache[h] = now
    return False
```

### 4. 互斥锁 -- 心跳让位于用户消息

心跳和用户消息共享一把 `threading.Lock`:

```python
class HeartbeatRunner:
    def __init__(self, ...):
        # 互斥锁: 心跳和用户消息共享
        self._lock = threading.Lock()
```

后台线程尝试用非阻塞方式获取锁:

```python
def _background_loop(self, agent_fn) -> None:
    while not self._stop_event.is_set():
        should, reason = self.should_run()
        if should:
            acquired = self._lock.acquire(blocking=False)
            if not acquired:
                # 用户消息正在处理, 跳过本轮
                self._stop_event.wait(1.0)
                continue

            try:
                self.running = True
                self.last_run = time.time()
                result = self.run_heartbeat_turn(agent_fn)
                if result:
                    with self._output_lock:
                        self._output_queue.append(result)
            except Exception as exc:
                with self._output_lock:
                    self._output_queue.append(f"[heartbeat error: {exc}]")
            finally:
                self.running = False
                self._lock.release()

        self._stop_event.wait(1.0)
```

主线程处理用户消息时阻塞式获取锁:

```python
# 主线程: 用户消息处理
heartbeat._lock.acquire()
try:
    # 处理用户输入, 调用 LLM...
finally:
    heartbeat._lock.release()
```

效果: 心跳遇到锁被占用就跳过 (非阻塞), 用户消息遇到锁被占用就等待 (阻塞)。用户消息优先级高于心跳。

### 5. 输出队列

心跳在后台线程运行, 但输出需要在主线程展示。通过线程安全的队列传递:

```python
# 后台线程写入
with self._output_lock:
    self._output_queue.append(result)

# 主线程读取 (在每次等待用户输入前)
def drain_output(self) -> list[str]:
    with self._output_lock:
        messages = self._output_queue[:]
        self._output_queue.clear()
        return messages
```

主循环中的集成:

```python
while True:
    # 在等待用户输入前, 输出心跳消息
    for msg in heartbeat.drain_output():
        print_heartbeat(msg)

    user_input = input(colored_prompt()).strip()
    # ...
```

### 6. 心跳执行

一次完整的心跳执行:

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

    # 检查 HEARTBEAT_OK
    is_ok, cleaned = self._strip_heartbeat_ok(response_text)
    if is_ok:
        return None

    # 去重检查
    if self.is_duplicate(cleaned):
        return None

    return cleaned
```

流程: 加载 HEARTBEAT.md -> 构建 prompt -> 调用 agent -> HEARTBEAT_OK 检查 -> 去重检查 -> 输出。

### 7. CronScheduler -- 简化版定时调度

除了心跳, 本节还展示了一个简化的 cron 调度器:

```python
class CronScheduler:
    def __init__(self):
        self.jobs: list[dict] = []

    def add_job(self, name: str, interval_seconds: int, callback) -> None:
        self.jobs.append({
            "name": name,
            "interval": interval_seconds,
            "callback": callback,
            "last_run": 0.0,
        })

    def tick(self) -> list[str]:
        now = time.time()
        executed = []
        for job in self.jobs:
            if now - job["last_run"] >= job["interval"]:
                try:
                    job["callback"]()
                    job["last_run"] = now
                    executed.append(job["name"])
                except Exception as exc:
                    job["last_run"] = now
        return executed
```

## 核心代码

心跳系统的核心 -- HeartbeatRunner 的初始化和后台线程启动:

```python
class HeartbeatRunner:
    def __init__(self, interval_seconds=1800, active_hours=(9, 22),
                 heartbeat_path=None):
        self.interval = interval_seconds
        self.active_start, self.active_end = active_hours
        self.heartbeat_path = heartbeat_path or (WORKSPACE_DIR / "HEARTBEAT.md")
        self.last_run: float = 0.0
        self.dedup_cache: dict[str, float] = {}
        self.running = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._output_queue: list[str] = []
        self._output_lock = threading.Lock()

    def start(self, agent_fn) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._background_loop,
            args=(agent_fn,),
            daemon=True,
            name="heartbeat-runner",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
```

心跳使用的 agent 函数 -- 单轮无工具调用, 降低 token 消耗:

```python
def run_agent_single_turn(prompt: str) -> str:
    system = build_system_prompt()
    try:
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = ""
        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
        return text.strip()
    except Exception as exc:
        return f"[agent error: {exc}]"
```

## 和上一节的区别

| 组件 | s07 | s08 |
|------|-----|-----|
| Agent 行为 | 纯被动: 用户说才回 | 主动 + 被动: 心跳定期检查并汇报 |
| 后台线程 | 无 | HeartbeatRunner 后台线程, 每秒检查 |
| HEARTBEAT.md | 无 | 定义心跳检查内容 |
| 互斥锁 | 无 | threading.Lock 保证心跳/用户消息互斥 |
| HEARTBEAT_OK | 无 | 静默信号, agent 没事报时不输出 |
| 去重 | 无 | SHA-256 哈希, 24 小时内不重复发送 |
| 活跃时段 | 无 | 只在配置的时间窗口内运行心跳 |
| 输出队列 | 无 | 后台线程产出 -> 主线程输出 |
| 命令 | /soul, /memory | /heartbeat (状态), /trigger (手动触发) |

关键转变: 从 "Agent 等待用户" 变成 "Agent 主动检查并通知"。这是从 chatbot 到 AI 助手的根本区别 -- 助手不需要你时刻关注它, 它会在需要的时候主动联系你。

## 设计解析

**为什么心跳用后台线程而不是定时器?**

后台线程以 1 秒间隔轮询, 好处:
- 停止信号能在 1 秒内响应 (定时器可能要等完整间隔)
- 6 步检查链可以在每次轮询中完整执行
- 与互斥锁配合更自然 (尝试获取 -> 失败则跳过 -> 下次再试)

**为什么 HEARTBEAT_OK 不直接用空字符串?**

因为 LLM 不可能返回完全空的响应。让模型返回一个明确的标记 (HEARTBEAT_OK) 比期待它返回空更可靠。这也给了模型一个清晰的 "退出机制": 检查完没事, 就说 HEARTBEAT_OK, 而不是硬编一些无意义的内容。

**为什么去重用内容哈希而不是时间?**

只靠时间去重会漏判: 同一件事在不同时间触发, 时间不同但内容相同。用内容的 SHA-256 哈希, 只要输出内容一样 (忽略大小写和空白), 就认为是重复。

**OpenClaw 生产版的不同之处:**

- 支持完整的 cron 表达式 (如 `0 9 * * 1` 表示每周一早上 9 点)
- 支持 `every` 语法 (如 `every 30m`, `every 2h`)
- 多 Agent 独立心跳配置, 每个 Agent 可以有不同的 HEARTBEAT.md 和间隔
- 系统事件触发: 异步命令完成、外部 webhook 等可以触发特殊心跳
- 心跳消息可以路由到不同渠道 (Telegram、Discord、Slack 等)
- Lane-based 互斥: 不只是一把锁, 而是基于 CommandLane 的队列深度判断
- HEARTBEAT_OK 处理更复杂: 支持 HTML/Markdown 包裹, 支持 ackMaxChars (附带少量文字的 OK 也视为静默)
- 持久化已执行状态, 重启后不会重复执行

## 试一试

```sh
cd mini-claw
python agents/s08_heartbeat.py
```

首次运行会自动创建 `workspace/SOUL.md` 和 `workspace/HEARTBEAT.md` 示例文件。

默认心跳间隔是 60 秒 (生产环境是 30 分钟), 可通过环境变量调整:

```sh
HEARTBEAT_INTERVAL=30 python agents/s08_heartbeat.py
```

可以尝试:

1. 启动后等待 60 秒, 观察心跳是否触发。如果没有需要汇报的事, 控制台不会有输出 (HEARTBEAT_OK 被静默)。

2. 先告诉 Agent 一些待办事项, 然后等心跳触发:

```
You > Remember: I need to submit my report by 5pm today.
```

3. 查看心跳状态:

```
You > /heartbeat
```

4. 手动触发一次心跳 (不等间隔):

```
You > /trigger
```

5. 编辑 `workspace/HEARTBEAT.md`, 添加更具体的检查指令, 观察心跳行为的变化。

6. 在心跳运行期间发送消息, 观察互斥锁的效果: 心跳完成后你的消息才会被处理, 或者心跳被跳过让你的消息优先。

7. 调整活跃时段, 观察非活跃时段心跳被跳过:

```sh
HEARTBEAT_ACTIVE_START=0 HEARTBEAT_ACTIVE_END=1 python agents/s08_heartbeat.py
# 只在 0:00 - 1:00 运行心跳, 其他时间静默
```
