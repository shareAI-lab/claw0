# s10: Delivery Queue (可靠消息投递)

> 先写后发, 失败重试 -- 让每条消息都不丢失。

## 问题

前面所有章节的消息发送都是 "fire and forget" -- 调用发送函数, 成功就好, 失败就丢:

1. **发送失败即丢失**: Telegram/Discord API 可能因为网络波动、限流 (rate limit)、API 故障而暂时不可用。当前实现直接 `try/except` 打印错误, 消息就没了。
2. **进程崩溃不可恢复**: Agent 生成了一条重要回复, 还没来得及发送进程就崩了。重启后没有任何记录, 这条消息永久丢失。
3. **无重试机制**: 一次发送失败后没有第二次机会。用户可能永远收不到那条关键提醒或定时任务的结果。

一个可靠的消息系统需要保证: **消息一旦生成, 最终一定会投递成功 (at-least-once delivery)**。

## 解决方案

```
  Agent Response
       |
       v
  enqueue() --> workspace/delivery-queue/{uuid}.json
       |         (atomic: .tmp -> os.replace)
       v
  attempt_delivery()
       |
  success? --yes--> ack() (delete file)
       |
      no
       |
  fail() -> retry_count++, compute backoff
       |
  retry > max? --yes--> move to failed/
       |
      no
       |
  wait: [5s, 25s, 2m, 10m, 10m]
       |
  attempt again...
```

核心思路: 消息先持久化到磁盘 (enqueue), 再尝试发送。成功则删除文件 (ack), 失败则计算退避时间后重试。超过最大重试次数后移入 `failed/` 目录供人工排查。进程重启时扫描 `delivery-queue/` 目录, 恢复所有未完成的投递。

## 工作原理

### 1. QueuedDelivery -- 投递条目数据结构

每条待投递的消息用一个 dataclass 表示:

```python
@dataclass
class QueuedDelivery:
    id: str                      # UUID, 同时作为文件名
    channel: str                 # 目标渠道: "telegram", "discord", ...
    recipient: str               # 接收者 ID
    content: str                 # 消息内容
    created_at: float            # 入队时间
    retry_count: int = 0         # 已重试次数
    last_attempt_at: float = 0.0 # 上次尝试时间
    last_error: str = ""         # 上次失败原因
    max_retries: int = 5         # 最大重试次数

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "channel": self.channel,
            "recipient": self.recipient,
            "content": self.content,
            "created_at": self.created_at,
            "retry_count": self.retry_count,
            "last_attempt_at": self.last_attempt_at,
            "last_error": self.last_error,
            "max_retries": self.max_retries,
        }

    @staticmethod
    def from_dict(d: dict) -> "QueuedDelivery":
        return QueuedDelivery(
            id=d["id"], channel=d["channel"], recipient=d["recipient"],
            content=d["content"], created_at=d["created_at"],
            retry_count=d.get("retry_count", 0),
            last_attempt_at=d.get("last_attempt_at", 0.0),
            last_error=d.get("last_error", ""),
            max_retries=d.get("max_retries", 5),
        )
```

每个字段都服务于投递的生命周期: `retry_count` 决定是否继续重试, `last_attempt_at` 配合退避计算决定何时重试, `last_error` 用于排查失败原因。

### 2. DeliveryQueue -- 磁盘队列

队列的核心操作: enqueue (入队), ack (确认成功), fail (记录失败), move_to_failed (放弃重试):

```python
class DeliveryQueue:
    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.failed_dir = queue_dir / "failed"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, channel: str, recipient: str, content: str) -> QueuedDelivery:
        delivery = QueuedDelivery(
            id=str(uuid.uuid4()),
            channel=channel,
            recipient=recipient,
            content=content,
            created_at=time.time(),
        )
        self._write_atomic(delivery)
        return delivery

    def ack(self, delivery_id: str) -> None:
        """投递成功, 删除文件。"""
        path = self.queue_dir / f"{delivery_id}.json"
        if path.exists():
            path.unlink()

    def fail(self, delivery: QueuedDelivery, error: str) -> None:
        """投递失败, 更新状态。"""
        delivery.retry_count += 1
        delivery.last_attempt_at = time.time()
        delivery.last_error = error
        if delivery.retry_count >= delivery.max_retries:
            self._move_to_failed(delivery)
        else:
            self._write_atomic(delivery)

    def _move_to_failed(self, delivery: QueuedDelivery) -> None:
        src = self.queue_dir / f"{delivery.id}.json"
        dst = self.failed_dir / f"{delivery.id}.json"
        content = json.dumps(delivery.to_dict(), indent=2, ensure_ascii=False)
        dst.write_text(content, encoding="utf-8")
        if src.exists():
            src.unlink()

    def load_pending(self) -> list[QueuedDelivery]:
        """加载所有待投递条目。"""
        pending = []
        for path in self.queue_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                pending.append(QueuedDelivery.from_dict(data))
            except (json.JSONDecodeError, KeyError):
                continue  # 跳过损坏的文件
        pending.sort(key=lambda d: d.created_at)
        return pending
```

文件系统即队列: 每条消息是一个 JSON 文件, 文件名是 UUID。`enqueue` 创建文件, `ack` 删除文件, `fail` 更新文件内容。`load_pending` 扫描目录即可恢复所有未完成的投递。

### 3. 原子写入 -- tmp + os.replace 模式

与 s09 的 CronStore 相同, 使用原子写入避免数据损坏:

```python
def _write_atomic(self, delivery: QueuedDelivery) -> None:
    path = self.queue_dir / f"{delivery.id}.json"
    tmp_path = path.with_suffix(".tmp")
    content = json.dumps(delivery.to_dict(), indent=2, ensure_ascii=False)
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)
```

为什么不直接 `write_text`? 因为 `write_text` 在写入过程中如果进程崩溃, 可能产生半写文件。`os.replace` 是原子操作: 文件要么是旧版本, 要么是新版本, 没有中间状态。

### 4. 指数退避 -- BACKOFF_MS 序列

失败后不应该立即重试 (可能还在限流), 而是等待递增的时间:

```python
BACKOFF_MS = [5000, 25000, 120000, 600000]

def compute_backoff_ms(retry_count: int) -> int:
    """根据重试次数计算退避时间 (毫秒)。

    序列: 5s -> 25s -> 2m -> 10m -> 10m (capped)
    """
    if retry_count <= 0:
        return BACKOFF_MS[0]
    idx = min(retry_count - 1, len(BACKOFF_MS) - 1)
    return BACKOFF_MS[idx]

def is_ready_for_retry(delivery: QueuedDelivery) -> bool:
    """检查是否已经等够退避时间。"""
    if delivery.retry_count == 0:
        return True
    backoff = compute_backoff_ms(delivery.retry_count)
    elapsed = (time.time() - delivery.last_attempt_at) * 1000
    return elapsed >= backoff
```

退避序列设计: 5s 足够应对瞬间网络抖动, 25s 覆盖短暂限流, 2m 应对 API 重启, 10m 是上限 (不值得等更久, 问题可能需要人工介入)。最后一个值 10m 会被重复使用, 直到达到最大重试次数。

### 5. DeliveryRunner -- 后台投递线程

后台线程负责取出待投递的条目, 尝试发送:

```python
class DeliveryRunner:
    def __init__(self, queue: DeliveryQueue, send_fn):
        self.queue = queue
        self.send_fn = send_fn  # (channel, recipient, content) -> None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="delivery-runner",
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            pending = self.queue.load_pending()
            for delivery in pending:
                if self._stop_event.is_set():
                    break
                if not is_ready_for_retry(delivery):
                    continue
                self._attempt(delivery)
            self._stop_event.wait(2.0)

    def _attempt(self, delivery: QueuedDelivery) -> None:
        try:
            self.send_fn(delivery.channel, delivery.recipient, delivery.content)
            self.queue.ack(delivery.id)
        except Exception as exc:
            self.queue.fail(delivery, str(exc))
```

`_loop` 每 2 秒扫描一次待投递列表。对每个条目检查退避时间是否已过, 然后尝试发送。成功调用 `ack`, 失败调用 `fail`。逻辑简单但可靠。

### 6. 启动恢复 -- recovery_scan()

进程重启时, 队列目录中可能有上次未完成的投递。启动时扫描并输出状态:

```python
def recovery_scan(queue: DeliveryQueue) -> dict:
    """启动时扫描队列目录, 报告待恢复的条目。"""
    pending = queue.load_pending()
    failed_dir = queue.failed_dir
    failed_count = len(list(failed_dir.glob("*.json")))

    stats = {
        "pending": len(pending),
        "failed": failed_count,
        "oldest_pending": None,
    }

    if pending:
        oldest = min(pending, key=lambda d: d.created_at)
        age_seconds = time.time() - oldest.created_at
        stats["oldest_pending"] = {
            "id": oldest.id,
            "channel": oldest.channel,
            "age_seconds": int(age_seconds),
            "retry_count": oldest.retry_count,
        }

    return stats
```

启动时的输出示例:

```
Delivery Queue Recovery:
  Pending: 3 items (oldest: 45s ago, 2 retries)
  Failed:  1 item (in failed/ directory)
  Resuming delivery...
```

这让运维人员一眼看到系统重启后有多少消息需要恢复。

## 核心代码

投递系统最核心的两个函数 -- 入队 (持久化保证) 和退避计算 (重试策略):

```python
def enqueue(self, channel: str, recipient: str, content: str) -> QueuedDelivery:
    """将消息入队: 先写磁盘, 再返回。

    这是 at-least-once 投递的关键: 消息写入磁盘后,
    即使进程崩溃也能在重启后恢复并继续投递。
    """
    delivery = QueuedDelivery(
        id=str(uuid.uuid4()),
        channel=channel,
        recipient=recipient,
        content=content,
        created_at=time.time(),
    )
    # 原子写入: 确保文件完整性
    path = self.queue_dir / f"{delivery.id}.json"
    tmp_path = path.with_suffix(".tmp")
    content_json = json.dumps(delivery.to_dict(), indent=2, ensure_ascii=False)
    tmp_path.write_text(content_json, encoding="utf-8")
    os.replace(tmp_path, path)
    return delivery


BACKOFF_MS = [5000, 25000, 120000, 600000]

def compute_backoff_ms(retry_count: int) -> int:
    """退避序列: 5s -> 25s -> 2m -> 10m -> 10m (capped)

    递增但有上限, 平衡 "快速恢复" 和 "不浪费资源" 两个目标。
    """
    if retry_count <= 0:
        return BACKOFF_MS[0]
    idx = min(retry_count - 1, len(BACKOFF_MS) - 1)
    return BACKOFF_MS[idx]
```

## 和上一节的区别

| 组件 | s09 | s10 |
|------|-----|-----|
| 消息投递 | 直接发送, 无保障 | 先入队再投递, at-least-once |
| 失败处理 | try/except 打印错误 | 退避重试 + 最终移入 failed/ |
| 持久化 | 仅 cron 任务持久化 | 每条待发消息都持久化到磁盘 |
| 重试机制 | cron 任务自动禁用 | 指数退避: 5s/25s/2m/10m |
| 恢复能力 | 重启后加载 cron 任务 | 重启后扫描队列目录, 恢复投递 |
| 命令 | /cron, /cron-log | /queue, /failed, /retry |

关键转变: 从 "尽力发送" 变成 "保证送达"。这是生产系统和原型的根本区别 -- 生产系统假设一切都会出错 (网络断开、API 限流、进程崩溃), 并为每种故障模式设计恢复路径。

## 设计解析

**为什么用文件系统而不是 SQLite 队列?**

文件系统队列的优势在教学场景中非常明显:

- **可观察**: `ls delivery-queue/` 就能看到所有待投递消息, 不需要 SQL 客户端。
- **可调试**: 直接用文本编辑器打开 JSON 文件查看和修改消息内容、重试次数。
- **原子性免费**: `os.replace` 提供了文件级别的原子性, 不需要事务。
- **零依赖**: 不需要安装 SQLite 或其他数据库。

在生产环境中, 当队列深度超过几千条或需要复杂查询 (按时间范围、按渠道聚合) 时, SQLite 是更好的选择。但对于教学和中小规模场景, 文件系统队列足够可靠且更容易理解。

**为什么退避是 [5s, 25s, 2m, 10m] 这个序列?**

这个序列是对常见故障恢复时间的经验拟合:

- **5s**: 网络抖动、DNS 偶发超时。大多数瞬间故障在 5 秒内恢复。
- **25s**: Telegram/Discord API 的短暂限流窗口通常在 10-30 秒。
- **2m**: API 部署更新、服务器重启通常在 1-3 分钟内完成。
- **10m**: 如果 10 分钟后还失败, 问题可能比较严重。继续以 10 分钟间隔重试直到达到最大次数, 避免无限等待。

不使用纯指数退避 (1s, 2s, 4s, 8s, 16s...) 是因为前几次退避太短 (可能还在限流), 后几次太长 (用户等不起)。手工选择的序列在实践中效果更好。

**OpenClaw 生产版的不同之处:**

- `src/delivery/delivery-queue.ts` 使用 SQLite 作为持久化后端, 支持高吞吐场景
- 退避策略支持 jitter (随机扰动), 避免多条消息在同一时刻重试造成 "雷鸣群" (thundering herd)
- 支持优先级队列: 用户直接回复的消息比 cron 任务的结果优先投递
- 支持批量投递: 多条消息合并为一次 API 调用, 减少限流风险
- 支持渠道级别的限流控制: 每个渠道独立的发送速率限制
- failed 条目支持通过 web UI 查看和手动重试, 不需要 SSH 到服务器
- 投递状态变化发送到 event bus, 可被监控系统订阅

## 试一试

```sh
cd mini-claw
python agents/s10_delivery.py
```

启动后, 系统会扫描 `workspace/delivery-queue/` 目录, 恢复未完成的投递。

可以尝试:

1. 正常对话, 观察消息先入队再投递的过程:

```
You > 你好, 今天天气怎么样?
```

控制台会显示:

```
[delivery] enqueued: a1b2c3d4 -> terminal
[delivery] sent: a1b2c3d4 (ack)
```

2. 查看当前队列状态:

```
You > /queue
```

正常情况下队列为空 (所有消息都已成功投递)。

3. 模拟发送失败, 观察重试行为:

```
You > /simulate-failure 3
```

接下来的 3 次发送会模拟失败。然后发一条消息:

```
You > 这条消息会重试 3 次。
```

控制台会显示退避和重试过程:

```
[delivery] enqueued: e5f6g7h8 -> terminal
[delivery] FAILED: e5f6g7h8 (simulated failure), retry 1, backoff 5s
[delivery] FAILED: e5f6g7h8 (simulated failure), retry 2, backoff 25s
[delivery] FAILED: e5f6g7h8 (simulated failure), retry 3, backoff 120s
[delivery] sent: e5f6g7h8 (ack, after 3 retries)
```

4. 查看失败的消息:

```
You > /failed
```

5. 手动重试失败的消息:

```
You > /retry e5f6g7h8
```

6. 模拟进程崩溃恢复: 在有待投递消息时按 Ctrl+C 退出, 然后重新启动。观察恢复扫描输出:

```sh
python agents/s10_delivery.py
```

```
Delivery Queue Recovery:
  Pending: 1 item (oldest: 30s ago, 1 retry)
  Failed:  0 items
  Resuming delivery...
```
