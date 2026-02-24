# s10: Delivery Queue & Reliable Messaging (可靠消息投递)

> "先写后发, 重试到底" -- 让每条消息都不丢失。

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

- **What we build**: 磁盘持久化的消息投递队列, 保证 at-least-once delivery
- **Core mechanism**: write-ahead (先写后发) + 指数退避重试 [5s, 25s, 2m, 10m, 10m]
- **Design pattern**: 文件系统即队列, 每条消息一个 JSON 文件, ack 删除, fail 更新

## The Problem

1. **发送失败即丢失**: Telegram/Discord API 可能因网络波动、限流、服务故障暂时不可用。之前的实现 `try/except` 打印错误, 消息就没了。
2. **进程崩溃不可恢复**: Agent 生成了回复, 还没发送进程就崩了。重启后没有任何记录, 消息永久丢失。
3. **无重试机制**: 一次发送失败后没有第二次机会。用户可能永远收不到那条关键的定时提醒。

## How It Works

### 1. QueuedDelivery -- 投递条目

每条待投递消息的完整生命周期信息:

```python
@dataclass
class QueuedDelivery:
    id: str                      # UUID, 同时作为文件名
    channel: str                 # 目标渠道: "telegram", "discord"
    to: str                      # 接收者 ID
    text: str                    # 消息内容
    retry_count: int = 0         # 已重试次数
    last_error: str | None = None
    enqueued_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0   # 下次可重试的时间

    def to_dict(self) -> dict: ...

    @staticmethod
    def from_dict(data: dict) -> "QueuedDelivery": ...
```

**每个字段都服务于投递生命周期: retry_count 决定是否继续重试, next_retry_at 控制退避间隔, last_error 用于排查。**

### 2. DeliveryQueue -- 磁盘队列的四个核心操作

```python
class DeliveryQueue:
    def __init__(self, queue_dir: Path):
        self.queue_dir = queue_dir
        self.failed_dir = queue_dir / "failed"
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        self.failed_dir.mkdir(parents=True, exist_ok=True)
```

**enqueue -- 原子写入磁盘**:

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

**ack -- 投递成功, 删除文件**:

```python
def ack(self, delivery_id: str) -> None:
    try:
        self._entry_path(delivery_id).unlink()
    except FileNotFoundError:
        pass
```

**fail -- 投递失败, 更新退避或移入 failed/**:

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

**move_to_failed -- 超过最大重试次数**:

```python
def move_to_failed(self, delivery_id: str) -> None:
    os.replace(
        str(self._entry_path(delivery_id)),
        str(self.failed_dir / f"{delivery_id}.json"),
    )
```

**文件系统即队列: enqueue 创建文件, ack 删除文件, fail 更新文件, move_to_failed 移动文件。`ls delivery-queue/` 就能看到所有待投递消息。**

### 3. 指数退避 -- backoff 序列

失败后不立即重试, 等待递增的时间:

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

重试 1-5 次的退避序列:

| retry_count | idx | backoff |
|-------------|-----|---------|
| 1 | 0 | 5s |
| 2 | 1 | 25s |
| 3 | 2 | 2m |
| 4 | 3 | 10m |
| 5 | 3 (capped) | 10m |

retry_count > 5 时调用 `move_to_failed`, 不再重试。

**5s 应对网络抖动, 25s 应对短暂限流, 2m 应对 API 重启, 10m 是上限 -- 再久就不值得自动等了。**

### 4. DeliveryRunner -- 后台投递线程

每秒扫描队列, 对到期的条目尝试投递:

```python
class DeliveryRunner:
    def __init__(self, queue: DeliveryQueue, deliver_fn):
        self.queue = queue
        self.deliver_fn = deliver_fn  # (channel, to, text) -> None, 失败抛异常

    def _process_pending(self) -> None:
        now = time.time()
        for entry in self.queue.load_pending():
            if entry.next_retry_at > now:
                continue  # 退避时间未到, 跳过
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

**逻辑简单但可靠: 扫描 -> 检查退避 -> 尝试发送 -> ack 或 fail。每秒循环。**

### 5. 启动恢复 -- recovery_scan

进程重启时扫描队列目录, 报告并恢复未完成的投递:

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

这是 write-ahead 模式的核心价值: 消息已经在磁盘上, 进程崩溃后重启, `load_pending()` 扫描目录自动恢复。

**不需要额外的恢复逻辑 -- 文件就是状态, 扫描文件就是恢复。**

### 6. Agent 回复走投递队列

在 s10 中, Agent 的回复不再直接打印, 而是 enqueue 到投递队列:

```python
# 之前 (s09): 直接输出
print_assistant(text)

# 现在 (s10): 先入队, 由 DeliveryRunner 投递
did = delivery_queue.enqueue(mock_channel.name, "user", text)
print_info(f"  enqueued -> delivery queue (id={did[:8]}..)")
```

心跳输出同样走投递队列:

```python
for msg in heartbeat.drain_output():
    did = delivery_queue.enqueue(mock_channel.name, "user", msg)
    print_heartbeat(f"enqueued heartbeat message (id={did[:8]}..)")
```

**所有出站消息 (用户回复、心跳通知) 都经过投递队列, 保证一致的可靠性。**

## What Changed from s09

| Component | s09 | s10 |
|-----------|-----|-----|
| 消息投递 | 直接发送, 无保障 | write-ahead queue, at-least-once |
| 失败处理 | try/except 打印错误 | 退避重试 [5s, 25s, 2m, 10m, 10m] |
| 持久化 | 仅 cron 任务持久化 | 每条待发消息持久化到磁盘 |
| 重试上限 | cron 连续 5 次失败自动禁用 | MAX_RETRIES=5 后移入 failed/ |
| 崩溃恢复 | 重启后加载 cron 任务 | 重启后扫描队列目录恢复投递 |
| 命令 | /cron, /cron-log | /queue, /failed, /retry, /simulate-failure |

**Key shift**: 从 "尽力发送" 变成 "保证送达"。生产系统假设一切都会出错 (网络断开、API 限流、进程崩溃), 并为每种故障设计恢复路径。

## Design Decisions

**Why file-system queue instead of SQLite?**

文件系统队列的优势在教学场景中非常明显: `ls delivery-queue/` 就能看到所有待投递消息, 用文本编辑器直接查看和修改 JSON 内容, `os.replace` 提供原子性, 零外部依赖。当队列深度超过几千条或需要复杂查询时, SQLite 是更好的选择。

**Why this specific backoff sequence [5s, 25s, 2m, 10m]?**

经验拟合常见故障恢复时间: 5s 应对 DNS 超时等瞬间故障, 25s 应对 Telegram 等平台的短暂限流窗口 (通常 10-30s), 2m 应对 API 部署更新和服务重启, 10m 是上限 -- 超过 10 分钟仍失败的问题很可能需要人工介入。不使用纯指数退避 (1s, 2s, 4s...) 是因为前几次太短 (可能还在限流), 后几次太长 (用户等不起)。

**Why MAX_RETRIES=5 and move_to_failed instead of infinite retry?**

无限重试会掩盖配置错误: 错误的 API token、不存在的接收者, 怎么重试都不会成功。5 次覆盖了退避序列的每个阶段, 总等待约 13 分钟。之后移入 failed/ 目录, 管理员可以通过 `/retry` 手动恢复。

**In production OpenClaw:** 投递队列在 `src/infra/outbound/delivery-queue.ts` 实现。生产版使用 SQLite 后端支持高吞吐, 退避策略增加了 jitter (随机扰动) 避免多条消息在同一时刻重试造成雷鸣群效应 (thundering herd)。支持优先级队列 (用户直接回复优先于 cron 结果), 渠道级限流控制, 以及通过 web UI 查看和手动重试 failed 条目。

## Try It

```sh
cd claw0
python agents/s10_delivery.py
```

正常对话, 观察消息先入队再投递:

```
You > Hello
  enqueued -> delivery queue (id=a1b2c3d4..)
  [delivery] delivered a1b2c3d4.. to telegram:user

[telegram -> user] Hello! How can I help you today?
```

开启故障模拟, 观察退避重试:

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

查看队列和 failed 状态:

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
