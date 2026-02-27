# 第 08 节: 消息投递

> 先写磁盘, 再尝试发送. 崩溃安全.

## 架构

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

## 本节要点

- **DeliveryQueue**: 磁盘持久化的预写队列. 入队时先写磁盘, 再尝试投递.
- **原子写入**: 临时文件 + `os.fsync()` + `os.replace()` -- 崩溃时不会产生半写文件.
- **DeliveryRunner**: 后台线程, 以指数退避处理待投递条目.
- **chunk_message()**: 按平台大小限制分片文本, 尊重段落边界.
- **启动恢复扫描**: 启动时自动重试上次崩溃前遗留的待投递条目.

## 核心代码走读

### 1. DeliveryQueue.enqueue() + 原子写入

基本规则: 先写磁盘, 再尝试投递. 如果进程在入队和投递之间崩溃,
消息仍然保存在磁盘上.

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
        os.fsync(f.fileno())        # 数据已落盘

    os.replace(str(tmp_path), str(final_path))  # POSIX 上的原子操作
```

三步保证:
- 第 1 步: 写入 `.tmp.{pid}.{id}.json` (崩溃 = 孤立的临时文件, 无害)
- 第 2 步: `fsync()` -- 数据已落盘
- 第 3 步: `os.replace()` -- 原子交换 (崩溃 = 旧文件或新文件, 绝不会是半写文件)

### 2. ack() / fail() -- 重试生命周期

```python
def ack(self, delivery_id: str) -> None:
    """投递成功. 删除队列文件."""
    (self.queue_dir / f"{delivery_id}.json").unlink()

def fail(self, delivery_id: str, error: str) -> None:
    """递增 retry_count, 计算下次重试时间, 或放弃."""
    entry = self._read_entry(delivery_id)
    entry.retry_count += 1
    entry.last_error = error
    if entry.retry_count >= MAX_RETRIES:
        self.move_to_failed(delivery_id)
        return
    backoff_ms = compute_backoff_ms(entry.retry_count)
    entry.next_retry_at = time.time() + backoff_ms / 1000.0
    self._write_entry(entry)  # 将新的重试状态更新到磁盘
```

带抖动的退避, 防止雷群效应:

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

### 3. DeliveryRunner -- 后台循环

每秒扫描待投递条目. 只处理 `next_retry_at` 已到期的条目.
启动时执行恢复扫描, 处理上次崩溃遗留的条目.

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

## 试一试

```sh
python zh/s08_delivery.py

# 发送一条消息 -- 观察它被入队并投递
# You > Hello!

# 开启 50% 失败率
# You > /simulate-failure

# 再发一条消息 -- 观察带退避的重试
# You > 在失败模式下的测试消息

# 查看队列
# You > /queue
# You > /failed

# 恢复正常, 观察待投递条目被送出
# You > /simulate-failure

# 查看统计数据
# You > /stats
```

## OpenClaw 中的对应实现

| 方面           | claw0 (本文件)                   | OpenClaw 生产代码                     |
|----------------|----------------------------------|---------------------------------------|
| 队列存储       | 目录中的 JSON 文件               | 相同的每条目一个文件模式              |
| 原子写入       | tmp + fsync + os.replace         | 相同方案                              |
| 退避           | [5s, 25s, 2min, 10min] + 抖动   | 相同的调度                            |
| 消息分片       | 段落边界分割                     | 相同 + 代码围栏感知                   |
| 恢复           | 启动时扫描队列目录               | 相同的扫描 + 孤立文件清理             |

## 全系列总结

10 个章节, 构成一个 agent 网关的核心机制:

```
    Section 01: while True + stop_reason        (循环)
    Section 02: TOOLS + TOOL_HANDLERS           (执行)
    Section 03: JSONL + ContextGuard            (持久化)
    Section 04: Channel ABC + InboundMessage    (通道)
    Section 05: BindingTable + session key      (路由)
    Section 06: 8-layer prompt + hybrid search  (智能)
    Section 07: Heartbeat + Cron                (自治)
    Section 08: DeliveryQueue + backoff         (可靠投递)
    Section 09: 3-layer retry onion + profiles  (韧性)
    Section 10: Named lanes + generation track  (并发)
```

第 01 节的 agent 循环在第 10 节的核心依然清晰可辨.
AI agent 就是一个 `while True` 循环加上一张分发表,
外面包裹着持久化、路由、智能、调度、可靠性、韧性和并发控制的层层机制.
