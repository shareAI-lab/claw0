# s04: Multi-Channel (多通道架构)

> "Same brain, many mouths" -- 同一个 agent, 同时服务 CLI、Telegram、Discord、WhatsApp...

## 问题

前三节的 agent 只能通过 CLI 交互. 但真实世界中, 用户可能从 Telegram 发消息, 也可能从 Discord, 也可能从 WhatsApp. 如果每个通道都要写一个独立的 agent, 代码会大量重复, 且行为不一致.

OpenClaw 和 Claude Code 最大的区别就在这里:

- **Claude Code** = 单通道 CLI 工具
- **OpenClaw** = 多通道 AI 网关

OpenClaw 的 gateway 同时连接 Telegram, Discord, Slack, WhatsApp, Signal, iMessage 等. 每个通道都是一个 "Channel plugin", 实现统一的收发接口.

这意味着:
1. Agent 逻辑只写一次, 但可以通过任意通道交互
2. 每个通道有自己的消息格式、长度限制、媒体能力
3. 消息在进入 agent 前被标准化为统一的 InboundMessage
4. 回复在发出前根据通道限制自动分块

## 解决方案

```
  Telegram     Discord      CLI       File (webhook sim)
     |            |           |            |
     v            v           v            v
  +------------ ChannelRegistry -----------+
  |          receive() / send()            |
  +----------------------------------------+
                    |
              InboundMessage
                    |
           Agent Loop + SessionStore
                    |
              response text
                    |
  +----------------------------------------+
  |     channel.send(chunked text)         |
  +------------ ChannelRegistry -----------+
     |            |           |            |
     v            v           v            v
  Telegram     Discord      CLI       File
```

核心架构由四层组成:

1. **Channel (通道抽象基类)** -- 定义收发接口
2. **InboundMessage (标准化消息)** -- 所有通道的消息统一为同一种格式
3. **ChannelRegistry (通道注册表)** -- 管理所有通道, 提供统一的轮询入口
4. **Gateway (消息分发)** -- 轮询所有通道 -> 构建 session key -> 调用 agent -> 发回回复

## 工作原理

### InboundMessage: 标准化的入站消息

不同通道的消息格式各异 (Telegram 有 chat_id, Discord 有 guild_id, CLI 只有文本), 但进入 agent 前统一为:

```python
@dataclass
class InboundMessage:
    channel: str           # 来源通道 ID (如 "cli", "telegram")
    sender: str            # 发送者标识 (如用户名, 用户 ID)
    text: str              # 消息文本
    media_urls: list[str] = field(default_factory=list)   # 附件 URL 列表
    thread_id: str | None = None                          # 线程/话题 ID
    timestamp: float = field(default_factory=time.time)   # Unix 时间戳
```

这种标准化是 OpenClaw 能用同一个 agent 服务多个通道的关键. 在真正的 OpenClaw 中, InboundMessage 还包含 quoted_text (引用回复)、reactions、metadata (通道特定元数据) 等字段.

### Channel: 通道抽象基类

每个通道插件必须实现这个接口:

```python
class Channel(ABC):
    @property
    @abstractmethod
    def id(self) -> str:
        """通道唯一标识, 如 'telegram', 'discord', 'cli'."""
        ...

    @property
    @abstractmethod
    def label(self) -> str:
        """人类可读的通道名称."""
        ...

    @property
    @abstractmethod
    def max_text_length(self) -> int:
        """单条消息的最大字符数. 超过此限制需要分块."""
        ...

    @abstractmethod
    def receive(self) -> InboundMessage | None:
        """非阻塞轮询: 有新消息返回 InboundMessage, 否则返回 None."""
        ...

    @abstractmethod
    def send(self, text: str, media: list | None = None) -> None:
        """发送消息到通道. 长文本会被自动分块."""
        ...
```

关键设计: **receive() 必须是非阻塞的**. 因为 gateway 需要在一个循环里轮询所有通道, 阻塞任何一个通道会饿死其他通道.

### chunk_text(): 智能文本分块

Channel 基类提供了默认的分块策略. 当回复文本超过通道的 max_text_length 时, 按优先级尝试在不同边界拆分:

```python
def chunk_text(self, text: str) -> list[str]:
    max_len = self.max_text_length
    if len(text) <= max_len:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        cut = remaining[:max_len]
        split_pos = cut.rfind("\n\n")        # 1. 优先: 段落边界

        if split_pos < max_len // 4:
            split_pos = cut.rfind("\n")       # 2. 其次: 换行处

        if split_pos < max_len // 4:
            split_pos = cut.rfind(" ")        # 3. 再次: 空格处

        if split_pos < max_len // 4:
            split_pos = max_len               # 4. 最后: 硬切

        chunks.append(remaining[:split_pos].rstrip())
        remaining = remaining[split_pos:].lstrip()

    return [c for c in chunks if c]
```

分块优先级: 段落 > 换行 > 空格 > 硬切. 这保证了拆分后的每段文本尽可能保持语义完整.

### 3 个通道实现

**CLIChannel** -- 最简单的通道, 从 stdin 读取, 向 stdout 输出:

```python
class CLIChannel(Channel):
    @property
    def id(self) -> str:
        return "cli"

    @property
    def max_text_length(self) -> int:
        return 8000

    def enqueue(self, text: str, sender: str = "user") -> None:
        """从外部将消息放入队列, 供 gateway 循环使用."""
        self._pending = InboundMessage(channel=self.id, sender=sender, text=text)

    def receive(self) -> InboundMessage | None:
        msg = self._pending
        self._pending = None
        return msg

    def send(self, text: str, media: list | None = None) -> None:
        chunks = self.chunk_text(text)
        for i, chunk in enumerate(chunks):
            if i > 0:
                print("---")
            print(chunk)
```

**FileChannel** -- 监视文件变化, 模拟 webhook 行为:

```python
class FileChannel(Channel):
    @property
    def id(self) -> str:
        return "file"

    @property
    def max_text_length(self) -> int:
        return 4000

    def receive(self) -> InboundMessage | None:
        current_size = self._inbox.stat().st_size
        if current_size <= self._read_offset:
            return None
        # 读取新增部分
        with open(self._inbox, "r", encoding="utf-8") as f:
            f.seek(self._read_offset)
            new_content = f.read()
        self._read_offset = current_size
        # 取最后一条非空行作为消息
        lines = [l.strip() for l in new_content.strip().splitlines() if l.strip()]
        if not lines:
            return None
        return InboundMessage(channel=self.id, sender="file_user", text=lines[-1])
```

FileChannel 通过记录 `_read_offset` (已读字节数) 来检测新内容, 模拟了真实 webhook 通道的 "增量读取" 模式.

**MockTelegramChannel** -- 模拟 Telegram Bot API 的行为特征:

```python
class MockTelegramChannel(Channel):
    @property
    def id(self) -> str:
        return "telegram"

    @property
    def max_text_length(self) -> int:
        return 4096  # Telegram Bot API 实际限制

    def send(self, text: str, media: list | None = None) -> None:
        chunks = self.chunk_text(text)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with open(self._outbox, "a", encoding="utf-8") as f:
            for chunk in chunks:
                envelope = {
                    "ok": True,
                    "result": {
                        "message_id": self._update_id,
                        "date": now,
                        "text": chunk,
                    },
                }
                f.write(json.dumps(envelope, ensure_ascii=False) + "\n")
```

MockTelegramChannel 模拟了 Telegram 的 4096 字符限制和消息信封格式, 让你不需要真正的 Bot Token 就能理解通道如何工作.

### ChannelRegistry: 通道注册表

管理所有已注册的通道插件, 提供统一的轮询入口:

```python
class ChannelRegistry:
    def __init__(self):
        self._channels: dict[str, Channel] = {}

    def register(self, channel: Channel) -> None:
        if channel.id in self._channels:
            raise ValueError(f"Channel already registered: {channel.id}")
        self._channels[channel.id] = channel

    def poll_all(self) -> list[InboundMessage]:
        """轮询所有通道, 收集新消息."""
        messages = []
        for channel in self._channels.values():
            msg = channel.receive()
            if msg is not None:
                messages.append(msg)
        return messages
```

`poll_all()` 是 gateway 主循环的核心: 每次迭代调用它收集所有通道的新消息, 然后逐个分发给 agent 处理.

### Gateway: 多通道消息分发

`gateway_poll_once()` 执行一次全通道轮询:

```python
def gateway_poll_once(
    registry: ChannelRegistry,
    session_store: SessionStore,
    client: Anthropic,
) -> int:
    messages = registry.poll_all()
    processed = 0

    for msg in messages:
        channel = registry.get(msg.channel)
        if not channel:
            continue

        # 根据通道和发送者构建 session key
        session_key = build_session_key(msg.channel, msg.sender)

        try:
            response = agent_loop(msg.text, session_key, session_store, client)
            channel.send(response)  # 通过来源通道发回回复
            processed += 1
        except Exception as exc:
            channel.send(f"[Error] {exc}")

    return processed
```

### Session Key 与通道的关系

session key 现在包含了通道信息:

```python
def build_session_key(channel_id: str, sender: str, agent_id: str = "main") -> str:
    safe_sender = sender.replace(":", "_").replace("/", "_")
    return f"{agent_id}:{channel_id}:{safe_sender}"
```

这意味着:
- 同一个用户在 CLI 和 Telegram 有不同的会话 (`main:cli:user` vs `main:telegram:tg_user_1`)
- 不同用户在同一个通道也有不同的会话
- 每个会话独立持久化, 互不干扰

## 核心代码

多通道架构的核心是 **Channel ABC + ChannelRegistry.poll_all() + gateway_poll_once()** 的组合 (来自 `agents/s04_multi_channel.py`):

```python
# 通道接口: 所有通道必须实现 receive/send
class Channel(ABC):
    @abstractmethod
    def receive(self) -> InboundMessage | None: ...
    @abstractmethod
    def send(self, text: str, media: list | None = None) -> None: ...

# 注册表: 管理所有通道, 提供统一轮询
class ChannelRegistry:
    def poll_all(self) -> list[InboundMessage]:
        messages = []
        for channel in self._channels.values():
            msg = channel.receive()
            if msg is not None:
                messages.append(msg)
        return messages

# Gateway: 轮询 -> 路由 -> agent -> 回复
def gateway_poll_once(registry, session_store, client) -> int:
    for msg in registry.poll_all():
        channel = registry.get(msg.channel)
        session_key = build_session_key(msg.channel, msg.sender)
        response = agent_loop(msg.text, session_key, session_store, client)
        channel.send(response)
```

这个三层结构可以概括为: **Channel 负责收发, Registry 负责管理, Gateway 负责路由**.

## 和上一节的区别

| 组件 | s03 | s04 |
|------|-----|-----|
| 消息来源 | 只有 stdin | 多通道 (CLI + File + MockTelegram) |
| 消息格式 | 原始字符串 | InboundMessage dataclass |
| Session key | 硬编码 `main:cli:user` | 根据 channel + sender 动态构建 |
| 回复发送 | 直接 print | 通过 channel.send() (自动分块) |
| 轮询模式 | 无 | ChannelRegistry.poll_all() |
| 新增抽象 | -- | Channel ABC, InboundMessage, ChannelRegistry |
| 核心函数 | agent_loop() | agent_loop() + gateway_poll_once() |

核心改动: 引入了 Channel 抽象层, 将 "消息从哪来" 和 "消息怎么处理" 解耦. agent_loop 本身完全不变, 只是被 gateway 在更高层调度.

## 设计解析

### 为什么 receive() 必须非阻塞?

如果 Telegram 通道的 receive() 阻塞等待消息, 那在等待期间 Discord 通道的消息就无法被处理. 非阻塞设计让 gateway 可以快速轮询所有通道, 不会因为某个通道没有消息而卡住.

真正的 OpenClaw 使用事件循环 (event loop) 和异步 I/O 来进一步提高效率, 但核心思想一样: 不要让任何一个通道阻塞其他通道.

### 为什么 chunk_text() 在 Channel 基类中?

不同通道有不同的消息长度限制:
- Telegram: 4096 字符
- Discord: 2000 字符
- SMS: 160 字符
- CLI: 无硬性限制

基类提供通用的分块算法, 各通道可以覆盖它来实现通道特定的分块策略 (比如 Telegram 需要避免在 Markdown 标记中间断开).

### 为什么用文件模拟 webhook?

FileChannel 用文件的 "追加-读取" 来模拟真实 webhook 的 "推送-消费" 模式. 你可以在另一个终端用 `echo "hello" >> workspace/.channels/file_inbox.txt` 来模拟外部系统推送消息, 而不需要搭建 HTTP 服务器或注册 Bot.

### OpenClaw 生产版本做了什么不同?

- **15+ 真实通道**: Telegram, Discord, Slack, Signal, WhatsApp, iMessage, Microsoft Teams, Matrix, IRC, Google Chat, Zalo 等
- **插件热加载**: 通道作为 npm 包, 可以运行时安装和卸载
- **WebSocket / Long-polling / Webhook**: 根据通道特性选择连接方式
- **速率限制**: 每个通道有独立的发送速率限制, 避免被平台封禁
- **媒体管道**: 统一的媒体上传/下载/转码/压缩流水线
- **消息路由**: 基于规则的消息路由 (如某些用户的消息转发到特定 agent)
- **并发处理**: 多条消息可以并行处理, 不需要等前一条处理完
- **优雅关机**: 收到 SIGTERM 时完成正在处理的消息, 然后关闭所有通道连接

## 试一试

```sh
cd mini-claw
python agents/s04_multi_channel.py
```

可以尝试的操作:

1. 直接输入文本和 agent 对话 -- 这走的是 CLIChannel
2. `/channels` 查看已注册的通道和它们的参数
3. 打开另一个终端, 向 File 通道发消息:
   ```sh
   echo "What time is it?" >> workspace/.channels/file_inbox.txt
   ```
   然后在 agent 终端输入 `/poll`, 观察消息被 FileChannel 接收并处理
4. `/send telegram Hello from test` 向 MockTelegram 通道注入消息, 然后 `/poll` 处理
5. 查看 `workspace/.channels/file_outbox.txt` 和 `workspace/.channels/telegram_outbox.txt` 中的回复
6. `/sessions` 查看不同通道产生的独立会话 (注意 session key 中的 channel 部分不同)
7. 观察 MockTelegram 的 outbox 文件, 回复被包装在 Telegram 消息信封中
