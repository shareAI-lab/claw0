# s04: Multi-Channel (多通道架构)

> "Same brain, many mouths" -- 同一个 agent, 同时服务 CLI、Telegram、Discord.

## At a Glance

```
Telegram    File (webhook sim)    CLI
   |              |                |
   v              v                v
+------------- ChannelRegistry -----------+
|  receive(): InboundMessage | None       |
+-----------------------------------------+
                  |
           InboundMessage
                  |
   gateway_poll_once() --> build_session_key()
                  |
           agent_loop() + SessionStore
                  |
           response text
                  |
+-----------------------------------------+
|  channel.send(): chunk_text() + output  |
+------------- ChannelRegistry -----------+
   |              |                |
   v              v                v
Telegram    File outbox           stdout
```

- **What we build**: Channel 插件接口 + ChannelRegistry + gateway 轮询循环, 让同一个 agent 通过任意通道收发消息.
- **Core mechanism**: 所有通道的消息统一为 InboundMessage, receive() 非阻塞轮询, send() 自动分块.
- **Design pattern**: Channel 负责收发, Registry 负责管理, Gateway 负责路由. agent_loop 本身不变.

## The Problem

1. **每个通道一套代码 = 重复和不一致.** CLI 一个 agent, Telegram 一个 agent, Discord 又一个. 逻辑重复, 行为不统一, bug 修一处漏两处.

2. **不同通道的消息格式不同.** Telegram 有 chat_id 和 update_id, Discord 有 guild_id, CLI 只有纯文本. 没有标准化, agent 就得为每个通道写不同的解析逻辑.

3. **不同通道的消息长度限制不同.** Telegram 4096 字符, Discord 2000 字符, SMS 160 字符. 一个 2000 字的回复在 Telegram 没问题, 在 Discord 就会被截断.

## How It Works

### 1. InboundMessage: 标准化入站消息

不同通道的消息格式各异, 但进入 agent 前统一为一种结构:

```python
@dataclass
class InboundMessage:
    channel: str           # 来源通道 ID
    sender: str            # 发送者标识
    text: str              # 消息文本
    media_urls: list[str] = field(default_factory=list)
    thread_id: str | None = None
    timestamp: float = field(default_factory=time.time)
```

**标准化是多通道架构的核心. 没有它, agent 就得为每个通道写不同的处理逻辑.**

### 2. Channel: 通道抽象基类

每个通道插件必须实现这个接口:

```python
class Channel(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...          # "telegram", "cli"

    @property
    @abstractmethod
    def max_text_length(self) -> int: ...  # 4096, 8000

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...  # 非阻塞轮询

    @abstractmethod
    def send(self, text: str, media: list | None = None) -> None: ...

    def chunk_text(self, text: str) -> list[str]:
        # 默认分块: 段落 > 换行 > 空格 > 硬切
        ...
```

**receive() 必须非阻塞. gateway 在一个循环里轮询所有通道, 阻塞任何一个就饿死其他通道.**

### 3. 三个通道实现

**CLIChannel** -- 最简单, 从 stdin 读, 向 stdout 写:

```python
class CLIChannel(Channel):
    def enqueue(self, text: str, sender: str = "user") -> None:
        self._pending = InboundMessage(channel=self.id, sender=sender, text=text)

    def receive(self) -> InboundMessage | None:
        msg = self._pending
        self._pending = None
        return msg

    def send(self, text: str, media: list | None = None) -> None:
        for chunk in self.chunk_text(text):
            print(chunk)
```

**FileChannel** -- 监视文件变化, 模拟 webhook. 通过 `_read_offset` 追踪已读位置, 只处理新增内容:

```python
class FileChannel(Channel):
    def receive(self) -> InboundMessage | None:
        current_size = self._inbox.stat().st_size
        if current_size <= self._read_offset:
            return None
        with open(self._inbox, "r") as f:
            f.seek(self._read_offset)
            new_content = f.read()
        self._read_offset = current_size
        lines = [l.strip() for l in new_content.strip().splitlines() if l.strip()]
        return InboundMessage(channel=self.id, sender="file_user", text=lines[-1])
```

**MockTelegramChannel** -- 模拟 Telegram 的 4096 字符限制和消息信封格式, 回复写入 outbox 文件.

### 4. ChannelRegistry 和 gateway_poll_once

Registry 管理通道, poll_all 收集消息; gateway_poll_once 路由到 agent:

```python
class ChannelRegistry:
    def register(self, channel: Channel) -> None:
        self._channels[channel.id] = channel

    def poll_all(self) -> list[InboundMessage]:
        messages = []
        for channel in self._channels.values():
            msg = channel.receive()
            if msg is not None:
                messages.append(msg)
        return messages

def gateway_poll_once(registry, session_store, client) -> int:
    for msg in registry.poll_all():
        channel = registry.get(msg.channel)
        session_key = build_session_key(msg.channel, msg.sender)
        response = agent_loop(msg.text, session_key, session_store, client)
        channel.send(response)
```

**session_key 包含通道信息: `main:cli:user` vs `main:telegram:tg_user_1`. 同一用户在不同通道有不同会话.**

## What Changed from s03

| Component | s03 | s04 |
|-----------|-----|-----|
| 消息来源 | 只有 stdin | CLI + File + MockTelegram |
| 消息格式 | 原始字符串 | InboundMessage dataclass |
| Session key | 硬编码 `main:cli:user` | 根据 channel + sender 动态构建 |
| 回复发送 | 直接 print | channel.send() (自动分块) |
| 新增抽象 | -- | Channel ABC, InboundMessage, ChannelRegistry |

**Key shift**: 引入 Channel 抽象层, 将 "消息从哪来" 和 "消息怎么处理" 解耦. agent_loop 完全不变, 只是被 gateway 在更高层调度.

## Design Decisions

**为什么 receive() 必须非阻塞?**

gateway 在一个循环中轮询所有通道. 如果 Telegram 的 receive() 阻塞等待, Discord 的消息就无法被处理. 非阻塞让 gateway 可以快速扫描所有通道, 有消息就处理, 没消息就跳过.

**为什么用文件模拟 webhook?**

FileChannel 用 "追加-读取" 模拟 "推送-消费". 在另一个终端 `echo "hello" >> inbox.txt` 就能模拟外部系统推送, 不需要搭建 HTTP 服务器或注册 Bot Token.

**In production OpenClaw:** 支持 15+ 真实通道 (Telegram, Discord, Slack, Signal, WhatsApp, iMessage, Teams, Matrix 等), 通道作为 npm 包可热加载, 每个通道有独立的速率限制, 支持统一的媒体上传/下载/转码管道, 消息可并行处理.

## Try It

```sh
cd claw0
python agents/s04_multi_channel.py
```

试试这些操作:

- 直接输入文本 -- 走 CLIChannel
- `/channels` 查看已注册通道
- 另一个终端: `echo "What time is it?" >> workspace/.channels/file_inbox.txt`
- 回到 agent 终端: `/poll` -- 观察 FileChannel 接收并处理消息
- `/send telegram Hello from test`, 然后 `/poll`
- `/sessions` -- 注意不同通道产生的独立会话 (session key 中的 channel 部分不同)
- 查看 `workspace/.channels/telegram_outbox.txt` -- 回复被包装在 Telegram 消息信封中
