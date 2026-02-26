# 第 04 节: 通道

> 每个平台都不同, 但它们都产生相同的 InboundMessage.

## 架构

```
    Telegram ----.                          .---- sendMessage API
    Feishu -------+-- InboundMessage ---+---- im/v1/messages
    CLI (stdin) --'    Agent Loop        '---- print(stdout)
                       (same brain)

    Telegram detail:
    getUpdates (long-poll, 30s)
        |
    offset persist (disk)
        |
    media_group_id? --yes--> buffer 500ms --> merge captions
        |no
    text buffer (1s silence) --> flush
        |
    InboundMessage --> allowed_chats filter --> agent turn
```

## 本节要点

- **InboundMessage**: 一个 dataclass, 将所有平台的消息负载统一为同一格式.
- **Channel ABC**: `receive()` + `send()` 就是全部接口契约.
- **TelegramChannel**: 长轮询, offset 持久化, 媒体组缓冲, 文本合并.
- **FeishuChannel**: 基于 webhook, token 认证, @提及检测, 多类型消息解析.
- **ChannelManager**: 持有所有活跃通道的注册中心.

## 核心代码走读

### 1. InboundMessage -- 统一的消息格式

每个通道都归一化为此格式. agent 循环只看到 `InboundMessage`,
永远不接触平台特定的负载.

```python
@dataclass
class InboundMessage:
    text: str
    sender_id: str
    channel: str = ""          # "cli", "telegram", "feishu"
    account_id: str = ""       # 接收消息的 bot
    peer_id: str = ""          # DM=user_id, group=chat_id, topic=chat_id:topic:thread_id
    is_group: bool = False
    media: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)
```

`peer_id` 编码了会话范围:

| 上下文            | peer_id 格式              |
|-------------------|---------------------------|
| Telegram 私聊     | `user_id`                 |
| Telegram 群组     | `chat_id`                 |
| Telegram 话题     | `chat_id:topic:thread_id` |
| 飞书单聊          | `user_id`                 |
| 飞书群组          | `chat_id`                 |

### 2. Channel 抽象基类

添加新平台只需实现两个方法:

```python
class Channel(ABC):
    name: str = "unknown"

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool: ...
```

`CLIChannel` 是最简单的实现 -- `receive()` 包装 `input()`,
`send()` 包装 `print()`:

```python
class CLIChannel(Channel):
    name = "cli"

    def receive(self) -> InboundMessage | None:
        text = input("You > ").strip()
        if not text:
            return None
        return InboundMessage(
            text=text, sender_id="cli-user", channel="cli",
            account_id="cli-local", peer_id="cli-user",
        )

    def send(self, to: str, text: str, **kwargs: Any) -> bool:
        print_assistant(text)
        return True
```

### 3. run_agent_turn -- 与通道无关的处理逻辑

agent turn 函数接收一个 `InboundMessage`, 运行标准的工具循环,
然后通过来源通道发送回复:

```python
def run_agent_turn(inbound: InboundMessage, conversations: dict, mgr: ChannelManager):
    sk = build_session_key(inbound.channel, inbound.account_id, inbound.peer_id)
    if sk not in conversations:
        conversations[sk] = []
    messages = conversations[sk]
    messages.append({"role": "user", "content": inbound.text})

    # Telegram 的输入指示器
    if inbound.channel == "telegram":
        tg = mgr.get("telegram")
        if isinstance(tg, TelegramChannel):
            tg.send_typing(inbound.peer_id.split(":topic:")[0])

    while True:
        response = client.messages.create(
            model=MODEL_ID, max_tokens=8096,
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            text = "".join(b.text for b in response.content if hasattr(b, "text"))
            ch = mgr.get(inbound.channel)
            if ch:
                ch.send(inbound.peer_id, text)
            break
        elif response.stop_reason == "tool_use":
            # 分发工具, 追加结果, 继续
            ...
```

## 试一试

```sh
# 仅 CLI (除了 API key 外不需要其他环境变量)
python zh/s04_channels.py

# 启用 Telegram -- 在 .env 中添加:
# TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
# TELEGRAM_ALLOWED_CHATS=12345,67890    (可选白名单)

# 启用飞书 -- 在 .env 中添加:
# FEISHU_APP_ID=cli_xxxxx
# FEISHU_APP_SECRET=xxxxx

# REPL 命令
# You > /channels      (列出已注册的通道)
# You > /accounts      (显示 bot 账号)
```

## OpenClaw 中的对应实现

| 方面            | claw0 (本文件)                   | OpenClaw 生产代码                        |
|-----------------|----------------------------------|------------------------------------------|
| Channel ABC     | `receive()` + `send()`           | 相同接口 + 生命周期钩子                  |
| 平台数量        | CLI, Telegram, 飞书              | 10+ (Telegram, Discord, Slack 等)        |
| 并发模型        | 每个通道一个线程 + 共享队列      | 相同的线程模型 + 异步网关                |
| 消息格式        | `InboundMessage` dataclass       | 相同的统一消息类型                       |
| Offset 存储     | 纯文本文件                       | 带版本号的 JSON + 原子写入               |
