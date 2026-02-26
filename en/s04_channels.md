# Section 04: Channels

> Every platform is different, but they all produce the same InboundMessage.

## Architecture

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

## Key Concepts

- **InboundMessage**: a dataclass that normalizes all platform payloads into one format.
- **Channel ABC**: `receive()` + `send()` is the entire contract.
- **TelegramChannel**: long-polling, offset persistence, media group buffering, text coalescing.
- **FeishuChannel**: webhook-based, token auth, mention detection, multi-type message parsing.
- **ChannelManager**: registry that holds all active channels.

## Key Code Walkthrough

### 1. InboundMessage -- the universal message format

Every channel normalizes into this. The agent loop only sees `InboundMessage`,
never platform-specific payloads.

```python
@dataclass
class InboundMessage:
    text: str
    sender_id: str
    channel: str = ""          # "cli", "telegram", "feishu"
    account_id: str = ""       # which bot received it
    peer_id: str = ""          # DM=user_id, group=chat_id, topic=chat_id:topic:thread_id
    is_group: bool = False
    media: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)
```

The `peer_id` encodes conversation scope:

| Context           | peer_id Format            |
|-------------------|---------------------------|
| Telegram DM       | `user_id`                 |
| Telegram group    | `chat_id`                 |
| Telegram topic    | `chat_id:topic:thread_id` |
| Feishu p2p        | `user_id`                 |
| Feishu group      | `chat_id`                 |

### 2. The Channel ABC

Adding a new platform means implementing exactly two methods:

```python
class Channel(ABC):
    name: str = "unknown"

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool: ...
```

`CLIChannel` is the simplest implementation -- `receive()` wraps `input()`,
`send()` wraps `print()`:

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

### 3. run_agent_turn -- channel-agnostic processing

The agent turn function takes an `InboundMessage`, runs the standard tool loop,
and sends the reply back through the source channel:

```python
def run_agent_turn(inbound: InboundMessage, conversations: dict, mgr: ChannelManager):
    sk = build_session_key(inbound.channel, inbound.account_id, inbound.peer_id)
    if sk not in conversations:
        conversations[sk] = []
    messages = conversations[sk]
    messages.append({"role": "user", "content": inbound.text})

    # Typing indicator for Telegram
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
            # dispatch tools, append results, continue
            ...
```

## Try It

```sh
# CLI only (no env vars needed beyond API key)
python en/s04_channels.py

# With Telegram -- add to .env:
# TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
# TELEGRAM_ALLOWED_CHATS=12345,67890    (optional whitelist)

# With Feishu -- add to .env:
# FEISHU_APP_ID=cli_xxxxx
# FEISHU_APP_SECRET=xxxxx

# REPL commands
# You > /channels      (list registered channels)
# You > /accounts      (show bot accounts)
```

## How OpenClaw Does It

| Aspect          | claw0 (this file)                | OpenClaw production                      |
|-----------------|----------------------------------|------------------------------------------|
| Channel ABC     | `receive()` + `send()`           | Same contract + lifecycle hooks          |
| Platforms       | CLI, Telegram, Feishu            | 10+ (Telegram, Discord, Slack, etc.)     |
| Concurrency     | Thread per channel + shared queue| Same threading model + async gateway     |
| Message format  | `InboundMessage` dataclass       | Same normalized message type             |
| Offset storage  | Plain text file                  | JSON with version + atomic write         |
