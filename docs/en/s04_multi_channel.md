# s04: Multi-Channel

> "Same brain, many mouths" -- One agent serving CLI, Telegram, and Discord simultaneously.

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

- **What we build**: A Channel plugin interface + ChannelRegistry + gateway polling loop, allowing the same agent to send and receive messages through any channel.
- **Core mechanism**: All channel messages are normalized into InboundMessage. receive() uses non-blocking polling. send() auto-chunks long text.
- **Design pattern**: Channels handle I/O, the Registry manages them, the Gateway routes messages. agent_loop itself does not change.

## The Problem

1. **One codebase per channel = duplication and inconsistency.** A separate agent for CLI, another for Telegram, yet another for Discord. Duplicated logic, inconsistent behavior, fix a bug in one place and miss it in two others.

2. **Different channels have different message formats.** Telegram has chat_id and update_id, Discord has guild_id, CLI has plain text. Without standardization, the agent needs channel-specific parsing logic everywhere.

3. **Different channels have different message length limits.** Telegram caps at 4096 characters, Discord at 2000, SMS at 160. A 2000-character reply works fine on Telegram but gets truncated on Discord.

## How It Works

### 1. InboundMessage: Standardized Inbound Messages

Different channels produce different message formats, but before entering the agent they are all normalized into one structure:

```python
@dataclass
class InboundMessage:
    channel: str           # source channel ID
    sender: str            # sender identifier
    text: str              # message text
    media_urls: list[str] = field(default_factory=list)
    thread_id: str | None = None
    timestamp: float = field(default_factory=time.time)
```

**Standardization is the core of multi-channel architecture. Without it, the agent needs different handling logic for every channel.**

### 2. Channel: The Abstract Base Class

Every channel plugin must implement this interface:

```python
class Channel(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...          # "telegram", "cli"

    @property
    @abstractmethod
    def max_text_length(self) -> int: ...  # 4096, 8000

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...  # non-blocking poll

    @abstractmethod
    def send(self, text: str, media: list | None = None) -> None: ...

    def chunk_text(self, text: str) -> list[str]:
        # Default chunking: paragraph > newline > space > hard cut
        ...
```

**receive() must be non-blocking. The gateway polls all channels in one loop; blocking any one channel starves all the others.**

### 3. Three Channel Implementations

**CLIChannel** -- The simplest. Reads from stdin, writes to stdout:

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

**FileChannel** -- Watches a file for changes, simulating a webhook. Uses `_read_offset` to track the read position and process only new content:

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

**MockTelegramChannel** -- Simulates Telegram's 4096-character limit and message envelope format. Replies are written to an outbox file.

### 4. ChannelRegistry and gateway_poll_once

The Registry manages channels and poll_all collects messages. gateway_poll_once routes each message to the agent:

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

**The session_key includes channel information: `main:cli:user` vs `main:telegram:tg_user_1`. The same user on different channels gets separate sessions.**

## What Changed from s03

| Component | s03 | s04 |
|-----------|-----|-----|
| Message source | stdin only | CLI + File + MockTelegram |
| Message format | Raw string | InboundMessage dataclass |
| Session key | Hardcoded `main:cli:user` | Dynamically built from channel + sender |
| Reply delivery | Direct print | channel.send() (auto-chunking) |
| New abstractions | -- | Channel ABC, InboundMessage, ChannelRegistry |

**Key shift**: The Channel abstraction layer decouples "where messages come from" and "how messages are processed". agent_loop is completely unchanged -- it is simply invoked by the gateway at a higher level.

## Design Decisions

**Why must receive() be non-blocking?**

The gateway polls all channels in a single loop. If Telegram's receive() blocks waiting for data, Discord messages cannot be processed. Non-blocking lets the gateway quickly scan all channels: process if there is a message, skip if there is not.

**Why simulate webhooks with a file?**

FileChannel uses "append-then-read" to simulate "push-then-consume". From another terminal, `echo "hello" >> inbox.txt` simulates an external system push. No HTTP server or bot token registration needed.

**In production OpenClaw:** Supports 15+ real channels (Telegram, Discord, Slack, Signal, WhatsApp, iMessage, Teams, Matrix, and more). Channels are hot-loadable as npm packages, each with independent rate limits. A unified media upload/download/transcoding pipeline is available. Messages can be processed in parallel.

## Try It

```sh
cd claw0
python agents/s04_multi_channel.py
```

Things to try:

- Type text directly -- goes through CLIChannel
- `/channels` to list registered channels
- In another terminal: `echo "What time is it?" >> workspace/.channels/file_inbox.txt`
- Back in the agent terminal: `/poll` -- observe FileChannel receiving and processing the message
- `/send telegram Hello from test`, then `/poll`
- `/sessions` -- notice the separate sessions from different channels (the channel segment of the session key differs)
- Check `workspace/.channels/telegram_outbox.txt` -- replies are wrapped in a Telegram message envelope
