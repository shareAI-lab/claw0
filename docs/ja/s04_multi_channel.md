# s04: Multi-Channel (マルチチャネルアーキテクチャ)

> "Same brain, many mouths" -- 同一のエージェントが CLI、Telegram、Discord を同時にサービスする。

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

- **What we build**: Channel プラグインインターフェース + ChannelRegistry + gateway ポーリングループにより、同一のエージェントが任意のチャネルを通じてメッセージを送受信できるようにする。
- **Core mechanism**: すべてのチャネルのメッセージを InboundMessage に統一し、receive() はノンブロッキングポーリング、send() は自動分割を行う。
- **Design pattern**: Channel が送受信を担当し、Registry が管理を担当し、Gateway がルーティングを担当する。agent_loop 自体は変更なし。

## The Problem

1. **チャネルごとに一式のコードを書けば、重複と不整合が生じる。** CLI 用のエージェント、Telegram 用のエージェント、Discord 用のエージェントと別々に実装すれば、ロジックが重複し、振る舞いが統一されず、一箇所のバグ修正が他を漏らす。

2. **チャネルによってメッセージフォーマットが異なる。** Telegram には chat_id と update_id があり、Discord には guild_id があり、CLI は純粋なテキストのみ。標準化がなければ、エージェントはチャネルごとに異なるパースロジックを書く必要がある。

3. **チャネルによってメッセージ長の制限が異なる。** Telegram は 4096 文字、Discord は 2000 文字、SMS は 160 文字。2000 文字の返答は Telegram では問題ないが、Discord では切り詰められる。

## How It Works

### 1. InboundMessage: 入力メッセージの標準化

チャネルごとに異なるメッセージフォーマットを、エージェントに渡す前に一つの構造に統一する:

```python
@dataclass
class InboundMessage:
    channel: str           # 送信元チャネル ID
    sender: str            # 送信者識別子
    text: str              # メッセージテキスト
    media_urls: list[str] = field(default_factory=list)
    thread_id: str | None = None
    timestamp: float = field(default_factory=time.time)
```

**標準化がマルチチャネルアーキテクチャの核心である。これがなければ、エージェントはチャネルごとに異なる処理ロジックを書くことになる。**

### 2. Channel: チャネル抽象基底クラス

各チャネルプラグインはこのインターフェースを実装する:

```python
class Channel(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...          # "telegram", "cli"

    @property
    @abstractmethod
    def max_text_length(self) -> int: ...  # 4096, 8000

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...  # ノンブロッキングポーリング

    @abstractmethod
    def send(self, text: str, media: list | None = None) -> None: ...

    def chunk_text(self, text: str) -> list[str]:
        # デフォルトの分割: 段落 > 改行 > 空白 > 強制切断
        ...
```

**receive() はノンブロッキングでなければならない。gateway は一つのループで全チャネルをポーリングするため、いずれか一つがブロックすれば他のチャネルが飢餓状態になる。**

### 3. 三つのチャネル実装

**CLIChannel** -- 最もシンプルな実装。stdin から読み、stdout に書く:

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

**FileChannel** -- ファイルの変更を監視し、webhook をシミュレートする。`_read_offset` で既読位置を追跡し、新規追加分のみを処理する:

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

**MockTelegramChannel** -- Telegram の 4096 文字制限とメッセージエンベロープフォーマットを模倣し、返信を outbox ファイルに書き込む。

### 4. ChannelRegistry と gateway_poll_once

Registry がチャネルを管理し、poll_all がメッセージを収集する。gateway_poll_once がエージェントにルーティングする:

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

**session_key にはチャネル情報が含まれる: `main:cli:user` vs `main:telegram:tg_user_1`。同一ユーザーでもチャネルが異なれば別のセッションになる。**

## What Changed from s03

| コンポーネント | s03 | s04 |
|-----------|-----|-----|
| メッセージソース | stdin のみ | CLI + File + MockTelegram |
| メッセージフォーマット | 生の文字列 | InboundMessage dataclass |
| Session key | ハードコード `main:cli:user` | channel + sender に基づき動的に構築 |
| 返信の送信 | 直接 print | channel.send() (自動分割) |
| 新規抽象 | -- | Channel ABC, InboundMessage, ChannelRegistry |

**Key shift**: Channel 抽象レイヤーの導入により、「メッセージがどこから来るか」と「メッセージをどう処理するか」を分離した。agent_loop は全く変更なし。gateway がより上位のレイヤーでスケジューリングするだけ。

## Design Decisions

**なぜ receive() はノンブロッキングでなければならないのか?**

gateway は一つのループで全チャネルをポーリングする。Telegram の receive() がブロックして待機すれば、Discord のメッセージは処理できなくなる。ノンブロッキングにすることで、gateway はすべてのチャネルを素早くスキャンし、メッセージがあれば処理し、なければスキップできる。

**なぜファイルで webhook をシミュレートするのか?**

FileChannel は「追記-読み取り」で「プッシュ-消費」を模倣する。別のターミナルで `echo "hello" >> inbox.txt` と打てば外部システムからのプッシュを模倣でき、HTTP サーバーの構築や Bot Token の登録は不要だ。

**In production OpenClaw:** 15 以上の実際のチャネル (Telegram, Discord, Slack, Signal, WhatsApp, iMessage, Teams, Matrix 等) をサポートし、チャネルは npm パッケージとしてホットロード可能である。各チャネルには独立のレート制限があり、統一されたメディアのアップロード/ダウンロード/トランスコードパイプラインをサポートし、メッセージは並列処理できる。

## Try It

```sh
cd claw0
python agents/s04_multi_channel.py
```

次の操作を試してみよう:

- テキストを直接入力する -- CLIChannel を経由する
- `/channels` で登録済みチャネルを確認する
- 別のターミナルで: `echo "What time is it?" >> workspace/.channels/file_inbox.txt`
- エージェントのターミナルに戻り: `/poll` -- FileChannel がメッセージを受信して処理する様子を観察する
- `/send telegram Hello from test` の後に `/poll`
- `/sessions` -- 異なるチャネルから生成された独立したセッションに注目する (session key の channel 部分が異なる)
- `workspace/.channels/telegram_outbox.txt` を確認する -- 返信が Telegram メッセージエンベロープに包まれている
