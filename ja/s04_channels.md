# 第04章: チャネル

> プラットフォームはそれぞれ異なるが、全て同じ InboundMessage を生成する。

## アーキテクチャ

```
    Telegram ----.                          .---- sendMessage API
    Feishu -------+-- InboundMessage ---+---- im/v1/messages
    CLI (stdin) --'    Agent Loop        '---- print(stdout)
                       (同じ頭脳)

    Telegram の詳細:
    getUpdates (long-poll, 30s)
        |
    offset 永続化 (ディスク)
        |
    media_group_id? --yes--> 500ms バッファ --> キャプションをマージ
        |no
    テキストバッファ (1秒の沈黙) --> flush
        |
    InboundMessage --> allowed_chats フィルタ --> エージェントターン
```

## 新しい概念

- **InboundMessage**: 全プラットフォームのペイロードを1つのフォーマットに正規化するデータクラス。
- **Channel ABC**: `receive()` + `send()` がコントラクトの全て。
- **TelegramChannel**: ロングポーリング、オフセット永続化、メディアグループバッファリング、テキスト結合。
- **FeishuChannel**: Webhookベース、トークン認証、メンション検出、複数タイプのメッセージ解析。
- **ChannelManager**: アクティブな全チャネルを保持するレジストリ。

## コードウォークスルー

### 1. InboundMessage -- 統一メッセージフォーマット

全てのチャネルがこの形式に正規化する。エージェントループは `InboundMessage` のみを扱い、プラットフォーム固有のペイロードは一切見ない。

```python
@dataclass
class InboundMessage:
    text: str
    sender_id: str
    channel: str = ""          # "cli", "telegram", "feishu"
    account_id: str = ""       # どのボットが受信したか
    peer_id: str = ""          # DM=user_id, グループ=chat_id, トピック=chat_id:topic:thread_id
    is_group: bool = False
    media: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)
```

`peer_id` は会話のスコープをエンコードする:

| コンテキスト      | peer_id のフォーマット    |
|-------------------|---------------------------|
| Telegram DM       | `user_id`                 |
| Telegram グループ | `chat_id`                 |
| Telegram トピック | `chat_id:topic:thread_id` |
| Feishu 1対1      | `user_id`                 |
| Feishu グループ  | `chat_id`                 |

### 2. Channel ABC

新しいプラットフォームの追加 = 2つのメソッドの実装のみ:

```python
class Channel(ABC):
    name: str = "unknown"

    @abstractmethod
    def receive(self) -> InboundMessage | None: ...

    @abstractmethod
    def send(self, to: str, text: str, **kwargs: Any) -> bool: ...
```

`CLIChannel` が最もシンプルな実装 -- `receive()` は `input()` をラップし、`send()` は `print()` をラップする:

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

### 3. run_agent_turn -- チャネル非依存の処理

エージェントターン関数は `InboundMessage` を受け取り、標準的なツールループを実行し、元のチャネルを通じて応答を返す:

```python
def run_agent_turn(inbound: InboundMessage, conversations: dict, mgr: ChannelManager):
    sk = build_session_key(inbound.channel, inbound.account_id, inbound.peer_id)
    if sk not in conversations:
        conversations[sk] = []
    messages = conversations[sk]
    messages.append({"role": "user", "content": inbound.text})

    # Telegram の場合、タイピングインジケーターを送信
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
            # ツールをディスパッチし、結果を追加して続行
            ...
```

## 試してみる

```sh
# CLIのみ (APIキー以外の環境変数は不要)
python en/s04_channels.py

# Telegram を使う場合 -- .env に追加:
# TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
# TELEGRAM_ALLOWED_CHATS=12345,67890    (オプションのホワイトリスト)

# Feishu を使う場合 -- .env に追加:
# FEISHU_APP_ID=cli_xxxxx
# FEISHU_APP_SECRET=xxxxx

# REPL コマンド
# You > /channels      (登録済みチャネルの一覧)
# You > /accounts      (ボットアカウントの表示)
```

## OpenClaw での実装

| 観点            | claw0 (本ファイル)               | OpenClaw 本番環境                        |
|-----------------|----------------------------------|------------------------------------------|
| Channel ABC     | `receive()` + `send()`           | 同じコントラクト + ライフサイクルフック   |
| プラットフォーム | CLI, Telegram, Feishu           | 10以上 (Telegram, Discord, Slack 等)     |
| 並行性          | チャネルごとのスレッド + 共有キュー | 同じスレッディングモデル + 非同期ゲートウェイ |
| メッセージ形式  | `InboundMessage` データクラス    | 同じ正規化メッセージ型                   |
| オフセット保存  | プレーンテキストファイル         | バージョン付きJSON + アトミック書き込み  |
