# s01: Agent Loop (エージェントループ)

> "One loop to rule them all" -- AI エージェントの秘密は、たった一つの while ループにある。

## At a Glance

```
User Input --> messages[] --> LLM API
                               |
                        stop_reason?
                       /           \
                 "end_turn"    "tool_use"
                    |              |
                  Print        (s02 で実装)
                    |
             append to messages
                    |
            wait for next input
```

- **What we build**: 最小限の対話式 REPL。ユーザーが一言入力し、モデルが一言返し、それを繰り返す。
- **Core mechanism**: `while True` ループ + `stop_reason` チェックで、毎ターンの制御フローを決定する。
- **Design pattern**: messages 配列を唯一の状態とし、user/assistant メッセージが厳密に交互する列構造。

## The Problem

1. **ループがなければ手動でコンテキストを貼り付ける羽目になる。** LLM はステートレスであり、プロンプトを渡せばテキストを返し、それで終わる。ループがなければ、過去の対話を毎回自分でモデルに渡す必要がある。

2. **コンテキストを失えば記憶喪失になる。** assistant の応答を messages に追加しなければ、次のターンでモデルは自分が以前何を言ったか分からず、マルチターンの推論ができない。

3. **stop_reason をチェックしなければ拡張できない。** 「応答を受け取ったら表示する」とハードコードしてしまうと、後からツール呼び出しを追加する際にフロー全体を書き直すことになる。stop_reason が制御フローの唯一の分岐点である。

## How It Works

### 1. クライアントとメッセージ履歴の初期化

Anthropic クライアントを作成し、空の messages リストを用意する。messages がエージェント全体の唯一の状態となる。

```python
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)
SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."
messages: list[dict] = []
```

**messages はフラットな配列であり、ツリーでもグラフでもない。**

### 2. ユーザー入力の取得

`while True` ループ内で入力を読み取り、終了シグナルを処理する。

```python
while True:
    try:
        user_input = input(colored_prompt()).strip()
    except (KeyboardInterrupt, EOFError):
        break
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        break
```

**空入力はスキップし、quit/exit/Ctrl+C で正常終了する。**

### 3. user メッセージの追加と API 呼び出し

ユーザーの発言を `role: "user"` として messages に追加し、配列全体を API に送信する。

```python
messages.append({"role": "user", "content": user_input})

try:
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=8096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
except Exception as exc:
    messages.pop()  # API 失敗時、user メッセージをロールバック
    continue
```

**API 呼び出しが失敗した場合、追加したメッセージを pop して、ユーザーがリトライできるようにする。**

### 4. stop_reason のチェックと応答の表示

stop_reason はループ全体の制御シグナルである。このセクションでは `end_turn` のみだが、分岐構造は後の拡張に対応している。

```python
if response.stop_reason == "end_turn":
    assistant_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            assistant_text += block.text
    print_assistant(assistant_text)

    messages.append({
        "role": "assistant",
        "content": response.content,
    })
```

**assistant の応答も messages に追加する必要がある。こうすることで、次のターンでモデルが自分の過去の発言を参照できる。**

## What Changed from s00

本セクションが出発点であり、先行バージョンは存在しない。ゼロから確立する基本パターン:

| 概念 | 状態 |
|------|------|
| 対話ループ | `while True` + `input()` |
| 状態管理 | `messages[]` 配列 |
| 制御フロー | `stop_reason` 分岐 |
| ツール | なし (次セクションで追加) |
| 永続化 | なし (終了時に消失) |

**Key shift**: 「単発 API 呼び出し」から「持続的な対話ループ」へ。パターン全体は一行で要約できる: `while True -> input -> append -> API -> check stop_reason -> print -> append -> loop`。

## Design Decisions

**なぜ messages はフラットな配列であり、より複雑な構造ではないのか?**

Anthropic API は messages を厳密に交互する user/assistant のリストとして要求する。つまり、コンテキスト管理は配列操作 -- 追加、切り詰め、クリーンアップ -- であり、ステートマシンは不要である。シンプルさこそが正しい。

**なぜ stop_reason が唯一の制御シグナルなのか?**

API が返す stop_reason はすべてのケースを網羅している: `end_turn` (完了)、`tool_use` (ツール呼び出し)、`max_tokens` (上限到達)。一つの if/elif で処理できる。「モデルが続けたがっているか」を自前で判定する必要はない -- API が既にその判断を下している。

**In production OpenClaw:** コアのループパターンは全く同じだが、lane ベースの並行実行 (複数対話の並行処理)、多層リトライ (API タイムアウト/レート制限/ネットワークエラー)、ストリーミング (トークン単位の出力)、およびトークン管理 (長すぎる履歴の自動切り詰め) が追加される。これらをすべて取り除けば、残るのはこの while True ループである。

## Try It

```sh
cd claw0
python agents/s01_agent_loop.py
```

まず `.env` に以下を設定する:

```sh
ANTHROPIC_API_KEY=sk-ant-xxxxx
MODEL_ID=claude-sonnet-4-20250514
```

次の入力を試してみよう:

- `What is Python?` -- 基本的な質疑応答
- 関連する質問を連続で投げる -- モデルが履歴コンテキストをどう活用するか観察する
- `Can you help me write a file?` -- モデルは回答しようとするが、ツールがないためテキストしか返せない
- `quit` -- 終了すると全履歴が消失する
