# 第01章: エージェントループ

> エージェントとは `while True` + `stop_reason` である。

## アーキテクチャ

```
    User Input
        |
        v
    messages[] <-- append {role: "user", ...}
        |
        v
    client.messages.create(model, system, messages)
        |
        v
    stop_reason?
      /        \
 "end_turn"  "tool_use"
     |            |
   Print      (第02章)
     |
     v
    messages[] <-- append {role: "assistant", ...}
     |
     +--- ループ先頭に戻り、次の入力を待つ
```

これ以降の全て -- ツール、セッション、ルーティング、配信 -- はこのループの上に積み重なるだけで、ループ自体は変わらない。

## 本章のポイント

- **messages[]** が唯一の状態。LLM は毎回この配列全体を参照する。
- **stop_reason** が各APIレスポンス後の唯一の分岐点。
- **end_turn** = 「テキストを表示」。**tool_use** = 「実行し、結果をフィードバック」(第02章)。
- ループ構造は一切変わらない。以降の章はこのループの周囲に機能を追加する。

## コードウォークスルー

### 1. 完全なエージェントループ

1ターンごとに3ステップ: 入力取得、API呼び出し、stop_reasonで分岐。

```python
def agent_loop() -> None:
    messages: list[dict] = []

    while True:
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            print(f"API Error: {exc}")
            messages.pop()   # ロールバックしてリトライ可能にする
            continue

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

### 2. stop_reason による分岐

第01章の時点でも `tool_use` のスタブを書いておく。ツールはまだ存在しないが、このスキャフォールディングのおかげで第02章では外側のループを一切変更せずに済む。

```python
        elif response.stop_reason == "tool_use":
            print_info("[stop_reason=tool_use] No tools in this section.")
            messages.append({"role": "assistant", "content": response.content})
```

| stop_reason    | 意味                         | アクション         |
|----------------|------------------------------|--------------------|
| `"end_turn"`   | モデルが応答を完了した       | 表示してループ継続 |
| `"tool_use"`   | モデルがツール呼び出しを要求 | 実行し結果を返す   |
| `"max_tokens"` | トークン上限で応答が途切れた | 部分テキストを表示 |

## 試してみる

```sh
# .env に API キーを設定
echo 'ANTHROPIC_API_KEY=sk-ant-xxxxx' > .env
echo 'MODEL_ID=claude-sonnet-4-20250514' >> .env

# エージェントを起動
python ja/s01_agent_loop.py

# 対話する -- messages[] が蓄積されるのでマルチターンが機能する
# You > What is the capital of France?
# You > And what is its population?
# (モデルは前のターンの「France」を覚えている)
```

## OpenClaw での実装

| 観点           | claw0 (本ファイル)             | OpenClaw 本番環境                     |
|----------------|--------------------------------|---------------------------------------|
| ループの場所   | 1ファイル内の `agent_loop()`   | `src/agent/` の `AgentLoop` クラス    |
| メッセージ     | メモリ上の `list[dict]`        | JSONL永続化された SessionStore        |
| stop_reason    | 同じ分岐ロジック               | 同じロジック + ストリーミング対応     |
| エラー処理     | 直前メッセージをpopして続行    | バックオフ付きリトライ + コンテキストガード |
| システムプロンプト | ハードコードされた文字列   | 8層の動的組み立て (第06章)            |
