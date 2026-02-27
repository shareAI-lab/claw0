# 第02章: ツール使用

> ツールとはデータ(スキーマ) + ハンドラマップ。モデルが名前を選び、こちらがルックアップする。

## アーキテクチャ

```
    User Input
        |
        v
    messages[] --> LLM API (tools=TOOLS)
                       |
                  stop_reason?
                  /          \
            "end_turn"    "tool_use"
               |              |
             Print    for each tool_use block:
                        TOOL_HANDLERS[name](**input)
                              |
                        tool_result
                              |
                        messages[] <-- {role:"user", content:[tool_result]}
                              |
                        back to LLM --> may chain more tools
                                          or "end_turn" --> Print
```

外側の `while True` は第01章と同一。唯一の追加は、`stop_reason == "tool_use"` の間LLMを繰り返し呼び出す**内側の**whileループ。

## 本章のポイント

- **TOOLS**: モデルに何が利用可能かを伝えるJSONスキーマ辞書のリスト。
- **TOOL_HANDLERS**: 名前をPython関数にマッピングする `dict[str, Callable]`。
- **process_tool_call()**: 辞書ルックアップ + `**kwargs` ディスパッチ。
- **内側ループ**: モデルはテキスト出力の前に複数のツール呼び出しを連鎖できる。
- **ツール結果はuserメッセージに格納** (Anthropic APIの要件)。

## コードウォークスルー

### 1. スキーマ + ディスパッチテーブル

2つの並行データ構造。`TOOLS` がモデルに伝え、`TOOL_HANDLERS` がコードに伝える。

```python
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command."},
                "timeout": {"type": "integer", "description": "Timeout in seconds."},
            },
            "required": ["command"],
        },
    },
    # ... read_file, write_file, edit_file (同じパターン)
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}
```

新しいツールの追加 = `TOOLS` に1エントリ + `TOOL_HANDLERS` に1エントリ。ループ自体は変更不要。

### 2. ディスパッチ関数

モデルはツール名と入力の辞書を返す。ディスパッチは辞書ルックアップ。エラーは(raiseではなく)文字列として返し、モデルがそれを見てリカバリできるようにする。

```python
def process_tool_call(tool_name: str, tool_input: dict) -> str:
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"
```

### 3. 内側のツール呼び出しループ

第01章からの唯一の構造的変更。モデルは最終的なテキスト応答を生成する前に、複数回ツールを呼び出す場合がある。

```python
while True:
    response = client.messages.create(
        model=MODEL_ID, max_tokens=8096,
        system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        # テキストを抽出、表示、break
        break

    elif response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = process_tool_call(block.name, block.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result,
            })
        # ツール結果はuserメッセージに格納 (API要件)
        messages.append({"role": "user", "content": tool_results})
        continue  # LLMに戻る
```

## 試してみる

```sh
python ja/s02_tool_use.py

# コマンド実行を依頼
# You > What files are in the current directory?

# ファイル読み取りを依頼
# You > Read the contents of en/s01_agent_loop.py

# ファイルの作成と編集を依頼
# You > Create a file called hello.txt with "Hello World"
# You > Change "World" to "claw0" in hello.txt

# ツールの連鎖を観察 (read -> edit -> verify)
# You > Add a comment at the top of hello.txt
```

## OpenClaw での実装

| 観点             | claw0 (本ファイル)            | OpenClaw 本番環境                      |
|------------------|-------------------------------|----------------------------------------|
| ツール定義       | Pythonの辞書リスト            | TypeBoxスキーマ、自動バリデーション     |
| ディスパッチ     | `dict[str, Callable]` ルックアップ | 同じパターン + ミドルウェアパイプライン |
| 安全性           | `safe_path()` でトラバーサルをブロック | サンドボックス実行、許可リスト       |
| ツール数         | 4個 (bash, read, write, edit) | 20以上 (Web検索、メディア、カレンダー等) |
| ツール結果       | プレーンな文字列を返す        | メタデータ付き構造化結果               |
