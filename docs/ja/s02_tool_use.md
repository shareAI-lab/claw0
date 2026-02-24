# s02: Tool Use (ツール呼び出し)

> "Give the model hands" -- ループは変わらない。ディスパッチテーブルを一つ追加しただけだ。

## At a Glance

```
User --> [messages + tools] --> LLM
                                 |
                          stop_reason?
                         /           \
                   "end_turn"    "tool_use"
                      |              |
                    Print    TOOL_HANDLERS[name](**input)
                                     |
                              tool_result
                                     |
                            append to messages
                                     |
                              back to LLM  <-- 内側 while ループ
```

- **What we build**: エージェントに 4 つのツール (bash, read_file, write_file, edit_file) を追加し、ファイルシステムの操作やコマンド実行を可能にする。
- **Core mechanism**: TOOLS スキーマがモデルに利用可能なツールを伝え、TOOL_HANDLERS ディスパッチテーブルがコード側で実行する関数を決める。
- **Design pattern**: 外側ループがユーザー入力を待ち、内側ループが連続ツール呼び出しを処理する。stop_reason がすべてを制御する。

## The Problem

1. **モデルは「話す」ことしかできず、「実行する」ことができない。** ユーザーが「config.json を読んで」と言っても、モデルは「`cat config.json` で読めます」と返すだけで、実際に実行できない。

2. **モデルは複数のツールを連続して呼び出す場合がある。** 例えば「ファイルを読み、その中の一行を修正し、結果を確認する」には read_file -> edit_file -> read_file の3回のツール呼び出しが必要で、単一リクエストでは完結しない。

3. **制御なきツール実行は危険である。** パス検査がなければモデルは `/etc/passwd` を読み取る可能性があり、出力の切り詰めがなければ `find /` 一つでコンテキストウィンドウがあふれる。

## How It Works

### 1. ツールスキーマとディスパッチテーブルの定義

二つのデータ構造が `name` フィールドで紐付けられる。TOOLS がモデルに何が使えるかを伝え、TOOL_HANDLERS がコード側で何を実行するかを決める。

```python
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "timeout": {"type": "integer", "description": "Timeout in seconds. Default 30."},
            },
            "required": ["command"],
        },
    },
    # ... read_file, write_file, edit_file (同様の構造)
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}
```

**TOOLS 配列は API に渡し、TOOL_HANDLERS 辞書はローカルに保持する。モデルがツールを選び、我々がツールを実行する。**

### 2. ツール関数の実装

各ツールはキーワード引数 (スキーマの properties に対応) を受け取り、文字列結果を返す:

```python
def tool_bash(command: str, timeout: int = 30) -> str:
    dangerous = ["rm -rf /", "mkfs", "> /dev/sd", "dd if="]
    for pattern in dangerous:
        if pattern in command:
            return f"Error: Refused to run dangerous command containing '{pattern}'"
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        timeout=timeout, cwd=str(WORKDIR),
    )
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
    return truncate(output) if output else "[no output]"
```

**tool_edit_file は old_string がファイル内に正確に一度だけ出現することを要求し、それ以外は置換を拒否する -- 一意性制約によって誤操作を防ぐ。**

### 3. ディスパッチ関数

ツール名から TOOL_HANDLERS を検索して実行する:

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

**すべてのエラーは例外を投げるのではなく、文字列として返してモデルに伝える。モデルはエラーメッセージを見て自己修正できる。**

### 4. 内側 while ループによるツールチェーンの処理

モデルは複数のツールを連続して呼び出す場合がある。内側ループは stop_reason が `tool_use` でなくなるまで続く:

```python
while True:
    response = client.messages.create(
        model=MODEL_ID, max_tokens=8096,
        system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        # テキストを抽出、表示、内側ループを抜ける
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
        # tool_result は user ロールのメッセージとして返す (API フォーマットの要件)
        messages.append({"role": "user", "content": tool_results})
```

**外側ループがユーザー入力を待ち、内側ループがツールチェーンを処理する。二重ループ、同一の stop_reason で制御。**

### 5. セキュリティ機構

二つのヘルパー関数がツール実行の境界を守る:

```python
def safe_path(raw: str) -> Path:
    target = (WORKDIR / raw).resolve()
    if not str(target).startswith(str(WORKDIR)):
        raise ValueError(f"Path traversal blocked: {raw}")
    return target

def truncate(text: str, limit: int = 50000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"
```

**safe_path がパストラバーサルを防ぎ、truncate が出力によるコンテキストの溢れを防ぐ。**

## What Changed from s01

| コンポーネント | s01 | s02 |
|-----------|-----|-----|
| API 呼び出し | tools を渡さない | `tools=TOOLS` を渡す |
| stop_reason | `end_turn` のみ | `tool_use` 分岐を追加 |
| ループ構造 | 単一の while True | 外側 (ユーザー) + 内側 (ツールチェーン) |
| セキュリティ | なし | safe_path + truncate + 危険コマンドブラックリスト |
| 新規コード | -- | TOOLS スキーマ + TOOL_HANDLERS + 4 つのツール関数 |

**Key shift**: API 呼び出しに `tools` パラメータを一つ追加し、stop_reason に `tool_use` 分岐を一つ追加した。ループ構造自体は変わっていない。

## Design Decisions

**なぜ tool_result は user ロールのメッセージに含まれるのか?**

Anthropic API は messages の厳密な交互配列 user -> assistant -> user を要求する。ツール呼び出しの応答は assistant メッセージ (tool_use ブロックを含む) であり、ツール結果は次の user メッセージとして返す必要がある。これは「ユーザーの発言」ではなく、API フォーマット上の要件である。

**なぜ 4 つのツールで、もっと多くないのか?**

bash が 90% のシステム操作をカバーし、read_file は `bash cat` より安全 (パス検査と切り詰めあり)、write_file は親ディレクトリの自動作成付き、edit_file は精密な置換を行う (一意性制約)。この 4 つのツールで、エージェントはほとんどのプログラミングタスクを遂行できる。

**In production OpenClaw:** 50 以上のツールがあり、複数の tool_use ブロックの並列実行をサポートし、各ツールには独立の権限ポリシーがある (bash はユーザー確認が必要、read_file は自動許可)。ツール結果は画像や構造化データに対応し、切り詰め閾値は残りトークン予算に応じて動的に調整される。

## Try It

```sh
cd claw0
python agents/s02_tool_use.py
```

次の入力を試してみよう:

- `List the files in the current directory` -- bash ツールが ls を実行する様子を観察する
- `Read the contents of agents/s01_agent_loop.py` -- read_file の動作を観察する
- `Create a file called hello.py that prints "Hello, World!"` -- write_file の動作を観察する
- `Read hello.py, then change the message to "Hello, claw0!"` -- read_file + edit_file のツールチェーンを観察する
