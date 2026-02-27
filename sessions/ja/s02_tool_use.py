"""
Section 02: ツール使用
「ツールとはデータ (スキーマ辞書) + ハンドラーマップである。モデルが名前を選び、こちらがそれを検索する。」

エージェントループは s01 から変更なし。追加点のみ:
  1. TOOLS 配列がモデルに利用可能なツールを伝える (JSON スキーマ)
  2. TOOL_HANDLERS 辞書がツール名を Python 関数にマッピング
  3. stop_reason == "tool_use" のとき、dispatch して結果を返す

    ユーザー --> LLM --> stop_reason == "tool_use"?
                          |
                  TOOL_HANDLERS[name](**input)
                          |
                  tool_result --> LLM に返す
                          |
                   stop_reason == "end_turn"? --> 表示

ツール:
    - bash        : シェルコマンドを実行
    - read_file   : ファイル内容を読み取り
    - write_file  : ファイルに書き込み
    - edit_file   : ファイル内の文字列を正確に置換

使い方:
    cd claw0
    python ja/s02_tool_use.py

.env に必要な設定:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# インポート
# ---------------------------------------------------------------------------
import os
import sys
import subprocess
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "Use the tools to help the user with file operations and shell commands.\n"
    "Always read a file before editing it.\n"
    "When using edit_file, the old_string must match EXACTLY (including whitespace)."
)

MAX_TOOL_OUTPUT = 50000
WORKDIR = Path.cwd()

# ---------------------------------------------------------------------------
# ANSI カラー
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# 安全性ヘルパー
# ---------------------------------------------------------------------------

def safe_path(raw: str) -> Path:
    """パスを解決し、WORKDIR 外へのディレクトリトラバーサルをブロックする。"""
    target = (WORKDIR / raw).resolve()
    if not str(target).startswith(str(WORKDIR)):
        raise ValueError(f"パストラバーサルをブロック: {raw} が WORKDIR 外に解決されます")
    return target


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [切り詰め, 合計 {len(text)} 文字]"


# ---------------------------------------------------------------------------
# ツール実装
# ---------------------------------------------------------------------------


def tool_bash(command: str, timeout: int = 30) -> str:
    """シェルコマンドを実行し、出力を返す。"""
    dangerous = ["rm -rf /", "mkfs", "> /dev/sd", "dd if="]
    for pattern in dangerous:
        if pattern in command:
            return f"エラー: '{pattern}' を含む危険なコマンドの実行を拒否しました"

    print_tool("bash", command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(WORKDIR),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        if result.returncode != 0:
            output += f"\n[終了コード: {result.returncode}]"
        return truncate(output) if output else "[出力なし]"
    except subprocess.TimeoutExpired:
        return f"エラー: コマンドが {timeout} 秒でタイムアウトしました"
    except Exception as exc:
        return f"エラー: {exc}"


def tool_read_file(file_path: str) -> str:
    """ファイルの内容を読み取る。"""
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"エラー: ファイルが見つかりません: {file_path}"
        if not target.is_file():
            return f"エラー: ファイルではありません: {file_path}"
        content = target.read_text(encoding="utf-8")
        return truncate(content)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"エラー: {exc}"


def tool_write_file(file_path: str, content: str) -> str:
    """ファイルに内容を書き込む。必要に応じて親ディレクトリを作成する。"""
    print_tool("write_file", file_path)
    try:
        target = safe_path(file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"{file_path} に {len(content)} 文字を書き込みました"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"エラー: {exc}"


def tool_edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """正確な文字列置換。old_string はファイル内に1回だけ出現する必要がある。"""
    print_tool("edit_file", f"{file_path} ({len(old_string)} 文字を置換)")
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"エラー: ファイルが見つかりません: {file_path}"

        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)

        if count == 0:
            return "エラー: old_string がファイル内に見つかりません。完全に一致することを確認してください。"
        if count > 1:
            return (
                f"エラー: old_string が {count} 回見つかりました。"
                "一意でなければなりません。より多くの前後の文脈を含めてください。"
            )

        new_content = content.replace(old_string, new_string, 1)
        target.write_text(new_content, encoding="utf-8")
        return f"{file_path} の編集に成功しました"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"エラー: {exc}"


# ---------------------------------------------------------------------------
# ツールスキーマ + dispatch テーブル
# ---------------------------------------------------------------------------
# TOOLS = モデルに利用可能なものを伝える (JSON スキーマ)
# TOOL_HANDLERS = コード側で何を呼び出すかを定義 (名前 -> 関数)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command and return its output. "
            "Use for system commands, git, package managers, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30.",
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file. Creates parent directories if needed. "
            "Overwrites existing content."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write.",
                },
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": (
            "Replace an exact string in a file with a new string. "
            "The old_string must appear exactly once in the file. "
            "Always read the file first to get the exact text to replace."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file (relative to working directory).",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace. Must be unique.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}


# ---------------------------------------------------------------------------
# ツール dispatch
# ---------------------------------------------------------------------------

def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """名前でハンドラーを検索し、入力を kwargs として呼び出す。"""
    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        return f"エラー: 不明なツール '{tool_name}'"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        return f"エラー: {tool_name} の引数が不正です: {exc}"
    except Exception as exc:
        return f"エラー: {tool_name} が失敗しました: {exc}"


# ---------------------------------------------------------------------------
# コア: エージェントループ
# ---------------------------------------------------------------------------
# s01 と同じ while True に加えて:
#   - API 呼び出しに tools=TOOLS を含める
#   - 内部 while ループでツール呼び出しチェーンを処理
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """メインのエージェントループ -- ツール付き REPL。"""

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 02: ツール使用")
    print_info(f"  モデル: {MODEL_ID}")
    print_info(f"  作業ディレクトリ: {WORKDIR}")
    print_info(f"  ツール: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  'quit' または 'exit' で終了。Ctrl+C でも可。")
    print_info("=" * 60)
    print()

    while True:
        # --- ユーザー入力を取得 ---
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}さようなら。{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}さようなら。{RESET}")
            break

        # --- ユーザーメッセージを追加 ---
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # --- 内部ループ: モデルが複数のツール呼び出しを連鎖させる場合がある ---
        while True:
            try:
                response = client.messages.create(
                    model=MODEL_ID,
                    max_tokens=8096,
                    system=SYSTEM_PROMPT,
                    tools=TOOLS,
                    messages=messages,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API エラー: {exc}{RESET}\n")
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            if response.stop_reason == "end_turn":
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
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

                # ツール結果はユーザーメッセージに入れる (Anthropic API の要件)
                messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                continue

            else:
                print_info(f"[stop_reason={response.stop_reason}]")
                assistant_text = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        assistant_text += block.text
                if assistant_text:
                    print_assistant(assistant_text)
                break


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}エラー: ANTHROPIC_API_KEY が設定されていません。{RESET}")
        print(f"{DIM}.env.example を .env にコピーして API キーを記入してください。{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
