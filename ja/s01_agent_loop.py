"""
Section 01: エージェントループ
「エージェントとは while True + stop_reason のことである」

    ユーザー入力 --> [messages[]] --> LLM API --> stop_reason?
                                                /        \
                                          "end_turn"  "tool_use"
                                              |           |
                                           表示      (次のセクション)

使い方:
    cd claw0
    python ja/s01_agent_loop.py

.env に必要な設定:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# インポート
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."

# ---------------------------------------------------------------------------
# ANSI カラー
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# コア: エージェントループ
# ---------------------------------------------------------------------------
#   1. ユーザー入力を受け取り、messages に追加
#   2. API を呼び出す
#   3. stop_reason を確認して次の動作を決定
#
#   ここでは stop_reason は常に "end_turn" (ツールなし)。
#   次のセクションで "tool_use" を追加 -- ループ構造はそのまま。
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """メインのエージェントループ -- 対話型 REPL。"""

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 01: エージェントループ")
    print_info(f"  モデル: {MODEL_ID}")
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

        # --- 履歴に追加 ---
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # --- LLM を呼び出す ---
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            print(f"\n{YELLOW}API エラー: {exc}{RESET}\n")
            messages.pop()
            continue

        # --- stop_reason を確認 ---
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

        elif response.stop_reason == "tool_use":
            print_info("[stop_reason=tool_use] このセクションにはツールがありません。")
            print_info("ツール対応は s02_tool_use.py を参照してください。")
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

        else:
            print_info(f"[stop_reason={response.stop_reason}]")
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text
            if assistant_text:
                print_assistant(assistant_text)
            messages.append({
                "role": "assistant",
                "content": response.content,
            })


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
