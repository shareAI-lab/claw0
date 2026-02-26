"""
Section 01: Agent 循环
"Agent 就是 while True + stop_reason"

    用户输入 --> [messages[]] --> LLM API --> stop_reason?
                                               /        \
                                         "end_turn"  "tool_use"
                                             |           |
                                          打印回复    (下一节)

用法:
    cd claw0
    python zh/s01_agent_loop.py

需要在 .env 中配置:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# 导入
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."

# ---------------------------------------------------------------------------
# ANSI 颜色
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
# 核心: Agent 循环
# ---------------------------------------------------------------------------
#   1. 收集用户输入, 追加到 messages
#   2. 调用 API
#   3. 检查 stop_reason 决定下一步
#
#   本节 stop_reason 永远是 "end_turn" (没有工具).
#   下一节加入 "tool_use" -- 循环结构保持不变.
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """主 agent 循环 -- 对话式 REPL."""

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 01: Agent 循环")
    print_info(f"  Model: {MODEL_ID}")
    print_info("  输入 'quit' 或 'exit' 退出. Ctrl+C 同样有效.")
    print_info("=" * 60)
    print()

    while True:
        # --- 获取用户输入 ---
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}再见.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}再见.{RESET}")
            break

        # --- 追加到历史 ---
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # --- 调用 LLM ---
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            messages.pop()
            continue

        # --- 检查 stop_reason ---
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
            print_info("[stop_reason=tool_use] 本节没有可用工具.")
            print_info("参见 s02_tool_use.py 了解工具支持.")
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
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY 未设置.{RESET}")
        print(f"{DIM}将 .env.example 复制为 .env 并填入你的 key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
