"""
Section 01: The Agent Loop
"One loop to rule them all"

AI Agent 的全部秘密就是一个 while 循环不断检查 stop_reason.
本节展示最纯粹的对话循环 -- 没有工具, 没有花活, 只有:
  用户输入 -> 历史消息 -> LLM -> 打印回复 -> 继续

架构图:

    User Input --> [messages[]] --> LLM API
                                     |
                              stop_reason?
                             /           \
                       "end_turn"    "tool_use"
                          |              |
                       Print          (next section)

运行方式:
    cd claw0
    python agents/s01_agent_loop.py

需要在 .env 中配置:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
    # ANTHROPIC_BASE_URL=https://...  (可选)
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

# 加载 .env -- 向上查找到 claw0 目录
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

# 系统提示: 保持简单, 这一节只关注循环本身
SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."

# ---------------------------------------------------------------------------
# ANSI 颜色 -- 让 REPL 看起来更清晰
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


def colored_prompt() -> str:
    """返回带颜色的输入提示符."""
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    """格式化打印 assistant 回复."""
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    """打印灰色提示信息."""
    print(f"{DIM}{text}{RESET}")


# ---------------------------------------------------------------------------
# 核心: Agent 循环
# ---------------------------------------------------------------------------
# 关键认知:
#   整个循环的 "智能" 来自 LLM, 我们只做三件事:
#   1. 收集用户输入, 追加到 messages
#   2. 调用 API, 拿到 response
#   3. 检查 stop_reason 决定下一步
#
#   本节 stop_reason 永远是 "end_turn" (没有工具可调用).
#   下一节加入工具后, 循环结构完全不变, 只多一个分支.
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """主 agent 循环 -- 对话式 REPL."""

    # messages 是整个 agent 的 "记忆"
    # 每轮对话的 user/assistant 消息都追加到这里
    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  Mini-Claw  |  Section 01: The Agent Loop")
    print_info(f"  Model: {MODEL_ID}")
    print_info("  Type 'quit' or 'exit' to leave. Ctrl+C also works.")
    print_info("=" * 60)
    print()

    while True:
        # --- Step 1: 获取用户输入 ---
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C 或 Ctrl+D: 优雅退出
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        # --- Step 2: 追加 user 消息到历史 ---
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # --- Step 3: 调用 LLM ---
        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            # API 错误不应该炸掉整个循环
            print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
            # 回滚刚追加的 user 消息, 让用户可以重试
            messages.pop()
            continue

        # --- Step 4: 检查 stop_reason ---
        # 在本节中, stop_reason 只会是 "end_turn"
        # 但我们把分支写全, 为下一节做准备
        if response.stop_reason == "end_turn":
            # 提取纯文本回复
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text

            # 打印回复
            print_assistant(assistant_text)

            # 追加 assistant 消息到历史 (保持上下文)
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

        elif response.stop_reason == "tool_use":
            # 本节没有工具, 但如果模型尝试调用工具, 给出提示
            print_info("[stop_reason=tool_use] No tools available in this section.")
            print_info("See s02_tool_use.py for tool support.")
            # 仍然把回复追加到历史
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

        else:
            # 其他 stop_reason: max_tokens 等
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

        # --- 循环继续, 等待下一次用户输入 ---


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    """程序入口."""
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
