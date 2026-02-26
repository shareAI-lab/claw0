"""
Section 01: The Agent Loop
"An agent is just while True + stop_reason"

    User Input --> [messages[]] --> LLM API --> stop_reason?
                                                /        \
                                          "end_turn"  "tool_use"
                                              |           |
                                           Print      (next section)

Usage:
    cd claw0
    python en/s01_agent_loop.py

Required .env config:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."

# ---------------------------------------------------------------------------
# ANSI colors
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
# Core: The Agent Loop
# ---------------------------------------------------------------------------
# 1. Collect user input, append to messages
# 2. Call the API
# 3. Check stop_reason -- "end_turn" means print, "tool_use" means dispatch
#
# Here stop_reason is always "end_turn" (no tools yet).
# Next section adds tools; the loop structure stays the same.
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """Main agent loop -- conversational REPL."""

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 01: The Agent Loop")
    print_info(f"  Model: {MODEL_ID}")
    print_info("  Type 'quit' or 'exit' to leave. Ctrl+C also works.")
    print_info("=" * 60)
    print()

    while True:
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye.{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            print(f"{DIM}Goodbye.{RESET}")
            break

        messages.append({
            "role": "user",
            "content": user_input,
        })

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

        # Check stop_reason to decide what happens next
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
            print_info("[stop_reason=tool_use] No tools in this section.")
            print_info("See s02_tool_use.py for tool support.")
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
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
