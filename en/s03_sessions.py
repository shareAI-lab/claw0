"""
Section 03: Sessions & Context Guard
"Sessions are JSONL files. Append on write, replay on read. When too big, summarize."

Two layers around the same agent loop:

  SessionStore -- JSONL persistence (append on write, replay on read)
  ContextGuard -- 3-stage overflow retry:
    try normal -> truncate tool results -> compact history (50%) -> fail

    User Input
        |
    load_session() --> rebuild messages[] from JSONL
        |
    guard_api_call() --> try -> truncate -> compact -> raise
        |
    save_turn() --> append to JSONL
        |
    Print response

Usage:
    cd claw0
    python en/s03_sessions.py

Required .env:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import sys
import json
import uuid
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

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

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "Use tools to help the user with file and time queries.\n"
    "Be concise. If a session has prior context, use it."
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"

CONTEXT_SAFE_LIMIT = 180000

MAX_TOOL_OUTPUT = 50000

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def print_warn(text: str) -> None:
    print(f"{YELLOW}{text}{RESET}")


def print_session(text: str) -> None:
    print(f"{MAGENTA}{text}{RESET}")


# ---------------------------------------------------------------------------
# Safe path helper
# ---------------------------------------------------------------------------

def safe_path(raw: str) -> Path:
    """Resolve path, block traversal outside WORKSPACE_DIR."""
    target = (WORKSPACE_DIR / raw).resolve()
    if not str(target).startswith(str(WORKSPACE_DIR.resolve())):
        raise ValueError(f"Path traversal blocked: {raw}")
    return target


# ---------------------------------------------------------------------------
# SessionStore -- JSONL-based conversation persistence
# ---------------------------------------------------------------------------
# Mental model: each session is a .jsonl file. Every event (user message,
# assistant response, tool call, tool result) is one JSON line. To restore
# a session, read lines and rebuild the API messages array.
# ---------------------------------------------------------------------------


class SessionStore:
    """Manages persistent storage for agent sessions."""

    def __init__(self, agent_id: str = "default"):
        self.agent_id = agent_id
        self.base_dir = WORKSPACE_DIR / ".sessions" / "agents" / agent_id / "sessions"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir.parent / "sessions.json"
        self._index: dict[str, dict] = self._load_index()
        self.current_session_id: str | None = None

    def _load_index(self) -> dict[str, dict]:
        if self.index_path.exists():
            try:
                return json.loads(self.index_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_index(self) -> None:
        self.index_path.write_text(
            json.dumps(self._index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _session_path(self, session_id: str) -> Path:
        return self.base_dir / f"{session_id}.jsonl"

    def create_session(self, label: str = "") -> str:
        session_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        self._index[session_id] = {
            "label": label,
            "created_at": now,
            "last_active": now,
            "message_count": 0,
        }
        self._save_index()
        self._session_path(session_id).touch()
        self.current_session_id = session_id
        return session_id

    def load_session(self, session_id: str) -> list[dict]:
        """Rebuild API-format messages[] from JSONL."""
        path = self._session_path(session_id)
        if not path.exists():
            return []
        self.current_session_id = session_id
        return self._rebuild_history(path)

    def save_turn(self, role: str, content: Any) -> None:
        if not self.current_session_id:
            return
        self.append_transcript(self.current_session_id, {
            "type": role,
            "content": content,
            "ts": time.time(),
        })

    def save_tool_result(self, tool_use_id: str, name: str,
                         tool_input: dict, result: str) -> None:
        if not self.current_session_id:
            return
        ts = time.time()
        self.append_transcript(self.current_session_id, {
            "type": "tool_use",
            "tool_use_id": tool_use_id,
            "name": name,
            "input": tool_input,
            "ts": ts,
        })
        self.append_transcript(self.current_session_id, {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": result,
            "ts": ts,
        })

    def append_transcript(self, session_id: str, record: dict) -> None:
        path = self._session_path(session_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if session_id in self._index:
            self._index[session_id]["last_active"] = (
                datetime.now(timezone.utc).isoformat()
            )
            self._index[session_id]["message_count"] += 1
            self._save_index()

    def _rebuild_history(self, path: Path) -> list[dict]:
        """
        Rebuild API-format messages from JSONL lines.
        Messages must alternate user/assistant; tool_use blocks go in
        assistant messages, tool_result blocks go in user messages.
        """
        messages: list[dict] = []
        lines = path.read_text(encoding="utf-8").strip().split("\n")

        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            rtype = record.get("type")

            if rtype == "user":
                messages.append({
                    "role": "user",
                    "content": record["content"],
                })

            elif rtype == "assistant":
                content = record["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                messages.append({
                    "role": "assistant",
                    "content": content,
                })

            elif rtype == "tool_use":
                block = {
                    "type": "tool_use",
                    "id": record["tool_use_id"],
                    "name": record["name"],
                    "input": record["input"],
                }
                if messages and messages[-1]["role"] == "assistant":
                    content = messages[-1]["content"]
                    if isinstance(content, list):
                        content.append(block)
                    else:
                        messages[-1]["content"] = [
                            {"type": "text", "text": str(content)},
                            block,
                        ]
                else:
                    messages.append({
                        "role": "assistant",
                        "content": [block],
                    })

            elif rtype == "tool_result":
                result_block = {
                    "type": "tool_result",
                    "tool_use_id": record["tool_use_id"],
                    "content": record["content"],
                }
                if (messages and messages[-1]["role"] == "user"
                        and isinstance(messages[-1]["content"], list)
                        and messages[-1]["content"]
                        and isinstance(messages[-1]["content"][0], dict)
                        and messages[-1]["content"][0].get("type") == "tool_result"):
                    messages[-1]["content"].append(result_block)
                else:
                    messages.append({
                        "role": "user",
                        "content": [result_block],
                    })

        return messages

    def list_sessions(self) -> list[tuple[str, dict]]:
        items = list(self._index.items())
        items.sort(key=lambda x: x[1].get("last_active", ""), reverse=True)
        return items


def _serialize_messages_for_summary(messages: list[dict]) -> str:
    """Flatten messages to plain text for LLM summarization."""
    parts: list[str] = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(f"[{role}]: {block['text']}")
                    elif btype == "tool_use":
                        parts.append(
                            f"[{role} called {block.get('name', '?')}]: "
                            f"{json.dumps(block.get('input', {}), ensure_ascii=False)}"
                        )
                    elif btype == "tool_result":
                        rc = block.get("content", "")
                        preview = rc[:500] if isinstance(rc, str) else str(rc)[:500]
                        parts.append(f"[tool_result]: {preview}")
                elif hasattr(block, "text"):
                    parts.append(f"[{role}]: {block.text}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# ContextGuard -- context overflow protection
# ---------------------------------------------------------------------------
# Three stages:
#   1. Truncate oversized tool results (head-only)
#   2. Compact old messages into an LLM summary (fixed 50%)
#   3. Raise if still overflowing
# ---------------------------------------------------------------------------


class ContextGuard:
    """Protect the agent from context window overflow."""

    def __init__(self, max_tokens: int = CONTEXT_SAFE_LIMIT):
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough estimate: 1 token per 4 characters."""
        return len(text) // 4

    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if "text" in block:
                            total += self.estimate_tokens(block["text"])
                        elif block.get("type") == "tool_result":
                            rc = block.get("content", "")
                            if isinstance(rc, str):
                                total += self.estimate_tokens(rc)
                        elif block.get("type") == "tool_use":
                            total += self.estimate_tokens(
                                json.dumps(block.get("input", {}))
                            )
                    elif hasattr(block, "text"):
                        total += self.estimate_tokens(block.text)
                    elif hasattr(block, "input"):
                        total += self.estimate_tokens(json.dumps(block.input))
        return total

    def truncate_tool_result(self, result: str, max_fraction: float = 0.3) -> str:
        """Head-only truncation: keep the first max_fraction of the context budget."""
        max_chars = int(self.max_tokens * 4 * max_fraction)
        if len(result) <= max_chars:
            return result
        head = result[:max_chars]
        return head + f"\n\n[... truncated ({len(result)} chars total, showing first {len(head)}) ...]"

    def compact_history(self, messages: list[dict],
                        api_client: Anthropic, model: str) -> list[dict]:
        """
        Compress the first 50% of messages into an LLM-generated summary.
        Keep the last 20% (min 4) as-is so recent context is preserved.
        """
        total = len(messages)
        if total <= 4:
            return messages

        keep_count = max(4, int(total * 0.2))
        compress_count = max(2, int(total * 0.5))
        compress_count = min(compress_count, total - keep_count)

        if compress_count < 2:
            return messages

        old_messages = messages[:compress_count]
        recent_messages = messages[compress_count:]

        old_text = _serialize_messages_for_summary(old_messages)

        summary_prompt = (
            "Summarize the following conversation concisely, "
            "preserving key facts and decisions. "
            "Output only the summary, no preamble.\n\n"
            f"{old_text}"
        )

        try:
            summary_resp = api_client.messages.create(
                model=model,
                max_tokens=2048,
                system="You are a conversation summarizer. Be concise and factual.",
                messages=[{"role": "user", "content": summary_prompt}],
            )
            summary_text = ""
            for block in summary_resp.content:
                if hasattr(block, "text"):
                    summary_text += block.text

            print_session(
                f"  [compact] {len(old_messages)} messages -> summary "
                f"({len(summary_text)} chars)"
            )
        except Exception as exc:
            print_warn(f"  [compact] Summary failed ({exc}), dropping old messages")
            return recent_messages

        compacted = [
            {
                "role": "user",
                "content": "[Previous conversation summary]\n" + summary_text,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Understood, I have the context from our previous conversation."}],
            },
        ]
        compacted.extend(recent_messages)
        return compacted

    def _truncate_large_tool_results(self, messages: list[dict]) -> list[dict]:
        """Walk through messages and truncate oversized tool_result blocks."""
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                new_blocks = []
                for block in content:
                    if (isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and isinstance(block.get("content"), str)):
                        block = dict(block)
                        block["content"] = self.truncate_tool_result(
                            block["content"]
                        )
                    new_blocks.append(block)
                result.append({"role": msg["role"], "content": new_blocks})
            else:
                result.append(msg)
        return result

    def guard_api_call(
        self,
        api_client: Anthropic,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        max_retries: int = 2,
    ) -> Any:
        """
        3-stage retry:
          Attempt 0: normal call
          Attempt 1: truncate oversized tool results
          Attempt 2: compact history via LLM summary
        """
        current_messages = messages

        for attempt in range(max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": 8096,
                    "system": system,
                    "messages": current_messages,
                }
                if tools:
                    kwargs["tools"] = tools
                result = api_client.messages.create(**kwargs)
                if current_messages is not messages:
                    messages.clear()
                    messages.extend(current_messages)
                return result

            except Exception as exc:
                error_str = str(exc).lower()
                is_overflow = ("context" in error_str or "token" in error_str)

                if not is_overflow or attempt >= max_retries:
                    raise

                if attempt == 0:
                    print_warn(
                        "  [guard] Context overflow, truncating tool results..."
                    )
                    current_messages = self._truncate_large_tool_results(
                        current_messages
                    )
                elif attempt == 1:
                    print_warn(
                        "  [guard] Still overflowing, compacting history..."
                    )
                    current_messages = self.compact_history(
                        current_messages, api_client, model
                    )

        raise RuntimeError("guard_api_call: exhausted retries")


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def tool_read_file(file_path: str) -> str:
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        if len(content) > MAX_TOOL_OUTPUT:
            return content[:MAX_TOOL_OUTPUT] + f"\n... [truncated, {len(content)} total chars]"
        return content
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_list_directory(directory: str = ".") -> str:
    print_tool("list_directory", directory)
    try:
        target = safe_path(directory)
        if not target.exists():
            return f"Error: Directory not found: {directory}"
        if not target.is_dir():
            return f"Error: Not a directory: {directory}"
        entries = sorted(target.iterdir())
        lines = []
        for entry in entries:
            prefix = "[dir]  " if entry.is_dir() else "[file] "
            lines.append(prefix + entry.name)
        return "\n".join(lines) if lines else "[empty directory]"
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


def tool_get_current_time() -> str:
    print_tool("get_current_time", "")
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%d %H:%M:%S UTC")


# ---------------------------------------------------------------------------
# Tool schema + dispatch table
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file under the workspace directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path relative to workspace directory.",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and subdirectories in a directory under workspace.",
        "input_schema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Path relative to workspace directory. Default is root.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time in UTC.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]

TOOL_HANDLERS: dict[str, Any] = {
    "read_file": tool_read_file,
    "list_directory": tool_list_directory,
    "get_current_time": tool_get_current_time,
}


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


# ---------------------------------------------------------------------------
# REPL commands
# ---------------------------------------------------------------------------

def handle_repl_command(
    command: str,
    store: SessionStore,
    guard: ContextGuard,
    messages: list[dict],
) -> tuple[bool, list[dict]]:
    """Handle /-prefixed commands. Returns (handled, messages)."""
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd == "/new":
        label = arg or ""
        sid = store.create_session(label)
        print_session(f"  Created new session: {sid}" + (f" ({label})" if label else ""))
        return True, []

    elif cmd == "/list":
        sessions = store.list_sessions()
        if not sessions:
            print_info("  No sessions found.")
            return True, messages
        print_info("  Sessions:")
        for sid, meta in sessions:
            active = " <-- current" if sid == store.current_session_id else ""
            label = meta.get("label", "")
            label_str = f" ({label})" if label else ""
            count = meta.get("message_count", 0)
            last = meta.get("last_active", "?")[:19]
            print_info(f"    {sid}{label_str}  msgs={count}  last={last}{active}")
        return True, messages

    elif cmd == "/switch":
        if not arg:
            print_warn("  Usage: /switch <session_id>")
            return True, messages
        target_id = arg.strip()
        matched = [sid for sid in store._index if sid.startswith(target_id)]
        if len(matched) == 0:
            print_warn(f"  Session not found: {target_id}")
            return True, messages
        if len(matched) > 1:
            print_warn(f"  Ambiguous prefix, matches: {', '.join(matched)}")
            return True, messages
        sid = matched[0]
        new_messages = store.load_session(sid)
        print_session(f"  Switched to session: {sid} ({len(new_messages)} messages)")
        return True, new_messages

    elif cmd == "/context":
        estimated = guard.estimate_messages_tokens(messages)
        pct = (estimated / guard.max_tokens) * 100
        bar_len = 30
        filled = int(bar_len * min(pct, 100) / 100)
        bar = "#" * filled + "-" * (bar_len - filled)
        color = GREEN if pct < 50 else (YELLOW if pct < 80 else RED)
        print_info(f"  Context usage: ~{estimated:,} / {guard.max_tokens:,} tokens")
        print(f"  {color}[{bar}] {pct:.1f}%{RESET}")
        print_info(f"  Messages: {len(messages)}")
        return True, messages

    elif cmd == "/compact":
        if len(messages) <= 4:
            print_info("  Too few messages to compact (need > 4).")
            return True, messages
        print_session("  Compacting history...")
        new_messages = guard.compact_history(messages, client, MODEL_ID)
        print_session(f"  {len(messages)} -> {len(new_messages)} messages")
        return True, new_messages

    elif cmd == "/help":
        print_info("  Commands:")
        print_info("    /new [label]       Create a new session")
        print_info("    /list              List all sessions")
        print_info("    /switch <id>       Switch to a session (prefix match)")
        print_info("    /context           Show context token usage")
        print_info("    /compact           Manually compact conversation history")
        print_info("    /help              Show this help")
        print_info("    quit / exit        Exit the REPL")
        return True, messages

    return False, messages


# ---------------------------------------------------------------------------
# Core: Agent loop (same while True, wrapped with SessionStore + ContextGuard)
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """Main agent loop with session persistence and context protection."""

    store = SessionStore(agent_id="claw0")
    guard = ContextGuard()

    # Resume most recent session or create a new one
    sessions = store.list_sessions()
    if sessions:
        sid = sessions[0][0]
        messages = store.load_session(sid)
        print_session(f"  Resumed session: {sid} ({len(messages)} messages)")
    else:
        sid = store.create_session("initial")
        messages = []
        print_session(f"  Created initial session: {sid}")

    print_info("=" * 60)
    print_info("  claw0  |  Section 03: Sessions & Context Guard")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Session: {store.current_session_id}")
    print_info(f"  Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  Type /help for commands, quit/exit to leave.")
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

        # REPL commands
        if user_input.startswith("/"):
            handled, messages = handle_repl_command(
                user_input, store, guard, messages
            )
            if handled:
                continue

        # Append user message
        messages.append({
            "role": "user",
            "content": user_input,
        })
        store.save_turn("user", user_input)

        # Inner loop: tool call chains
        while True:
            try:
                response = guard.guard_api_call(
                    api_client=client,
                    model=MODEL_ID,
                    system=SYSTEM_PROMPT,
                    messages=messages,
                    tools=TOOLS,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Serialize content blocks for JSONL storage
            serialized_content = []
            for block in response.content:
                if hasattr(block, "text"):
                    serialized_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    serialized_content.append({
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })
            store.save_turn("assistant", serialized_content)

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
                    store.save_tool_result(
                        block.id, block.name, block.input, result
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

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
