"""
Section 09: Resilience
"When one call fails, rotate and retry."

The 3-layer retry onion wraps every agent execution. Each layer handles
a different class of failure:

    Layer 1 -- Auth Rotation: cycle through API key profiles, skip cooldowns.
    Layer 2 -- Overflow Recovery: compact messages on context overflow.
    Layer 3 -- Tool-Use Loop: the standard while True + stop_reason dispatch.

    Profiles: [main-key, backup-key, emergency-key]
         |
    for each non-cooldown profile:          LAYER 1: Auth Rotation
         |
    create client(profile.api_key)
         |
    for compact_attempt in 0..2:            LAYER 2: Overflow Recovery
         |
    _run_attempt(client, model, ...)        LAYER 3: Tool-Use Loop
         |              |
       success       exception
         |              |
    mark_success    classify_failure()
    return result       |
                   overflow? --> compact, retry Layer 2
                   auth/rate? -> mark_failure, break to Layer 1
                   timeout?  --> mark_failure(60s), break to Layer 1
                        |
                   all profiles exhausted?
                        |
                   try fallback models

Usage:
    cd claw0
    python en/s09_resilience.py

Required in .env:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to tools.\n"
    "Use tools to help the user with file operations and shell commands.\n"
    "Be concise."
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent.parent / "workspace"

# Retry limits
BASE_RETRY = 24
PER_PROFILE = 8
MAX_OVERFLOW_COMPACTION = 3

# Context guard settings
CONTEXT_SAFE_LIMIT = 180000
MAX_TOOL_OUTPUT = 50000

# ---------------------------------------------------------------------------
# ANSI colors
# ---------------------------------------------------------------------------
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"
MAGENTA = "\033[35m"
RED = "\033[31m"
BLUE = "\033[34m"
ORANGE = "\033[38;5;208m"


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def print_resilience(text: str) -> None:
    print(f"  {MAGENTA}[resilience]{RESET} {text}")


def print_warn(text: str) -> None:
    print(f"  {YELLOW}[warn]{RESET} {text}")


def print_error(text: str) -> None:
    print(f"  {RED}[error]{RESET} {text}")


def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


# ---------------------------------------------------------------------------
# 1. FailoverReason -- classify why an API call failed
# ---------------------------------------------------------------------------


class FailoverReason(Enum):
    """Each reason maps to a different retry strategy."""
    rate_limit = "rate_limit"
    auth = "auth"
    timeout = "timeout"
    billing = "billing"
    overflow = "overflow"
    unknown = "unknown"


def classify_failure(exc: Exception) -> FailoverReason:
    """Examine the exception string to determine failure category.

    The classification drives retry behavior:
      - overflow  -> compact messages and retry with the same profile
      - auth      -> skip this profile, try next
      - rate_limit -> skip this profile with cooldown, try next
      - timeout   -> short cooldown on this profile, try next
      - billing   -> skip this profile, try next
      - unknown   -> skip this profile, try next
    """
    msg = str(exc).lower()

    if "rate" in msg or "429" in msg:
        return FailoverReason.rate_limit
    if "auth" in msg or "401" in msg or "key" in msg:
        return FailoverReason.auth
    if "timeout" in msg or "timed out" in msg:
        return FailoverReason.timeout
    if "billing" in msg or "quota" in msg or "402" in msg:
        return FailoverReason.billing
    if "context" in msg or "token" in msg or "overflow" in msg:
        return FailoverReason.overflow

    return FailoverReason.unknown


# ---------------------------------------------------------------------------
# 2. AuthProfile -- one API key with cooldown tracking
# ---------------------------------------------------------------------------


@dataclass
class AuthProfile:
    """Represents a single API key that can be rotated into service.

    Fields:
      name            -- human-readable label
      provider        -- which LLM provider (e.g. "anthropic")
      api_key         -- the actual API key string
      cooldown_until  -- unix timestamp; skip this profile until then
      failure_reason  -- last failure reason string, or None if healthy
      last_good_at    -- unix timestamp of last successful call
    """
    name: str
    provider: str
    api_key: str
    cooldown_until: float = 0.0
    failure_reason: str | None = None
    last_good_at: float = 0.0


# ---------------------------------------------------------------------------
# 3. ProfileManager -- select, mark, and list profiles
# ---------------------------------------------------------------------------


class ProfileManager:
    """Manages a pool of AuthProfiles with cooldown-aware selection."""

    def __init__(self, profiles: list[AuthProfile]):
        self.profiles = profiles

    def select_profile(self) -> AuthProfile | None:
        """Return the first profile whose cooldown has expired.

        Profiles are checked in order. A profile is available if
        time.time() >= cooldown_until. Returns None if all are on cooldown.
        """
        now = time.time()
        for profile in self.profiles:
            if now >= profile.cooldown_until:
                return profile
        return None

    def select_all_available(self) -> list[AuthProfile]:
        """Return all non-cooldown profiles in order."""
        now = time.time()
        return [p for p in self.profiles if now >= p.cooldown_until]

    def mark_failure(
        self,
        profile: AuthProfile,
        reason: FailoverReason,
        cooldown_seconds: float = 300.0,
    ) -> None:
        """Put a profile on cooldown after a failure.

        Default cooldown is 5 minutes. Timeout failures use a shorter
        cooldown (caller passes cooldown_seconds=60).
        """
        profile.cooldown_until = time.time() + cooldown_seconds
        profile.failure_reason = reason.value
        print_resilience(
            f"Profile '{profile.name}' -> cooldown {cooldown_seconds:.0f}s "
            f"(reason: {reason.value})"
        )

    def mark_success(self, profile: AuthProfile) -> None:
        """Clear failure state and record last success time."""
        profile.failure_reason = None
        profile.last_good_at = time.time()

    def list_profiles(self) -> list[dict[str, Any]]:
        """Return status of all profiles for display."""
        now = time.time()
        result = []
        for p in self.profiles:
            remaining = max(0, p.cooldown_until - now)
            status = "available" if remaining == 0 else f"cooldown ({remaining:.0f}s)"
            result.append({
                "name": p.name,
                "provider": p.provider,
                "status": status,
                "failure_reason": p.failure_reason,
                "last_good": (
                    time.strftime("%H:%M:%S", time.localtime(p.last_good_at))
                    if p.last_good_at > 0 else "never"
                ),
            })
        return result


# ---------------------------------------------------------------------------
# 4. Simplified ContextGuard (inline, from the s03 pattern)
# ---------------------------------------------------------------------------
# Provides token estimation and message compaction. The full ContextGuard
# lives in s03_sessions.py; here we only need the compaction path that
# Layer 2 of the retry onion uses on overflow errors.
# ---------------------------------------------------------------------------


class ContextGuard:
    """Lightweight context overflow protection for the resilience runner."""

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

    def truncate_tool_results(self, messages: list[dict]) -> list[dict]:
        """Truncate oversized tool_result blocks to reduce context usage."""
        max_chars = int(self.max_tokens * 4 * 0.3)
        result = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                new_blocks = []
                for block in content:
                    if (isinstance(block, dict)
                            and block.get("type") == "tool_result"
                            and isinstance(block.get("content"), str)
                            and len(block["content"]) > max_chars):
                        block = dict(block)
                        original_len = len(block["content"])
                        block["content"] = (
                            block["content"][:max_chars]
                            + f"\n\n[... truncated ({original_len} chars total, "
                            f"showing first {max_chars}) ...]"
                        )
                    new_blocks.append(block)
                result.append({"role": msg["role"], "content": new_blocks})
            else:
                result.append(msg)
        return result

    def compact_history(
        self,
        messages: list[dict],
        api_client: Anthropic,
        model: str,
    ) -> list[dict]:
        """Compress the first 50% of messages into an LLM-generated summary.

        Keeps the last 20% (min 4) of messages intact so recent context
        is preserved.
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

        # Flatten old messages to plain text for summarization
        parts: list[str] = []
        for msg in old_messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"[{role}]: {content}")
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            parts.append(f"[{role}]: {block['text']}")
                        elif block.get("type") == "tool_use":
                            parts.append(
                                f"[{role} called {block.get('name', '?')}]: "
                                f"{json.dumps(block.get('input', {}), ensure_ascii=False)}"
                            )
                        elif block.get("type") == "tool_result":
                            rc = block.get("content", "")
                            preview = rc[:500] if isinstance(rc, str) else str(rc)[:500]
                            parts.append(f"[tool_result]: {preview}")
                    elif hasattr(block, "text"):
                        parts.append(f"[{role}]: {block.text}")
        old_text = "\n".join(parts)

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

            print_resilience(
                f"Compacted {len(old_messages)} messages -> summary "
                f"({len(summary_text)} chars)"
            )
        except Exception as exc:
            print_warn(f"Summary failed ({exc}), dropping old messages")
            return recent_messages

        compacted = [
            {
                "role": "user",
                "content": "[Previous conversation summary]\n" + summary_text,
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Understood, I have the context from our previous conversation."}
                ],
            },
        ]
        compacted.extend(recent_messages)
        return compacted


# ---------------------------------------------------------------------------
# 5. Tool definitions (simplified bash + read_file from s02)
# ---------------------------------------------------------------------------


WORKDIR = Path.cwd()


def safe_path(raw: str) -> Path:
    """Resolve path, block traversal outside WORKDIR."""
    target = (WORKDIR / raw).resolve()
    if not str(target).startswith(str(WORKDIR)):
        raise ValueError(f"Path traversal blocked: {raw} resolves outside WORKDIR")
    return target


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"


def tool_bash(command: str, timeout: int = 30) -> str:
    """Run a shell command and return its output."""
    dangerous = ["rm -rf /", "mkfs", "> /dev/sd", "dd if="]
    for pattern in dangerous:
        if pattern in command:
            return f"Error: Refused to run dangerous command containing '{pattern}'"

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
            output += f"\n[exit code: {result.returncode}]"
        return truncate(output) if output else "[no output]"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as exc:
        return f"Error: {exc}"


def tool_read_file(file_path: str) -> str:
    """Read the contents of a file."""
    print_tool("read_file", file_path)
    try:
        target = safe_path(file_path)
        if not target.exists():
            return f"Error: File not found: {file_path}"
        if not target.is_file():
            return f"Error: Not a file: {file_path}"
        content = target.read_text(encoding="utf-8")
        return truncate(content)
    except ValueError as exc:
        return str(exc)
    except Exception as exc:
        return f"Error: {exc}"


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
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
}


def process_tool_call(tool_name: str, tool_input: dict) -> str:
    """Look up handler by name, call it with the input kwargs."""
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
# 6. SimulatedFailure -- lets users trigger specific error types
# ---------------------------------------------------------------------------
# The REPL command /simulate-failure <reason> sets this flag so the next
# API call raises a synthetic error. This lets users observe how the
# 3-layer onion handles each failure class without needing real failures.
# ---------------------------------------------------------------------------


class SimulatedFailure:
    """Holds a pending simulated failure that fires on the next API call."""

    TEMPLATES: dict[str, str] = {
        "rate_limit": "Error code: 429 -- rate limit exceeded",
        "auth": "Error code: 401 -- authentication failed, invalid API key",
        "timeout": "Request timed out after 30s",
        "billing": "Error code: 402 -- billing quota exceeded",
        "overflow": "Error: context window token overflow, too many tokens",
        "unknown": "Error: unexpected internal server error",
    }

    def __init__(self) -> None:
        self._pending: str | None = None

    def arm(self, reason: str) -> str:
        """Arm a failure for the next API call. Returns confirmation message."""
        if reason not in self.TEMPLATES:
            return (
                f"Unknown reason '{reason}'. "
                f"Valid: {', '.join(self.TEMPLATES.keys())}"
            )
        self._pending = reason
        return f"Armed: next API call will fail with '{reason}'"

    def check_and_fire(self) -> None:
        """If armed, raise the simulated error and disarm."""
        if self._pending is not None:
            reason = self._pending
            self._pending = None
            raise RuntimeError(self.TEMPLATES[reason])

    @property
    def is_armed(self) -> bool:
        return self._pending is not None

    @property
    def pending_reason(self) -> str | None:
        return self._pending


# ---------------------------------------------------------------------------
# 7. ResilienceRunner -- the 3-layer retry onion
# ---------------------------------------------------------------------------
# This is the core of the resilience system. It wraps every agent execution
# in three nested retry layers:
#
#   Layer 1 (outermost): iterate through API key profiles, skipping any
#           on cooldown. If a profile fails with auth/rate/timeout, mark
#           it and move to the next.
#
#   Layer 2 (middle): on context overflow errors, compact the message
#           history and retry up to MAX_OVERFLOW_COMPACTION times.
#
#   Layer 3 (innermost): the standard tool-use loop (while True + stop_reason).
#           This is what s01/s02 teach -- it runs until end_turn or error.
#
# If all profiles are exhausted, try fallback models with the first
# available profile. If everything fails, raise RuntimeError.
# ---------------------------------------------------------------------------


class ResilienceRunner:
    """Execute an agent turn with automatic failover, compaction, and retries."""

    def __init__(
        self,
        profile_manager: ProfileManager,
        model_id: str,
        fallback_models: list[str] | None = None,
        context_guard: ContextGuard | None = None,
        simulated_failure: SimulatedFailure | None = None,
    ):
        self.profile_manager = profile_manager
        self.model_id = model_id
        self.fallback_models = fallback_models or []
        self.guard = context_guard or ContextGuard()
        self.simulated_failure = simulated_failure

        num_profiles = len(profile_manager.profiles)
        self.max_iterations = min(
            max(BASE_RETRY + PER_PROFILE * num_profiles, 32),
            160,
        )

        # Stats
        self.total_attempts = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_compactions = 0
        self.total_rotations = 0

    def run(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[Any, list[dict]]:
        """Execute the 3-layer retry onion.

        Returns (final_response, updated_messages).
        Raises RuntimeError if all profiles and fallbacks are exhausted.
        """
        current_messages = list(messages)
        profiles_tried: set[str] = set()

        # ---- LAYER 1: Auth Rotation ----
        # Iterate through available profiles. On auth/rate/timeout failures,
        # mark the profile and try the next one.
        for _rotation in range(len(self.profile_manager.profiles)):
            profile = self.profile_manager.select_profile()
            if profile is None:
                print_warn("All profiles on cooldown")
                break
            if profile.name in profiles_tried:
                break
            profiles_tried.add(profile.name)

            if len(profiles_tried) > 1:
                self.total_rotations += 1
                print_resilience(
                    f"Rotating to profile '{profile.name}'"
                )

            api_client = Anthropic(
                api_key=profile.api_key,
                base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
            )

            # ---- LAYER 2: Overflow Recovery ----
            # On context overflow, compact messages and retry.
            layer2_messages = list(current_messages)
            for compact_attempt in range(MAX_OVERFLOW_COMPACTION):
                try:
                    self.total_attempts += 1

                    # Check simulated failure before real API call
                    if self.simulated_failure:
                        self.simulated_failure.check_and_fire()

                    # ---- LAYER 3: Tool-Use Loop ----
                    result, layer2_messages = self._run_attempt(
                        api_client, self.model_id, system,
                        layer2_messages, tools,
                    )
                    self.profile_manager.mark_success(profile)
                    self.total_successes += 1
                    return result, layer2_messages

                except Exception as exc:
                    reason = classify_failure(exc)
                    self.total_failures += 1

                    if reason == FailoverReason.overflow:
                        if compact_attempt < MAX_OVERFLOW_COMPACTION - 1:
                            self.total_compactions += 1
                            print_resilience(
                                f"Context overflow (attempt {compact_attempt + 1}/"
                                f"{MAX_OVERFLOW_COMPACTION}), compacting..."
                            )
                            # Stage 1: truncate tool results
                            layer2_messages = self.guard.truncate_tool_results(
                                layer2_messages
                            )
                            # Stage 2: compact history via LLM summary
                            layer2_messages = self.guard.compact_history(
                                layer2_messages, api_client, self.model_id,
                            )
                            continue
                        else:
                            print_error(
                                f"Overflow not resolved after "
                                f"{MAX_OVERFLOW_COMPACTION} compaction attempts"
                            )
                            self.profile_manager.mark_failure(
                                profile, reason, cooldown_seconds=600
                            )
                            break

                    elif reason in (FailoverReason.auth, FailoverReason.billing):
                        self.profile_manager.mark_failure(
                            profile, reason, cooldown_seconds=300
                        )
                        break  # try next profile

                    elif reason == FailoverReason.rate_limit:
                        self.profile_manager.mark_failure(
                            profile, reason, cooldown_seconds=120
                        )
                        break  # try next profile

                    elif reason == FailoverReason.timeout:
                        self.profile_manager.mark_failure(
                            profile, reason, cooldown_seconds=60
                        )
                        break  # try next profile

                    else:
                        # Unknown failure -- cooldown and try next
                        self.profile_manager.mark_failure(
                            profile, reason, cooldown_seconds=120
                        )
                        break

        # ---- Fallback models ----
        # All primary profiles exhausted. Try fallback models with any
        # available profile (cooldowns may have expired during retries).
        if self.fallback_models:
            print_resilience("Primary profiles exhausted, trying fallback models...")
            for fallback_model in self.fallback_models:
                profile = self.profile_manager.select_profile()
                if profile is None:
                    # Try resetting cooldowns for fallback attempt
                    for p in self.profile_manager.profiles:
                        if p.failure_reason in (
                            FailoverReason.rate_limit.value,
                            FailoverReason.timeout.value,
                        ):
                            p.cooldown_until = 0.0
                    profile = self.profile_manager.select_profile()
                if profile is None:
                    continue

                print_resilience(
                    f"Fallback: model='{fallback_model}', profile='{profile.name}'"
                )

                api_client = Anthropic(
                    api_key=profile.api_key,
                    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
                )

                try:
                    self.total_attempts += 1
                    if self.simulated_failure:
                        self.simulated_failure.check_and_fire()

                    result, updated = self._run_attempt(
                        api_client, fallback_model, system,
                        current_messages, tools,
                    )
                    self.profile_manager.mark_success(profile)
                    self.total_successes += 1
                    return result, updated

                except Exception as exc:
                    reason = classify_failure(exc)
                    self.total_failures += 1
                    print_warn(
                        f"Fallback model '{fallback_model}' failed: "
                        f"{reason.value} -- {exc}"
                    )
                    continue

        raise RuntimeError(
            "All profiles and fallback models exhausted. "
            f"Tried {len(profiles_tried)} profiles, "
            f"{len(self.fallback_models)} fallback models."
        )

    def _run_attempt(
        self,
        api_client: Anthropic,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> tuple[Any, list[dict]]:
        """Layer 3: the standard tool-use loop.

        Runs the while True + stop_reason pattern from s01/s02.
        Returns (final_response, updated_messages) on end_turn.
        Re-raises any API exception for the outer layers to handle.
        """
        current_messages = list(messages)
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            response = api_client.messages.create(
                model=model,
                max_tokens=8096,
                system=system,
                tools=tools,
                messages=current_messages,
            )

            current_messages.append({
                "role": "assistant",
                "content": response.content,
            })

            if response.stop_reason == "end_turn":
                return response, current_messages

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
                current_messages.append({
                    "role": "user",
                    "content": tool_results,
                })
                continue

            else:
                # Unexpected stop_reason (e.g. max_tokens) -- treat as end_turn
                return response, current_messages

        raise RuntimeError(
            f"Tool-use loop exceeded {self.max_iterations} iterations"
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "total_compactions": self.total_compactions,
            "total_rotations": self.total_rotations,
            "max_iterations": self.max_iterations,
        }


# ---------------------------------------------------------------------------
# 8. REPL commands
# ---------------------------------------------------------------------------


def handle_repl_command(
    cmd: str,
    profile_manager: ProfileManager,
    runner: ResilienceRunner,
    sim_failure: SimulatedFailure,
) -> bool:
    """Handle REPL commands. Returns True if command was handled."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if command == "/profiles":
        profiles = profile_manager.list_profiles()
        print_info("  Profiles:")
        for p in profiles:
            status_color = GREEN if p["status"] == "available" else YELLOW
            print_info(
                f"    {p['name']:16s} "
                f"{status_color}{p['status']:20s}{RESET} "
                f"last_good={p['last_good']}"
                + (f"  failure={p['failure_reason']}" if p["failure_reason"] else "")
            )
        return True

    if command == "/cooldowns":
        now = time.time()
        any_active = False
        print_info("  Active cooldowns:")
        for p in profile_manager.profiles:
            remaining = max(0, p.cooldown_until - now)
            if remaining > 0:
                any_active = True
                print_info(
                    f"    {p.name}: {remaining:.0f}s remaining "
                    f"(reason: {p.failure_reason or 'unknown'})"
                )
        if not any_active:
            print_info("    No active cooldowns.")
        return True

    if command == "/simulate-failure":
        if not arg:
            valid = ", ".join(SimulatedFailure.TEMPLATES.keys())
            print_info(f"  Usage: /simulate-failure <reason>")
            print_info(f"  Valid reasons: {valid}")
            if sim_failure.is_armed:
                print_info(f"  Currently armed: {sim_failure.pending_reason}")
            return True
        result = sim_failure.arm(arg)
        print_resilience(result)
        return True

    if command == "/fallback":
        if runner.fallback_models:
            print_info("  Fallback model chain:")
            for i, model in enumerate(runner.fallback_models, 1):
                print_info(f"    {i}. {model}")
        else:
            print_info("  No fallback models configured.")
        print_info(f"  Primary model: {runner.model_id}")
        return True

    if command == "/stats":
        stats = runner.get_stats()
        print_info("  Resilience stats:")
        print_info(f"    Attempts:    {stats['total_attempts']}")
        print_info(f"    Successes:   {stats['total_successes']}")
        print_info(f"    Failures:    {stats['total_failures']}")
        print_info(f"    Compactions: {stats['total_compactions']}")
        print_info(f"    Rotations:   {stats['total_rotations']}")
        print_info(f"    Max iter:    {stats['max_iterations']}")
        return True

    if command == "/context":
        # Display is handled via guard; we don't have messages here, so
        # this is a no-op hint. The actual context display is in the main loop.
        return False

    if command == "/help":
        print_info("  Commands:")
        print_info("    /profiles               Show all profiles with status")
        print_info("    /cooldowns              Show active cooldowns")
        print_info("    /simulate-failure <r>   Arm a simulated failure for next call")
        print_info("    /fallback               Show fallback model chain")
        print_info("    /stats                  Show resilience statistics")
        print_info("    /help                   Show this help")
        print_info("    quit / exit             Exit the REPL")
        return True

    return False


# ---------------------------------------------------------------------------
# 9. Agent loop + REPL
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """Main agent loop with resilience runner."""

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ANTHROPIC_BASE_URL") or None

    # Create demo profiles. In production, each would have a different key.
    # Here we use the same key for all three to keep the demo self-contained.
    profiles = [
        AuthProfile(
            name="main-key",
            provider="anthropic",
            api_key=api_key,
        ),
        AuthProfile(
            name="backup-key",
            provider="anthropic",
            api_key=api_key,
        ),
        AuthProfile(
            name="emergency-key",
            provider="anthropic",
            api_key=api_key,
        ),
    ]

    profile_manager = ProfileManager(profiles)
    sim_failure = SimulatedFailure()
    guard = ContextGuard()

    # Fallback models: try smaller/cheaper models if the primary fails
    fallback_models = [
        "claude-haiku-4-20250514",
    ]

    runner = ResilienceRunner(
        profile_manager=profile_manager,
        model_id=MODEL_ID,
        fallback_models=fallback_models,
        context_guard=guard,
        simulated_failure=sim_failure,
    )

    messages: list[dict] = []

    print_info("=" * 64)
    print_info("  claw0  |  Section 09: Resilience")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Profiles: {', '.join(p.name for p in profiles)}")
    print_info(f"  Fallback: {', '.join(fallback_models) or 'none'}")
    print_info(f"  Tools: {', '.join(TOOL_HANDLERS.keys())}")
    print_info("  Commands:")
    print_info("    /profiles               Show all profiles")
    print_info("    /cooldowns              Show active cooldowns")
    print_info("    /simulate-failure <r>   Arm simulated failure")
    print_info("    /fallback               Show fallback chain")
    print_info("    /stats                  Resilience statistics")
    print_info("    /help                   All commands")
    print_info("  Type 'quit' or 'exit' to leave.")
    print_info("=" * 64)
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
            if handle_repl_command(
                user_input, profile_manager, runner, sim_failure
            ):
                continue
            print_info(f"  Unknown command: {user_input}")
            continue

        # Append user message
        messages.append({"role": "user", "content": user_input})

        # Run through the 3-layer retry onion
        try:
            response, messages = runner.run(
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=TOOLS,
            )

            # Extract and print assistant text
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text
            if assistant_text:
                print_assistant(assistant_text)

        except RuntimeError as exc:
            print_error(str(exc))
            # Roll back the failed user message
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()

        except Exception as exc:
            print(f"\n{YELLOW}Unexpected error: {exc}{RESET}\n")
            while messages and messages[-1]["role"] != "user":
                messages.pop()
            if messages:
                messages.pop()


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
