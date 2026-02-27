"""
Section 09: レジリエンス
「1つの呼び出しが失敗したら、ローテーションしてリトライ」

3層リトライオニオンが全てのエージェント実行をラップする。各層は
異なるクラスの障害を処理する:

    Layer 1 -- 認証ローテーション: APIキープロファイルを巡回、クールダウンをスキップ。
    Layer 2 -- オーバーフロー回復: コンテキストオーバーフロー時にメッセージを圧縮。
    Layer 3 -- ツール使用ループ: 標準的な while True + stop_reason ディスパッチ。

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

使い方:
    cd claw0
    python ja/s09_resilience.py

.env に必要な設定:
    ANTHROPIC_API_KEY=sk-ant-xxxxx
    MODEL_ID=claude-sonnet-4-20250514
"""

# ---------------------------------------------------------------------------
# インポート
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
# 設定
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
# ANSI カラー
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
# 1. FailoverReason -- API 呼び出し失敗の理由を分類
# ---------------------------------------------------------------------------


class FailoverReason(Enum):
    """各理由が異なるリトライ戦略にマッピングされる。"""
    rate_limit = "rate_limit"
    auth = "auth"
    timeout = "timeout"
    billing = "billing"
    overflow = "overflow"
    unknown = "unknown"


def classify_failure(exc: Exception) -> FailoverReason:
    """例外文字列を調べて障害カテゴリを判定する。

    分類結果がリトライ動作を決定する:
      - overflow  -> メッセージを圧縮して同じプロファイルでリトライ
      - auth      -> このプロファイルをスキップし、次へ
      - rate_limit -> クールダウン付きでこのプロファイルをスキップし、次へ
      - timeout   -> 短いクールダウンでこのプロファイルをスキップし、次へ
      - billing   -> このプロファイルをスキップし、次へ
      - unknown   -> このプロファイルをスキップし、次へ
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
# 2. AuthProfile -- クールダウン追跡付きの1つの API キー
# ---------------------------------------------------------------------------


@dataclass
class AuthProfile:
    """サービスにローテーション投入できる単一の API キーを表す。

    フィールド:
      name            -- 人間が読めるラベル
      provider        -- LLM プロバイダー (例: "anthropic")
      api_key         -- 実際の API キー文字列
      cooldown_until  -- Unix タイムスタンプ; この時刻までプロファイルをスキップ
      failure_reason  -- 最後の障害理由文字列、正常なら None
      last_good_at    -- 最後に成功した呼び出しの Unix タイムスタンプ
    """
    name: str
    provider: str
    api_key: str
    cooldown_until: float = 0.0
    failure_reason: str | None = None
    last_good_at: float = 0.0


# ---------------------------------------------------------------------------
# 3. ProfileManager -- プロファイルの選択、マーク、一覧表示
# ---------------------------------------------------------------------------


class ProfileManager:
    """クールダウンを考慮した AuthProfile プールの管理。"""

    def __init__(self, profiles: list[AuthProfile]):
        self.profiles = profiles

    def select_profile(self) -> AuthProfile | None:
        """クールダウンが切れた最初のプロファイルを返す。

        プロファイルは順番にチェックされる。time.time() >= cooldown_until
        であれば利用可能。全てクールダウン中なら None を返す。
        """
        now = time.time()
        for profile in self.profiles:
            if now >= profile.cooldown_until:
                return profile
        return None

    def select_all_available(self) -> list[AuthProfile]:
        """クールダウンしていない全プロファイルを順番に返す。"""
        now = time.time()
        return [p for p in self.profiles if now >= p.cooldown_until]

    def mark_failure(
        self,
        profile: AuthProfile,
        reason: FailoverReason,
        cooldown_seconds: float = 300.0,
    ) -> None:
        """障害後にプロファイルをクールダウン状態にする。

        デフォルトのクールダウンは5分。タイムアウト障害は短い
        クールダウンを使用 (呼び出し元が cooldown_seconds=60 を渡す)。
        """
        profile.cooldown_until = time.time() + cooldown_seconds
        profile.failure_reason = reason.value
        print_resilience(
            f"Profile '{profile.name}' -> cooldown {cooldown_seconds:.0f}s "
            f"(reason: {reason.value})"
        )

    def mark_success(self, profile: AuthProfile) -> None:
        """障害状態をクリアし、最後の成功時刻を記録する。"""
        profile.failure_reason = None
        profile.last_good_at = time.time()

    def list_profiles(self) -> list[dict[str, Any]]:
        """表示用に全プロファイルのステータスを返す。"""
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
# 4. 簡易版 ContextGuard (インライン、s03 パターンより)
# ---------------------------------------------------------------------------
# トークン推定とメッセージ圧縮を提供する。完全版の ContextGuard は
# s03_sessions.py にある。ここではリトライオニオンの Layer 2 が
# オーバーフローエラー時に使用する圧縮パスのみ必要。
# ---------------------------------------------------------------------------


class ContextGuard:
    """レジリエンスランナー用の軽量コンテキストオーバーフロー保護。"""

    def __init__(self, max_tokens: int = CONTEXT_SAFE_LIMIT):
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """概算: 4文字につき1トークン。"""
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
        """サイズ超過の tool_result ブロックを切り詰めてコンテキスト使用量を削減する。"""
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
        """最初の50%のメッセージを LLM 生成の要約に圧縮する。

        最後の20% (最低4件) のメッセージはそのまま保持し、
        直近のコンテキストを維持する。
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
# 5. ツール定義 (s02 からの簡易版 bash + read_file)
# ---------------------------------------------------------------------------


WORKDIR = Path.cwd()


def safe_path(raw: str) -> Path:
    """パスを解決し、WORKDIR 外へのトラバーサルをブロックする。"""
    target = (WORKDIR / raw).resolve()
    if not str(target).startswith(str(WORKDIR)):
        raise ValueError(f"Path traversal blocked: {raw} resolves outside WORKDIR")
    return target


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"


def tool_bash(command: str, timeout: int = 30) -> str:
    """シェルコマンドを実行し、出力を返す。"""
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
    """ファイルの内容を読み取る。"""
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
    """名前でハンドラを検索し、入力 kwargs で呼び出す。"""
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
# 6. SimulatedFailure -- 特定のエラータイプをトリガーする機能
# ---------------------------------------------------------------------------
# REPL コマンド /simulate-failure <reason> がこのフラグを設定し、次の
# API 呼び出しで合成エラーを発生させる。実際の障害なしに3層オニオンが
# 各障害クラスをどう処理するかを観察できる。
# ---------------------------------------------------------------------------


class SimulatedFailure:
    """次の API 呼び出しで発火する保留中のシミュレート障害を保持する。"""

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
        """次の API 呼び出し用に障害を準備する。確認メッセージを返す。"""
        if reason not in self.TEMPLATES:
            return (
                f"Unknown reason '{reason}'. "
                f"Valid: {', '.join(self.TEMPLATES.keys())}"
            )
        self._pending = reason
        return f"Armed: next API call will fail with '{reason}'"

    def check_and_fire(self) -> None:
        """準備済みの場合、シミュレートエラーを発生させて解除する。"""
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
# 7. ResilienceRunner -- 3層リトライオニオン
# ---------------------------------------------------------------------------
# レジリエンスシステムの中核。全てのエージェント実行を3つのネストされた
# リトライ層でラップする:
#
#   Layer 1 (最外層): API キープロファイルを順次試行。クールダウン中の
#           ものはスキップ。auth/rate/timeout 障害でマークして次へ。
#
#   Layer 2 (中間層): コンテキストオーバーフローエラー時に、メッセージ
#           履歴を圧縮し MAX_OVERFLOW_COMPACTION 回までリトライ。
#
#   Layer 3 (最内層): 標準的なツール使用ループ (while True + stop_reason)。
#           s01/s02 で学ぶ内容 -- end_turn またはエラーまで実行。
#
# 全プロファイルが使い切られた場合、最初の利用可能なプロファイルで
# フォールバックモデルを試行。全て失敗した場合は RuntimeError を送出。
# ---------------------------------------------------------------------------


class ResilienceRunner:
    """自動フェイルオーバー、圧縮、リトライ付きでエージェントターンを実行する。"""

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
        """3層リトライオニオンを実行する。

        (final_response, updated_messages) を返す。
        全プロファイルとフォールバックが使い切られた場合は RuntimeError を送出。
        """
        current_messages = list(messages)
        profiles_tried: set[str] = set()

        # ---- LAYER 1: Auth Rotation ----
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
            layer2_messages = list(current_messages)
            for compact_attempt in range(MAX_OVERFLOW_COMPACTION):
                try:
                    self.total_attempts += 1

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
                            layer2_messages = self.guard.truncate_tool_results(
                                layer2_messages
                            )
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
                        break

                    elif reason == FailoverReason.rate_limit:
                        self.profile_manager.mark_failure(
                            profile, reason, cooldown_seconds=120
                        )
                        break

                    elif reason == FailoverReason.timeout:
                        self.profile_manager.mark_failure(
                            profile, reason, cooldown_seconds=60
                        )
                        break

                    else:
                        self.profile_manager.mark_failure(
                            profile, reason, cooldown_seconds=120
                        )
                        break

        # ---- Fallback models ----
        if self.fallback_models:
            print_resilience("Primary profiles exhausted, trying fallback models...")
            for fallback_model in self.fallback_models:
                profile = self.profile_manager.select_profile()
                if profile is None:
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
        """Layer 3: 標準的なツール使用ループ。

        s01/s02 の while True + stop_reason パターンを実行する。
        end_turn で (final_response, updated_messages) を返す。
        API 例外は外側の層が処理するために再送出される。
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
# 8. REPL コマンド
# ---------------------------------------------------------------------------


def handle_repl_command(
    cmd: str,
    profile_manager: ProfileManager,
    runner: ResilienceRunner,
    sim_failure: SimulatedFailure,
) -> bool:
    """REPL コマンドを処理する。コマンドが処理された場合は True を返す。"""
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
# 9. エージェントループ + REPL
# ---------------------------------------------------------------------------


def agent_loop() -> None:
    """レジリエンスランナー付きのメインエージェントループ。"""

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    base_url = os.getenv("ANTHROPIC_BASE_URL") or None

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
# エントリーポイント
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}Error: ANTHROPIC_API_KEY not set.{RESET}")
        print(f"{DIM}Copy .env.example to .env and fill in your key.{RESET}")
        sys.exit(1)

    agent_loop()


if __name__ == "__main__":
    main()
