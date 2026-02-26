r"""
Section 06: インテリジェンス
「魂を与え、記憶を教える」

システムプロンプトはディスク上のファイルから組み立てられる。
ファイルを入れ替えれば性格が変わる。コード変更は不要。

    [SOUL.md]  [IDENTITY.md]  [TOOLS.md]  [MEMORY.md]  ...
         \          |            |           /
        +-------------------------------+
        |     BootstrapLoader           |
        +-------------------------------+
                    |
        +-------------------------------+        +-------------------+
        |   build_system_prompt()       | <----> | SkillsManager     |
        +-------------------------------+        +-------------------+
                    |                                     ^
                    v                                     |
        +-------------------------------+        +-------------------+
        |   エージェントループ (ターンごと) | <----> | MemoryStore       |
        |   検索 -> 構築 -> LLM 呼び出し  |        | (書き込み, 検索)   |
        +-------------------------------+        +-------------------+

使い方:
    cd claw0
    python ja/s06_intelligence.py

REPL コマンド:
    /soul /skills /memory /search <q> /prompt /bootstrap
"""

# ---------------------------------------------------------------------------
# インポートと設定
# ---------------------------------------------------------------------------
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent / "workspace"

BOOTSTRAP_FILES = [
    "SOUL.md", "IDENTITY.md", "TOOLS.md", "USER.md",
    "HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "MEMORY.md",
]

MAX_FILE_CHARS = 20000
MAX_TOTAL_CHARS = 150000
MAX_SKILLS = 150
MAX_SKILLS_PROMPT = 30000

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


def colored_prompt() -> str:
    return f"{CYAN}{BOLD}You > {RESET}"


def print_assistant(text: str) -> None:
    print(f"\n{GREEN}{BOLD}Assistant:{RESET} {text}\n")


def print_tool(name: str, detail: str) -> None:
    print(f"  {DIM}[tool: {name}] {detail}{RESET}")


def print_info(text: str) -> None:
    print(f"{DIM}{text}{RESET}")


def print_section(title: str) -> None:
    print(f"\n{MAGENTA}{BOLD}--- {title} ---{RESET}")


# ---------------------------------------------------------------------------
# 1. ブートストラップファイルローダー
# ---------------------------------------------------------------------------
# エージェント起動時にワークスペースのブートストラップファイルを読み込む。
# ロードモード: full (メインエージェント) | minimal (サブエージェント / cron) | none (素の状態)

class BootstrapLoader:

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir

    def load_file(self, name: str) -> str:
        path = self.workspace_dir / name
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def truncate_file(self, content: str, max_chars: int = MAX_FILE_CHARS) -> str:
        if len(content) <= max_chars:
            return content
        cut = content.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        return content[:cut] + f"\n\n[... 切り詰め (合計 {len(content)} 文字、先頭 {cut} 文字を表示) ...]"

    def load_all(self, mode: str = "full") -> dict[str, str]:
        if mode == "none":
            return {}
        names = ["AGENTS.md", "TOOLS.md"] if mode == "minimal" else list(BOOTSTRAP_FILES)
        result: dict[str, str] = {}
        total = 0
        for name in names:
            raw = self.load_file(name)
            if not raw:
                continue
            truncated = self.truncate_file(raw)
            if total + len(truncated) > MAX_TOTAL_CHARS:
                remaining = MAX_TOTAL_CHARS - total
                if remaining > 0:
                    truncated = self.truncate_file(raw, remaining)
                else:
                    break
            result[name] = truncated
            total += len(truncated)
        return result


# ---------------------------------------------------------------------------
# 2. ソウルシステム
# ---------------------------------------------------------------------------
# SOUL.md はエージェントの性格を定義する。システムプロンプトの早い位置に
# 注入される -- 早い位置ほど行動への影響が強い。


def load_soul(workspace_dir: Path) -> str:
    path = workspace_dir / "SOUL.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 3. スキル発見とインジェクション
# ---------------------------------------------------------------------------
# スキル = SKILL.md (フロントマター付き) を含むディレクトリ。
# 優先順位に従ってスキャンされ、同名のスキルは後から見つかったもので上書きされる。


class SkillsManager:

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.skills: list[dict[str, str]] = []

    def _parse_frontmatter(self, text: str) -> dict[str, str]:
        """pyyaml に依存せず、シンプルな YAML フロントマターを解析する。"""
        meta: dict[str, str] = {}
        if not text.startswith("---"):
            return meta
        parts = text.split("---", 2)
        if len(parts) < 3:
            return meta
        for line in parts[1].strip().splitlines():
            if ":" not in line:
                continue
            key, _, value = line.strip().partition(":")
            meta[key.strip()] = value.strip()
        return meta

    def _scan_dir(self, base: Path) -> list[dict[str, str]]:
        found: list[dict[str, str]] = []
        if not base.is_dir():
            return found
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.is_file():
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
            except Exception:
                continue
            meta = self._parse_frontmatter(content)
            if not meta.get("name"):
                continue
            body = ""
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    body = parts[2].strip()
            found.append({
                "name": meta.get("name", ""),
                "description": meta.get("description", ""),
                "invocation": meta.get("invocation", ""),
                "body": body,
                "path": str(child),
            })
        return found

    def discover(self, extra_dirs: list[Path] | None = None) -> None:
        """優先順位に従ってスキルディレクトリをスキャンする。同名のスキルは後から見つかったもので上書き。"""
        scan_order: list[Path] = []
        if extra_dirs:
            scan_order.extend(extra_dirs)
        scan_order.append(self.workspace_dir / "skills")
        scan_order.append(self.workspace_dir / ".skills")
        scan_order.append(self.workspace_dir / ".agents" / "skills")
        scan_order.append(Path.cwd() / ".agents" / "skills")
        scan_order.append(Path.cwd() / "skills")

        seen: dict[str, dict[str, str]] = {}
        for d in scan_order:
            for skill in self._scan_dir(d):
                seen[skill["name"]] = skill
        self.skills = list(seen.values())[:MAX_SKILLS]

    def format_prompt_block(self) -> str:
        if not self.skills:
            return ""
        lines = ["## Available Skills", ""]
        total = 0
        for skill in self.skills:
            block = (
                f"### Skill: {skill['name']}\n"
                f"Description: {skill['description']}\n"
                f"Invocation: {skill['invocation']}\n"
            )
            if skill.get("body"):
                block += f"\n{skill['body']}\n"
            block += "\n"
            if total + len(block) > MAX_SKILLS_PROMPT:
                lines.append(f"(... 以降のスキルは切り詰め)")
                break
            lines.append(block)
            total += len(block)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. メモリシステム
# ---------------------------------------------------------------------------
# 2層ストレージ:
#   MEMORY.md = 恒久的な事実 (手動で管理)
#   daily/{date}.jsonl = 日次ログ (エージェントツール経由で自動記録)
# 検索: TF-IDF + コサイン類似度、純 Python 実装


class MemoryStore:

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.memory_dir = workspace_dir / "memory" / "daily"
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    def write_memory(self, content: str, category: str = "general") -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = self.memory_dir / f"{today}.jsonl"
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "category": category,
            "content": content,
        }
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return f"メモリを {today}.jsonl に保存しました ({category})"
        except Exception as exc:
            return f"メモリ書き込みエラー: {exc}"

    def load_evergreen(self) -> str:
        path = self.workspace_dir / "MEMORY.md"
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _load_all_chunks(self) -> list[dict[str, str]]:
        """全てのメモリを読み込み、チャンク (パス + テキスト) に分割する。"""
        chunks: list[dict[str, str]] = []
        evergreen = self.load_evergreen()
        if evergreen:
            for para in evergreen.split("\n\n"):
                para = para.strip()
                if para:
                    chunks.append({"path": "MEMORY.md", "text": para})
        if self.memory_dir.is_dir():
            for jf in sorted(self.memory_dir.glob("*.jsonl")):
                try:
                    for line in jf.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        text = entry.get("content", "")
                        if text:
                            cat = entry.get("category", "")
                            label = f"{jf.name} [{cat}]" if cat else jf.name
                            chunks.append({"path": label, "text": text})
                except Exception:
                    continue
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """小文字英語 + 個別の CJK 文字、短いトークンを除外。"""
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())
        return [t for t in tokens if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]

    def search_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """TF-IDF + コサイン類似度による検索。"""
        chunks = self._load_all_chunks()
        if not chunks:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        chunk_tokens = [self._tokenize(c["text"]) for c in chunks]

        # 文書頻度
        df: dict[str, int] = {}
        for tokens in chunk_tokens:
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        n = len(chunks)

        def tfidf(tokens: list[str]) -> dict[str, float]:
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            return {t: c * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1) for t, c in tf.items()}

        def cosine(a: dict[str, float], b: dict[str, float]) -> float:
            common = set(a) & set(b)
            if not common:
                return 0.0
            dot = sum(a[k] * b[k] for k in common)
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return dot / (na * nb) if na and nb else 0.0

        qvec = tfidf(query_tokens)
        scored: list[dict[str, Any]] = []
        for i, tokens in enumerate(chunk_tokens):
            if not tokens:
                continue
            score = cosine(qvec, tfidf(tokens))
            if score > 0.0:
                snippet = chunks[i]["text"]
                if len(snippet) > 200:
                    snippet = snippet[:200] + "..."
                scored.append({"path": chunks[i]["path"], "score": round(score, 4), "snippet": snippet})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def get_stats(self) -> dict[str, Any]:
        evergreen = self.load_evergreen()
        daily_files = list(self.memory_dir.glob("*.jsonl")) if self.memory_dir.is_dir() else []
        total_entries = 0
        for f in daily_files:
            try:
                total_entries += sum(1 for line in f.read_text(encoding="utf-8").splitlines() if line.strip())
            except Exception:
                pass
        return {"evergreen_chars": len(evergreen), "daily_files": len(daily_files), "daily_entries": total_entries}


# ---------------------------------------------------------------------------
# メモリツール: memory_write + memory_search
# ---------------------------------------------------------------------------

memory_store = MemoryStore(WORKSPACE_DIR)


def tool_memory_write(content: str, category: str = "general") -> str:
    print_tool("memory_write", f"[{category}] {content[:60]}...")
    return memory_store.write_memory(content, category)


def tool_memory_search(query: str, top_k: int = 5) -> str:
    print_tool("memory_search", query)
    results = memory_store.search_memory(query, top_k)
    if not results:
        return "関連するメモリが見つかりません。"
    return "\n".join(f"[{r['path']}] (score: {r['score']}) {r['snippet']}" for r in results)


# ---------------------------------------------------------------------------
# ツール定義: スキーマ + ハンドラー
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "memory_write",
        "description": (
            "Save an important fact or observation to long-term memory. "
            "Use when you learn something worth remembering about the user or context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The fact or observation to remember."},
                "category": {"type": "string", "description": "Category: preference, fact, context, etc."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search stored memories for relevant information, ranked by similarity.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {"type": "integer", "description": "Max results. Default: 5."},
            },
            "required": ["query"],
        },
    },
]

TOOL_HANDLERS: dict[str, Any] = {
    "memory_write": tool_memory_write,
    "memory_search": tool_memory_search,
}


def process_tool_call(tool_name: str, tool_input: dict) -> str:
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
# 5. システムプロンプトの組み立て
# ---------------------------------------------------------------------------
# ターンごとに再構築 -- 前のターンでメモリが更新されている可能性がある。


def build_system_prompt(
    mode: str = "full",
    bootstrap: dict[str, str] | None = None,
    skills_block: str = "",
    memory_context: str = "",
    agent_id: str = "main",
    channel: str = "terminal",
) -> str:
    if bootstrap is None:
        bootstrap = {}
    sections: list[str] = []

    # レイヤー 1: アイデンティティ
    identity = bootstrap.get("IDENTITY.md", "").strip()
    sections.append(identity if identity else "You are a helpful personal AI assistant.")

    # レイヤー 2: ソウル -- 性格の注入、早いほど影響が強い
    if mode == "full":
        soul = bootstrap.get("SOUL.md", "").strip()
        if soul:
            sections.append(f"## Personality\n\n{soul}")

    # レイヤー 3: ツールガイダンス
    tools_md = bootstrap.get("TOOLS.md", "").strip()
    if tools_md:
        sections.append(f"## Tool Usage Guidelines\n\n{tools_md}")

    # レイヤー 4: スキル
    if mode == "full" and skills_block:
        sections.append(skills_block)

    # レイヤー 5: メモリ -- 恒久的 + このターンの自動検索結果
    if mode == "full":
        mem_md = bootstrap.get("MEMORY.md", "").strip()
        parts: list[str] = []
        if mem_md:
            parts.append(f"### Evergreen Memory\n\n{mem_md}")
        if memory_context:
            parts.append(f"### Recalled Memories (auto-searched)\n\n{memory_context}")
        if parts:
            sections.append("## Memory\n\n" + "\n\n".join(parts))
        sections.append(
            "## Memory Instructions\n\n"
            "- Use memory_write to save important user facts and preferences.\n"
            "- Reference remembered facts naturally in conversation.\n"
            "- Use memory_search to recall specific past information."
        )

    # レイヤー 6: ブートストラップコンテキスト -- 残りのブートストラップファイル
    if mode in ("full", "minimal"):
        for name in ["HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "USER.md"]:
            content = bootstrap.get(name, "").strip()
            if content:
                sections.append(f"## {name.replace('.md', '')}\n\n{content}")

    # レイヤー 7: ランタイムコンテキスト
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        f"## Runtime Context\n\n"
        f"- Agent ID: {agent_id}\n- Model: {MODEL_ID}\n"
        f"- Channel: {channel}\n- Current time: {now}\n- Prompt mode: {mode}"
    )

    # レイヤー 8: チャネルヒント
    hints = {
        "terminal": "You are responding via a terminal REPL. Markdown is supported.",
        "telegram": "You are responding via Telegram. Keep messages concise.",
        "discord": "You are responding via Discord. Keep messages under 2000 characters.",
        "slack": "You are responding via Slack. Use Slack mrkdwn formatting.",
    }
    sections.append(f"## Channel\n\n{hints.get(channel, f'You are responding via {channel}.')}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 6. エージェントループ + REPL
# ---------------------------------------------------------------------------

def handle_repl_command(
    cmd: str,
    bootstrap_data: dict[str, str],
    skills_mgr: SkillsManager,
    skills_block: str,
) -> bool:
    """REPL スラッシュコマンドを処理する。処理した場合は True を返す。"""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/soul":
        print_section("SOUL.md")
        soul = bootstrap_data.get("SOUL.md", "")
        print(soul if soul else f"{DIM}(SOUL.md が見つかりません){RESET}")
        return True

    if command == "/skills":
        print_section("発見されたスキル")
        if not skills_mgr.skills:
            print(f"{DIM}(スキルが見つかりません){RESET}")
        else:
            for s in skills_mgr.skills:
                print(f"  {BLUE}{s['invocation']}{RESET}  {s['name']} - {s['description']}")
                print(f"    {DIM}パス: {s['path']}{RESET}")
        return True

    if command == "/memory":
        print_section("メモリ統計")
        stats = memory_store.get_stats()
        print(f"  恒久的 (MEMORY.md): {stats['evergreen_chars']} 文字")
        print(f"  日次ファイル: {stats['daily_files']}")
        print(f"  日次エントリ: {stats['daily_entries']}")
        return True

    if command == "/search":
        if not arg:
            print(f"{YELLOW}使い方: /search <クエリ>{RESET}")
            return True
        print_section(f"メモリ検索: {arg}")
        results = memory_store.search_memory(arg)
        if not results:
            print(f"{DIM}(結果なし){RESET}")
        else:
            for r in results:
                color = GREEN if r["score"] > 0.3 else DIM
                print(f"  {color}[{r['score']:.4f}]{RESET} {r['path']}")
                print(f"    {r['snippet']}")
        return True

    if command == "/prompt":
        print_section("完全なシステムプロンプト")
        prompt = build_system_prompt(
            mode="full", bootstrap=bootstrap_data,
            skills_block=skills_block, memory_context=_auto_recall("show prompt"),
        )
        if len(prompt) > 3000:
            print(prompt[:3000])
            print(f"\n{DIM}... (残り {len(prompt) - 3000} 文字、合計 {len(prompt)}){RESET}")
        else:
            print(prompt)
        print(f"\n{DIM}プロンプト合計長: {len(prompt)} 文字{RESET}")
        return True

    if command == "/bootstrap":
        print_section("ブートストラップファイル")
        if not bootstrap_data:
            print(f"{DIM}(ブートストラップファイルが読み込まれていません){RESET}")
        else:
            for name, content in bootstrap_data.items():
                print(f"  {BLUE}{name}{RESET}: {len(content)} 文字")
        total = sum(len(v) for v in bootstrap_data.values())
        print(f"\n  {DIM}合計: {total} 文字 (上限: {MAX_TOTAL_CHARS}){RESET}")
        return True

    return False


def _auto_recall(user_message: str) -> str:
    """ユーザーメッセージに基づいて関連メモリを自動検索し、システムプロンプトに注入する。"""
    results = memory_store.search_memory(user_message, top_k=3)
    if not results:
        return ""
    return "\n".join(f"- [{r['path']}] {r['snippet']}" for r in results)


def agent_loop() -> None:
    loader = BootstrapLoader(WORKSPACE_DIR)
    bootstrap_data = loader.load_all(mode="full")

    skills_mgr = SkillsManager(WORKSPACE_DIR)
    skills_mgr.discover()
    skills_block = skills_mgr.format_prompt_block()

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 06: インテリジェンス")
    print_info(f"  モデル: {MODEL_ID}")
    print_info(f"  ワークスペース: {WORKSPACE_DIR}")
    print_info(f"  ブートストラップファイル: {len(bootstrap_data)}")
    print_info(f"  発見されたスキル: {len(skills_mgr.skills)}")
    stats = memory_store.get_stats()
    print_info(f"  メモリ: 恒久的 {stats['evergreen_chars']}文字, {stats['daily_files']} 日次ファイル")
    print_info("  コマンド: /soul /skills /memory /search /prompt /bootstrap")
    print_info("  'quit' または 'exit' で終了。")
    print_info("=" * 60)
    print()

    while True:
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

        if user_input.startswith("/"):
            if handle_repl_command(user_input, bootstrap_data, skills_mgr, skills_block):
                continue

        # 自動メモリ検索 -- 関連メモリをシステムプロンプトに注入
        memory_context = _auto_recall(user_input)
        if memory_context:
            print_info("  [自動想起] 関連メモリが見つかりました")

        # ターンごとにシステムプロンプトを再構築 (メモリが更新されている可能性がある)
        system_prompt = build_system_prompt(
            mode="full", bootstrap=bootstrap_data,
            skills_block=skills_block, memory_context=memory_context,
        )

        messages.append({"role": "user", "content": user_input})

        # エージェント内部ループ: end_turn まで連続するツール呼び出しを処理
        while True:
            try:
                response = client.messages.create(
                    model=MODEL_ID, max_tokens=8096,
                    system=system_prompt, tools=TOOLS, messages=messages,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API エラー: {exc}{RESET}\n")
                while messages and messages[-1]["role"] != "user":
                    messages.pop()
                if messages:
                    messages.pop()
                break

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                if text:
                    print_assistant(text)
                break
            elif response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    result = process_tool_call(block.name, block.input)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})
                messages.append({"role": "user", "content": tool_results})
                continue
            else:
                print_info(f"[stop_reason={response.stop_reason}]")
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                if text:
                    print_assistant(text)
                break


# ---------------------------------------------------------------------------
# エントリーポイント
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}エラー: ANTHROPIC_API_KEY が設定されていません。{RESET}")
        print(f"{DIM}.env.example を .env にコピーして API キーを記入してください。{RESET}")
        sys.exit(1)
    if not WORKSPACE_DIR.is_dir():
        print(f"{YELLOW}エラー: ワークスペースディレクトリが見つかりません: {WORKSPACE_DIR}{RESET}")
        print(f"{DIM}claw0 プロジェクトルートから実行してください。{RESET}")
        sys.exit(1)
    agent_loop()


if __name__ == "__main__":
    main()
