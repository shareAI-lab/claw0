r"""
Section 06: Intelligence (智能)
"赋予灵魂, 教会记忆"

每轮对话前, agent 的"大脑"是如何组装的?
本节是整个教学项目的核心集成点 -- 演示系统提示词的分层构建过程.

在 s01-s02 中, 系统提示词是硬编码的字符串.
在真实的 agent 框架中, 系统提示词由多个层级动态组装:
  Identity / 灵魂 / Tools / 技能 / Memory / Bootstrap / Runtime / Channel

架构:

    [SOUL.md]  [IDENTITY.md]  [TOOLS.md]  [MEMORY.md]  ...
         \          |            |           /
          v         v            v          v
        +-------------------------------+
        |     BootstrapLoader           |
        |  (load, truncate, cap)        |
        +-------------------------------+
                    |
                    v
        +-------------------------------+        +-------------------+
        |   build_system_prompt()       | <----> | SkillsManager     |
        |   (8 层组装)                  |        | (discover, parse) |
        +-------------------------------+        +-------------------+
                    |                                     ^
                    v                                     |
        +-------------------------------+        +-------------------+
        |   Agent Loop (每轮)           | <----> | MemoryStore       |
        |   search -> build -> call LLM |        | (write, search)   |
        +-------------------------------+        +-------------------+

用法:
    cd claw0
    python zh/s06_intelligence.py

REPL 命令:
    /soul /skills /memory /search <q> /prompt /bootstrap
"""

# ---------------------------------------------------------------------------
# 导入与配置
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

# Bootstrap 文件名 -- 每个 agent 启动时加载这 8 个文件
BOOTSTRAP_FILES = [
    "SOUL.md", "IDENTITY.md", "TOOLS.md", "USER.md",
    "HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "MEMORY.md",
]

MAX_FILE_CHARS = 20000
MAX_TOTAL_CHARS = 150000
MAX_SKILLS = 150
MAX_SKILLS_PROMPT = 30000

# ---------------------------------------------------------------------------
# ANSI 颜色
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
# 1. Bootstrap 文件加载器
# ---------------------------------------------------------------------------
# 在 agent 启动时加载工作区的 Bootstrap 文件.
# 不同加载模式 (full/minimal/none) 适用于不同场景:
#   full = 主 agent | minimal = 子 agent / cron | none = 最小化

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
        """截断超长文件内容. 仅保留头部, 在行边界处截断."""
        if len(content) <= max_chars:
            return content
        cut = content.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        return content[:cut] + f"\n\n[... truncated ({len(content)} chars total, showing first {cut}) ...]"

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
# 2. 灵魂系统
# ---------------------------------------------------------------------------
# SOUL.md 定义 agent 的人格. 不同 agent 可以有不同的 SOUL.md 文件.
# 注入到系统提示词的靠前位置 -- 越靠前影响力越强.


def load_soul(workspace_dir: Path) -> str:
    path = workspace_dir / "SOUL.md"
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# 3. 技能发现与注入
# ---------------------------------------------------------------------------
# 一个技能 = 一个包含 SKILL.md (带 frontmatter) 的目录.
# 按优先级顺序扫描; 同名技能会被后发现的覆盖.


class SkillsManager:

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir
        self.skills: list[dict[str, str]] = []

    def _parse_frontmatter(self, text: str) -> dict[str, str]:
        """解析简单的 YAML frontmatter, 不依赖 pyyaml."""
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
        """按优先级扫描技能目录; 同名技能后者覆盖前者."""
        scan_order: list[Path] = []
        if extra_dirs:
            scan_order.extend(extra_dirs)
        scan_order.append(self.workspace_dir / "skills")           # 内置技能
        scan_order.append(self.workspace_dir / ".skills")          # 托管技能
        scan_order.append(self.workspace_dir / ".agents" / "skills")  # 个人 agent 技能
        scan_order.append(Path.cwd() / ".agents" / "skills")      # 项目 agent 技能
        scan_order.append(Path.cwd() / "skills")                  # 工作区技能

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
                lines.append(f"(... more skills truncated)")
                break
            lines.append(block)
            total += len(block)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. 记忆系统
# ---------------------------------------------------------------------------
# 两层存储:
#   MEMORY.md = 长期事实 (手动维护)
#   daily/{date}.jsonl = 每日日志 (通过 agent 工具自动写入)
# 搜索: TF-IDF + 余弦相似度, 纯 Python 实现


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
            return f"Memory saved to {today}.jsonl ({category})"
        except Exception as exc:
            return f"Error writing memory: {exc}"

    def load_evergreen(self) -> str:
        path = self.workspace_dir / "MEMORY.md"
        if not path.is_file():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _load_all_chunks(self) -> list[dict[str, str]]:
        """加载所有记忆并拆分为块 (path + text)."""
        chunks: list[dict[str, str]] = []
        # 按段落拆分长期记忆
        evergreen = self.load_evergreen()
        if evergreen:
            for para in evergreen.split("\n\n"):
                para = para.strip()
                if para:
                    chunks.append({"path": "MEMORY.md", "text": para})
        # 每日记忆: 每条 JSONL 记录作为一个块
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
        """分词: 小写英文 + 单个 CJK 字符, 过滤短 token."""
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())
        return [t for t in tokens if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]

    def search_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """TF-IDF + 余弦相似度搜索, 纯 Python 实现."""
        chunks = self._load_all_chunks()
        if not chunks:
            return []
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        chunk_tokens = [self._tokenize(c["text"]) for c in chunks]

        # 文档频率
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
# 记忆工具: memory_write + memory_search
# ---------------------------------------------------------------------------

memory_store = MemoryStore(WORKSPACE_DIR)


def tool_memory_write(content: str, category: str = "general") -> str:
    print_tool("memory_write", f"[{category}] {content[:60]}...")
    return memory_store.write_memory(content, category)


def tool_memory_search(query: str, top_k: int = 5) -> str:
    print_tool("memory_search", query)
    results = memory_store.search_memory(query, top_k)
    if not results:
        return "No relevant memories found."
    return "\n".join(f"[{r['path']}] (score: {r['score']}) {r['snippet']}" for r in results)


# ---------------------------------------------------------------------------
# 工具定义: Schema + Handler
# ---------------------------------------------------------------------------
# 工具 schema 设计说明:
#
# 每个章节 (s02, s06 等) 为了教学清晰度定义了自己的工具集.
# 在生产环境中, 工具 schema 会从共享注册表继承/组合.
#
# s06 中的工具 (memory_write, memory_search) 是对 s02 工具
# (bash, read_file, write_file, edit_file) 的补充 -- 而非替代.
# 完整的 agent 会将两组工具合并为一个列表传递给 LLM.

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
        return f"Error: Unknown tool '{tool_name}'"
    try:
        return handler(**tool_input)
    except TypeError as exc:
        return f"Error: Invalid arguments for {tool_name}: {exc}"
    except Exception as exc:
        return f"Error: {tool_name} failed: {exc}"


# ---------------------------------------------------------------------------
# 5. 系统提示词组装 -- 核心函数
# ---------------------------------------------------------------------------
# 教学演示 8 个关键提示词层级.
# 每轮重建 -- 上一轮可能更新了记忆.
# 模式: full (主 agent) / minimal (子 agent / cron) / none (最小化)


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

    # 第 1 层: 身份 -- 来自 IDENTITY.md 或默认值
    identity = bootstrap.get("IDENTITY.md", "").strip()
    sections.append(identity if identity else "You are a helpful personal AI assistant.")

    # 第 2 层: 灵魂 -- 人格注入, 越靠前影响力越强
    if mode == "full":
        soul = bootstrap.get("SOUL.md", "").strip()
        if soul:
            sections.append(f"## Personality\n\n{soul}")

    # 第 3 层: 工具使用指南
    tools_md = bootstrap.get("TOOLS.md", "").strip()
    if tools_md:
        sections.append(f"## Tool Usage Guidelines\n\n{tools_md}")

    # 第 4 层: 技能
    if mode == "full" and skills_block:
        sections.append(skills_block)

    # 第 5 层: 记忆 -- 长期记忆 + 本轮自动搜索结果
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

    # 第 6 层: Bootstrap 上下文 -- 剩余的 Bootstrap 文件
    if mode in ("full", "minimal"):
        for name in ["HEARTBEAT.md", "BOOTSTRAP.md", "AGENTS.md", "USER.md"]:
            content = bootstrap.get(name, "").strip()
            if content:
                sections.append(f"## {name.replace('.md', '')}\n\n{content}")

    # 第 7 层: 运行时上下文
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        f"## Runtime Context\n\n"
        f"- Agent ID: {agent_id}\n- Model: {MODEL_ID}\n"
        f"- Channel: {channel}\n- Current time: {now}\n- Prompt mode: {mode}"
    )

    # 第 8 层: 渠道提示
    hints = {
        "terminal": "You are responding via a terminal REPL. Markdown is supported.",
        "telegram": "You are responding via Telegram. Keep messages concise.",
        "discord": "You are responding via Discord. Keep messages under 2000 characters.",
        "slack": "You are responding via Slack. Use Slack mrkdwn formatting.",
    }
    sections.append(f"## Channel\n\n{hints.get(channel, f'You are responding via {channel}.')}")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# 6. Agent 循环 + REPL
# ---------------------------------------------------------------------------

def handle_repl_command(
    cmd: str,
    bootstrap_data: dict[str, str],
    skills_mgr: SkillsManager,
    skills_block: str,
) -> bool:
    """处理 REPL 斜杠命令. 返回 True 表示已处理."""
    parts = cmd.strip().split(maxsplit=1)
    command = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if command == "/soul":
        print_section("SOUL.md")
        soul = bootstrap_data.get("SOUL.md", "")
        print(soul if soul else f"{DIM}(未找到 SOUL.md){RESET}")
        return True

    if command == "/skills":
        print_section("已发现的技能")
        if not skills_mgr.skills:
            print(f"{DIM}(未找到技能){RESET}")
        else:
            for s in skills_mgr.skills:
                print(f"  {BLUE}{s['invocation']}{RESET}  {s['name']} - {s['description']}")
                print(f"    {DIM}path: {s['path']}{RESET}")
        return True

    if command == "/memory":
        print_section("记忆统计")
        stats = memory_store.get_stats()
        print(f"  长期记忆 (MEMORY.md): {stats['evergreen_chars']} 字符")
        print(f"  每日文件: {stats['daily_files']}")
        print(f"  每日条目: {stats['daily_entries']}")
        return True

    if command == "/search":
        if not arg:
            print(f"{YELLOW}用法: /search <query>{RESET}")
            return True
        print_section(f"记忆搜索: {arg}")
        results = memory_store.search_memory(arg)
        if not results:
            print(f"{DIM}(无结果){RESET}")
        else:
            for r in results:
                color = GREEN if r["score"] > 0.3 else DIM
                print(f"  {color}[{r['score']:.4f}]{RESET} {r['path']}")
                print(f"    {r['snippet']}")
        return True

    if command == "/prompt":
        print_section("完整系统提示词")
        prompt = build_system_prompt(
            mode="full", bootstrap=bootstrap_data,
            skills_block=skills_block, memory_context=_auto_recall("show prompt"),
        )
        if len(prompt) > 3000:
            print(prompt[:3000])
            print(f"\n{DIM}... ({len(prompt) - 3000} more chars, total {len(prompt)}){RESET}")
        else:
            print(prompt)
        print(f"\n{DIM}提示词总长度: {len(prompt)} 字符{RESET}")
        return True

    if command == "/bootstrap":
        print_section("Bootstrap 文件")
        if not bootstrap_data:
            print(f"{DIM}(未加载 Bootstrap 文件){RESET}")
        else:
            for name, content in bootstrap_data.items():
                print(f"  {BLUE}{name}{RESET}: {len(content)} chars")
        total = sum(len(v) for v in bootstrap_data.values())
        print(f"\n  {DIM}总计: {total} 字符 (上限: {MAX_TOTAL_CHARS}){RESET}")
        return True

    return False


def _auto_recall(user_message: str) -> str:
    """根据用户消息自动搜索相关记忆, 注入到系统提示词中."""
    results = memory_store.search_memory(user_message, top_k=3)
    if not results:
        return ""
    return "\n".join(f"- [{r['path']}] {r['snippet']}" for r in results)


def agent_loop() -> None:
    # 启动阶段: 加载 Bootstrap 文件, 发现技能 (技能仅在启动时发现一次)
    loader = BootstrapLoader(WORKSPACE_DIR)
    bootstrap_data = loader.load_all(mode="full")

    skills_mgr = SkillsManager(WORKSPACE_DIR)
    skills_mgr.discover()
    skills_block = skills_mgr.format_prompt_block()

    messages: list[dict] = []

    print_info("=" * 60)
    print_info("  claw0  |  Section 06: Intelligence")
    print_info(f"  Model: {MODEL_ID}")
    print_info(f"  Workspace: {WORKSPACE_DIR}")
    print_info(f"  Bootstrap 文件: {len(bootstrap_data)}")
    print_info(f"  已发现技能: {len(skills_mgr.skills)}")
    stats = memory_store.get_stats()
    print_info(f"  记忆: 长期 {stats['evergreen_chars']}字符, {stats['daily_files']} 个每日文件")
    print_info("  命令: /soul /skills /memory /search /prompt /bootstrap")
    print_info("  输入 'quit' 或 'exit' 退出.")
    print_info("=" * 60)
    print()

    while True:
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

        # REPL 命令
        if user_input.startswith("/"):
            if handle_repl_command(user_input, bootstrap_data, skills_mgr, skills_block):
                continue

        # 自动记忆搜索 -- 将相关记忆注入系统提示词
        memory_context = _auto_recall(user_input)
        if memory_context:
            print_info("  [自动召回] 找到相关记忆")

        # 每轮重建系统提示词 (记忆可能在上一轮被更新)
        system_prompt = build_system_prompt(
            mode="full", bootstrap=bootstrap_data,
            skills_block=skills_block, memory_context=memory_context,
        )

        messages.append({"role": "user", "content": user_input})

        # Agent 内循环: 处理连续的工具调用直到 end_turn
        while True:
            try:
                response = client.messages.create(
                    model=MODEL_ID, max_tokens=8096,
                    system=system_prompt, tools=TOOLS, messages=messages,
                )
            except Exception as exc:
                print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
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
# 入口
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print(f"{YELLOW}错误: 未设置 ANTHROPIC_API_KEY.{RESET}")
        print(f"{DIM}将 .env.example 复制为 .env 并填入你的密钥.{RESET}")
        sys.exit(1)
    if not WORKSPACE_DIR.is_dir():
        print(f"{YELLOW}错误: 未找到工作区目录: {WORKSPACE_DIR}{RESET}")
        print(f"{DIM}请从 claw0 项目根目录运行.{RESET}")
        sys.exit(1)
    agent_loop()


if __name__ == "__main__":
    main()
