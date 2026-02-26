[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# claw0

**From Zero to One: Build an AI Agent Gateway**

> 8 progressive sections, each a runnable Python file.
> 3 languages (English, Chinese, Japanese) -- code + docs co-located.

---

## What is this?

A teaching repository that builds a minimal AI agent gateway from scratch, section by section. 8 sections, 8 mental models, ~5,000 lines of Python. Each section gives you one clear "aha" moment -- after all 8, you can read OpenClaw's production codebase and feel at home.

```sh
s01: Agent Loop           -- The foundation: while + stop_reason
s02: Tool Use             -- Give the model hands: dispatch table
s03: Sessions & Context   -- Persist conversations, handle overflow
s04: Channels             -- Telegram + Feishu: real channel pipelines
s05: Gateway & Routing    -- 5-tier binding, session isolation
s06: Intelligence         -- Soul, memory, skills, prompt assembly
s07: Heartbeat & Cron     -- Proactive agent + scheduled tasks
s08: Delivery             -- Reliable message queue with backoff
```

## Architecture

```
+------------------- claw0 layers -------------------+
|                                                     |
|  s08: Delivery     (write-ahead queue, backoff)     |
|  s07: Heartbeat    (lane lock, cron scheduler)      |
|  s06: Intelligence (8-layer prompt, TF-IDF memory)  |
|  s05: Gateway      (WebSocket, 5-tier routing)      |
|  s04: Channels     (Telegram pipeline, Feishu hook) |
|  s03: Sessions     (JSONL persistence, 3-stage retry)|
|  s02: Tools        (dispatch table, 4 tools)        |
|  s01: Agent Loop   (while True + stop_reason)       |
|                                                     |
+-----------------------------------------------------+
```

## Section Dependencies

```
s01 --> s02 --> s03 --> s04 --> s05
                 |               |
                 v               v
                s06 ----------> s07 --> s08
```

- s01-s02: Foundation (no dependencies)
- s03: Builds on s02 (adds persistence to the tool loop)
- s04: Builds on s03 (channels produce InboundMessages for sessions)
- s05: Builds on s04 (routes channel messages to agents)
- s06: Builds on s03 (uses sessions for context, adds prompt layers)
- s07: Builds on s06 (heartbeat uses soul/memory for prompt)
- s08: Builds on s07 (heartbeat output flows through delivery queue)

## Quick Start

```sh
# 1. Clone and enter
git clone https://github.com/anthropics/claw0.git && cd claw0

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY and MODEL_ID

# 4. Run any section (pick your language)
python en/s01_agent_loop.py    # English
python zh/s01_agent_loop.py    # Chinese
python ja/s01_agent_loop.py    # Japanese
```

## Learning Path

```
Phase 1: FOUNDATION     Phase 2: CONNECTIVITY     Phase 3: BRAIN        Phase 4: AUTONOMY
+----------------+      +-------------------+     +-----------------+   +-----------------+
| s01: Loop      |      | s03: Sessions     |     | s06: Intelligence|  | s07: Heartbeat  |
| s02: Tools     | ---> | s04: Channels     | --> |   soul, memory, | ->|   & Cron        |
|                |      | s05: Gateway      |     |   skills, prompt |  | s08: Delivery   |
+----------------+      +-------------------+     +-----------------+   +-----------------+
 while + dispatch        persist + route            personality + recall  proactive + reliable
```

## Section Details

| # | Section | Mental Model | Lines |
|---|---------|-------------|-------|
| 01 | Agent Loop | `while True` + `stop_reason` -- that's an agent | ~175 |
| 02 | Tool Use | Tools = schema dict + handler map. Model picks a name, you look it up | ~445 |
| 03 | Sessions | JSONL: append on write, replay on read. Too big? Summarize old parts | ~890 |
| 04 | Channels | Every platform differs, but they all produce the same `InboundMessage` | ~780 |
| 05 | Gateway | Binding table maps (channel, peer) to agent. Most specific wins | ~625 |
| 06 | Intelligence | System prompt = files on disk. Swap files, change personality | ~750 |
| 07 | Heartbeat & Cron | Timer thread: "should I run?" + queue work alongside user messages | ~660 |
| 08 | Delivery | Write to disk first, then send. Crashes can't lose messages | ~870 |

## Repository Structure

```
claw0/
  README.md              English README
  README.zh.md           Chinese README
  README.ja.md           Japanese README
  .env.example           Configuration template
  requirements.txt       Python dependencies
  workspace/             Shared workspace samples
    SOUL.md  IDENTITY.md  TOOLS.md  USER.md
    HEARTBEAT.md  BOOTSTRAP.md  AGENTS.md  MEMORY.md
    CRON.json
    skills/example-skill/SKILL.md
  en/                    English (code + docs)
    s01_agent_loop.py    s01_agent_loop.md
    s02_tool_use.py      s02_tool_use.md
    ...                  (8 .py + 8 .md)
  zh/                    Chinese (code + docs)
    s01_agent_loop.py    s01_agent_loop.md
    ...                  (8 .py + 8 .md)
  ja/                    Japanese (code + docs)
    s01_agent_loop.py    s01_agent_loop.md
    ...                  (8 .py + 8 .md)
```

Each language folder is self-contained: runnable Python code + documentation side by side. Code logic is identical across languages; comments and docs differ.

## Prerequisites

- Python 3.11+
- An API key for Anthropic (or compatible provider)

## Dependencies

```
anthropic>=0.39.0
python-dotenv>=1.0.0
websockets>=12.0
croniter>=2.0.0
python-telegram-bot>=21.0
httpx>=0.27.0
```

## Related Projects

- **[learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)** -- A companion teaching repo that builds an agent **framework** (nano Claude Code) from scratch in 12 progressive sessions. Where claw0 focuses on gateway routing, channels, and proactive behavior, learn-claude-code dives deep into the agent's internal design: structured planning (TodoManager + nag), context compression (3-layer compact), file-based task persistence with dependency graphs, team coordination (JSONL mailboxes, shutdown/plan-approval FSM), autonomous self-organization, and git worktree isolation for parallel execution. If you want to understand how a production-grade unit agent works inside, start there.

## License

MIT
