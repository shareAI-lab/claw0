# claw0

**From Zero to One: Build an OpenClaw-like AI Gateway**

> 10 progressive sections, each introducing one core mechanism.
> Every section is a runnable Python file you can execute immediately.

---

## What is this?

This is a teaching repository that walks you through building a minimal AI agent gateway from scratch, inspired by the [OpenClaw](https://github.com/openclaw/openclaw) architecture. Each section adds exactly one mechanism without changing the core loop.

```
s01: Agent Loop        -- The foundation: while + stop_reason
s02: Tool Use          -- Give the model hands: dispatch map
s03: Sessions          -- Conversations that survive restarts
s04: Multi-Channel     -- Same brain, many mouths
s05: Gateway Server    -- The switchboard: WebSocket + JSON-RPC
s06: Routing           -- Every message finds its home
s07: Soul & Memory     -- Give it a soul, let it remember
s08: Heartbeat         -- Not just reactive - proactive
s09: Cron Scheduler    -- The right thing at the right time
s10: Delivery Queue    -- Messages never get lost
```

## Architecture at a Glance

```
+--------- claw0 architecture ---------+
|                                           |
|  s10: Delivery Queue (reliable delivery)  |
|  s09: Cron Scheduler (timed tasks)        |
|  s08: Heartbeat (proactive behavior)      |
|  s07: Soul & Memory (personality + recall)|
|  s06: Routing (multi-agent binding)       |
|  s05: Gateway (WebSocket/HTTP server)     |
|  s04: Multi-Channel (channel plugins)     |
|  s03: Sessions (persistent state)         |
|  s02: Tools (bash/read/write/edit)        |
|  s01: Agent Loop (while + stop_reason)    |
|                                           |
+-------------------------------------------+
```

## Quick Start

```sh
# 1. Clone and enter
git clone https://github.com/shareAI-lab/claw0.git && cd claw0

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env with your API key and model

# 4. Run any section
python agents/s01_agent_loop.py
python agents/s02_tool_use.py
# ... etc
```

## Learning Path

```
Phase 1: THE LOOP       Phase 2: STATE        Phase 3: GATEWAY      Phase 4: INTELLIGENCE  Phase 5: OPERATIONS
+----------------+      +----------------+    +----------------+    +----------------+     +----------------+
| s01: Agent Loop|      | s03: Sessions  |    | s05: Gateway   |    | s07: Soul/Mem  |     | s09: Cron      |
| s02: Tool Use  | ---> | s04: Multi-Ch  | -> | s06: Routing   | -> | s08: Heartbeat | --> | s10: Delivery  |
| (0 -> 1 tools) |      | (state+channel)|    | (server+route) |    | (persona+auto) |     | (schedule+rely)|
+----------------+      +----------------+    +----------------+    +----------------+     +----------------+
    2 tools                2 mechanisms           2 mechanisms          2 mechanisms           2 mechanisms
```

## Section Details

| # | Section | Motto | Key Mechanism | New Concepts |
|---|---------|-------|---------------|--------------|
| 01 | Agent Loop | "One loop to rule them all" | while + stop_reason | LLM API, message history |
| 02 | Tool Use | "Give the model hands" | TOOL_HANDLERS dispatch | Tool schemas, safe execution |
| 03 | Sessions | "Conversations that survive restarts" | SessionStore + JSONL | Persistence, session keys |
| 04 | Multi-Channel | "Same brain, many mouths" | Channel plugin interface | Abstraction, normalization |
| 05 | Gateway Server | "The switchboard" | WebSocket + JSON-RPC | Server architecture, RPC |
| 06 | Routing | "Every message finds its home" | Binding resolution | Multi-agent, routing priority |
| 07 | Soul & Memory | "Give it a soul, let it remember" | SOUL.md + MemoryStore | Personality, vector search |
| 08 | Heartbeat | "Not just reactive - proactive" | HeartbeatRunner | Autonomous behavior |
| 09 | Cron Scheduler | "The right thing at the right time" | CronService + 3 schedule types | at/every/cron, auto-disable |
| 10 | Delivery Queue | "Messages never get lost" | DeliveryQueue + backoff | At-least-once, disk-backed |

## How OpenClaw Compares

| Concept | claw0 (Teaching) | OpenClaw (Production) |
|---------|---------------------|----------------------|
| Agent Loop | Simple while loop | Lane-based concurrency, retry onion |
| Tools | 4 basic tools | 50+ tools with security policies |
| Sessions | JSON file | JSONL transcripts + sessions.json metadata |
| Channels | CLI + File mock | Telegram, Discord, Slack, Signal, WhatsApp, 15+ channels |
| Gateway | websockets library | Raw http + ws, plugin HTTP routes |
| Routing | Priority bindings | Multi-level: peer/guild/team/account/channel + identity links |
| Memory | Keyword search | SQLite-vec + FTS5 + embedding cache |
| Heartbeat | Thread + timer | 6-step check chain, lane mutual exclusion, 24h dedup |
| Cron | 3 schedule types (at/every/cron) | Full cron parser, timezone support, SQLite run log |
| Delivery | File-based queue + backoff | SQLite queue, jitter, priority, batch delivery |

## Prerequisites

- Python 3.11+
- An API key for Anthropic (or compatible provider)

## License

MIT - Use freely for learning and teaching.
