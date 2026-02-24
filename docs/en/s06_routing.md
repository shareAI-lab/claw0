# s06: Message Routing & Bindings

> "Every message finds its home" -- the router does the matchmaking.

## At a Glance

```
  Inbound Message
  {channel:"telegram", sender:"user123", kind:"direct"}
       |
       v
  +--- MessageRouter.resolve() ----------------+
  |                                             |
  |  Bindings (priority desc):                  |
  |  [P4] peer_id:user123           -> alice    |
  |  [P3] guild_id:server1          -> bob      |
  |  [P2] account_id:bot-account    -> main     |
  |  [P1] channel:telegram          -> main     |
  |  [P0] default                   -> main     |
  |                                             |
  +---------------------------------------------+
       |
       v
  AgentConfig(id="alice") + session_key
       |
       v
  run_agent(config, session_key, message)
```

- **What we build**: A single gateway serving multiple agents, automatically dispatching messages via binding rules
- **Core mechanism**: 5-tier priority matching -- peer > guild > account > channel > default
- **Design pattern**: Declarative bindings + dm_scope for session isolation granularity

## The Problem

1. **Multiple agents cannot coexist**: The s05 gateway can only host a single agent. Running a creative-writing agent and a technical-QA agent simultaneously requires deploying two separate gateways -- ops cost grows linearly.
2. **Messages have no destination**: Messages from Telegram, Discord, specific users, and general users all pour into the same agent with no way to split traffic by source.
3. **Session isolation is out of control**: Conversations from different users bleed into each other. The same user on different channels cannot have independent contexts, nor can they choose to share one.

## How It Works

### 1. AgentConfig -- Multi-Agent Configuration

Each agent has its own model and system_prompt, exhibiting a distinct "personality" through different prompts:

```python
@dataclass
class AgentConfig:
    id: str
    model: str
    system_prompt: str
    tools: list[dict] = field(default_factory=list)
```

Configuration example -- three agents sharing one gateway:

```python
DEFAULT_CONFIG = {
    "agents": [
        {"id": "main",  "system_prompt": "You are a helpful assistant."},
        {"id": "alice", "system_prompt": "You are Alice, a creative writing assistant..."},
        {"id": "bob",   "system_prompt": "You are Bob, a technical assistant..."},
    ],
    "bindings": [
        {"peer_id": "user-alice-fan", "agent_id": "alice", "priority": 40},
        {"guild_id": "dev-server",    "agent_id": "bob",   "priority": 30},
        {"channel": "telegram",       "agent_id": "main",  "priority": 10},
    ],
    "default_agent": "main",
    "dm_scope": "per-peer",
}
```

**A single JSON config defines all agents and routing rules -- no code changes required.**

### 2. Binding -- 5-Tier Priority Matching

A Binding maps "match conditions" to "target agent". A `None` field means wildcard; higher priority wins:

```python
@dataclass
class Binding:
    channel: str | None = None       # P1: channel level
    account_id: str | None = None    # P2: account level
    guild_id: str | None = None      # P3: guild level
    peer_id: str | None = None       # P4: peer level (most specific)
    peer_kind: str | None = None
    agent_id: str = "main"
    priority: int = 0                # higher = matched first
```

The router iterates bindings in descending priority order, stopping at the first match:

```python
class MessageRouter:
    def resolve(self, channel, sender, peer_kind="direct",
                guild_id=None, account_id=None) -> tuple[AgentConfig, str]:
        matched_agent_id = self.default_agent
        for binding in self.bindings:  # sorted by priority desc
            if self._matches(binding, channel, sender, peer_kind, guild_id, account_id):
                matched_agent_id = binding.agent_id
                break
        agent = self.agents.get(matched_agent_id) or self.agents[self.default_agent]
        session_key = build_session_key(
            agent_id=agent.id, channel=channel,
            peer_kind=peer_kind,
            peer_id=sender if peer_kind == "direct" else (guild_id or sender),
            dm_scope=self.dm_scope,
        )
        return agent, session_key
```

Match logic: every non-None field must match; all conditions must be satisfied:

```python
def _matches(self, binding, channel, sender, peer_kind, guild_id, account_id) -> bool:
    if binding.channel and binding.channel.lower() != channel.lower():
        return False
    if binding.peer_id and binding.peer_id.lower() != sender.lower():
        return False
    # guild_id, account_id, peer_kind follow the same pattern...
    return True
```

**The priority model is immediately readable: the most specific rule (peer) has the highest priority; the fallback (default) has the lowest.**

### 3. Session Key -- Session Isolation

The session key encodes agent + channel + peer information. `dm_scope` controls DM isolation granularity:

```python
def build_session_key(agent_id, channel, peer_kind, peer_id, dm_scope="per-peer") -> str:
    if peer_kind != "direct":
        return f"agent:{agent_id}:{channel}:{peer_kind}:{peer_id}"
    if dm_scope == "main":
        return f"agent:{agent_id}:main"
    elif dm_scope == "per-peer":
        return f"agent:{agent_id}:direct:{peer_id}"
    elif dm_scope == "per-channel-peer":
        return f"agent:{agent_id}:{channel}:direct:{peer_id}"
```

| dm_scope | session key | Use case |
|----------|-------------|----------|
| `main` | `agent:alice:main` | Personal assistant, all DMs share one session |
| `per-peer` | `agent:alice:direct:user123` | Multi-user bot, each user isolated |
| `per-channel-peer` | `agent:alice:telegram:direct:user123` | Same user on different channels isolated |

**Session keys are constructed automatically -- the client never specifies them manually.**

### 4. RoutingGateway -- Gateway with Routing

Built on top of s05, adding `identify` (declare identity) and routing diagnostics:

```python
class RoutingGateway:
    def __init__(self, host, port, router, sessions, token=""):
        self._methods = {
            "health": ...,
            "chat.send": self._handle_chat_send,
            "identify": self._handle_identify,
            "routing.resolve": self._handle_routing_resolve,
            "routing.bindings": self._handle_routing_bindings,
            "sessions.list": self._handle_sessions_list,
        }
```

`chat.send` automatically resolves through the router:

```python
async def _handle_chat_send(self, client, params):
    channel = params.get("channel", client.channel)
    sender = params.get("sender", client.sender)
    agent_config, session_key = self.router.resolve(
        channel=channel, sender=sender, ...
    )
    session = self.sessions.get_or_create(session_key, agent_id=agent_config.id)
    return run_agent(agent_config, session, text)
```

**The client just sends messages; the router decides who handles them and which session to use.**

## What Changed from s05

| Component | s05 | s06 |
|-----------|-----|-----|
| Agent | Single, fixed system_prompt | Multiple agents, each independently configured |
| Routing | None, all messages to one agent | Binding priority matching, automatic dispatch |
| Session key | Client-specified | Auto-constructed from dm_scope |
| Client identity | None | `identify` declares channel/sender |
| Configuration | Hardcoded | JSON file or default config |
| RPC methods | 4 | 7 (added routing/sessions/identify) |

**Key shift**: From "one gateway, one agent" to "one gateway, many agents, routing decides."

## Design Decisions

**Why priority ordering instead of rule chains?**

The priority model is simple and transparent: an operator can see at a glance which rule will fire. No conditional-combination logic or decision-tree traversal is needed. Production OpenClaw uses the same priority-sorted approach.

**Why not use sender ID directly as session key?**

The same user may be chatting on both Telegram and Discord, or interacting with different agents. The session key must encode agent_id + channel + peer across three dimensions to properly isolate these scenarios. `dm_scope` lets administrators choose the granularity they need.

**In production OpenClaw:** Bindings are defined in `src/routing/bindings.ts`; session key construction lives in `src/routing/session-key.ts`. The production version adds identity links (cross-channel user association), team-level bindings (one binding targets a group of agents), and finer-grained dm_scope values like `per-account-channel-peer`. Route resolution also factors in allow lists, mute lists, and command gating.

## Try It

```sh
cd claw0
python agents/s06_routing.py
```

After the gateway starts, run the test client in another terminal:

```sh
python agents/s06_routing.py --test-client
```

Sample output:

```
--- Routing Diagnostics ---
   telegram | sender=random-user     | kind=direct -> agent=main   session=agent:main:direct:random-user
   telegram | sender=user-alice-fan  | kind=direct -> agent=alice  session=agent:alice:direct:user-alice-fan
    discord | sender=dev-person      | kind=group  -> agent=bob    session=agent:bob:discord:group:dev-server
      slack | sender=someone         | kind=direct -> agent=main   session=agent:main:direct:someone
```

You can also use REPL mode to debug routing locally (no gateway, no LLM calls):

```sh
python agents/s06_routing.py --repl
```

```
route> telegram user-alice-fan
  Agent:       alice
  Session Key: agent:alice:direct:user-alice-fan

route> discord dev-person group dev-server
  Agent:       bob
  Session Key: agent:bob:discord:group:dev-server
```
