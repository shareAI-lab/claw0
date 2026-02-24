# s03: Sessions

> "Conversations that survive restarts" -- An agent that forgets everything on restart is not truly usable.

## At a Glance

```
                sessions.json (metadata index)
                     |
User --> agent_loop() --> SessionStore --> transcripts/
              |                             session_abc.jsonl
         load_session()                     session_def.jsonl
         save_turn()
              |
         session_key = agent:channel:peer
```

- **What we build**: A JSONL persistence layer that fully restores conversation history after a process restart.
- **Core mechanism**: The append-only JSONL transcript is the source of truth; `_rebuild_history()` reconstructs API-formatted messages from it.
- **Design pattern**: agent_loop evolves from a self-contained function into a pure function that receives external state -- loading history, processing messages, and saving results are all handled through SessionStore.

## The Problem

1. **Exit means amnesia.** All conversation history in s01/s02 lives only in memory. When the process exits, everything is gone. You debug an agent for half an hour, restart, and it remembers nothing.

2. **No way to distinguish different conversations.** A single agent needs to maintain multiple independent sessions -- different users, different topics -- each with its own history, isolated from the others.

3. **Tool call results are lost.** If the model previously read a file, after a restart it has no idea what it read. It may re-read the same file or give contradictory answers.

## How It Works

### 1. JSONL Transcript Format

Each session corresponds to a `.jsonl` file, one JSON object per line:

```
{"type":"session","id":"abc123","key":"main:cli:user","created":"2025-01-01T00:00:00Z"}
{"type":"user","content":"hello","ts":"2025-01-01T00:00:01Z"}
{"type":"assistant","content":"Hi there!","ts":"2025-01-01T00:00:02Z"}
{"type":"tool_use","name":"read_file","tool_use_id":"tu_001","input":{"path":"config.json"},"ts":"..."}
{"type":"tool_result","tool_use_id":"tu_001","output":"{\"key\": \"value\"}","ts":"..."}
```

JSONL is append-only: only appends, never modifies. A process crash loses at most the last line. It can be monitored in real time with `tail -f`. No locking needed.

**JSONL is the source of truth; sessions.json is just an index.**

### 2. SessionStore Creates and Loads Sessions

Creating a session generates a unique session_id and a corresponding JSONL file. Loading a session rebuilds messages from the JSONL:

```python
def create_session(self, session_key: str) -> dict:
    session_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    transcript_file = f"{session_key.replace(':', '_')}_{session_id}.jsonl"
    metadata = {
        "session_key": session_key, "session_id": session_id,
        "created_at": now, "updated_at": now,
        "message_count": 0, "transcript_file": transcript_file,
    }
    self._index[session_key] = metadata
    self._save_index()
    self.append_transcript(session_key, {
        "type": "session", "id": session_id, "key": session_key, "created": now,
    })
    return metadata

def load_session(self, session_key: str) -> dict:
    if session_key not in self._index:
        metadata = self.create_session(session_key)
        return {"metadata": metadata, "history": []}
    metadata = self._index[session_key]
    history = self._rebuild_history(metadata["transcript_file"])
    return {"metadata": metadata, "history": history}
```

**load_session auto-creates non-existent sessions. The caller never needs to distinguish "create" from "restore".**

### 3. The Core: _rebuild_history()

Rebuilds an Anthropic API-formatted messages list from the JSONL. The tricky part is reassembling tool_use/tool_result -- in JSONL they are independent lines, but the API requires tool_use inside assistant messages and tool_result inside user messages:

```python
def _rebuild_history(self, transcript_file: str) -> list[dict]:
    messages: list[dict] = []
    pending_tool_uses: list[dict] = []

    for line in filepath.read_text(encoding="utf-8").strip().splitlines():
        entry = json.loads(line)
        entry_type = entry.get("type")

        if entry_type == "session":
            continue

        if entry_type == "user":
            if pending_tool_uses:
                messages.append({"role": "assistant", "content": pending_tool_uses})
                pending_tool_uses = []
            messages.append({"role": "user", "content": entry.get("content", "")})

        elif entry_type == "tool_use":
            pending_tool_uses.append({
                "type": "tool_use", "id": entry.get("tool_use_id", ""),
                "name": entry.get("name", ""), "input": entry.get("input", {}),
            })

        elif entry_type == "tool_result":
            if pending_tool_uses:
                messages.append({"role": "assistant", "content": pending_tool_uses})
                pending_tool_uses = []
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": entry.get("tool_use_id", ""),
                "content": entry.get("output", ""),
            }]})

    return messages
```

**The pending_tool_uses buffer is the key: it merges consecutive tool_use lines into a single assistant message's content array.**

### 4. agent_loop Becomes a Pure Function

agent_loop no longer manages messages internally. It loads from SessionStore, processes, then saves:

```python
def agent_loop(user_input: str, session_key: str,
               session_store: SessionStore, client: Anthropic) -> str:
    session_data = session_store.load_session(session_key)
    messages = session_data["history"]
    messages.append({"role": "user", "content": user_input})

    all_assistant_blocks: list = []
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=4096,
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        all_assistant_blocks.extend(response.content)
        # ... tool loop logic same as s02 ...

    session_store.save_turn(session_key, user_input, all_assistant_blocks)
    return final_text
```

**agent_loop shifts from "owning state" to "receiving state" -- this prepares for the multi-channel architecture.**

### 5. Session Key Format

`<agent_id>:<channel>:<peer_id>`, for example `main:cli:user`. The three-segment design:

- **agent**: supports multiple agent instances
- **channel**: the same user on different channels gets different sessions
- **peer**: different users on the same channel get different sessions

**This format is used directly in the s04 multi-channel architecture.**

## What Changed from s02

| Component | s02 | s03 |
|-----------|-----|-----|
| History storage | Memory only (lost on exit) | JSONL files (persistent) |
| agent_loop signature | No parameters, manages messages internally | Receives session_key + session_store |
| Tool results | In memory only | Also written to transcript |
| Multiple sessions | Not supported | Differentiated by session_key |
| Session commands | None | /new, /sessions, /switch, /history, /delete |

**Key shift**: agent_loop transforms from a self-contained function into a pure function, with state management delegated to SessionStore.

## Design Decisions

**Why JSONL instead of SQLite?**

Append-only writes are a natural fit for message logs. A crash cannot corrupt existing data. It is human-readable -- just `cat` the file. Zero dependencies, no database driver needed. Lines can be processed incrementally without loading everything at once.

**Why separate sessions.json from the JSONL files?**

sessions.json is the index; JSONL files are the content. Listing all sessions only requires reading the index (fast). Restoring a specific session reads the JSONL on demand. This is analogous to database index/data separation.

**In production OpenClaw:** The session key format is `agent:<agentId>:<channel>:<peerKind>:<peerId>` (adding peerKind to distinguish direct/group/thread). JSONL files are stored under `~/.openclaw/agents/<agentId>/sessions/`. History restoration automatically truncates old messages based on the model's context window size. Sessions can be given a TTL for automatic expiration.

## Try It

```sh
cd claw0
python agents/s03_sessions.py
```

Things to try:

- Chat with the agent for a few turns, then `/quit` to exit
- Restart -- observe the "Restored: N previous turns" message; context has been recovered
- `/new` to create a new session, `/sessions` to list all sessions
- `/switch <key>` to switch back to a previous session and verify correct context
- Inspect the JSONL files under `workspace/.sessions/transcripts/`
