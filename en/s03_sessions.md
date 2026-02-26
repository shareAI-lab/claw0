# Section 03: Sessions & Context Guard

> Sessions are JSONL files. Append, replay, summarize when too big.

## Architecture

```
    User Input
        |
        v
    SessionStore.load_session()  --> rebuild messages[] from JSONL
        |
        v
    ContextGuard.guard_api_call()
        |
        +-- Attempt 0: normal call
        |       |
        |   overflow? --no--> success
        |       |yes
        +-- Attempt 1: truncate oversized tool results
        |       |
        |   overflow? --no--> success
        |       |yes
        +-- Attempt 2: compact history via LLM summary
        |       |
        |   overflow? --yes--> raise
        |
    SessionStore.save_turn()  --> append to JSONL
        |
        v
    Print response

    File layout:
    workspace/.sessions/agents/{agent_id}/sessions/{session_id}.jsonl
    workspace/.sessions/agents/{agent_id}/sessions.json  (index)
```

## What's New

- **SessionStore**: JSONL persistence. Append on write, replay on read.
- **_rebuild_history()**: converts flat JSONL back into API-compatible messages[].
- **ContextGuard**: 3-stage overflow retry (normal -> truncate -> compact -> fail).
- **compact_history()**: LLM-generated summary replaces old messages.
- **REPL commands**: `/new`, `/switch`, `/context`, `/compact` for session management.

## Key Code Walkthrough

### 1. JSONL append and replay

Each session is a `.jsonl` file -- one JSON record per line. Append-only writes
are atomic (no rewriting the whole file). Four record types:

```python
{"type": "user", "content": "Hello", "ts": 1234567890}
{"type": "assistant", "content": [{"type": "text", "text": "Hi!"}], "ts": ...}
{"type": "tool_use", "tool_use_id": "toolu_...", "name": "read_file", "input": {...}, "ts": ...}
{"type": "tool_result", "tool_use_id": "toolu_...", "content": "file contents", "ts": ...}
```

The `_rebuild_history()` method converts these flat records back into the Anthropic
API format (strict alternating user/assistant, tool_use inside assistant, tool_result
inside user):

```python
def _rebuild_history(self, path: Path) -> list[dict]:
    messages: list[dict] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        record = json.loads(line)
        rtype = record.get("type")

        if rtype == "user":
            messages.append({"role": "user", "content": record["content"]})
        elif rtype == "assistant":
            content = record["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            messages.append({"role": "assistant", "content": content})
        elif rtype == "tool_use":
            # Merge into last assistant message
            block = {"type": "tool_use", "id": record["tool_use_id"],
                     "name": record["name"], "input": record["input"]}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "assistant", "content": [block]})
        elif rtype == "tool_result":
            # Merge consecutive results into one user message
            result_block = {"type": "tool_result",
                            "tool_use_id": record["tool_use_id"],
                            "content": record["content"]}
            if (messages and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                    and messages[-1]["content"][0].get("type") == "tool_result"):
                messages[-1]["content"].append(result_block)
            else:
                messages.append({"role": "user", "content": [result_block]})
    return messages
```

### 2. The 3-stage guard

`guard_api_call()` wraps every API call. If the context overflows, it retries
with increasingly aggressive strategies:

```python
def guard_api_call(self, api_client, model, system, messages,
                   tools=None, max_retries=2):
    current_messages = messages
    for attempt in range(max_retries + 1):
        try:
            result = api_client.messages.create(
                model=model, max_tokens=8096,
                system=system, messages=current_messages,
                **({"tools": tools} if tools else {}),
            )
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
                current_messages = self._truncate_large_tool_results(current_messages)
            elif attempt == 1:
                current_messages = self.compact_history(
                    current_messages, api_client, model)
```

### 3. History compaction

Serialize the oldest 50% of messages to plain text, ask the LLM to summarize,
replace with a summary + recent messages:

```python
def compact_history(self, messages, api_client, model):
    keep_count = max(4, int(len(messages) * 0.2))
    compress_count = max(2, int(len(messages) * 0.5))
    compress_count = min(compress_count, len(messages) - keep_count)

    old_text = _serialize_messages_for_summary(messages[:compress_count])
    summary_resp = api_client.messages.create(
        model=model, max_tokens=2048,
        system="You are a conversation summarizer. Be concise and factual.",
        messages=[{"role": "user", "content": summary_prompt}],
    )
    # Replace old messages with summary + "Understood" pair
    compacted = [
        {"role": "user", "content": "[Previous conversation summary]\n" + summary},
        {"role": "assistant", "content": [{"type": "text",
         "text": "Understood, I have the context."}]},
    ]
    compacted.extend(messages[compress_count:])
    return compacted
```

## Try It

```sh
python en/s03_sessions.py

# Create sessions and switch between them
# You > /new my-project
# You > Tell me about Python generators
# You > /new experiments
# You > What is 2+2?
# You > /switch my-p     (prefix match)

# Check context usage
# You > /context
# Context usage: ~1,234 / 180,000 tokens
# [####--------------------------] 0.7%

# Manually compact when context gets large
# You > /compact
```

## How OpenClaw Does It

| Aspect            | claw0 (this file)              | OpenClaw production                     |
|-------------------|--------------------------------|-----------------------------------------|
| Storage format    | JSONL files, one per session   | Same JSONL format                       |
| Replay            | `_rebuild_history()`           | Same reconstruction logic               |
| Overflow handling | 3-stage guard                  | Same pattern + token counting API       |
| Compaction        | LLM summary of old messages    | Same approach, adaptive compression     |
| Token estimation  | `len(text) // 4` heuristic     | API-provided token counts               |
