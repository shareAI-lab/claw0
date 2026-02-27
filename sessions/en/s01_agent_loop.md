# Section 01: The Agent Loop

> An agent is just `while True` + `stop_reason`.

## Architecture

```
    User Input
        |
        v
    messages[] <-- append {role: "user", ...}
        |
        v
    client.messages.create(model, system, messages)
        |
        v
    stop_reason?
      /        \
 "end_turn"  "tool_use"
     |            |
   Print      (Section 02)
     |
     v
    messages[] <-- append {role: "assistant", ...}
     |
     +--- loop back, wait for next input
```

Everything else -- tools, sessions, routing, delivery -- layers on top
without changing this loop.

## Key Concepts

- **messages[]** is the only state. The LLM sees the full array every call.
- **stop_reason** is the single decision point after each API response.
- **end_turn** = "print the text." **tool_use** = "execute, feed result back" (Section 02).
- The loop structure never changes. Later sections add features around it.

## Key Code Walkthrough

### 1. The complete agent loop

Three steps per turn: collect input, call API, branch on stop_reason.

```python
def agent_loop() -> None:
    messages: list[dict] = []

    while True:
        try:
            user_input = input(colored_prompt()).strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            break

        messages.append({"role": "user", "content": user_input})

        try:
            response = client.messages.create(
                model=MODEL_ID,
                max_tokens=8096,
                system=SYSTEM_PROMPT,
                messages=messages,
            )
        except Exception as exc:
            print(f"API Error: {exc}")
            messages.pop()   # roll back so user can retry
            continue

        if response.stop_reason == "end_turn":
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text
            print_assistant(assistant_text)

            messages.append({
                "role": "assistant",
                "content": response.content,
            })
```

### 2. The stop_reason branch

Even in Section 01, the code stubs out `tool_use`. No tools exist yet,
but the scaffolding means Section 02 requires zero changes to the outer loop.

```python
        elif response.stop_reason == "tool_use":
            print_info("[stop_reason=tool_use] No tools in this section.")
            messages.append({"role": "assistant", "content": response.content})
```

| stop_reason    | Meaning                      | Action             |
|----------------|------------------------------|--------------------|
| `"end_turn"`   | Model finished its reply     | Print, loop        |
| `"tool_use"`   | Model wants to call a tool   | Execute, feed back |
| `"max_tokens"` | Reply cut off by token limit | Print partial text |

## Try It

```sh
# Make sure .env has your key
echo 'ANTHROPIC_API_KEY=sk-ant-xxxxx' > .env
echo 'MODEL_ID=claude-sonnet-4-20250514' >> .env

# Run the agent
python en/s01_agent_loop.py

# Talk to it -- multi-turn works because messages[] accumulates
# You > What is the capital of France?
# You > And what is its population?
# (The model remembers "France" from the previous turn.)
```

## How OpenClaw Does It

| Aspect         | claw0 (this file)              | OpenClaw production                   |
|----------------|--------------------------------|---------------------------------------|
| Loop location  | `agent_loop()` in one file     | `AgentLoop` class in `src/agent/`     |
| Messages       | Plain `list[dict]` in memory   | JSONL-persisted SessionStore          |
| stop_reason    | Same branching logic           | Same logic + streaming support        |
| Error handling | Pop last message, continue     | Retry with backoff + context guard    |
| System prompt  | Hardcoded string               | 8-layer dynamic assembly (Section 06) |
