# s01: Agent Loop

> "One loop to rule them all" -- The entire secret of an AI Agent is a single while loop.

## At a Glance

```
User Input --> messages[] --> LLM API
                               |
                        stop_reason?
                       /           \
                 "end_turn"    "tool_use"
                    |              |
                  Print        (s02 implements)
                    |
             append to messages
                    |
            wait for next input
```

- **What we build**: A minimal conversational REPL -- the user says something, the model replies, and the cycle repeats.
- **Core mechanism**: A `while True` loop plus a `stop_reason` check that determines control flow each turn.
- **Design pattern**: The messages array is the sole state, maintained as a strictly alternating user/assistant message sequence.

## The Problem

1. **No loop = manual context pasting.** LLMs are stateless: you send a prompt, get text back, and the interaction ends. Without a loop, you would have to manually copy prior conversation into every request.

2. **Lost context = amnesia.** If you do not append the assistant's reply back into messages, the model cannot see what it said previously on the next turn. Multi-turn reasoning becomes impossible.

3. **No stop_reason check = no extensibility.** If you hard-code "print whatever comes back", adding tool calls later forces a full rewrite. The stop_reason is the single branch point for all control flow.

## How It Works

### 1. Initialize the Client and Message History

Create an Anthropic client and prepare an empty messages list. The messages list is the agent's only state.

```python
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)
SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."
messages: list[dict] = []
```

**messages is a flat array -- not a tree, not a graph.**

### 2. Read User Input

Inside a `while True` loop, read input and handle exit signals.

```python
while True:
    try:
        user_input = input(colored_prompt()).strip()
    except (KeyboardInterrupt, EOFError):
        break
    if not user_input:
        continue
    if user_input.lower() in ("quit", "exit"):
        break
```

**Empty input is skipped. quit/exit/Ctrl+C trigger a graceful exit.**

### 3. Append the User Message and Call the API

Each user utterance is appended as `role: "user"`, then the entire array is sent to the API.

```python
messages.append({"role": "user", "content": user_input})

try:
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=8096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
except Exception as exc:
    messages.pop()  # Roll back on API failure
    continue
```

**On API failure, pop the just-appended message so the user can retry.**

### 4. Check stop_reason and Print the Reply

stop_reason is the control signal for the entire loop. This section only handles `end_turn`, but the branch structure is ready for future extension.

```python
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

**The assistant reply must also be appended to messages -- this is how the model sees its own prior output on the next turn.**

## Design Decisions

**Why is messages a flat array instead of something more complex?**

The Anthropic API requires messages to be a strictly alternating user/assistant list. This means context management is just array operations -- append, truncate, trim. No state machine needed. Simple is correct.

**Why is stop_reason the only control signal?**

The API's stop_reason exhaustively covers all outcomes: `end_turn` (done talking), `tool_use` (wants to call a tool), `max_tokens` (hit the limit). A single if/elif handles everything. You do not need to guess whether the model "wants to continue" -- the API has already made that decision.

**In production OpenClaw:** The core loop pattern is identical, but adds lane-based concurrency (parallel conversations), multi-layer retry (API timeouts / rate limits / network errors), streaming (token-by-token output), and token management (automatic truncation of overly long histories). Strip all of that away, and what remains is this while True loop.

## Try It

```sh
cd claw0
python agents/s01_agent_loop.py
```

You need to configure `.env` first:

```sh
ANTHROPIC_API_KEY=sk-ant-xxxxx
MODEL_ID=claude-sonnet-4-20250514
```

Things to try:

- `What is Python?` -- basic Q&A
- Ask follow-up questions -- observe how the model uses conversation history
- `Can you help me write a file?` -- the model will try to answer, but without tools it can only produce text
- `quit` -- all history is lost on exit
