# s02: Tool Use

> "Give the model hands" -- The loop stays the same; we just add a dispatch table.

## At a Glance

```
User --> [messages + tools] --> LLM
                                 |
                          stop_reason?
                         /           \
                   "end_turn"    "tool_use"
                      |              |
                    Print    TOOL_HANDLERS[name](**input)
                                     |
                              tool_result
                                     |
                            append to messages
                                     |
                              back to LLM  <-- inner while loop
```

- **What we build**: An agent with 4 tools (bash, read_file, write_file, edit_file) that can operate on the file system and execute commands.
- **Core mechanism**: The TOOLS schema tells the model what tools exist; the TOOL_HANDLERS dispatch table tells our code which function to run.
- **Design pattern**: The outer loop waits for user input; the inner loop handles chains of consecutive tool calls. stop_reason controls everything.

## The Problem

1. **The model can talk but cannot act.** The user says "read config.json for me" and the model can only reply "you could use `cat config.json`" -- it cannot actually execute anything.

2. **The model may need to call multiple tools in sequence.** For example, "read a file, change one line, then verify the result" requires read_file -> edit_file -> read_file -- three tool invocations that a single request cannot complete.

3. **Uncontrolled tool execution is dangerous.** Without path checking, the model could read `/etc/passwd`. Without output truncation, a `find /` could blow up the context window.

## How It Works

### 1. Define the Tool Schema and Dispatch Table

Two data structures linked by the `name` field: TOOLS tells the model what is available; TOOL_HANDLERS tells our code what to execute.

```python
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."},
                "timeout": {"type": "integer", "description": "Timeout in seconds. Default 30."},
            },
            "required": ["command"],
        },
    },
    # ... read_file, write_file, edit_file (same structure)
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}
```

**The TOOLS array is sent to the API. The TOOL_HANDLERS dict stays local. The model chooses tools; we execute them.**

### 2. Implement Tool Functions

Each tool receives keyword arguments (matching the schema's properties) and returns a string result:

```python
def tool_bash(command: str, timeout: int = 30) -> str:
    dangerous = ["rm -rf /", "mkfs", "> /dev/sd", "dd if="]
    for pattern in dangerous:
        if pattern in command:
            return f"Error: Refused to run dangerous command containing '{pattern}'"
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        timeout=timeout, cwd=str(WORKDIR),
    )
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
    return truncate(output) if output else "[no output]"
```

**tool_edit_file requires old_string to appear exactly once in the file; otherwise the replacement is rejected -- a uniqueness constraint that prevents accidental edits.**

### 3. Dispatch Function

Look up the tool name in TOOL_HANDLERS and execute:

```python
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
```

**All errors are returned as strings to the model rather than raised as exceptions. The model sees the error message and can self-correct.**

### 4. Inner While Loop for Tool Chains

The model may call multiple tools in a row. The inner loop continues until stop_reason is no longer `tool_use`:

```python
while True:
    response = client.messages.create(
        model=MODEL_ID, max_tokens=8096,
        system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        # Extract text, print, break out of inner loop
        break

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
        # tool_result must be returned as a user-role message (API format requirement)
        messages.append({"role": "user", "content": tool_results})
```

**The outer loop waits for user input. The inner loop handles tool chains. Two loops, one stop_reason controlling both.**

### 5. Safety Mechanisms

Two helper functions guard the tool execution boundary:

```python
def safe_path(raw: str) -> Path:
    target = (WORKDIR / raw).resolve()
    if not str(target).startswith(str(WORKDIR)):
        raise ValueError(f"Path traversal blocked: {raw}")
    return target

def truncate(text: str, limit: int = 50000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"
```

**safe_path prevents path traversal. truncate prevents output from blowing up the context window.**

## What Changed from s01

| Component | s01 | s02 |
|-----------|-----|-----|
| API call | No tools passed | `tools=TOOLS` parameter added |
| stop_reason | Only `end_turn` | `tool_use` branch added |
| Loop structure | Single while True | Outer (user) + inner (tool chain) |
| Safety | None | safe_path + truncate + dangerous command blocklist |
| New code | -- | TOOLS schema + TOOL_HANDLERS + 4 tool functions |

**Key shift**: The API call gains one extra `tools` parameter and stop_reason gains one extra `tool_use` branch. The loop structure itself does not change.

## Design Decisions

**Why does tool_result go inside a user-role message?**

The Anthropic API requires messages to strictly alternate user -> assistant -> user. A tool call response is an assistant message (containing a tool_use block), so the tool result must be returned as the next user message. This is not "something the user said" -- it is an API format requirement.

**Why 4 tools and not more?**

bash covers 90% of system operations. read_file is safer than `bash cat` (it has path checking and truncation). write_file auto-creates parent directories. edit_file does precise replacement (uniqueness constraint). These 4 tools are sufficient for an agent to complete most programming tasks.

**In production OpenClaw:** There are 50+ tools with support for parallel execution of multiple tool_use blocks, independent permission policies per tool (bash requires user confirmation, read_file auto-allows), tool results that support images and structured data, and truncation thresholds dynamically adjusted based on remaining token budget.

## Try It

```sh
cd claw0
python agents/s02_tool_use.py
```

Things to try:

- `List the files in the current directory` -- observe the bash tool running ls
- `Read the contents of agents/s01_agent_loop.py` -- observe read_file
- `Create a file called hello.py that prints "Hello, World!"` -- observe write_file
- `Read hello.py, then change the message to "Hello, claw0!"` -- observe the read_file + edit_file tool chain
