# Section 02: Tool Use

> Tools are data (schema) + a handler map. Model picks a name, you look it up.

## Architecture

```
    User Input
        |
        v
    messages[] --> LLM API (tools=TOOLS)
                       |
                  stop_reason?
                  /          \
            "end_turn"    "tool_use"
               |              |
             Print    for each tool_use block:
                        TOOL_HANDLERS[name](**input)
                              |
                        tool_result
                              |
                        messages[] <-- {role:"user", content:[tool_result]}
                              |
                        back to LLM --> may chain more tools
                                          or "end_turn" --> Print
```

The outer `while True` is identical to Section 01. The only addition is an
**inner** while loop that keeps calling the LLM while `stop_reason == "tool_use"`.

## Key Concepts

- **TOOLS**: a list of JSON-schema dicts that tell the model what exists.
- **TOOL_HANDLERS**: a `dict[str, Callable]` that maps names to Python functions.
- **process_tool_call()**: dict lookup + `**kwargs` dispatch.
- **Inner loop**: the model may chain multiple tool calls before producing text.
- **Tool results go in a user message** (Anthropic API requirement).

## Key Code Walkthrough

### 1. Schema + dispatch table

Two parallel data structures. `TOOLS` tells the model, `TOOL_HANDLERS` tells your code.

```python
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command."},
                "timeout": {"type": "integer", "description": "Timeout in seconds."},
            },
            "required": ["command"],
        },
    },
    # ... read_file, write_file, edit_file (same pattern)
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}
```

Adding a new tool = one entry in `TOOLS` + one entry in `TOOL_HANDLERS`. The loop itself does not change.

### 2. Dispatch function

The model returns a tool name and a dict of inputs. Dispatch is a dict lookup.
Errors are returned as strings (not raised) so the model can see them and recover.

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

### 3. The inner tool-call loop

The only structural change from Section 01. The model may call tools multiple
times before producing a final text response.

```python
while True:
    response = client.messages.create(
        model=MODEL_ID, max_tokens=8096,
        system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        # extract text, print, break
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
        # Tool results go in a user message (API requirement)
        messages.append({"role": "user", "content": tool_results})
        continue  # back to LLM
```

## Try It

```sh
python en/s02_tool_use.py

# Ask it to run a command
# You > What files are in the current directory?

# Ask it to read a file
# You > Read the contents of en/s01_agent_loop.py

# Ask it to create and edit a file
# You > Create a file called hello.txt with "Hello World"
# You > Change "World" to "claw0" in hello.txt

# Watch it chain tools (read -> edit -> verify)
# You > Add a comment at the top of hello.txt
```

## How OpenClaw Does It

| Aspect           | claw0 (this file)             | OpenClaw production                    |
|------------------|-------------------------------|----------------------------------------|
| Tool definitions | Plain Python dicts in a list  | TypeBox schemas, auto-validated        |
| Dispatch         | `dict[str, Callable]` lookup  | Same pattern + middleware pipeline     |
| Safety           | `safe_path()` blocks traversal| Sandboxed execution, allowlists        |
| Tool count       | 4 (bash, read, write, edit)   | 20+ (web search, media, calendar, etc.)|
| Tool results     | Return plain strings          | Structured results with metadata       |
