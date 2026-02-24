# s02: Tool Use (工具调用)

> "Give the model hands" -- 循环没变, 只加了一张调度表.

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
                              back to LLM  <-- 内层 while 循环
```

- **What we build**: 给 agent 加上 4 个工具 (bash, read_file, write_file, edit_file), 让它能操作文件系统和执行命令.
- **Core mechanism**: TOOLS schema 告诉模型有什么工具, TOOL_HANDLERS 调度表告诉我们的代码执行什么函数.
- **Design pattern**: 外层循环等用户输入, 内层循环处理连续工具调用, stop_reason 控制一切.

## The Problem

1. **模型只能说, 不能做.** 用户说 "帮我读一下 config.json", 模型只能回复 "你可以用 `cat config.json`", 无法真正执行.

2. **模型可能需要连续调用多个工具.** 比如 "读取文件, 修改其中一行, 再验证结果" -- 这需要 read_file -> edit_file -> read_file 三次工具调用, 单次请求无法完成.

3. **不受控的工具执行是危险的.** 没有路径检查, 模型可能读取 `/etc/passwd`; 没有输出截断, 一个 `find /` 就能撑爆上下文窗口.

## How It Works

### 1. 定义工具 Schema 和调度表

两个数据结构通过 `name` 字段关联: TOOLS 告诉模型有什么可用, TOOL_HANDLERS 告诉代码执行什么.

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
    # ... read_file, write_file, edit_file (结构相同)
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}
```

**TOOLS 数组传给 API, TOOL_HANDLERS 字典留在本地. 模型选择工具, 我们执行工具.**

### 2. 实现工具函数

每个工具接收关键字参数 (对应 schema 的 properties), 返回字符串结果:

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

**tool_edit_file 要求 old_string 在文件中恰好出现一次, 否则拒绝替换 -- 通过唯一性约束避免误操作.**

### 3. 调度函数

根据工具名从 TOOL_HANDLERS 查找并执行:

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

**所有错误都通过返回字符串传递给模型, 而不是抛异常. 模型看到错误信息后可以自行修正.**

### 4. 内层 while 循环处理工具链

模型可能连续调用多个工具. 内层循环持续到 stop_reason 不再是 `tool_use`:

```python
while True:
    response = client.messages.create(
        model=MODEL_ID, max_tokens=8096,
        system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        # 提取文本, 打印, 跳出内循环
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
        # tool_result 必须作为 user 角色消息返回 (API 格式要求)
        messages.append({"role": "user", "content": tool_results})
```

**外层循环等用户输入, 内层循环处理工具链. 两层循环, 同一个 stop_reason 控制.**

### 5. 安全机制

两个辅助函数保护工具执行边界:

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

**safe_path 防止路径穿越, truncate 防止输出撑爆上下文.**

## What Changed from s01

| Component | s01 | s02 |
|-----------|-----|-----|
| API 调用 | 不传 tools | 传入 `tools=TOOLS` |
| stop_reason | 只有 `end_turn` | 增加 `tool_use` 分支 |
| 循环结构 | 单层 while True | 外层 (用户) + 内层 (工具链) |
| 安全机制 | 无 | safe_path + truncate + 危险命令黑名单 |
| 新增代码 | -- | TOOLS schema + TOOL_HANDLERS + 4 个工具函数 |

**Key shift**: API 调用多传了一个 `tools` 参数, stop_reason 多了一个 `tool_use` 分支. 循环的结构本身没变.

## Design Decisions

**为什么 tool_result 在 user 角色消息中?**

Anthropic API 要求 messages 严格交替 user -> assistant -> user. 工具调用的 response 是 assistant 消息 (含 tool_use block), 工具结果必须作为下一条 user 消息返回. 这不是 "用户说的话", 是 API 格式的要求.

**为什么 4 个工具而不是更多?**

bash 覆盖 90% 的系统操作; read_file 比 `bash cat` 更安全 (有路径检查和截断); write_file 带自动创建父目录; edit_file 做精确替换 (唯一性约束). 这 4 个工具足以让 agent 完成大部分编程任务.

**In production OpenClaw:** 有 50+ 工具, 支持并行执行多个 tool_use block, 每个工具有独立的权限策略 (bash 需要用户确认, read_file 自动允许), 工具结果支持图片和结构化数据, 截断阈值根据剩余 token 预算动态调整.

## Try It

```sh
cd claw0
python agents/s02_tool_use.py
```

试试这些输入:

- `List the files in the current directory` -- 观察 bash 工具执行 ls
- `Read the contents of agents/s01_agent_loop.py` -- 观察 read_file
- `Create a file called hello.py that prints "Hello, World!"` -- 观察 write_file
- `Read hello.py, then change the message to "Hello, claw0!"` -- 观察 read_file + edit_file 工具链
