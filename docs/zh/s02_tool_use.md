# s02: Tool Use (工具调用)

> "Give the model hands" -- Agent 循环本身没变, 我们只是加了一张调度表.

## 问题

上一节的 agent 只能对话, 不能做事. 当用户说 "帮我读一下 config.json" 时, 模型只能说 "好的, 你可以用 `cat config.json` 来读取", 而不能真正去读文件.

LLM 本身没有手脚. 它只能生成文本. 要让它真正操作文件系统、执行命令, 需要一个 **工具调度层**: 模型告诉我们 "我想调用什么工具, 传什么参数", 我们执行后把结果送回去.

这就是 function calling (tool use) 的核心思想: 模型不执行工具, 它只**选择**工具和参数. 执行权在我们手里.

## 解决方案

```
    User --> LLM --> stop_reason == "tool_use"?
                          |
                  TOOL_HANDLERS[name](**input)
                          |
                  tool_result --> back to LLM
                          |
                   stop_reason == "end_turn"?
                          |
                       Print

    TOOLS (schema)          TOOL_HANDLERS (dispatch)
    +-----------------+     +-------------------------+
    | name: "bash"    |     | "bash"      -> tool_bash      |
    | input_schema:   | <-> | "read_file" -> tool_read_file  |
    |   {command: str}|     | "write_file"-> tool_write_file |
    +-----------------+     | "edit_file" -> tool_edit_file  |
                            +-------------------------+
```

两个数据结构的关系:
- **TOOLS 数组**: 传给 API, 告诉模型 "你有哪些工具可用, 每个工具接受什么参数"
- **TOOL_HANDLERS 字典**: 我们自己的代码用来分发 -- "模型说要调 bash, 就执行 tool_bash 函数"

两者通过 `name` 字段关联.

## 工作原理

### Step 1: 定义工具 Schema

每个工具需要一个 JSON Schema 描述, 告诉模型工具的名称、用途和参数结构:

```python
TOOLS = [
    {
        "name": "bash",
        "description": (
            "Run a shell command and return its output. "
            "Use for system commands, git, package managers, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds. Default 30.",
                },
            },
            "required": ["command"],
        },
    },
    # ... read_file, write_file, edit_file
]
```

### Step 2: 实现工具函数

每个工具函数接收关键字参数 (和 schema 中的 properties 对应), 返回字符串结果:

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
    if result.returncode != 0:
        output += f"\n[exit code: {result.returncode}]"
    return truncate(output) if output else "[no output]"
```

### Step 3: 建立调度表

一个简单的字典将工具名映射到处理函数:

```python
TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}
```

### Step 4: 调度函数

根据工具名分发到对应的处理函数:

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

### Step 5: 内层 while 循环处理连续工具调用

模型可能连续调用多个工具才最终给出文本回复. 所以在外层用户输入循环内部, 还有一个 while True 循环:

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
        # tool_result 必须在 user 角色消息中返回 (Anthropic API 要求)
        messages.append({"role": "user", "content": tool_results})
        continue  # 继续内循环, 模型会看到结果并决定下一步
```

### 安全机制

两个关键的安全辅助函数:

**safe_path()** -- 防止路径穿越:

```python
def safe_path(raw: str) -> Path:
    target = (WORKDIR / raw).resolve()
    if not str(target).startswith(str(WORKDIR)):
        raise ValueError(f"Path traversal blocked: {raw} resolves outside WORKDIR")
    return target
```

**truncate()** -- 防止超长输出撑爆上下文:

```python
def truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text)} total chars]"
```

## 核心代码

整个工具调用的核心是 **TOOL_HANDLERS 调度表 + 内层 while 循环** (来自 `agents/s02_tool_use.py`):

```python
TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}

# 内层循环: 处理连续工具调用
while True:
    response = client.messages.create(
        model=MODEL_ID, max_tokens=8096,
        system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
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
        messages.append({"role": "user", "content": tool_results})
```

这个模式可以概括为: **外层循环等用户输入, 内层循环处理工具链, stop_reason 控制一切**.

## 和上一节的区别

| 组件 | s01 | s02 |
|------|-----|-----|
| API 调用 | 不传 tools 参数 | 传入 `tools=TOOLS` |
| stop_reason 处理 | 只有 `end_turn` | 增加 `tool_use` 分支 |
| 循环结构 | 单层 while True | 外层 (用户输入) + 内层 (工具链) |
| 安全机制 | 无 | safe_path() + truncate() + 危险命令黑名单 |
| 新增代码 | -- | TOOLS schema + TOOL_HANDLERS + 4 个工具函数 |

核心改动: API 调用时多传了一个 `tools` 参数, 然后在 `stop_reason == "tool_use"` 时执行工具并把结果送回去. **循环的结构本身没变**.

## 设计解析

### 4 个工具的设计考量

- **bash**: 通用 shell 命令, 覆盖 90% 的系统操作
- **read_file**: 比 `bash cat file` 更安全, 因为有路径检查和截断
- **write_file**: 带自动创建父目录, 比 `bash echo > file` 更可靠
- **edit_file**: 精确替换文件中的文本, 要求 old_string 唯一出现一次. 这个设计直接来自 OpenClaw 的 edit 工具 -- 通过唯一性约束避免误替换

### 为什么 tool_result 在 user 角色中?

Anthropic API 的设计: messages 必须严格交替 user -> assistant -> user -> assistant. 工具调用的 response 是 assistant 消息 (包含 tool_use block), 工具结果必须作为下一条 user 消息返回. 这不是 "用户的消息", 而是 API 格式的要求.

### OpenClaw 生产版本做了什么不同?

- **50+ 工具**: 包括文件搜索 (glob/grep)、git 操作、网络请求、浏览器控制等
- **并行执行**: 多个 tool_use block 可以并行执行, 而非顺序处理
- **安全策略**: 每个工具有独立的权限策略 (如 bash 需要用户确认, read_file 自动允许)
- **工具结果格式**: 支持图片、结构化数据等, 不仅仅是纯文本
- **Token 感知截断**: 根据剩余 token 预算动态调整截断阈值

## 试一试

```sh
cd mini-claw
python agents/s02_tool_use.py
```

可以尝试的操作:

1. `List the files in the current directory` -- 观察模型调用 bash 执行 ls
2. `Read the contents of agents/s01_agent_loop.py` -- 观察 read_file 工具
3. `Create a file called hello.py that prints "Hello, World!"` -- 观察 write_file
4. `Read hello.py, then change the message to "Hello, Mini-Claw!"` -- 观察 read_file + edit_file 的工具链
5. `What is the current git branch?` -- 观察模型如何选择 bash 并解释结果
