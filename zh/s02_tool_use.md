# 第 02 节: 工具使用

> 工具 = 数据 (schema) + 处理函数映射表. 模型选一个名字, 你查表执行.

## 架构

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

外层 `while True` 与第 01 节完全相同. 唯一的新增是一个**内层** while 循环,
在 `stop_reason == "tool_use"` 时持续调用 LLM.

## 本节要点

- **TOOLS**: JSON schema 字典列表, 告诉模型有哪些工具可用.
- **TOOL_HANDLERS**: `dict[str, Callable]`, 将工具名映射到 Python 函数.
- **process_tool_call()**: 字典查找 + `**kwargs` 分发.
- **内层循环**: 模型可能连续调用多个工具, 然后才生成文本.
- **工具结果放在 user 消息中** (Anthropic API 的要求).

## 核心代码走读

### 1. Schema + 分发表

两个平行的数据结构. `TOOLS` 告诉模型, `TOOL_HANDLERS` 告诉你的代码.

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
    # ... read_file, write_file, edit_file (相同模式)
]

TOOL_HANDLERS: dict[str, Any] = {
    "bash": tool_bash,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "edit_file": tool_edit_file,
}
```

添加新工具 = 在 `TOOLS` 中加一项 + 在 `TOOL_HANDLERS` 中加一项. 循环本身不需要改动.

### 2. 分发函数

模型返回工具名和输入字典. 分发就是一次字典查找.
错误作为字符串返回 (而非抛出异常), 这样模型可以看到错误并自行修正.

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

### 3. 内层工具调用循环

相比第 01 节唯一的结构变化. 模型可能连续调用多次工具, 最后才产生文本回复.

```python
while True:
    response = client.messages.create(
        model=MODEL_ID, max_tokens=8096,
        system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
    )
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        # 提取文本, 打印, break
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
        # 工具结果放在 user 消息中 (API 要求)
        messages.append({"role": "user", "content": tool_results})
        continue  # 回到 LLM
```

## 试一试

```sh
python zh/s02_tool_use.py

# 让它执行命令
# You > 当前目录下有哪些文件?

# 让它读取文件
# You > 读取 en/s01_agent_loop.py 的内容

# 让它创建和编辑文件
# You > 创建一个名为 hello.txt 的文件, 内容是 "Hello World"
# You > 把 hello.txt 中的 "World" 改成 "claw0"

# 观察它链式调用工具 (读取 -> 编辑 -> 验证)
# You > 在 hello.txt 顶部添加一行注释
```

## OpenClaw 中的对应实现

| 方面             | claw0 (本文件)                 | OpenClaw 生产代码                      |
|------------------|-------------------------------|----------------------------------------|
| 工具定义         | Python 字典列表               | TypeBox schema, 自动校验               |
| 分发             | `dict[str, Callable]` 查表    | 相同模式 + 中间件管线                   |
| 安全性           | `safe_path()` 阻止目录穿越    | 沙箱执行, 白名单                       |
| 工具数量         | 4 个 (bash, read, write, edit)| 20+ (网页搜索, 媒体, 日历等)            |
| 工具结果         | 返回纯字符串                  | 带元数据的结构化结果                    |
