# 第 01 节: Agent 循环

> Agent 就是 `while True` + `stop_reason`.

## 架构

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
   Print      (第 02 节)
     |
     v
    messages[] <-- append {role: "assistant", ...}
     |
     +--- 回到循环, 等待下一次输入
```

后续所有功能 -- 工具、会话、路由、投递 -- 都是在这个循环之上叠加的层,
循环本身不会改变.

## 本节要点

- **messages[]** 是唯一的状态. 每次 API 调用时, LLM 都会看到完整数组.
- **stop_reason** 是每次 API 响应后的唯一决策点.
- **end_turn** = "打印文本." **tool_use** = "执行工具, 将结果反馈回去" (第 02 节).
- 循环结构永远不变. 后续章节围绕它添加功能.

## 核心代码走读

### 1. 完整的 agent 循环

每轮三个步骤: 收集输入, 调用 API, 根据 stop_reason 分支.

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
            messages.pop()   # 回滚, 让用户可以重试
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

### 2. stop_reason 分支

即使在第 01 节, 代码也预留了 `tool_use` 分支. 虽然还没有工具,
但这个脚手架意味着第 02 节不需要修改外层循环.

```python
        elif response.stop_reason == "tool_use":
            print_info("[stop_reason=tool_use] No tools in this section.")
            messages.append({"role": "assistant", "content": response.content})
```

| stop_reason    | 含义                   | 动作           |
|----------------|------------------------|----------------|
| `"end_turn"`   | 模型完成了回复         | 打印, 继续循环 |
| `"tool_use"`   | 模型想调用工具         | 执行, 反馈结果 |
| `"max_tokens"` | 回复被 token 限制截断  | 打印部分文本   |

## 试一试

```sh
# 确保 .env 中有你的密钥
echo 'ANTHROPIC_API_KEY=sk-ant-xxxxx' > .env
echo 'MODEL_ID=claude-sonnet-4-20250514' >> .env

# 运行 agent
python zh/s01_agent_loop.py

# 和它对话 -- 多轮对话有效, 因为 messages[] 会累积
# You > 法国的首都是哪里?
# You > 它的人口是多少?
# (模型记得上一轮提到的"法国".)
```

## OpenClaw 中的对应实现

| 方面           | claw0 (本文件)                  | OpenClaw 生产代码                      |
|----------------|--------------------------------|---------------------------------------|
| 循环位置       | 单文件中的 `agent_loop()`       | `src/agent/` 中的 `AgentLoop` 类      |
| 消息存储       | 内存中的 `list[dict]`          | JSONL 持久化的 SessionStore           |
| stop_reason    | 相同的分支逻辑                 | 相同逻辑 + 流式支持                    |
| 错误处理       | 弹出最后一条消息, 继续         | 带退避的重试 + 上下文保护              |
| 系统提示词     | 硬编码字符串                   | 8 层动态组装 (第 06 节)                |
