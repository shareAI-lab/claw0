# s01: Agent Loop (智能体循环)

> "One loop to rule them all" -- AI Agent 的全部秘密就是一个 while 循环不断检查 stop_reason.

## 问题

大语言模型本身是无状态的: 你给它一段 prompt, 它返回一段文本, 交互就结束了. 但我们希望 agent 能和用户持续对话 -- 用户说一句, agent 回一句, 来回交替, 直到用户主动退出.

没有循环, 你就得每次手动把上下文复制粘贴给模型. 用户自己变成了那个循环.

Agent loop 把这个过程自动化: 读取用户输入, 追加到消息历史, 调用 LLM, 检查 stop_reason 决定下一步, 打印回复, 然后继续等待下一次输入. 整个 "智能" 来自 LLM, 我们的代码只做三件事:

1. 收集用户输入, 追加到 messages
2. 调用 API, 拿到 response
3. 检查 stop_reason 决定下一步

本节中 stop_reason 永远是 `end_turn` (因为没有工具可调用). 下一节加入工具后, 循环结构完全不变, 只多一个分支.

## 解决方案

```
    User Input --> [messages[]] --> LLM API
                                     |
                              stop_reason?
                             /           \
                       "end_turn"    "tool_use"
                          |              |
                       Print          (next section)
                          |
                    append to messages
                          |
                  wait for next input
```

messages 数组是整个 agent 的 "记忆". 每轮对话的 user/assistant 消息都追加到这里, 下次调用 API 时一并发送, 使模型能看到完整的对话上下文.

## 工作原理

### Step 1: 初始化

创建 Anthropic 客户端和空的 messages 列表. messages 就是对话历史, 是 agent 唯一的状态.

```python
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)

SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."

messages: list[dict] = []
```

### Step 2: 获取用户输入

在 `while True` 循环中读取用户输入. 处理 Ctrl+C / Ctrl+D 优雅退出, 以及 `quit` / `exit` 命令.

```python
while True:
    try:
        user_input = input(colored_prompt()).strip()
    except (KeyboardInterrupt, EOFError):
        print(f"\n{DIM}Goodbye.{RESET}")
        break

    if not user_input:
        continue

    if user_input.lower() in ("quit", "exit"):
        print(f"{DIM}Goodbye.{RESET}")
        break
```

### Step 3: 追加 user 消息到历史

用户每说一句话, 就作为 `role: "user"` 追加到 messages 数组. 这保证模型能看到完整的对话上下文.

```python
messages.append({
    "role": "user",
    "content": user_input,
})
```

### Step 4: 调用 LLM

将 messages 数组发送给 Anthropic API. 注意 `system` 参数是单独传递的, 不在 messages 中.

```python
response = client.messages.create(
    model=MODEL_ID,
    max_tokens=8096,
    system=SYSTEM_PROMPT,
    messages=messages,
)
```

如果 API 调用失败, 回滚刚追加的 user 消息, 让用户可以重试:

```python
except Exception as exc:
    print(f"\n{YELLOW}API Error: {exc}{RESET}\n")
    messages.pop()
    continue
```

### Step 5: 检查 stop_reason

stop_reason 是控制流的全部. 本节只需关注 `end_turn`:

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

关键点: assistant 的回复也要追加到 messages, 这样下次调用时模型能看到自己之前说了什么.

## 核心代码

整个 agent loop 的核心在 `agent_loop()` 函数 (来自 `agents/s01_agent_loop.py`, 第 95-188 行). 精简到本质就是:

```python
def agent_loop() -> None:
    messages: list[dict] = []

    while True:
        # 1. 获取用户输入
        user_input = input("You > ").strip()
        if user_input.lower() in ("quit", "exit"):
            break

        # 2. 追加到历史
        messages.append({"role": "user", "content": user_input})

        # 3. 调用 LLM
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=8096,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        # 4. 检查 stop_reason, 提取文本, 打印
        if response.stop_reason == "end_turn":
            assistant_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    assistant_text += block.text
            print(assistant_text)

        # 5. 追加 assistant 回复到历史
        messages.append({
            "role": "assistant",
            "content": response.content,
        })
```

整个模式可以用一句话概括: **while True -> 用户输入 -> messages.append -> API call -> check stop_reason -> print -> messages.append -> loop**.

## 设计解析

### 为什么 messages 是一个平坦的数组?

Anthropic API 要求 messages 是一个严格交替的 user/assistant 消息列表. 这个设计看起来简单, 但它意味着整个对话的上下文管理就是数组操作 -- 追加、截断、清理, 不需要复杂的状态机.

### 为什么 stop_reason 是唯一的控制信号?

API 返回的 stop_reason 只有几种可能: `end_turn` (模型说完了), `tool_use` (模型要调用工具), `max_tokens` (达到 token 上限). 我们的代码只需要一个 if/elif 就能处理所有情况. 不需要自己判断 "模型是否想继续" -- API 已经帮我们做了这个决定.

### OpenClaw 生产版本做了什么不同?

生产版本的核心循环模式完全一样, 但增加了大量基础设施:

- **Lane-based 并发**: 多个对话可以并行处理, 不会互相阻塞
- **Retry onion**: 多层重试机制 (API 超时重试、速率限制退避、网络错误恢复)
- **Streaming**: 逐 token 输出, 而非等待完整回复
- **Token 管理**: 自动截断过长的历史, 避免超出模型上下文窗口
- **Error recovery**: API 调用失败时的优雅降级和用户提示

但如果你把所有这些剥掉, 剩下的就是这个 while True 循环.

## 试一试

```sh
cd mini-claw
python agents/s01_agent_loop.py
```

需要先在 `.env` 文件中配置:

```sh
ANTHROPIC_API_KEY=sk-ant-xxxxx
MODEL_ID=claude-sonnet-4-20250514
# ANTHROPIC_BASE_URL=https://...  (可选, 用于代理)
```

可以尝试的对话:

1. `What is Python?` -- 观察基本的问答循环
2. 连续提问几个相关问题 -- 观察模型如何利用历史上下文
3. `Can you help me write a file?` -- 模型会尝试回答, 但没有工具, 只能给出代码文本 (下一节解决这个问题)
4. 输入 `quit` 退出 -- 观察所有历史在退出后丢失 (第三节解决持久化问题)
