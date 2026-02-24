# s01: Agent Loop (智能体循环)

> "One loop to rule them all" -- AI Agent 的全部秘密就是一个 while 循环.

## At a Glance

```
User Input --> messages[] --> LLM API
                               |
                        stop_reason?
                       /           \
                 "end_turn"    "tool_use"
                    |              |
                  Print        (s02 实现)
                    |
             append to messages
                    |
            wait for next input
```

- **What we build**: 一个最小的对话式 REPL -- 用户说一句, 模型回一句, 循环往复.
- **Core mechanism**: `while True` 循环 + `stop_reason` 检查, 决定每轮的控制流.
- **Design pattern**: messages 数组作为唯一状态, 严格交替的 user/assistant 消息序列.

## The Problem

1. **无循环 = 手动粘贴上下文.** LLM 是无状态的: 给它一段 prompt, 返回一段文本, 交互就结束. 没有循环, 你得每次手动把之前的对话复制给模型.

2. **丢失上下文 = 失忆.** 如果不把 assistant 的回复追加回 messages, 模型下一轮看不到自己之前说了什么, 无法进行多轮推理.

3. **无 stop_reason 检查 = 无法扩展.** 如果只硬编码 "收到回复就打印", 之后加工具调用时整个流程得重写. stop_reason 是控制流的唯一分支点.

## How It Works

### 1. 初始化客户端和消息历史

创建 Anthropic 客户端, 准备空的 messages 列表. messages 是整个 agent 的唯一状态.

```python
MODEL_ID = os.getenv("MODEL_ID", "claude-sonnet-4-20250514")
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
)
SYSTEM_PROMPT = "You are a helpful AI assistant. Answer questions directly."
messages: list[dict] = []
```

**messages 是一个平坦数组, 不是树, 不是图.**

### 2. 获取用户输入

在 `while True` 循环中读取输入, 处理退出信号.

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

**空输入跳过, quit/exit/Ctrl+C 优雅退出.**

### 3. 追加 user 消息并调用 API

用户每说一句话, 作为 `role: "user"` 追加到 messages, 然后整个数组发给 API.

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
    messages.pop()  # API 失败, 回滚 user 消息
    continue
```

**API 调用失败时 pop 掉刚追加的消息, 让用户可以重试.**

### 4. 检查 stop_reason 并打印回复

stop_reason 是整个循环的控制信号. 本节只有 `end_turn`, 但分支结构为后续扩展做好了准备.

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

**assistant 回复也要追加到 messages -- 这样模型下一轮才能看到自己之前说了什么.**

## What Changed from s00

本节是起点, 没有前置版本. 从零开始建立的核心模式:

| 概念 | 状态 |
|------|------|
| 对话循环 | `while True` + `input()` |
| 状态管理 | `messages[]` 数组 |
| 控制流 | `stop_reason` 分支 |
| 工具 | 无 (下节加入) |
| 持久化 | 无 (退出即丢) |

**Key shift**: 从 "单次 API 调用" 到 "持续对话循环", 整个模式可以一句话概括: `while True -> input -> append -> API -> check stop_reason -> print -> append -> loop`.

## Design Decisions

**为什么 messages 是一个平坦数组而不是更复杂的结构?**

Anthropic API 要求 messages 是严格交替的 user/assistant 列表. 这意味着上下文管理就是数组操作 -- 追加、截断、清理, 不需要状态机. 简单即正确.

**为什么 stop_reason 是唯一的控制信号?**

API 返回的 stop_reason 穷举了所有情况: `end_turn` (说完了), `tool_use` (要调工具), `max_tokens` (达到上限). 一个 if/elif 就能处理. 不需要自己判断 "模型是否想继续" -- API 已经做了这个决定.

**In production OpenClaw:** 核心循环模式完全一样, 但增加了 lane-based 并发 (多对话并行)、多层 retry (API 超时/速率限制/网络错误)、streaming (逐 token 输出)、以及 token 管理 (自动截断过长历史). 把这些全剥掉, 剩下的就是这个 while True 循环.

## Try It

```sh
cd claw0
python agents/s01_agent_loop.py
```

需要先在 `.env` 中配置:

```sh
ANTHROPIC_API_KEY=sk-ant-xxxxx
MODEL_ID=claude-sonnet-4-20250514
```

试试这些输入:

- `What is Python?` -- 基本问答
- 连续提问相关问题 -- 观察模型如何利用历史上下文
- `Can you help me write a file?` -- 模型会尝试回答, 但没有工具, 只能给出文本
- `quit` -- 退出后所有历史丢失
