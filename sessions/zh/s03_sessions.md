# 第 03 节: 会话与上下文保护

> 会话就是 JSONL 文件. 追加写入, 重放恢复, 太大就摘要压缩.

## 架构

```
    User Input
        |
        v
    SessionStore.load_session()  --> rebuild messages[] from JSONL
        |
        v
    ContextGuard.guard_api_call()
        |
        +-- Attempt 0: normal call
        |       |
        |   overflow? --no--> success
        |       |yes
        +-- Attempt 1: truncate oversized tool results
        |       |
        |   overflow? --no--> success
        |       |yes
        +-- Attempt 2: compact history via LLM summary
        |       |
        |   overflow? --yes--> raise
        |
    SessionStore.save_turn()  --> append to JSONL
        |
        v
    Print response

    File layout:
    workspace/.sessions/agents/{agent_id}/sessions/{session_id}.jsonl
    workspace/.sessions/agents/{agent_id}/sessions.json  (index)
```

## 本节要点

- **SessionStore**: JSONL 持久化. 写入时追加, 读取时重放.
- **_rebuild_history()**: 将扁平的 JSONL 转换回 API 兼容的 messages[].
- **ContextGuard**: 3 阶段溢出重试 (正常 -> 截断 -> 压缩 -> 失败).
- **compact_history()**: LLM 生成摘要替换旧消息.
- **REPL 命令**: `/new`, `/switch`, `/context`, `/compact` 用于会话管理.

## 核心代码走读

### 1. JSONL 追加与重放

每个会话是一个 `.jsonl` 文件 -- 每行一条 JSON 记录. 追加写入是原子的
(不需要重写整个文件). 四种记录类型:

```python
{"type": "user", "content": "Hello", "ts": 1234567890}
{"type": "assistant", "content": [{"type": "text", "text": "Hi!"}], "ts": ...}
{"type": "tool_use", "tool_use_id": "toolu_...", "name": "read_file", "input": {...}, "ts": ...}
{"type": "tool_result", "tool_use_id": "toolu_...", "content": "file contents", "ts": ...}
```

`_rebuild_history()` 方法将这些扁平记录转换回 Anthropic API 格式
(严格交替 user/assistant, tool_use 在 assistant 内, tool_result 在 user 内):

```python
def _rebuild_history(self, path: Path) -> list[dict]:
    messages: list[dict] = []
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        record = json.loads(line)
        rtype = record.get("type")

        if rtype == "user":
            messages.append({"role": "user", "content": record["content"]})
        elif rtype == "assistant":
            content = record["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            messages.append({"role": "assistant", "content": content})
        elif rtype == "tool_use":
            # 合并到最后一条 assistant 消息
            block = {"type": "tool_use", "id": record["tool_use_id"],
                     "name": record["name"], "input": record["input"]}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "assistant", "content": [block]})
        elif rtype == "tool_result":
            # 将连续的结果合并到同一条 user 消息
            result_block = {"type": "tool_result",
                            "tool_use_id": record["tool_use_id"],
                            "content": record["content"]}
            if (messages and messages[-1]["role"] == "user"
                    and isinstance(messages[-1]["content"], list)
                    and messages[-1]["content"][0].get("type") == "tool_result"):
                messages[-1]["content"].append(result_block)
            else:
                messages.append({"role": "user", "content": [result_block]})
    return messages
```

### 2. 3 阶段保护

`guard_api_call()` 包裹每次 API 调用. 如果上下文溢出, 它会用越来越激进的策略重试:

```python
def guard_api_call(self, api_client, model, system, messages,
                   tools=None, max_retries=2):
    current_messages = messages
    for attempt in range(max_retries + 1):
        try:
            result = api_client.messages.create(
                model=model, max_tokens=8096,
                system=system, messages=current_messages,
                **({"tools": tools} if tools else {}),
            )
            if current_messages is not messages:
                messages.clear()
                messages.extend(current_messages)
            return result
        except Exception as exc:
            error_str = str(exc).lower()
            is_overflow = ("context" in error_str or "token" in error_str)
            if not is_overflow or attempt >= max_retries:
                raise
            if attempt == 0:
                current_messages = self._truncate_large_tool_results(current_messages)
            elif attempt == 1:
                current_messages = self.compact_history(
                    current_messages, api_client, model)
```

### 3. 历史压缩

将最早 50% 的消息序列化为纯文本, 让 LLM 生成摘要,
用摘要 + 近期消息替换:

```python
def compact_history(self, messages, api_client, model):
    keep_count = max(4, int(len(messages) * 0.2))
    compress_count = max(2, int(len(messages) * 0.5))
    compress_count = min(compress_count, len(messages) - keep_count)

    old_text = _serialize_messages_for_summary(messages[:compress_count])
    summary_resp = api_client.messages.create(
        model=model, max_tokens=2048,
        system="You are a conversation summarizer. Be concise and factual.",
        messages=[{"role": "user", "content": summary_prompt}],
    )
    # 用摘要 + "Understood" 对替换旧消息
    compacted = [
        {"role": "user", "content": "[Previous conversation summary]\n" + summary},
        {"role": "assistant", "content": [{"type": "text",
         "text": "Understood, I have the context."}]},
    ]
    compacted.extend(messages[compress_count:])
    return compacted
```

## 试一试

```sh
python zh/s03_sessions.py

# 创建会话并在会话之间切换
# You > /new my-project
# You > 给我讲讲 Python 生成器
# You > /new experiments
# You > 2+2 等于多少?
# You > /switch my-p     (前缀匹配)

# 查看上下文使用情况
# You > /context
# Context usage: ~1,234 / 180,000 tokens
# [####--------------------------] 0.7%

# 上下文过大时手动压缩
# You > /compact
```

## OpenClaw 中的对应实现

| 方面              | claw0 (本文件)                 | OpenClaw 生产代码                       |
|-------------------|--------------------------------|-----------------------------------------|
| 存储格式          | JSONL 文件, 每个会话一个       | 相同的 JSONL 格式                       |
| 重放              | `_rebuild_history()`           | 相同的重建逻辑                          |
| 溢出处理          | 3 阶段保护                     | 相同模式 + token 计数 API               |
| 压缩              | LLM 摘要替换旧消息             | 相同方案, 自适应压缩                    |
| token 估算        | `len(text) // 4` 启发式        | API 提供的 token 计数                   |
