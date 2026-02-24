# s07: Soul System & Memory (人格系统与记忆)

> 赋予它灵魂, 让它记住 -- SOUL.md 定义 Agent 是谁, Memory 让它记得发生过什么。

## 问题

前面章节的 Agent 有一个根本缺陷: **没有身份, 没有记忆**。

1. **没有人格**: 每个 Agent 都是 "You are a helpful assistant", 千篇一律。用户无法感受到与一个 "独特角色" 对话。
2. **没有长期记忆**: 关掉终端, 所有对话消失。即使在同一个 session 中, Agent 也不会主动记住重要信息 (比如用户的名字、偏好、待办事项)。
3. **没有跨会话连续性**: 今天告诉 Agent 你喜欢什么, 明天它全忘了。

OpenClaw 用两个机制解决: **Soul System** 赋予人格, **Memory System** 赋予记忆。

## 解决方案

```
  +--- SOUL.md ---+     +--- MEMORY.md ---+
  | personality    |     | evergreen facts  |
  | values         |     | preferences      |
  | language style |     +------------------+
  +----------------+            |
         |                      |
         v                      v
  +-- System Prompt Builder --------+
  |  [soul] + [base] + [memories]   |
  +---------------------------------+
         |
         v
  +-- Agent Loop ---+     +-- memory/daily/ --+
  |  tools:         | --> |  2026-02-24.md     |
  |  memory_write   |     |  2026-02-23.md     |
  |  memory_search  |     +--------------------+
  +------------------+
```

核心思路: SOUL.md 作为人格定义注入 system prompt 最前面, 影响 Agent 的所有回复风格。Memory 分双层 -- MEMORY.md 存永久事实, `memory/YYYY-MM-DD.md` 存每日日志。Agent 通过 `memory_write` 和 `memory_search` 两个工具主动操作记忆。

## 工作原理

### 1. SoulSystem -- 人格注入

SOUL.md 是一个 Markdown 文件, 定义 Agent 的性格、价值观和语言风格:

```md
# Soul
You are Koda, a thoughtful AI assistant.

## Personality
- Warm but not overly enthusiastic
- Prefer concise, clear explanations
- Use analogies from nature and engineering

## Values
- Honesty over comfort
- Depth over breadth
- Action over speculation

## Language Style
- Chinese for casual chat, English for technical terms
- End complex explanations with a one-line summary
```

加载和注入逻辑:

```python
class SoulSystem:
    def __init__(self, soul_dir: Path):
        self.soul_path = soul_dir / "SOUL.md"

    def load_soul(self) -> str:
        if self.soul_path.exists():
            return self.soul_path.read_text(encoding="utf-8").strip()
        return ""

    def build_system_prompt(self, base_prompt: str) -> str:
        soul = self.load_soul()
        if soul:
            return f"{soul}\n\n---\n\n{base_prompt}"
        return base_prompt
```

关键设计: Soul 内容放在 system prompt 的最前面。LLM 对 system prompt 开头部分的 "注意力" 最强, 所以人格定义会强烈影响后续所有输出。

### 2. MemoryStore -- 双层记忆

记忆分两层, 对应不同的生命周期:

**MEMORY.md (常驻记忆)**: 手动维护的长期事实, 比如 "用户是一名后端工程师, 偏好 TypeScript"。

**memory/YYYY-MM-DD.md (每日日志)**: Agent 通过 `memory_write` 工具自动写入, 每天一个文件。

```python
class MemoryStore:
    def __init__(self, memory_dir: Path):
        self.memory_dir = memory_dir
        self.evergreen_path = memory_dir / "MEMORY.md"
        self.daily_dir = memory_dir / "memory"
        self.daily_dir.mkdir(parents=True, exist_ok=True)
```

写入每日记忆:

```python
def write_memory(self, content: str, category: str = "general") -> str:
    today = date.today().isoformat()
    path = self.daily_dir / f"{today}.md"

    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"\n## [{timestamp}] {category}\n\n{content}\n"

    if not path.exists():
        header = f"# Memory Log: {today}\n"
        path.write_text(header, encoding="utf-8")

    with open(path, "a", encoding="utf-8") as f:
        f.write(entry)

    return f"memory/{today}.md"
```

每条记忆带有时间戳和分类标签, 追加到当天的日志文件。文件结构示例:

```md
# Memory Log: 2026-02-24

## [14:30:15] preference

User prefers dark mode and vim keybindings.

## [15:20:03] fact

User's project uses Python 3.12 with FastAPI.
```

### 3. TF-IDF 搜索

当 Agent 需要回忆过去的信息时, 调用 `memory_search` 工具。搜索使用 TF-IDF + 余弦相似度:

**分词**:

```python
def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())
    return [t for t in tokens if len(t) > 1]
```

**余弦相似度**:

```python
def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    if not common_keys:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
```

**搜索主流程**:

```python
def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
    chunks = self._load_all_chunks()
    if not chunks:
        return []

    # 建立文档集合的词频
    doc_freq: Counter = Counter()
    chunk_tokens_list = []
    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        unique_tokens = set(tokens)
        for t in unique_tokens:
            doc_freq[t] += 1
        chunk_tokens_list.append(tokens)

    n_docs = len(chunks)

    def _idf(term: str) -> float:
        df = doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(n_docs / df)

    # query 的 TF-IDF 向量
    query_tokens = _tokenize(query)
    query_tf = Counter(query_tokens)
    query_vec = {t: (count / max(len(query_tokens), 1)) * _idf(t)
                 for t, count in query_tf.items()}

    # 对每个 chunk 计算相似度
    scored = []
    for i, chunk in enumerate(chunks):
        tokens = chunk_tokens_list[i]
        if not tokens:
            continue
        tf = Counter(tokens)
        chunk_vec = {t: (count / len(tokens)) * _idf(t)
                     for t, count in tf.items()}
        score = _cosine_similarity(query_vec, chunk_vec)
        if score > 0.01:
            scored.append({
                "path": chunk["path"],
                "score": round(score, 4),
                "snippet": chunk["text"][:300],
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
```

TF-IDF 的工作原理:
1. **TF (词频)**: 一个词在文档中出现越频繁, 相关性越高
2. **IDF (逆文档频率)**: 一个词在越少的文档中出现, 区分度越高 (比如 "the" 到处出现, IDF 低; "FastAPI" 只在特定文档出现, IDF 高)
3. **TF-IDF 向量**: 把每个文档表示为一个 {词: TF*IDF} 的稀疏向量
4. **余弦相似度**: 计算 query 向量和文档向量的夹角余弦, 越接近 1 越相似

这是 embedding 的教学替代: 原理相同 (文本 -> 向量 -> 相似度比较), 只是向量质量不如 embedding model 生成的高维密集向量。

### 4. 工具定义

Agent 通过两个工具操作记忆:

```python
TOOLS = [
    {
        "name": "memory_write",
        "description": (
            "Write a memory to persistent storage. Use this to remember important "
            "information the user shares: preferences, facts, decisions, names, dates. "
            "Each memory is timestamped and categorized."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The information to remember."},
                "category": {"type": "string", "description": "Category: preference, fact, decision, todo, person."},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": (
            "Search through stored memories using keyword matching. "
            "Use this before answering questions about prior conversations, "
            "user preferences, past decisions, or any previously discussed topics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "top_k": {"type": "integer", "description": "Max results. Default 5."},
            },
            "required": ["query"],
        },
    },
]
```

### 5. System Prompt 分层构建

最终 system prompt 的分层结构:

```python
def build_full_system_prompt() -> str:
    # 基础提示
    base = BASE_SYSTEM_PROMPT.format(date=date.today().isoformat())

    # 注入 soul 人格 (最前面)
    prompt = soul_system.build_system_prompt(base)

    # 注入常驻记忆
    evergreen = memory_store.load_evergreen()
    if evergreen:
        prompt += f"\n\n---\n\n## Evergreen Memory\n\n{evergreen}"

    # 注入近期记忆摘要
    recent = memory_store.get_recent_memories(days=3)
    if recent:
        prompt += "\n\n---\n\n## Recent Memory Context\n"
        for entry in recent:
            snippet = entry["content"][:500]
            prompt += f"\n### {entry['date']}\n{snippet}\n"

    return prompt
```

生成的 system prompt 结构:

```
[SOUL.md 内容]           <-- 人格定义, 优先级最高
---
[Base system prompt]     <-- 功能说明 + 当前日期
---
## Evergreen Memory      <-- 常驻记忆
[MEMORY.md 内容]
---
## Recent Memory Context <-- 近期记忆摘要
### 2026-02-24
[当天记忆片段]
### 2026-02-23
[昨天记忆片段]
```

每轮对话都重新构建 system prompt, 因为记忆可能已经更新。

### 6. Agent 循环

与 s01 的循环结构相同, 但增加了工具处理:

```python
while True:
    response = client.messages.create(
        model=MODEL_ID, max_tokens=8096,
        system=system_prompt, messages=messages, tools=TOOLS,
    )

    if response.stop_reason == "end_turn":
        # 提取并打印回复
        # ...
        break

    elif response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                if handler:
                    result = handler(block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })
        messages.append({"role": "user", "content": tool_results})
        # 继续内循环
```

## 核心代码

记忆的完整读写闭环 -- 这是 Soul + Memory 系统最核心的模式:

```python
# 写入: Agent 调用 memory_write 工具
def _handle_memory_write(params: dict) -> str:
    content = params.get("content", "")
    category = params.get("category", "general")
    if not content.strip():
        return json.dumps({"error": "Empty content"})
    path = memory_store.write_memory(content, category)
    return json.dumps({"status": "saved", "path": path, "category": category})

# 搜索: Agent 调用 memory_search 工具
def _handle_memory_search(params: dict) -> str:
    query = params.get("query", "")
    top_k = params.get("top_k", 5)
    if not query.strip():
        return json.dumps({"results": [], "error": "Empty query"})
    results = memory_store.search_memory(query, top_k=top_k)
    return json.dumps({"results": results, "total_found": len(results)})
```

记忆 chunk 按 markdown heading 拆分:

```python
@staticmethod
def _split_by_heading(content: str, path: str) -> list[dict]:
    lines = content.split("\n")
    chunks = []
    current_lines: list[str] = []
    current_start = 1

    for i, line in enumerate(lines):
        if line.startswith("#") and current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                chunks.append({
                    "path": path,
                    "text": text,
                    "line_start": current_start,
                    "line_end": current_start + len(current_lines) - 1,
                })
            current_lines = [line]
            current_start = i + 1
        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            chunks.append({"path": path, "text": text,
                           "line_start": current_start,
                           "line_end": current_start + len(current_lines) - 1})
    return chunks
```

## 和上一节的区别

| 组件 | s06 | s07 |
|------|-----|-----|
| 人格 | 无, 或硬编码 system_prompt | SOUL.md 文件定义, 动态加载 |
| 记忆 | 无持久化 | 双层: MEMORY.md (常驻) + memory/daily (每日) |
| 工具 | 无 | memory_write + memory_search |
| System prompt | 固定字符串 | 分层构建: soul + base + evergreen + recent |
| 搜索 | 无 | TF-IDF + 余弦相似度 |
| 运行模式 | WebSocket 网关 | 交互式 REPL (带 /soul 和 /memory 命令) |

关键转变: 从 "路由到正确的 Agent" 变成 "给 Agent 一个灵魂和记忆"。Agent 不再是无状态的函数, 而是一个有持续身份和可检索记忆的实体。

## 设计解析

**为什么人格用文件而不是代码?**

文件可以被非程序员编辑。你的产品经理、内容运营、甚至最终用户都可以修改 SOUL.md 来调整 Agent 的性格, 不需要改一行代码、不需要重新部署。这是 OpenClaw 的核心设计哲学: **配置驱动, 而不是代码驱动**。

**为什么记忆要分两层?**

- MEMORY.md 存放 "确认的事实": 用户偏好、项目信息、长期有效的知识。由人类审核和维护, 质量高。
- 每日日志存放 "原始发生的事": Agent 自动写入, 量大但质量参差。搜索时按相关性排序, 自动过滤噪音。

这种分层让记忆系统既有高质量的核心知识, 又有丰富的细节可供检索。

**为什么用 TF-IDF 而不是直接 embedding?**

本节是教学用途。TF-IDF 的优势:
- 不需要额外的 API 调用 (embedding model)
- 不需要数据库 (sqlite-vec)
- 算法透明, 可以逐步跟踪每一步计算

原理与 embedding 搜索相同: 文本 -> 向量 -> 相似度。只是向量的质量不同。

**OpenClaw 生产版的不同之处:**

- 搜索使用 sqlite-vec + embedding cache + FTS5 全文搜索三管齐下
- 人格文件除了 SOUL.md, 还有 IDENTITY.md (身份信息) 和 AGENTS.md (Agent 间协作规则)
- System prompt 构建更复杂, 包含身份链接、工具描述、MCP 上下文等
- 记忆写入不是独立工具, 而是通过 bash/write 工具写入文件系统
- 记忆文件支持行范围精确读取 (`memory_get`)
- 记忆分块使用固定大小 + overlap 策略, 而不是按 heading 拆分

## 试一试

```sh
cd mini-claw
python agents/s07_soul_memory.py
```

首次运行会自动创建 `workspace/SOUL.md` 示例文件。

可以尝试的对话:

1. 告诉 Agent 一些关于你的信息, 观察它是否自动调用 `memory_write`:

```
You > My name is Alex and I'm working on a Rust project called tundra.
```

2. 过一会儿问它记不记得, 观察 `memory_search` 调用:

```
You > What project am I working on?
```

3. 查看当前人格:

```
You > /soul
```

4. 查看记忆状态:

```
You > /memory
```

5. 编辑 `workspace/SOUL.md` 改变 Agent 的性格, 下次回复就会生效 (因为每轮都重新构建 system prompt)。

6. 手动在 `workspace/MEMORY.md` 写入一些事实, 观察 Agent 的回复中是否体现这些知识。
