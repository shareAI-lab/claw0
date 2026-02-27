# 第 06 节: 智能层

> 系统提示词从磁盘上的文件组装. 换文件, 换性格.

## 架构

```
    Startup                              Per-Turn
    =======                              ========

    BootstrapLoader                      User Input
    load SOUL.md, IDENTITY.md, ...           |
    truncate per file (20k)                  v
    cap total (150k)                    _auto_recall(user_input)
         |                              search memory by TF-IDF
         v                                   |
    SkillsManager                            v
    scan directories for SKILL.md       build_system_prompt()
    parse frontmatter                   assemble 8 layers:
    deduplicate by name                     1. Identity
         |                                  2. Soul (personality)
         v                                  3. Tools guidance
    bootstrap_data + skills_block           4. Skills
    (cached for all turns)                  5. Memory (evergreen + recalled)
                                            6. Bootstrap (remaining files)
                                            7. Runtime context
                                            8. Channel hints
                                                |
                                                v
                                            LLM API call

    Earlier layers = stronger influence on behavior.
    SOUL.md is at layer 2 for exactly this reason.
```

## 本节要点

- **BootstrapLoader**: 从工作区加载最多 8 个 markdown 文件, 有单文件和总量上限.
- **SkillsManager**: 扫描多个目录查找带 YAML frontmatter 的 `SKILL.md` 文件.
- **MemoryStore**: 双层存储 (常驻 MEMORY.md + 每日 JSONL), TF-IDF 搜索.
- **_auto_recall()**: 用用户消息搜索记忆, 将结果注入提示词.
- **build_system_prompt()**: 将 8 个层组装为一个字符串, 每轮重新构建.

## 核心代码走读

### 1. build_system_prompt() -- 8 层组装

这个函数是智能系统的核心. 它每轮都产生不同的系统提示词,
因为记忆可能已经被更新.

```python
def build_system_prompt(mode="full", bootstrap=None, skills_block="",
                        memory_context="", agent_id="main", channel="terminal"):
    sections: list[str] = []

    # 第 1 层: 身份
    identity = bootstrap.get("IDENTITY.md", "").strip()
    sections.append(identity if identity else "You are a helpful AI assistant.")

    # 第 2 层: 灵魂 (性格) -- 越靠前 = 影响力越强
    if mode == "full":
        soul = bootstrap.get("SOUL.md", "").strip()
        if soul:
            sections.append(f"## Personality\n\n{soul}")

    # 第 3 层: 工具使用指南
    tools_md = bootstrap.get("TOOLS.md", "").strip()
    if tools_md:
        sections.append(f"## Tool Usage Guidelines\n\n{tools_md}")

    # 第 4 层: 技能
    if mode == "full" and skills_block:
        sections.append(skills_block)

    # 第 5 层: 记忆 (常驻 + 自动搜索的)
    if mode == "full":
        # ... 合并 MEMORY.md 和召回的记忆

    # 第 6 层: 引导上下文 (HEARTBEAT.md, BOOTSTRAP.md, AGENTS.md, USER.md)
    # 第 7 层: 运行时上下文 (agent ID, 模型, 通道, 时间)
    # 第 8 层: 通道提示 ("You are responding via Telegram.")

    return "\n\n".join(sections)
```

### 2. MemoryStore.search_memory() -- TF-IDF 搜索

纯 Python 实现, 不需要外部向量数据库. 加载所有记忆片段, 计算 TF-IDF 向量,
按余弦相似度排序.

```python
def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
    chunks = self._load_all_chunks()   # MEMORY.md 段落 + 每日 JSONL 条目
    query_tokens = self._tokenize(query)
    chunk_tokens = [self._tokenize(c["text"]) for c in chunks]

    # 所有片段的文档频率
    df: dict[str, int] = {}
    for tokens in chunk_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    def tfidf(tokens):
        tf = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        return {t: c * (math.log((n + 1) / (df.get(t, 0) + 1)) + 1)
                for t, c in tf.items()}

    def cosine(a, b):
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[k] * b[k] for k in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    qvec = tfidf(query_tokens)
    scored = []
    for i, tokens in enumerate(chunk_tokens):
        score = cosine(qvec, tfidf(tokens))
        if score > 0.0:
            scored.append({"path": chunks[i]["path"], "score": score,
                           "snippet": chunks[i]["text"][:200]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
```

### 3. 混合搜索管道 -- 向量 + 关键词 + MMR

完整的搜索管道串联五个阶段:

1. **关键词搜索** (TF-IDF): 与上面相同的算法, 按余弦相似度返回 top-10
2. **向量搜索** (哈希投影): 通过基于哈希的随机投影模拟嵌入向量, 返回 top-10
3. **合并**: 按文本前缀取并集, 加权组合 (`vector_weight=0.7, text_weight=0.3`)
4. **时间衰减**: `score *= exp(-decay_rate * age_days)`, 越近的记忆得分越高
5. **MMR 重排序**: `MMR = lambda * relevance - (1-lambda) * max_similarity_to_selected`, 用 token 集合的 Jaccard 相似度保证多样性

基于哈希的向量嵌入展示了双通道搜索的**模式**, 不需要外部嵌入 API.

### 4. _auto_recall() -- 自动记忆注入

每次 LLM 调用之前, 自动搜索相关记忆并注入到系统提示词中.
用户不需要显式请求.

```python
def _auto_recall(user_message: str) -> str:
    results = memory_store.search_memory(user_message, top_k=3)
    if not results:
        return ""
    return "\n".join(f"- [{r['path']}] {r['snippet']}" for r in results)

# 在 agent 循环中, 每轮:
memory_context = _auto_recall(user_input)
system_prompt = build_system_prompt(
    mode="full", bootstrap=bootstrap_data,
    skills_block=skills_block, memory_context=memory_context,
)
```

## 试一试

```sh
python zh/s06_intelligence.py

# 创建工作区文件以体验完整系统:
# workspace/SOUL.md       -- "You are warm, curious, and encouraging."
# workspace/IDENTITY.md   -- "You are Luna, a personal AI companion."
# workspace/MEMORY.md     -- "User prefers Python over JavaScript."

# 查看组装好的提示词
# You > /prompt

# 检查加载了哪些引导文件
# You > /bootstrap

# 搜索记忆
# You > /search python

# 告诉它一些信息, 然后过一会再问
# You > 我最喜欢的颜色是蓝色.
# You > 你知道我的偏好吗?
# (auto-recall 找到颜色记忆并注入提示词)
```

## OpenClaw 中的对应实现

| 方面             | claw0 (本文件)                | OpenClaw 生产代码                       |
|------------------|------------------------------|-----------------------------------------|
| 提示词组装       | 8 层 `build_system_prompt`   | 相同的分层方案                          |
| 引导文件         | 从工作区目录加载             | 相同的文件集 + 每个 agent 的覆盖配置    |
| 记忆搜索         | 混合管道 (TF-IDF + 向量 + MMR) | 相同方案 + 可选的 embedding API         |
| 技能发现         | 扫描目录查找 SKILL.md        | 相同的扫描 + 插件系统                   |
| 自动召回         | 每条用户消息都搜索           | 相同模式, top_k 可配置                  |
