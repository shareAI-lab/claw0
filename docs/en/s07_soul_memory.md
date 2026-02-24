# s07: Soul & Memory

> "Give it a soul, let it remember" -- from stateless function to persistent identity.

## At a Glance

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
         |
    TF-IDF search
    query -> tokenize -> TF*IDF vectors -> cosine similarity -> top_k
```

- **What we build**: SOUL.md defines the agent personality; a two-layer memory system (evergreen + daily) lets the agent remember across sessions
- **Core mechanism**: Soul injected at the very start of the system prompt; memories retrieved via TF-IDF + cosine similarity
- **Design pattern**: File-driven config (non-programmers can edit personality) + tool-driven writes (the agent decides what to remember)

## The Problem

1. **Cookie-cutter agents**: Every agent is "You are a helpful assistant" -- users never feel they are talking to a distinct character. Switch agents, same style.
2. **Amnesia on restart**: All conversations live only in memory. Restart the process and everything vanishes. The agent you told your name to yesterday has forgotten today.
3. **No autonomous memory**: Even within a single session the agent does not persist user preferences, project details, or to-do items for use in future conversations.

## How It Works

### 1. SoulSystem -- Personality Injection

SOUL.md is a Markdown file defining the agent's character and language style:

```md
# Soul
You are Koda, a thoughtful AI assistant.

## Personality
- Warm but not overly enthusiastic
- Prefer concise, clear explanations

## Values
- Honesty over comfort
- Depth over breadth

## Language Style
- Chinese for casual chat, English for technical terms
- End complex explanations with a one-line summary
```

After loading, the soul is prepended to the system prompt:

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

**Soul sits at the very beginning of the system prompt because LLMs attend most strongly to the opening content -- the personality definition shapes all subsequent output.**

### 2. MemoryStore -- Two-Layer Memory

Memory is split into two layers:

- **MEMORY.md (evergreen memory)**: Manually curated long-term facts, e.g. "User prefers TypeScript, uses Vim"
- **memory/YYYY-MM-DD.md (daily logs)**: Automatically written by the agent via the `memory_write` tool, one file per day

```python
class MemoryStore:
    def __init__(self, memory_dir: Path):
        self.evergreen_path = memory_dir / "MEMORY.md"
        self.daily_dir = memory_dir / "memory"

    def write_memory(self, content: str, category: str = "general") -> str:
        today = date.today().isoformat()
        path = self.daily_dir / f"{today}.md"
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"\n## [{timestamp}] {category}\n\n{content}\n"
        if not path.exists():
            path.write_text(f"# Memory Log: {today}\n", encoding="utf-8")
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry)
        return f"memory/{today}.md"
```

Daily log file structure:

```md
# Memory Log: 2026-02-24

## [14:30:15] preference

User prefers dark mode and vim keybindings.

## [15:20:03] fact

User's project uses Python 3.12 with FastAPI.
```

**Evergreen memory is high quality (human-reviewed); daily logs are high volume but searchable. The two layers complement each other.**

### 3. TF-IDF Search -- Keyword-Based Memory Retrieval

When the agent calls `memory_search`, the system finds the most relevant memory chunks using TF-IDF + cosine similarity:

```python
def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
    chunks = self._load_all_chunks()
    if not chunks:
        return []

    # compute document frequency
    doc_freq: Counter = Counter()
    chunk_tokens_list = []
    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        for t in set(tokens):
            doc_freq[t] += 1
        chunk_tokens_list.append(tokens)

    n_docs = len(chunks)

    def _idf(term: str) -> float:
        df = doc_freq.get(term, 0)
        return math.log(n_docs / df) if df > 0 else 0.0

    # TF-IDF vector for the query
    query_tokens = _tokenize(query)
    query_tf = Counter(query_tokens)
    query_vec = {t: (count / max(len(query_tokens), 1)) * _idf(t)
                 for t, count in query_tf.items()}

    # compute cosine similarity for each chunk, take top_k
    scored = []
    for i, chunk in enumerate(chunks):
        tokens = chunk_tokens_list[i]
        if not tokens:
            continue
        tf = Counter(tokens)
        chunk_vec = {t: (c / len(tokens)) * _idf(t) for t, c in tf.items()}
        score = _cosine_similarity(query_vec, chunk_vec)
        if score > 0.01:
            scored.append({"path": chunk["path"], "score": round(score, 4),
                           "snippet": chunk["text"][:300]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
```

Tokenization supports mixed Chinese and English:

```python
def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())
    return [t for t in tokens if len(t) > 1]
```

**TF-IDF is the teaching substitute for embedding search: the principle is the same (text -> vector -> similarity), with zero external dependencies and every step traceable by hand.**

### 4. Layered System Prompt Construction

The system prompt is rebuilt every turn because memory may have been updated:

```python
def build_full_system_prompt() -> str:
    base = BASE_SYSTEM_PROMPT.format(date=date.today().isoformat())
    prompt = soul_system.build_system_prompt(base)  # soul + base

    evergreen = memory_store.load_evergreen()
    if evergreen:
        prompt += f"\n\n---\n\n## Evergreen Memory\n\n{evergreen}"

    recent = memory_store.get_recent_memories(days=3)
    if recent:
        prompt += "\n\n---\n\n## Recent Memory Context\n"
        for entry in recent:
            prompt += f"\n### {entry['date']}\n{entry['content'][:500]}\n"
    return prompt
```

Final prompt structure:

```
[SOUL.md]               <-- personality, highest priority
---
[Base system prompt]     <-- capabilities + date
---
## Evergreen Memory      <-- MEMORY.md long-term facts
---
## Recent Memory Context <-- last 3 days of log snippets
```

**Personality first, memory last -- ensures character consistency while still providing context.**

### 5. Tool Definitions and Agent Loop

The agent operates on memory through two tools:

```python
TOOLS = [
    {
        "name": "memory_write",
        "description": "Write a memory to persistent storage...",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "category": {"type": "string"},
            },
            "required": ["content"],
        },
    },
    {
        "name": "memory_search",
        "description": "Search through stored memories...",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
]
```

The agent loop follows the same structure as s01, but adds tool handling -- when `stop_reason == "tool_use"`, the tool is executed and results are fed back so the agent can continue reasoning.

**The agent autonomously decides when to memorize and when to search -- no hard-coded rules required.**

## What Changed from s06

| Component | s06 | s07 |
|-----------|-----|-----|
| Personality | Hardcoded system_prompt | SOUL.md file, dynamically loaded |
| Memory | No persistence | Two-layer: MEMORY.md + memory/daily |
| Tools | None | memory_write + memory_search |
| System prompt | Fixed string | Layered build: soul + base + evergreen + recent |
| Search | None | TF-IDF + cosine similarity |
| Run mode | WebSocket gateway | Interactive REPL (with /soul and /memory commands) |

**Key shift**: From "route to the right agent" to "give the agent a soul and a memory." The agent is no longer a stateless function but an entity with a persistent identity and searchable recall.

## Design Decisions

**Why file-based personality instead of code?**

Files can be edited by non-programmers. Product managers, content operators, even end users can tweak SOUL.md to adjust the agent's character without touching code or redeploying. This is the core idea of configuration-driven design.

**Why two-layer memory?**

MEMORY.md holds human-reviewed, high-confidence facts; daily logs hold raw records auto-written by the agent -- large volume but variable quality. Search ranks by relevance, automatically filtering noise. The two layers give the system both reliable core knowledge and rich retrievable detail.

**Why TF-IDF instead of embedding?**

For teaching purposes. TF-IDF requires no extra API calls, no database, and the algorithm is fully transparent. The principle is the same as embedding search (text -> vector -> similarity); only the vector quality differs.

**In production OpenClaw:** Search uses sqlite-vec + embedding cache + FTS5 full-text search in a three-pronged approach. Personality files include SOUL.md, IDENTITY.md, and AGENTS.md. Memory chunking uses a fixed-size + overlap strategy rather than heading-based splitting. Memory writes are not a standalone tool but go through the file-system tool.

## Try It

```sh
cd claw0
python agents/s07_soul_memory.py
```

On the first run, a sample `workspace/SOUL.md` is created automatically.

Try the following conversation:

```
You > My name is Alex and I'm working on a Rust project called tundra.
```

Observe whether the agent calls `memory_write` automatically. Then ask:

```
You > What project am I working on?
```

Watch the `memory_search` call. Use `/soul` to view the current personality, `/memory` to view memory status. Editing `workspace/SOUL.md` takes effect on the next turn.
