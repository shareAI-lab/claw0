# s07: Soul & Memory (パーソナリティシステムと記憶)

> "Give it a soul, let it remember" -- ステートレスな関数からアイデンティティを持つ存在へ。

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

- **What we build**: SOUL.md でエージェントのパーソナリティを定義し、二層記憶システム (常駐 + 日次) でエージェントがセッションをまたいで情報を記憶する。
- **Core mechanism**: Soul を system prompt の先頭に注入。記憶は TF-IDF + コサイン類似度で検索する。
- **Design pattern**: ファイル駆動設定 (非エンジニアでもパーソナリティを編集可能) + ツール駆動書き込み (エージェントが何を記憶するか自律的に判断)。

## The Problem

1. **画一的な応答**: すべてのエージェントが "You are a helpful assistant" であり、ユーザーはユニークなキャラクターと対話している実感を得られない。エージェントを変えても話し方がまったく同じになる。
2. **終了すれば記憶喪失**: すべての会話はメモリ上にのみ存在し、再起動すればすべて消える。昨日エージェントに名前を伝えても、今日にはすべて忘れている。
3. **能動的な記憶をしない**: 同一セッション内であっても、エージェントはユーザーの好み、プロジェクト情報、TODO を永続化して保存せず、後続の会話でこれらの情報を活用できない。

## How It Works

### 1. SoulSystem -- パーソナリティ注入

SOUL.md は Markdown ファイルで、エージェントの性格と言語スタイルを定義する:

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

読み込み後、system prompt の先頭に結合する:

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

**Soul を system prompt の先頭に配置する理由は、LLM が冒頭部分に最も強い注意を払うためであり、パーソナリティ定義がそれ以降のすべての出力に強く影響する。**

### 2. MemoryStore -- 二層記憶

記憶は二層に分かれる:

- **MEMORY.md (常駐記憶)**: 手動で管理する長期的な事実。例: 「ユーザーは TypeScript を好み、Vim を使用」
- **memory/YYYY-MM-DD.md (日次ログ)**: エージェントが `memory_write` ツールで自動書き込み、1日1ファイル

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

日次ログのファイル構造:

```md
# Memory Log: 2026-02-24

## [14:30:15] preference

User prefers dark mode and vim keybindings.

## [15:20:03] fact

User's project uses Python 3.12 with FastAPI.
```

**常駐記憶は品質が高く (人手でレビュー済み)、日次ログは量が多いが検索可能。二層が相互に補完する。**

### 3. TF-IDF 検索 -- キーワードベースの記憶検索

エージェントが `memory_search` を呼び出すと、システムは TF-IDF + コサイン類似度で最も関連性の高い記憶チャンクを見つける:

```python
def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
    chunks = self._load_all_chunks()
    if not chunks:
        return []

    # ドキュメント頻度の集計
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

    # クエリの TF-IDF ベクトル
    query_tokens = _tokenize(query)
    query_tf = Counter(query_tokens)
    query_vec = {t: (count / max(len(query_tokens), 1)) * _idf(t)
                 for t, count in query_tf.items()}

    # 各チャンクとのコサイン類似度を計算し top_k を取得
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

トークナイズは中英混在テキストに対応する:

```python
def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", text.lower())
    return [t for t in tokens if len(t) > 1]
```

**TF-IDF はエンベディング検索の教育的な代替手段: 原理は同じ (テキスト -> ベクトル -> 類似度) だが、外部依存ゼロで各ステップを手動で追跡できる。**

### 4. System Prompt の階層構築

毎ターンごとに system prompt を再構築する。記憶が更新されている可能性があるためだ:

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

最終的なプロンプト構造:

```
[SOUL.md]               <-- パーソナリティ、最高優先度
---
[Base system prompt]     <-- 機能説明 + 日付
---
## Evergreen Memory      <-- MEMORY.md の常駐事実
---
## Recent Memory Context <-- 直近 3 日間のログ断片
```

**パーソナリティが先頭、記憶が末尾 -- パーソナリティの一貫性を保ちつつコンテキストを提供する。**

### 5. ツール定義とエージェントループ

エージェントは二つのツールで記憶を操作する:

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

エージェントループは s01 と同じ構造だが、ツール処理が追加されている -- `stop_reason == "tool_use"` の場合にツールを実行して結果を返し、エージェントの推論を続行させる。

**エージェントがいつ記憶し、いつ検索するかを自律的に判断する。ハードコードされたルールは不要。**

## What Changed from s06

| コンポーネント | s06 | s07 |
|-----------|-----|-----|
| パーソナリティ | ハードコードされた system_prompt | SOUL.md ファイル、動的ロード |
| 記憶 | 永続化なし | 二層: MEMORY.md + memory/daily |
| ツール | なし | memory_write + memory_search |
| System prompt | 固定文字列 | 階層構築: soul + base + evergreen + recent |
| 検索 | なし | TF-IDF + コサイン類似度 |
| 実行モード | WebSocket ゲートウェイ | 対話式 REPL (/soul と /memory コマンド付き) |

**Key shift**: 「正しいエージェントにルーティングする」から「エージェントに魂と記憶を与える」へ。エージェントはもはやステートレスな関数ではなく、永続的なアイデンティティと検索可能な記憶を持つ存在となった。

## Design Decisions

**なぜコードではなくファイルベースのパーソナリティなのか?**

ファイルは非エンジニアでも編集できる。プロダクトマネージャー、コンテンツ運用担当、さらにはエンドユーザーでも SOUL.md を修正してエージェントの性格を調整でき、コード変更も再デプロイも不要だ。これが設定駆動の核心的な考え方である。

**なぜ二層の記憶なのか?**

MEMORY.md には人手でレビューされた確定的な事実を格納し、品質が高い。日次ログにはエージェントが自動書き込みした生の記録を格納し、量は多いが品質にばらつきがある。検索時には関連性でソートし、ノイズは自動的にフィルタリングされる。二層により、信頼性の高いコア知識と検索可能な豊富な詳細の両方を備えることができる。

**なぜエンベディングではなく TF-IDF なのか?**

教育目的である。TF-IDF は追加の API 呼び出しもデータベースも不要で、アルゴリズムが完全に透明だ。原理はエンベディング検索と同じ (テキスト -> ベクトル -> 類似度) であり、ベクトルの品質が異なるだけだ。

**In production OpenClaw:** 検索では sqlite-vec + エンベディングキャッシュ + FTS5 全文検索の三つを併用している。パーソナリティファイルは SOUL.md に加えて IDENTITY.md と AGENTS.md がある。記憶のチャンク分割は heading による分割ではなく、固定サイズ + オーバーラップ戦略を使用している。記憶の書き込みは専用ツールではなく、ファイルシステムツールを通じて行われる。

## Try It

```sh
cd claw0
python agents/s07_soul_memory.py
```

初回実行時に `workspace/SOUL.md` のサンプルファイルが自動生成される。

以下の会話を試してみよう:

```
You > My name is Alex and I'm working on a Rust project called tundra.
```

エージェントが `memory_write` を自動的に呼び出すかどうかを観察する。次に以下を尋ねる:

```
You > What project am I working on?
```

`memory_search` の呼び出しを観察する。`/soul` で現在のパーソナリティを確認し、`/memory` で記憶の状態を確認できる。`workspace/SOUL.md` を編集すれば、次のターンの応答から即座に反映される。
