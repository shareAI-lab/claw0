# 第06章: インテリジェンス

> システムプロンプトはディスク上のファイルから組み立てる。ファイルを入れ替えれば性格が変わる。

## アーキテクチャ

```
    起動時                                ターンごと
    =======                              ========

    BootstrapLoader                      User Input
    SOUL.md, IDENTITY.md, ... を読み込み      |
    ファイルごとに切り詰め (20k)               v
    合計を制限 (150k)                    _auto_recall(user_input)
         |                              TF-IDFでメモリを検索
         v                                   |
    SkillsManager                            v
    SKILL.md のディレクトリを走査        build_system_prompt()
    フロントマターを解析                 8層を組み立て:
    名前で重複排除                           1. アイデンティティ
         |                                  2. ソウル (性格)
         v                                  3. ツールガイダンス
    bootstrap_data + skills_block           4. スキル
    (全ターンでキャッシュ)                  5. メモリ (常駐 + 想起)
                                            6. ブートストラップ (残りのファイル)
                                            7. ランタイムコンテキスト
                                            8. チャネルヒント
                                                |
                                                v
                                            LLM API 呼び出し

    前のレイヤーほど行動への影響が強い。
    SOUL.md がレイヤー2にあるのはまさにこの理由。
```

## 本章のポイント

- **BootstrapLoader**: ワークスペースから最大8つのMarkdownファイルを読み込み、ファイルごとおよび合計の上限付き。
- **SkillsManager**: 複数のディレクトリから `SKILL.md` ファイルを走査し、YAMLフロントマターを解析。
- **MemoryStore**: 2層のストレージ (常駐 MEMORY.md + 日次 JSONL)、TF-IDF検索。
- **_auto_recall()**: ユーザーのメッセージを使ってメモリを検索し、結果をプロンプトに注入。
- **build_system_prompt()**: 8つのレイヤーを1つの文字列に組み立て、毎ターン再構築。

## コードウォークスルー

### 1. build_system_prompt() -- 8層の組み立て

この関数がインテリジェンスシステムの核心。メモリが更新される可能性があるため、毎ターン異なるシステムプロンプトを生成する。

```python
def build_system_prompt(mode="full", bootstrap=None, skills_block="",
                        memory_context="", agent_id="main", channel="terminal"):
    sections: list[str] = []

    # レイヤー 1: アイデンティティ
    identity = bootstrap.get("IDENTITY.md", "").strip()
    sections.append(identity if identity else "You are a helpful AI assistant.")

    # レイヤー 2: ソウル (性格) -- 前にあるほど影響が強い
    if mode == "full":
        soul = bootstrap.get("SOUL.md", "").strip()
        if soul:
            sections.append(f"## Personality\n\n{soul}")

    # レイヤー 3: ツールガイダンス
    tools_md = bootstrap.get("TOOLS.md", "").strip()
    if tools_md:
        sections.append(f"## Tool Usage Guidelines\n\n{tools_md}")

    # レイヤー 4: スキル
    if mode == "full" and skills_block:
        sections.append(skills_block)

    # レイヤー 5: メモリ (常駐 + 自動検索)
    if mode == "full":
        # ... MEMORY.md と想起されたメモリを結合

    # レイヤー 6: ブートストラップコンテキスト (HEARTBEAT.md, BOOTSTRAP.md, AGENTS.md, USER.md)
    # レイヤー 7: ランタイムコンテキスト (エージェントID、モデル、チャネル、時刻)
    # レイヤー 8: チャネルヒント ("You are responding via Telegram.")

    return "\n\n".join(sections)
```

### 2. MemoryStore.search_memory() -- TF-IDF検索

純粋なPython実装、外部ベクトルデータベースは不要。全メモリチャンクを読み込み、TF-IDFベクトルを計算し、コサイン類似度でランク付けする。

```python
def search_memory(self, query: str, top_k: int = 5) -> list[dict]:
    chunks = self._load_all_chunks()   # MEMORY.md の段落 + 日次JSONLのエントリ
    query_tokens = self._tokenize(query)
    chunk_tokens = [self._tokenize(c["text"]) for c in chunks]

    # 全チャンクにわたる文書頻度
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

### 3. ハイブリッド検索パイプライン -- ベクトル + キーワード + MMR

完全な検索パイプラインは5つのステージを連鎖させる:

1. **キーワード検索** (TF-IDF): 上記と同じアルゴリズム、コサイン類似度でtop-10を返す
2. **ベクトル検索** (ハッシュ射影): ハッシュベースのランダム射影による模擬埋め込み、top-10を返す
3. **マージ**: テキスト先頭による和集合、重み付き結合 (`vector_weight=0.7, text_weight=0.3`)
4. **時間減衰**: `score *= exp(-decay_rate * age_days)`, 新しいメモリほど高スコア
5. **MMR再ランク付け**: `MMR = lambda * relevance - (1-lambda) * max_similarity_to_selected`, トークン集合のJaccard類似度で多様性を確保

ハッシュベースのベクトル埋め込みは、外部の埋め込みAPIを必要とせずにデュアルチャネル検索の**パターン**を教える。

### 4. _auto_recall() -- 自動メモリ注入

各LLM呼び出しの前に、関連するメモリを検索してシステムプロンプトに注入する。ユーザーが明示的に依頼する必要はない。

```python
def _auto_recall(user_message: str) -> str:
    results = memory_store.search_memory(user_message, top_k=3)
    if not results:
        return ""
    return "\n".join(f"- [{r['path']}] {r['snippet']}" for r in results)

# エージェントループ内で、毎ターン:
memory_context = _auto_recall(user_input)
system_prompt = build_system_prompt(
    mode="full", bootstrap=bootstrap_data,
    skills_block=skills_block, memory_context=memory_context,
)
```

## 試してみる

```sh
python ja/s06_intelligence.py

# ワークスペースにファイルを作成してシステム全体を確認:
# workspace/SOUL.md       -- "You are warm, curious, and encouraging."
# workspace/IDENTITY.md   -- "You are Luna, a personal AI companion."
# workspace/MEMORY.md     -- "User prefers Python over JavaScript."

# 組み立てられたプロンプトを確認
# You > /prompt

# 読み込まれたブートストラップファイルを確認
# You > /bootstrap

# メモリを検索
# You > /search python

# 何かを伝え、後でそれについて質問する
# You > My favorite color is blue.
# You > What do you know about my preferences?
# (auto-recall が色のメモリを見つけて注入する)
```

## OpenClaw での実装

| 観点             | claw0 (本ファイル)           | OpenClaw 本番環境                       |
|------------------|------------------------------|-----------------------------------------|
| プロンプト組み立て | 8層の `build_system_prompt` | 同じレイヤードアプローチ                |
| ブートストラップ | ワークスペースから読み込み   | 同じファイルセット + エージェントごとのオーバーライド |
| メモリ検索       | ハイブリッドパイプライン (TF-IDF + ベクトル + MMR) | 同じアプローチ + オプションの埋め込みAPI |
| スキル検出       | SKILL.md のディレクトリ走査  | 同じ走査 + プラグインシステム           |
| 自動想起         | 毎ユーザーメッセージで検索   | 同じパターン、設定可能な top_k          |
