# s03: Sessions (セッション永続化)

> "Conversations that survive restarts" -- 再起動してもコンテキストを失わない。それが本当に使えるエージェントだ。

## At a Glance

```
                sessions.json (metadata index)
                     |
User --> agent_loop() --> SessionStore --> transcripts/
              |                             session_abc.jsonl
         load_session()                     session_def.jsonl
         save_turn()
              |
         session_key = agent:channel:peer
```

- **What we build**: JSONL 永続化レイヤーにより、対話履歴をプロセス再起動後も完全に復元する。
- **Core mechanism**: append-only の JSONL transcript が source of truth であり、`_rebuild_history()` がそこから API フォーマットの messages を再構築する。
- **Design pattern**: agent_loop が自己完結型関数から外部状態を受け取る純粋関数へ変化する。履歴の読み込み、メッセージの処理、結果の保存はすべて SessionStore を通じて行う。

## The Problem

1. **終了すれば記憶喪失。** s01/s02 のすべての対話履歴はメモリ上にしかない。プロセスが終了すればゼロに戻る。30 分かけてデバッグしたエージェントも、再起動すれば何も覚えていない。

2. **異なる会話を区別できない。** 一つのエージェントが複数の独立したセッションを同時に維持する必要がある。異なるユーザー、異なるトピック、それぞれが独自の履歴を持ち、互いに干渉しないべきだ。

3. **ツール呼び出し結果が失われる。** モデルが以前ファイルを読んでいた場合、再起動後にそれを知らず、重複読み取りや矛盾した回答をする可能性がある。

## How It Works

### 1. JSONL Transcript フォーマット

各セッションに対応する `.jsonl` ファイルがあり、各行が一つの JSON オブジェクトである:

```
{"type":"session","id":"abc123","key":"main:cli:user","created":"2025-01-01T00:00:00Z"}
{"type":"user","content":"hello","ts":"2025-01-01T00:00:01Z"}
{"type":"assistant","content":"Hi there!","ts":"2025-01-01T00:00:02Z"}
{"type":"tool_use","name":"read_file","tool_use_id":"tu_001","input":{"path":"config.json"},"ts":"..."}
{"type":"tool_result","tool_use_id":"tu_001","output":"{\"key\": \"value\"}","ts":"..."}
```

JSONL は append-only: 追記のみで変更しない。プロセスがクラッシュしても最後の一行を失うだけであり、`tail -f` でリアルタイム監視でき、ロックも不要だ。

**JSONL が source of truth であり、sessions.json は単なるインデックスである。**

### 2. SessionStore によるセッションの作成と読み込み

セッション作成時に一意の session_id と対応する JSONL ファイルを生成し、読み込み時に JSONL から messages を再構築する:

```python
def create_session(self, session_key: str) -> dict:
    session_id = uuid.uuid4().hex[:12]
    now = datetime.now(timezone.utc).isoformat()
    transcript_file = f"{session_key.replace(':', '_')}_{session_id}.jsonl"
    metadata = {
        "session_key": session_key, "session_id": session_id,
        "created_at": now, "updated_at": now,
        "message_count": 0, "transcript_file": transcript_file,
    }
    self._index[session_key] = metadata
    self._save_index()
    self.append_transcript(session_key, {
        "type": "session", "id": session_id, "key": session_key, "created": now,
    })
    return metadata

def load_session(self, session_key: str) -> dict:
    if session_key not in self._index:
        metadata = self.create_session(session_key)
        return {"metadata": metadata, "history": []}
    metadata = self._index[session_key]
    history = self._rebuild_history(metadata["transcript_file"])
    return {"metadata": metadata, "history": history}
```

**load_session は存在しないセッションを自動作成する。呼び出し側は「新規作成」と「復元」を区別する必要がない。**

### 3. 核心: _rebuild_history()

JSONL から Anthropic API フォーマットの messages リストを再構築する。難所は tool_use/tool_result の再組立にある。JSONL では独立した行だが、API は tool_use を assistant メッセージに、tool_result を user メッセージに含めることを要求する:

```python
def _rebuild_history(self, transcript_file: str) -> list[dict]:
    messages: list[dict] = []
    pending_tool_uses: list[dict] = []

    for line in filepath.read_text(encoding="utf-8").strip().splitlines():
        entry = json.loads(line)
        entry_type = entry.get("type")

        if entry_type == "session":
            continue

        if entry_type == "user":
            if pending_tool_uses:
                messages.append({"role": "assistant", "content": pending_tool_uses})
                pending_tool_uses = []
            messages.append({"role": "user", "content": entry.get("content", "")})

        elif entry_type == "tool_use":
            pending_tool_uses.append({
                "type": "tool_use", "id": entry.get("tool_use_id", ""),
                "name": entry.get("name", ""), "input": entry.get("input", {}),
            })

        elif entry_type == "tool_result":
            if pending_tool_uses:
                messages.append({"role": "assistant", "content": pending_tool_uses})
                pending_tool_uses = []
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": entry.get("tool_use_id", ""),
                "content": entry.get("output", ""),
            }]})

    return messages
```

**pending_tool_uses バッファが鍵となる。連続する tool_use 行を一つの assistant メッセージの content 配列にまとめる。**

### 4. agent_loop が純粋関数に変化

agent_loop は自身で messages を管理せず、SessionStore から読み込み、処理後に保存する:

```python
def agent_loop(user_input: str, session_key: str,
               session_store: SessionStore, client: Anthropic) -> str:
    session_data = session_store.load_session(session_key)
    messages = session_data["history"]
    messages.append({"role": "user", "content": user_input})

    all_assistant_blocks: list = []
    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=4096,
            system=SYSTEM_PROMPT, tools=TOOLS, messages=messages,
        )
        all_assistant_blocks.extend(response.content)
        # ... ツールループのロジックは s02 と同様 ...

    session_store.save_turn(session_key, user_input, all_assistant_blocks)
    return final_text
```

**agent_loop は「状態を所有する」から「状態を受け取る」へ変化した。これがマルチチャネルアーキテクチャへの準備となる。**

### 5. Session Key フォーマット

`<agent_id>:<channel>:<peer_id>`、例えば `main:cli:user`。三段構成の設計:

- **agent**: 複数のエージェントインスタンスをサポート
- **channel**: 同一ユーザーでも異なるチャネルでは異なるセッションを持つ
- **peer**: 同一チャネルの異なるユーザーが異なるセッションを持つ

**このフォーマットは s04 のマルチチャネルアーキテクチャでそのまま使用される。**

## What Changed from s02

| コンポーネント | s02 | s03 |
|-----------|-----|-----|
| 履歴保存 | メモリのみ (終了時に消失) | JSONL ファイル (永続化) |
| agent_loop のシグネチャ | 引数なし、内部で messages を管理 | session_key + session_store を受け取る |
| ツール結果 | メモリ内のみ | transcript にも書き込み |
| マルチセッション | 非対応 | session_key で区別 |
| セッションコマンド | なし | /new, /sessions, /switch, /history, /delete |

**Key shift**: agent_loop が自己完結型関数から純粋関数へ変化し、状態管理を SessionStore に委譲した。

## Design Decisions

**なぜ SQLite ではなく JSONL なのか?**

append-only の書き込みはメッセージログに自然に適合し、クラッシュしても既存データを破損しない。人間が読める形式であり、直接 `cat` で確認できる。依存関係がなく、データベースドライバも不要。行単位で処理でき、全体を一度に読み込む必要がない。

**なぜ sessions.json と JSONL を分離しているのか?**

sessions.json はインデックスであり、JSONL はコンテンツである。全セッションの一覧はインデックスを読むだけ (高速)、特定セッションの復元時にのみ JSONL を読む (オンデマンド)。データベースにおけるインデックスとデータの分離に似ている。

**In production OpenClaw:** session key フォーマットは `agent:<agentId>:<channel>:<peerKind>:<peerId>` (peerKind で direct/group/thread を区別する)、JSONL は `~/.openclaw/agents/<agentId>/sessions/` に保存される。履歴の復元時にはモデルのコンテキストウィンドウサイズに応じて古いメッセージを自動的に切り詰め、セッションには TTL を設定して自動的に期限切れにできる。

## Try It

```sh
cd claw0
python agents/s03_sessions.py
```

次の操作を試してみよう:

- エージェントと数回会話し、`/quit` で終了する
- 再起動する -- "Restored: N previous turns" という表示を確認し、コンテキストが復元されていることを確認する
- `/new` で新しいセッションを作成し、`/sessions` で全セッションを確認する
- `/switch <key>` で以前のセッションに切り替え、コンテキストが正しいことを検証する
- `workspace/.sessions/transcripts/` 配下の JSONL ファイルを確認する
