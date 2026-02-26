# 第03章: セッションとコンテキストガード

> セッションはJSONLファイル。追記し、再生し、大きくなったら要約する。

## アーキテクチャ

```
    User Input
        |
        v
    SessionStore.load_session()  --> JSONL から messages[] を再構築
        |
        v
    ContextGuard.guard_api_call()
        |
        +-- 試行 0: 通常の呼び出し
        |       |
        |   オーバーフロー? --no--> 成功
        |       |yes
        +-- 試行 1: 巨大なツール結果を切り詰め
        |       |
        |   オーバーフロー? --no--> 成功
        |       |yes
        +-- 試行 2: LLM要約で履歴を圧縮
        |       |
        |   オーバーフロー? --yes--> raise
        |
    SessionStore.save_turn()  --> JSONL に追記
        |
        v
    レスポンスを表示

    ファイル配置:
    workspace/.sessions/agents/{agent_id}/sessions/{session_id}.jsonl
    workspace/.sessions/agents/{agent_id}/sessions.json  (インデックス)
```

## 新しい概念

- **SessionStore**: JSONL永続化。書き込みは追記、読み込みは再生。
- **_rebuild_history()**: フラットなJSONLをAPI互換の messages[] に変換。
- **ContextGuard**: 3段階のオーバーフローリトライ (通常 -> 切り詰め -> 圧縮 -> 失敗)。
- **compact_history()**: LLMが生成した要約で古いメッセージを置換。
- **REPLコマンド**: `/new`、`/switch`、`/context`、`/compact` でセッション管理。

## コードウォークスルー

### 1. JSONL追記と再生

各セッションは `.jsonl` ファイル -- 1行に1つのJSONレコード。追記のみの書き込みはアトミック (ファイル全体の書き換えが不要)。4種類のレコードタイプ:

```python
{"type": "user", "content": "Hello", "ts": 1234567890}
{"type": "assistant", "content": [{"type": "text", "text": "Hi!"}], "ts": ...}
{"type": "tool_use", "tool_use_id": "toolu_...", "name": "read_file", "input": {...}, "ts": ...}
{"type": "tool_result", "tool_use_id": "toolu_...", "content": "file contents", "ts": ...}
```

`_rebuild_history()` メソッドはこれらのフラットなレコードをAnthropic APIフォーマット (user/assistantの厳密な交互配置、assistant内のtool_use、user内のtool_result) に変換する:

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
            # 直前のassistantメッセージにマージ
            block = {"type": "tool_use", "id": record["tool_use_id"],
                     "name": record["name"], "input": record["input"]}
            if messages and messages[-1]["role"] == "assistant":
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "assistant", "content": [block]})
        elif rtype == "tool_result":
            # 連続するresultを1つのuserメッセージにマージ
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

### 2. 3段階のガード

`guard_api_call()` は全てのAPI呼び出しをラップする。コンテキストがオーバーフローした場合、段階的に積極的な戦略でリトライする:

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

### 3. 履歴の圧縮

最も古い50%のメッセージをプレーンテキストに変換し、LLMに要約を依頼し、要約 + 最近のメッセージで置換する:

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
    # 古いメッセージを要約 + "Understood" ペアで置換
    compacted = [
        {"role": "user", "content": "[Previous conversation summary]\n" + summary},
        {"role": "assistant", "content": [{"type": "text",
         "text": "Understood, I have the context."}]},
    ]
    compacted.extend(messages[compress_count:])
    return compacted
```

## 試してみる

```sh
python en/s03_sessions.py

# セッションの作成と切り替え
# You > /new my-project
# You > Tell me about Python generators
# You > /new experiments
# You > What is 2+2?
# You > /switch my-p     (前方一致)

# コンテキスト使用量を確認
# You > /context
# Context usage: ~1,234 / 180,000 tokens
# [####--------------------------] 0.7%

# コンテキストが大きくなったら手動で圧縮
# You > /compact
```

## OpenClaw での実装

| 観点              | claw0 (本ファイル)             | OpenClaw 本番環境                       |
|-------------------|--------------------------------|-----------------------------------------|
| 保存形式          | JSONL ファイル、セッションごと | 同じ JSONL フォーマット                 |
| 再生              | `_rebuild_history()`           | 同じ再構築ロジック                      |
| オーバーフロー処理 | 3段階のガード                 | 同じパターン + トークンカウントAPI      |
| 圧縮              | 古いメッセージのLLM要約        | 同じアプローチ、適応的圧縮              |
| トークン推定      | `len(text) // 4` ヒューリスティック | API提供のトークンカウント           |
