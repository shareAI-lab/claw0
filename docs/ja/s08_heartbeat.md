# s08: Heartbeat & Proactive Behavior (ハートビートと能動的行動)

> "Not just reactive -- proactive" -- 受動的なチャットから能動的な監視へ。

## At a Glance

```
  +--- HeartbeatRunner (background thread) -------+
  |  loop every 1s:                                |
  |    should_run()?                                |
  |      [1] enabled?        (HEARTBEAT.md exists)  |
  |      [2] interval?       (>= N seconds elapsed) |
  |      [3] active hours?   (09:00-22:00)          |
  |      [4] has content?    (skip empty headings)   |
  |      [5] main lane idle? (lock.acquire nowait)   |
  |      [6] not running?    (self.running == False)  |
  |    all pass -> acquire lock -> run agent          |
  +---------------------------------------------------+
            |                         |
            v                    (mutual exclusion)
     Agent(HEARTBEAT.md)              |
            |                         v
            v                    User Message
     +------+------+            (holds same lock)
     |             |
  HEARTBEAT_OK   Content
  (suppress)       |
                   v
              SHA-256 dedup
              (24h window)
                   |
                   v
              Output to user
```

- **What we build**: バックグラウンドスレッドにより、エージェントが定期的に報告すべき事項がないか確認する。
- **Core mechanism**: 6 ステップの should_run チェックチェーン + HEARTBEAT_OK サイレント信号 + SHA-256 重複排除。
- **Design pattern**: ハートビートはユーザーメッセージに譲歩する (排他ロック)。安全なタイミングでのみ実行される。

## The Problem

1. **完全に受動的**: s07 のエージェントはユーザーが発話するまで応答できない。ユーザーが「3時までにレポートを提出」と伝えても、エージェントは 2:50 に能動的にリマインドしない。
2. **深夜の妨害**: 時間ウィンドウの制御がなく、バックグラウンドタスクが深夜 3 時にユーザーへメッセージを送信する可能性がある。
3. **重複通知**: エージェントが同じ内容を繰り返し報告する可能性がある -- 「TODO があります」を 30 秒ごとに送り続け、ユーザーが疲弊する。

## How It Works

### 1. HEARTBEAT.md -- ハートビート指示ファイル

SOUL.md と同様に、HEARTBEAT.md は Markdown ファイルであり、ハートビート時にエージェントが何を確認すべきかを定義する:

```md
# Heartbeat Instructions

Check the following and report ONLY if action is needed:

1. Review today's memory log for any unfinished tasks or pending items.
2. If the user mentioned a deadline or reminder, check if it is approaching.
3. If there are new daily memories, summarize any actionable items.

If nothing needs attention, respond with exactly: HEARTBEAT_OK
```

ファイルの存在がハートビートの有効/無効を決定する。ファイルを削除すればハートビートは無効になり、ファイルを作成すれば有効になる。コード変更は不要。

**ファイルの存在 = スイッチ。非エンジニアでもハートビートを制御できる。**

### 2. 6 ステップの should_run チェックチェーン

毎秒チェックし、6 つの条件すべてが通過した場合のみハートビートを実行する:

```python
def should_run(self) -> tuple[bool, str]:
    if not self._is_enabled():
        return False, "disabled (no HEARTBEAT.md)"
    if not self._interval_elapsed():
        return False, "interval not elapsed"
    if not self._is_active_hours():
        return False, "outside active hours"
    if not self._heartbeat_has_content():
        return False, "HEARTBEAT.md has no actionable content"
    if not self._main_lane_idle():
        return False, "main lane busy (user message in progress)"
    if self.running:
        return False, "heartbeat already running"
    return True, "ok"
```

`_heartbeat_has_content()` は見出しのみの行や空のチェックボックスをスキップし、ファイルに実質的な指示があることを確認する:

```python
def _heartbeat_has_content(self) -> bool:
    if not self.heartbeat_path.exists():
        return False
    content = self.heartbeat_path.read_text(encoding="utf-8")
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^#+(\s|$)", stripped):
            continue
        if re.match(r"^[-*+]\s*(\[[\sXx]?\]\s*)?$", stripped):
            continue
        return True
    return False
```

**各ステップの失敗が具体的な理由を返すため、デバッグが容易: ハートビートが発火しない理由を推測する必要がない。**

### 3. レーン排他制御 -- ハートビートはユーザーに譲歩する

ハートビートとユーザーメッセージは同一の `threading.Lock` を共有する。重要な違いはロックの取得方法にある:

```python
# ハートビートスレッド: ノンブロッキング取得、失敗ならスキップ
def _background_loop(self, agent_fn) -> None:
    while not self._stop_event.is_set():
        should, reason = self.should_run()
        if should:
            acquired = self._lock.acquire(blocking=False)
            if not acquired:
                self._stop_event.wait(1.0)
                continue
            try:
                self.running = True
                self.last_run = time.time()
                result = self.run_heartbeat_turn(agent_fn)
                if result:
                    with self._output_lock:
                        self._output_queue.append(result)
            finally:
                self.running = False
                self._lock.release()
        self._stop_event.wait(1.0)

# メインスレッド: ブロッキング取得、ハートビート完了後に即座に入る
heartbeat._lock.acquire()
try:
    # ユーザーメッセージの処理...
finally:
    heartbeat._lock.release()
```

**ユーザー優先: ハートビートはノンブロッキング (失敗ならスキップ)、ユーザーメッセージはブロッキング (ハートビート完了を待つ)。ユーザーがハートビートを待つ必要は決してない。**

### 4. HEARTBEAT_OK -- サイレント信号

エージェントが報告すべき事項がないと判断した場合に `HEARTBEAT_OK` を返し、システムはユーザーに出力しない:

```python
def _strip_heartbeat_ok(self, text: str) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return True, ""
    without_token = stripped.replace(HEARTBEAT_OK_TOKEN, "").strip()
    # トークン除去後に実質的な内容がないか残留 5 文字以下 -> サイレント
    if not without_token or len(without_token) <= 5:
        return True, ""
    if HEARTBEAT_OK_TOKEN in stripped:
        return False, without_token
    return False, stripped
```

**HEARTBEAT_OK によりエージェント自身が「報告すべき事項があるか」を判断する。ハートビートのたびにユーザーを妨害することがない。**

### 5. SHA-256 重複排除 -- 24 時間ウィンドウ

エージェントが報告すべき事項があると判断しても、同じ内容が 24 時間以内に既に送信されていれば再送しない:

```python
def _content_hash(self, content: str) -> str:
    normalized = content.strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

def is_duplicate(self, content: str) -> bool:
    h = self._content_hash(content)
    now = time.time()
    expired = [k for k, v in self.dedup_cache.items()
               if now - v > DEDUP_WINDOW_SECONDS]
    for k in expired:
        del self.dedup_cache[k]
    if h in self.dedup_cache:
        return True
    self.dedup_cache[h] = now
    return False
```

**ハッシュは 16 文字のみ。正規化 (strip + lower) 後にハッシュ化し、大文字小文字やスペースの違いによる誤判定を防ぐ。**

### 6. ハートビート実行の全フロー

一回の完全なハートビート: 指示読み込み -> プロンプト構築 -> エージェント呼び出し -> サイレント判定 -> 重複判定 -> 出力:

```python
def run_heartbeat_turn(self, agent_fn) -> str | None:
    heartbeat_content = self.heartbeat_path.read_text(encoding="utf-8").strip()
    heartbeat_prompt = (
        "This is a scheduled heartbeat check. "
        "Follow the HEARTBEAT.md instructions below strictly.\n"
        "Do NOT infer or repeat old tasks from prior context.\n"
        "If nothing needs attention, respond with exactly: HEARTBEAT_OK\n\n"
        f"--- HEARTBEAT.md ---\n{heartbeat_content}\n--- end ---"
    )
    response_text = agent_fn(heartbeat_prompt)
    if not response_text:
        return None
    is_ok, cleaned = self._strip_heartbeat_ok(response_text)
    if is_ok:
        return None
    if self.is_duplicate(cleaned):
        return None
    return cleaned
```

ハートビートはシングルターン/ツールなしの呼び出しを使用し、トークン消費を抑え、ハートビート中の記憶書き込みなどの副作用も回避する。

**ハートビートは低コストなプローブ: 簡潔な質問、簡潔な回答、ツールチェーンを起動しない。**

## What Changed from s07

| コンポーネント | s07 | s08 |
|-----------|-----|-----|
| 行動モデル | 受動的 (ユーザー入力待ち) | 能動的 + 受動的 (バックグラウンドスレッドが周期的にトリガー) |
| 設定ファイル | SOUL.md + MEMORY.md | HEARTBEAT.md を追加 |
| スレッドモデル | シングルスレッド | メインスレッド + バックグラウンドハートビートスレッド |
| 並行制御 | なし | threading.Lock 排他制御 |
| 出力フィルタリング | なし | HEARTBEAT_OK サイレント + SHA-256 重複排除 |
| 時間制御 | なし | active_hours アクティブ時間帯 |

**Key shift**: エージェントが「ユーザーが話しかけてから動く」から「自ら報告すべき事項がないか確認する」へ。ハートビートシステムは OpenClaw の最も特徴的な機能の一つである。

## Design Decisions

**なぜ単なるタイマーではなく 6 ステップのチェーンなのか?**

純粋なタイマーでは複雑な制約を処理できない: ユーザーがチャット中に割り込むべきではなく、深夜にメッセージを送るべきではなく、ファイルが削除されたら実行すべきではない。6 ステップチェーンはすべての前提条件を明示的にリストアップし、いずれかのステップが失敗すれば具体的な理由を返すため、デバッグが容易になる。

**なぜ空文字列ではなく HEARTBEAT_OK なのか?**

LLM は完全に空のレスポンスを返さない。モデルに明確なトークンを返させる方が、空を期待するよりも確実だ。これによりモデルに明確な終了メカニズムも与えられる: 確認して問題なければ HEARTBEAT_OK と言えばよく、無意味な内容を無理に出力する必要がない。

**なぜ時間ベースではなく SHA-256 コンテンツハッシュで重複排除するのか?**

時間のみによる重複排除は見逃しが生じる: 同じ事象が異なる時刻にトリガーされた場合、時刻は異なるが内容は同じだ。コンテンツハッシュを使えば、出力が同一 (大文字小文字とスペースを無視) であれば重複と判定する。

**In production OpenClaw:** ハートビートは `src/heartbeat/` 内の `HeartbeatRunner` で実装されており、should_run チェックチェーンは本節と完全に一致する。本番版は cron 式によるトリガー時刻定義、HTML/Markdown でラップされた形式 (`<b>HEARTBEAT_OK</b>`) にも対応する HEARTBEAT_OK 検出、少量のテキストを伴う OK もサイレントとして扱う ackMaxChars をサポートしている。排他制御は単純な排他ロックではなく CommandLane のキュー深度による判定を使用し、より精緻である。

## Try It

```sh
cd claw0
python agents/s08_heartbeat.py
```

デフォルトのハートビート間隔は 60 秒 (`HEARTBEAT_INTERVAL` 環境変数で調整可能)。

まず TODO を書き込み、ハートビートが自動的に検出するか観察する:

```
You > Remember that I need to submit the report by 3pm tomorrow.
Assistant: I've saved that to memory...

(ハートビートのトリガーを待つ)
[Heartbeat] Reminder: you need to submit the report by 3pm tomorrow.
```

ハートビートの状態確認と手動トリガー:

```
You > /heartbeat
--- Heartbeat Status ---
  Enabled: True
  Active hours: True
  Interval: 60s
  Last run: 45s ago
  Next in: ~15s
  Should run: False (interval not elapsed)
  Dedup cache: 1 entries

You > /trigger
[Heartbeat] Your memory log shows a pending task...
```
