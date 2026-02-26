# 第07章: ハートビートとCron

> タイマースレッドが「実行すべきか?」を確認し、ユーザーメッセージと並行して作業をキューに入れる。

## アーキテクチャ

```
    メインレーン (ユーザー入力):
        User Input --> lane_lock.acquire() -------> LLM --> Print
                       (ブロッキング: 常に優先)

    ハートビートレーン (バックグラウンドスレッド, 1秒ポーリング):
        should_run()?
            |no --> sleep 1s
            |yes
        _execute():
            lane_lock.acquire(blocking=False)
                |fail --> 譲歩 (ユーザーが優先)
                |success
            HEARTBEAT.md + SOUL.md + MEMORY.md からプロンプトを構築
                |
            run_agent_single_turn()
                |
            解析: "HEARTBEAT_OK"? --> 抑制
                   意味のあるテキスト? --> 重複? --> 抑制
                                           |no
                                       output_queue.append()

    Cronサービス (バックグラウンドスレッド, 1秒ティック):
        CRON.json --> ジョブを読み込み --> tick() 毎秒
            |
        各ジョブ: enabled? --> due? --> _run_job()
            |
        エラー? --> consecutive_errors++ --> >=5? --> 自動無効化
            |ok
        consecutive_errors = 0 --> cron-runs.jsonl に記録
```

## 本章のポイント

- **レーンの排他制御**: `threading.Lock` をユーザーとハートビートで共有。ユーザーは常に優先 (ブロッキング取得); ハートビートは譲歩 (ノンブロッキング)。
- **should_run()**: 各ハートビート試行前に4つの前提条件をチェック。
- **HEARTBEAT_OK**: エージェントが「報告事項なし」を示す規約。
- **CronService**: 3つのスケジュール種別 (`at`, `every`, `cron`)、5回連続エラーで自動無効化。
- **出力キュー**: バックグラウンド結果はスレッドセーフなリスト経由でREPLにドレインされる。

## コードウォークスルー

### 1. レーンの排他制御

最も重要な設計原則: ユーザーメッセージが常に優先。

```python
lane_lock = threading.Lock()

# メインレーン: ブロッキング取得。ユーザーは常にロックを得る。
lane_lock.acquire()
try:
    # ユーザーメッセージを処理、LLMを呼び出す
finally:
    lane_lock.release()

# ハートビートレーン: ノンブロッキング取得。ユーザーがアクティブなら譲歩する。
def _execute(self) -> None:
    acquired = self.lane_lock.acquire(blocking=False)
    if not acquired:
        return   # ユーザーがロックを保持中、今回のハートビートをスキップ
    self.running = True
    try:
        instructions, sys_prompt = self._build_heartbeat_prompt()
        response = run_agent_single_turn(instructions, sys_prompt)
        meaningful = self._parse_response(response)
        if meaningful and meaningful.strip() != self._last_output:
            self._last_output = meaningful.strip()
            with self._queue_lock:
                self._output_queue.append(meaningful)
    finally:
        self.running = False
        self.last_run_at = time.time()
        self.lane_lock.release()
```

### 2. should_run() -- 前提条件チェーンの連鎖

4つのチェックが全てパスする必要がある。ロックは `_execute()` 内で別途テストし、TOCTOUレースを回避する。

```python
def should_run(self) -> tuple[bool, str]:
    if not self.heartbeat_path.exists():
        return False, "HEARTBEAT.md not found"
    if not self.heartbeat_path.read_text(encoding="utf-8").strip():
        return False, "HEARTBEAT.md is empty"

    elapsed = time.time() - self.last_run_at
    if elapsed < self.interval:
        return False, f"interval not elapsed ({self.interval - elapsed:.0f}s remaining)"

    hour = datetime.now().hour
    s, e = self.active_hours
    in_hours = (s <= hour < e) if s <= e else not (e <= hour < s)
    if not in_hours:
        return False, f"outside active hours ({s}:00-{e}:00)"

    if self.running:
        return False, "already running"
    return True, "all checks passed"
```

### 3. CronService -- 3つのスケジュール種別

ジョブは `CRON.json` で定義する。各ジョブは `schedule.kind` と `payload` を持つ:

```python
@dataclass
class CronJob:
    id: str
    name: str
    enabled: bool
    schedule_kind: str       # "at" | "every" | "cron"
    schedule_config: dict
    payload: dict            # {"kind": "agent_turn", "message": "..."}
    consecutive_errors: int = 0

def _compute_next(self, job, now):
    if job.schedule_kind == "at":
        ts = datetime.fromisoformat(cfg.get("at", "")).timestamp()
        return ts if ts > now else 0.0
    if job.schedule_kind == "every":
        every = cfg.get("every_seconds", 3600)
        # 予測可能な発火時刻のためにアンカーに揃える
        steps = int((now - anchor) / every) + 1
        return anchor + steps * every
    if job.schedule_kind == "cron":
        return croniter(expr, datetime.fromtimestamp(now)).get_next(datetime).timestamp()
```

5回連続エラーで自動無効化:

```python
if status == "error":
    job.consecutive_errors += 1
    if job.consecutive_errors >= 5:
        job.enabled = False
else:
    job.consecutive_errors = 0
```

## 試してみる

```sh
python ja/s07_heartbeat_cron.py

# workspace/HEARTBEAT.md にインストラクションを作成:
# "Check if there are any unread reminders. Reply HEARTBEAT_OK if nothing to report."

# ハートビートの状態を確認
# You > /heartbeat

# ハートビートを強制実行
# You > /trigger

# Cronジョブの一覧 (workspace/CRON.json が必要)
# You > /cron

# レーンロックの状態を確認
# You > /lanes
# main_locked: False  heartbeat_running: False
```

`CRON.json` の例:

```json
{
  "jobs": [
    {
      "id": "daily-check",
      "name": "Daily Check",
      "enabled": true,
      "schedule": {"kind": "cron", "expr": "0 9 * * *"},
      "payload": {"kind": "agent_turn", "message": "Generate a daily summary."}
    }
  ]
}
```

## OpenClaw での実装

| 観点             | claw0 (本ファイル)            | OpenClaw 本番環境                       |
|------------------|-------------------------------|-----------------------------------------|
| レーン排他制御   | `threading.Lock`, ノンブロッキング | 同じロックパターン                   |
| ハートビート設定 | ワークスペースの `HEARTBEAT.md` | 同じファイル + 環境変数オーバーライド |
| Cronスケジュール | `CRON.json`, 3種類            | 同じフォーマット + Webhookトリガー      |
| 自動無効化       | 5回連続エラー                 | 同じ閾値、設定可能                      |
| 出力配信         | メモリ内キュー、REPLにドレイン | 配信キュー (第08章)                     |
