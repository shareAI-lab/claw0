# s09: Cron Scheduler (タスクスケジューラ)

> "When the agent knows time itself" -- ファジーな周期的ポーリングから精密な時刻駆動スケジューリングへ。

## At a Glance

```
  +--- CronService (background thread) -------+
  |  every 1s:                                  |
  |  for each enabled job:                      |
  |    compute_next_run_at(schedule, now)        |
  |    if next_run_at <= now:                    |
  |      execute_job(job)                        |
  |        -> call agent_fn(payload.message)     |
  |        -> update state (last_run, status)    |
  |        -> if delete_after_run: disable       |
  |        -> if errors >= 5: auto-disable       |
  |        -> append to run log                  |
  +---------------------------------------------+
           |
           v
    +--- CronStore ---+     +--- CronRunLog ---+
    | jobs.json       |     | run-log.jsonl     |
    | (atomic write)  |     | (append + prune)  |
    +------------------+     +------------------+

  +--- HeartbeatRunner (simplified) -----------+
  |  every N seconds:                           |
  |  should_run() -> agent_fn(HEARTBEAT.md)     |
  |  -> HEARTBEAT_OK = suppress                 |
  |  -> content = output to user                |
  +---------------------------------------------+
```

- **What we build**: 三種類のスケジュールタイプ (at/every/cron) をサポートし、永続化ストレージと自動障害処理を備えた Cron スケジューラ。
- **Core mechanism**: 1 秒ポーリング + compute_next_run_at アルゴリズム + アンカーポイント式 (ドリフト防止) + 連続エラー自動無効化。
- **Design pattern**: CronStore (JSON ファイル永続化) + CronRunLog (JSONL 追記ログ) + CronService (バックグラウンドスレッド) + ツール呼び出し (エージェントが自らスケジュール操作)。

## The Problem

1. **ハートビートだけでは不十分**: s08 のハートビートは「定期的にチェック」するものであり、特定の時刻に特定のタスクを実行する能力がない。ユーザーが「明日午後 5 時にリマインドして」と言っても、ハートビートでは保証できない。
2. **時刻精度がない**: ハートビートの間隔が 30 分だとすると、「午後 5 時ちょうど」のリマインドは 30 分の誤差を持つ。cron 式のように「毎週月曜 9:00」という精密なスケジュールを記述できない。
3. **永続化がない**: ハートビートの状態はメモリ上にしかなく、プロセスが再起動すれば「いつ最後に実行したか」を失う。スケジュールされたタスクがプロセス再起動をまたいで存続することもできない。

## How It Works

### 1. 三種類のスケジュール -- at / every / cron

```
  1. "at" -- ワンショット絶対時刻
     "明日午後 5 時にリマインドして"
     -> schedule: { kind: "at", at_time: "2026-02-25T17:00:00" }
     -> delete_after_run = True
     -> 一度実行したら自動無効化

  2. "every" -- アンカーポイントベースの等間隔
     "毎時サーバーの状態を確認して"
     -> schedule: { kind: "every", every_seconds: 3600, anchor: None }
     -> anchor のデフォルトは作成時刻
     -> 公式: anchor + ceil((now - anchor) / interval) * interval
     -> last_run を使わない理由: 実行時間による累積ドリフトを防止

  3. "cron" -- 標準 cron 式
     "毎週月曜 9 時に依存関係の脆弱性をチェック"
     -> schedule: { kind: "cron", expr: "0 9 * * 1", tz: None }
     -> croniter ライブラリで解析
     -> 同秒ループ防止: 結果が <= now なら now+1s から再試行
```

### 2. compute_next_run_at -- コアスケジューリングアルゴリズム

三種類のスケジュールそれぞれの次回トリガー時刻計算:

```python
def compute_next_run_at(schedule: dict, now_ts: float) -> float | None:
    kind = schedule.get("kind")

    if kind == "at":
        at_time_str = schedule.get("at_time", "")
        if not at_time_str:
            return None
        dt = datetime.fromisoformat(at_time_str)
        if dt.tzinfo is None:
            dt = dt.astimezone()
        at_ts = dt.timestamp()
        # まだ期限前なら返す、期限切れなら None
        return at_ts if at_ts > now_ts else None

    if kind == "every":
        every_seconds = max(1, int(schedule.get("every_seconds", 60)))
        anchor_str = schedule.get("anchor")
        if anchor_str:
            anchor_dt = datetime.fromisoformat(anchor_str)
            if anchor_dt.tzinfo is None:
                anchor_dt = anchor_dt.astimezone()
            anchor_ts = anchor_dt.timestamp()
        else:
            anchor_ts = now_ts

        if now_ts < anchor_ts:
            return anchor_ts

        # アンカーポイント公式: 累積ドリフトを防止
        elapsed = now_ts - anchor_ts
        steps = max(1, math.ceil(elapsed / every_seconds))
        return anchor_ts + steps * every_seconds

    if kind == "cron":
        expr = schedule.get("expr", "").strip()
        if not expr or croniter is None:
            return None
        base_dt = datetime.fromtimestamp(now_ts).astimezone()
        cron = croniter(expr, base_dt)
        next_dt = cron.get_next(datetime)
        next_ts = next_dt.timestamp()
        if next_ts > now_ts:
            return next_ts
        # 同秒ループ防止: now+1s から再試行
        retry_dt = datetime.fromtimestamp(math.floor(now_ts) + 1.0).astimezone()
        cron2 = croniter(expr, retry_dt)
        return cron2.get_next(datetime).timestamp()
```

**`every` タイプが `last_run` ではなくアンカーポイント公式を使う理由: `last_run + interval` だと実行時間が 3 秒かかれば毎回 3 秒ずつ遅延し、1 日で数分の累積ドリフトが生じる。アンカーポイント公式は作成時の基準点から数学的に計算するため、実行時間がどれだけかかっても次回のトリガー時刻がドリフトしない。**

### 3. CronStore -- JSON ファイル永続化

CronJob はすべて `workspace/cron/jobs.json` に保存され、アトミック書き込み (tmp + rename) でデータ破損を防止する:

```python
class CronStore:
    def save(self, jobs: list[dict]) -> None:
        with self._lock:
            data = {"version": 1, "jobs": jobs}
            content = json.dumps(data, indent=2, ensure_ascii=False, default=str)
            tmp_path = self.store_path.with_suffix(f".{os.getpid()}.tmp")
            try:
                tmp_path.write_text(content, encoding="utf-8")
                tmp_path.replace(self.store_path)  # アトミック rename
            except OSError:
                self.store_path.write_text(content, encoding="utf-8")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
```

**tmp ファイルに書き込んでから rename するのはなぜか: `write_text` は部分的に書き込まれた時点でクラッシュする可能性があり、rename は OS レベルでアトミックであるため、読み取り側は常に完全なファイルを見る。**

### 4. CronRunLog -- JSONL 追記ログ

各 Job の実行結果は JSONL 形式 (1 行 1 レコード) でログに追記される:

```python
class CronRunLog:
    MAX_SIZE_BYTES = 2 * 1024 * 1024  # 2MB

    def append(self, entry: dict) -> None:
        with self._lock:
            line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)
            self._prune_if_needed()

    def _prune_if_needed(self) -> None:
        if self.log_path.stat().st_size <= self.MAX_SIZE_BYTES:
            return
        raw = self.log_path.read_text(encoding="utf-8")
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        kept = lines[-(self.MAX_LINES // 2):]
        self.log_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
```

**JSONL の利点: 追記のみで既存データを読み込む必要がない (パフォーマンス)、1 行が破損しても他の行に影響しない (堅牢性)、grep/jq で直接照会できる (運用性)。**

### 5. CronService -- バックグラウンドスケジューリングエンジン

バックグラウンドスレッドが毎秒すべての enabled job をチェックし、期限到来時に実行する:

```python
class CronService:
    AUTO_DISABLE_THRESHOLD = 5

    def _background_loop(self) -> None:
        while not self._stop_event.is_set():
            self._compute_all_next_runs()
            now = time.time()
            due_jobs = self._find_due_jobs(now)

            for job in due_jobs:
                if self._stop_event.is_set():
                    break
                result = self._execute_job(job)
                # 実行直後に next_run_at を再計算
                self._compute_all_next_runs()

            self._stop_event.wait(1.0)
```

`_execute_job` の処理フロー:

```python
def _execute_job(self, job: dict) -> dict:
    # 1. エージェント呼び出し
    agent_fn = self.agent_fn_factory()
    response_text = agent_fn(message)

    # 2. state 更新
    state_patch = {
        "last_run_at": time.time(),
        "last_status": "ok" or "error",
        "consecutive_errors": 0 or prev + 1,
    }
    self.store.update_state(job_id, state_patch)

    # 3. delete_after_run の job は実行後に無効化
    if job.get("delete_after_run") and result["status"] == "ok":
        self.store.update_job(job_id, {"enabled": False})

    # 4. 連続 5 回エラーで自動無効化
    if consecutive_errors >= self.AUTO_DISABLE_THRESHOLD:
        self.store.update_job(job_id, {"enabled": False})

    # 5. ログ追記
    self.run_log.append(log_entry)
```

**自動無効化は暴走防止のための安全弁: 連続 5 回失敗は通常設定エラーを示しており、自動的に停止させて人間のレビューを待つ。**

### 6. ツール定義 -- エージェントが自らスケジュール操作

三つのツールによりエージェントが Cron Job を管理する:

```python
TOOLS = [
    {
        "name": "cron_create",
        "description": "Create a new scheduled cron job...",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "schedule_type": {"type": "string"},  # "at" / "every" / "cron"
                "schedule_value": {"type": "string"},  # ISO datetime / seconds / cron expr
                "message": {"type": "string"},
                "delete_after_run": {"type": "boolean"},
            },
            "required": ["name", "schedule_type", "schedule_value", "message"],
        },
    },
    {"name": "cron_list",   "description": "List all cron jobs..."},
    {"name": "cron_delete", "description": "Delete a cron job by ID..."},
]
```

使用例:

```
You > Remind me to submit the report at 5pm tomorrow.
[tool:cron_create] {"name":"report-reminder","schedule_type":"at",
                     "schedule_value":"2026-02-25T17:00:00",
                     "message":"Remind user to submit the report","delete_after_run":true}
Assistant: I've set a one-shot reminder for 5pm tomorrow.
```

**エージェントが自律的にスケジュールを作成・管理する。人間がツールパラメータを手動で組み立てる必要はない。**

## What Changed from s08

| コンポーネント | s08 | s09 |
|-----------|-----|-----|
| トリガー方式 | 固定間隔のハートビートのみ | at (ワンショット) + every (等間隔) + cron (式) |
| 時刻精度 | 間隔レベル (例: 30 分粒度) | 秒レベル (1 秒ポーリング) |
| 永続化 | なし (メモリ上の状態のみ) | CronStore (JSON) + CronRunLog (JSONL) |
| 障害処理 | なし | 連続エラー自動無効化 + エラーログ |
| エージェント操作 | 不可 | cron_create / cron_list / cron_delete ツール |
| ワンショットタスク | 不可 | delete_after_run で自動無効化 |
| ドリフト防止 | なし | アンカーポイント公式 |

**Key shift**: 「定期的に確認事項がないかチェック」(ハートビート) から「正確な時刻に正確なタスクを実行」(Cron) へ。ハートビートはファジーな周期的ポーリングであり、Cron は決定性のある時刻スケジューリングである。

## Design Decisions

**なぜ last_run ではなくアンカーポイント公式なのか?**

`last_run + interval` の方式だと実行時間による累積ドリフトが発生する。アンカーポイント公式 `anchor + ceil((now - anchor) / interval) * interval` は作成時の基準点に基づいて計算し、一度に計算を飛ばしても累積しない。これは OpenClaw の本番版が採用しているのと同じ戦略だ。

**なぜ既存のスケジューリングライブラリを使わないのか?**

教育目的のため、コアアルゴリズムを透明にする: compute_next_run_at の各分岐を手動でデバッグでき、ブラックボックスのスケジューリングエンジンに隠蔽されない。本番版では同じアルゴリズムが TypeScript で実装されている (`src/cron/schedule.ts`)。

**なぜ 1 秒ポーリングで cron 式を処理するのか?**

秒レベルの精度は cron にとって十分だ。1 秒のスリープ + `compute_next_run_at` の比較は CPU 負荷がほぼゼロであり、ウォッチドッグタイマーやタイマーホイールの複雑さを持ち込む必要がない。

**In production OpenClaw:** `src/cron/service.ts` のアーキテクチャは本節と同じだ。本番版は `agentId`、`sessionTarget`、`wakeMode`、`delivery` などのフィールドを持つ完全な CronJob 型を使用し、Cron Job のトリガーが特定のチャネルにメッセージを送信できる。スケジュールの検証、タイムゾーン処理、cron 式のパースも含めてより強固に実装されている。

## Try It

```sh
cd claw0
python agents/s09_cron.py
```

デモ用の cron job が自動的に作成される (毎分実行)。実行状態の確認コマンド:

```
/cron           -- すべての cron job を一覧表示
/cron-log       -- 最近の実行ログを表示
/trigger-cron <id>  -- 手動で cron job をトリガー
/heartbeat      -- ハートビートの状態を確認
/trigger        -- 手動でハートビートをトリガー
```

croniter がインストールされていない場合、`cron` タイプのスケジュールは動作しない:

```sh
pip install croniter
```
