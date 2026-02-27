# 第10章: 並行処理

> 名前付きレーンが混沌を直列化する。

## アーキテクチャ

```
    Incoming Work
        |
    CommandQueue.enqueue(lane, fn)
        |
    +---v---+    +--------+    +-----------+
    | main  |    |  cron  |    | heartbeat |
    | max=1 |    | max=1  |    |   max=1   |
    | FIFO  |    | FIFO   |    |   FIFO    |
    +---+---+    +---+----+    +-----+-----+
        |            |              |
    [active]     [active]       [active]
        |            |              |
    _task_done   _task_done     _task_done
        |            |              |
    _pump()      _pump()        _pump()
    (dequeue     (dequeue       (dequeue
     next if      next if        next if
     active<max)  active<max)    active<max)
```

各レーンは `LaneQueue`: `threading.Condition` で保護された FIFO デキュー。タスクは通常のコーラブルとして入り、`concurrent.futures.Future` を通じて結果を返す。`CommandQueue` が名前によって正しいレーンに作業をディスパッチし、ライフサイクル全体を管理する。

## 本章のポイント

- **名前付きレーン**: 各レーンは名前 (例: `"main"`、`"cron"`、`"heartbeat"`) と独立した FIFO キューを持つ。レーンは最初の使用時に遅延作成される。
- **max_concurrency**: 各レーンで同時に実行できるタスク数の上限。デフォルトは 1 (直列実行)。レーン内で並列作業を許可するには増加させる。
- **_pump() ループ**: 各タスク完了時 (`_task_done`)、レーンはさらにタスクをデキューできるかチェックする。この自己ポンピング設計により外部スケジューラが不要になる。
- **Future ベースの結果**: 全ての `enqueue()` は `concurrent.futures.Future` を返す。呼び出し元は `future.result()` でブロックするか、`add_done_callback()` でコールバックを付与できる。
- **世代追跡**: 各レーンは整数の世代カウンターを持つ。`reset_all()` で全世代がインクリメントされる。古いタスクが完了した時 (世代が現在と一致しない)、`_pump()` は呼ばれない -- リスタート後のゾンビタスクがキューを排出するのを防ぐ。
- **Condition ベースの同期**: `threading.Condition` が Section 07 の `threading.Lock` を置き換える。これにより `wait_for_idle()` がポーリングではなく、通知を受けるまで効率的にスリープできる。
- **ユーザー優先**: ユーザー入力は `main` レーンに入り結果をブロック待ちする。バックグラウンド作業 (ハートビート、cron) は別レーンに入り REPL をブロックしない。

## コードウォークスルー

### 1. LaneQueue -- コアプリミティブ

レーンはデキュー + 条件変数 + アクティブカウンター。`_pump()` がエンジン:

```python
class LaneQueue:
    def __init__(self, name: str, max_concurrency: int = 1) -> None:
        self.name = name
        self.max_concurrency = max(1, max_concurrency)
        self._deque = deque()           # [(fn, future, generation), ...]
        self._condition = threading.Condition()
        self._active_count = 0
        self._generation = 0

    def enqueue(self, fn, generation=None):
        future = concurrent.futures.Future()
        with self._condition:
            gen = generation if generation is not None else self._generation
            self._deque.append((fn, future, gen))
            self._pump()
        return future

    def _pump(self):
        """Pop and start tasks while active < max_concurrency."""
        while self._active_count < self.max_concurrency and self._deque:
            fn, future, gen = self._deque.popleft()
            self._active_count += 1
            threading.Thread(
                target=self._run_task, args=(fn, future, gen), daemon=True
            ).start()

    def _task_done(self, gen):
        with self._condition:
            self._active_count -= 1
            if gen == self._generation:  # stale tasks do not re-pump
                self._pump()
            self._condition.notify_all()
```

### 2. CommandQueue -- ディスパッチャー

`CommandQueue` は lane_name から `LaneQueue` への辞書を保持する。レーンは遅延作成される:

```python
class CommandQueue:
    def __init__(self):
        self._lanes: dict[str, LaneQueue] = {}
        self._lock = threading.Lock()

    def get_or_create_lane(self, name, max_concurrency=1):
        with self._lock:
            if name not in self._lanes:
                self._lanes[name] = LaneQueue(name, max_concurrency)
            return self._lanes[name]

    def enqueue(self, lane_name, fn):
        lane = self.get_or_create_lane(lane_name)
        return lane.enqueue(fn)

    def reset_all(self):
        """Increment generation on all lanes for restart recovery."""
        with self._lock:
            for lane in self._lanes.values():
                with lane._condition:
                    lane._generation += 1
```

### 3. 世代追跡 -- リスタートリカバリ

世代カウンターは微妙な問題を解決する: タスクがインフライト中にシステムがリスタートすると、それらのタスクが完了して古い状態でキューをポンプしようとする可能性がある。世代をインクリメントすることで、全ての古いコールバックは無害なノーオペレーションになる:

```python
def _task_done(self, gen):
    with self._condition:
        self._active_count -= 1
        if gen == self._generation:
            self._pump()       # current generation: normal flow
        # else: stale task -- do NOT pump, let it die quietly
        self._condition.notify_all()
```

### 4. HeartbeatRunner -- レーン対応スキップ

`lock.acquire(blocking=False)` の代わりに、ハートビートはレーン統計をチェックする:

```python
def heartbeat_tick(self):
    ok, reason = self.should_run()
    if not ok:
        return

    lane_stats = self.command_queue.get_or_create_lane(LANE_HEARTBEAT).stats()
    if lane_stats["active"] > 0:
        return  # lane is busy, skip this tick

    future = self.command_queue.enqueue(LANE_HEARTBEAT, _do_heartbeat)
    future.add_done_callback(_on_done)
```

これはノンブロッキングロックパターンと機能的に同等だが、レーン抽象化を用いて表現されている。

## 試してみる

```sh
python ja/s10_concurrency.py

# 全レーンと現在のステータスを表示
# You > /lanes
#   main          active=[.]  queued=0  max=1  gen=0
#   cron          active=[.]  queued=0  max=1  gen=0
#   heartbeat     active=[.]  queued=0  max=1  gen=0

# 名前付きレーンに手動で作業をエンキュー
# You > /enqueue main What is the capital of France?

# カスタムレーンを作成して作業をエンキュー
# You > /enqueue research Summarize recent AI developments

# レーンの max_concurrency を変更
# You > /concurrency research 3

# 世代カウンターを表示
# You > /generation

# リスタートをシミュレート (全世代をインクリメント)
# You > /reset

# レーンごとの保留アイテムを表示
# You > /queue
```

## OpenClaw での実装

| 観点                | claw0 (本ファイル)                        | OpenClaw 本番環境                              |
|---------------------|-------------------------------------------|------------------------------------------------|
| レーンプリミティブ  | `threading.Condition` 付き `LaneQueue`    | 同じパターン + メトリクス計装                  |
| ディスパッチャー    | レーン辞書の `CommandQueue`               | 同じ遅延作成ディスパッチャー                   |
| 並行性制御          | レーンごとの `max_concurrency`、デフォルト1 | 同じ、デプロイメントごとに設定可能             |
| タスク実行          | タスクごとに `threading.Thread`            | 上限付きワーカーのスレッドプール               |
| 結果配信            | `concurrent.futures.Future`               | 同じ Future ベースのインターフェース           |
| 世代追跡            | 整数カウンター、古いタスクはポンプをスキップ | 同じ世代パターンでリスタート安全性を確保       |
| アイドル検出        | `Condition.wait()` 付き `wait_for_idle()` | 同じ、グレースフルシャットダウンに使用         |
| 標準レーン          | main, cron, heartbeat                     | 同じデフォルト + プラグイン定義のカスタムレーン |
| ユーザー優先        | main レーンが結果をブロック待ち            | ユーザー入力に対する同じブロッキングセマンティクス |
