# 第08章: 配信

> まずディスクに書き込み、その後に送信を試みる。クラッシュに対して安全。

## アーキテクチャ

```
    Agent Reply / Heartbeat / Cron
              |
        chunk_message()          プラットフォームの制限で分割
              |                  (telegram=4096, discord=2000, etc.)
              v
        DeliveryQueue.enqueue()
          1. 一意のIDを生成
          2. .tmp.{pid}.{id}.json に書き込み
          3. fsync()
          4. os.replace() で {id}.json に    <-- 先行書き込み
              |
              v
        DeliveryRunner (バックグラウンドスレッド, 1秒スキャン)
              |
        deliver_fn(channel, to, text)
           /          \
        成功          失敗
          |              |
        ack()         fail()
        (.json を     (retry_count++, バックオフを計算,
         削除)        ディスク上の .json を更新)
                         |
                    retry_count >= 5?
                      |yes
                    failed/ に移動

    バックオフ: [5秒, 25秒, 2分, 10分] +/-20% のジッター
```

## 新しい概念

- **DeliveryQueue**: ディスク永続化された先行書き込みキュー。配信を試みる前にディスクに書き込む。
- **アトミック書き込み**: tmp ファイル + `os.fsync()` + `os.replace()` -- クラッシュ時に中途半端なファイルが残らない。
- **DeliveryRunner**: 指数バックオフで保留エントリを処理するバックグラウンドスレッド。
- **chunk_message()**: プラットフォームのサイズ制限に従い、段落境界を尊重してテキストを分割。
- **リカバリスキャン**: 起動時に、前回クラッシュからの保留エントリを自動的にリトライ。

## コードウォークスルー

### 1. DeliveryQueue.enqueue() + アトミック書き込み

基本ルール: まずディスクに書き込み、その後に配信を試みる。エンキューと配信の間でプロセスがクラッシュしても、メッセージはディスク上に残る。

```python
def enqueue(self, channel: str, to: str, text: str) -> str:
    delivery_id = uuid.uuid4().hex[:12]
    entry = QueuedDelivery(
        id=delivery_id, channel=channel, to=to, text=text,
        enqueued_at=time.time(), next_retry_at=0.0,
    )
    self._write_entry(entry)
    return delivery_id

def _write_entry(self, entry: QueuedDelivery) -> None:
    final_path = self.queue_dir / f"{entry.id}.json"
    tmp_path = self.queue_dir / f".tmp.{os.getpid()}.{entry.id}.json"

    data = json.dumps(entry.to_dict(), indent=2, ensure_ascii=False)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())        # データがディスク上にある

    os.replace(str(tmp_path), str(final_path))  # POSIXではアトミック
```

3ステップの保証:
- ステップ 1: `.tmp.{pid}.{id}.json` に書き込み (クラッシュ = 孤立した一時ファイル、無害)
- ステップ 2: `fsync()` -- データがディスク上に確定
- ステップ 3: `os.replace()` -- アトミックスワップ (クラッシュ = 旧ファイルか新ファイル、部分的なものは決してない)

### 2. ack() / fail() -- リトライのライフサイクル

```python
def ack(self, delivery_id: str) -> None:
    """配信成功。キューファイルを削除する。"""
    (self.queue_dir / f"{delivery_id}.json").unlink()

def fail(self, delivery_id: str, error: str) -> None:
    """retry_count をインクリメントし、次のリトライ時刻を計算するか、諦める。"""
    entry = self._read_entry(delivery_id)
    entry.retry_count += 1
    entry.last_error = error
    if entry.retry_count >= MAX_RETRIES:
        self.move_to_failed(delivery_id)
        return
    backoff_ms = compute_backoff_ms(entry.retry_count)
    entry.next_retry_at = time.time() + backoff_ms / 1000.0
    self._write_entry(entry)  # 新しいリトライ状態でディスク上を更新
```

ジッター付きバックオフでサンダリングハード問題を防止:

```python
BACKOFF_MS = [5_000, 25_000, 120_000, 600_000]
MAX_RETRIES = 5

def compute_backoff_ms(retry_count: int) -> int:
    if retry_count <= 0:
        return 0
    idx = min(retry_count - 1, len(BACKOFF_MS) - 1)
    base = BACKOFF_MS[idx]
    jitter = random.randint(-base // 5, base // 5)   # +/- 20%
    return max(0, base + jitter)
```

### 3. DeliveryRunner -- バックグラウンドループ

保留エントリを毎秒スキャンする。`next_retry_at` が経過したものだけを処理する。起動時に、前回クラッシュからのエントリのリカバリスキャンを実行する。

```python
class DeliveryRunner:
    def start(self) -> None:
        self._recovery_scan()
        self._thread = threading.Thread(
            target=self._background_loop, daemon=True)
        self._thread.start()

    def _process_pending(self) -> None:
        pending = self.queue.load_pending()
        now = time.time()
        for entry in pending:
            if entry.next_retry_at > now:
                continue
            self.total_attempted += 1
            try:
                self.deliver_fn(entry.channel, entry.to, entry.text)
                self.queue.ack(entry.id)
                self.total_succeeded += 1
            except Exception as exc:
                self.queue.fail(entry.id, str(exc))
                self.total_failed += 1
```

## 試してみる

```sh
python en/s08_delivery.py

# メッセージを送信 -- エンキューと配信を観察
# You > Hello!

# 50% の障害率を有効化
# You > /simulate-failure

# 障害状態で別のメッセージを送信 -- バックオフ付きリトライを観察
# You > Test message under failure

# キューを確認
# You > /queue
# You > /failed

# 信頼性を復元し、保留エントリが配信されるのを観察
# You > /simulate-failure

# 統計を確認
# You > /stats
```

## OpenClaw での実装

| 観点           | claw0 (本ファイル)              | OpenClaw 本番環境                     |
|----------------|---------------------------------|---------------------------------------|
| キューストレージ | ディレクトリ内のJSONファイル   | 同じファイル単位パターン              |
| アトミック書き込み | tmp + fsync + os.replace     | 同じアプローチ                        |
| バックオフ     | [5秒, 25秒, 2分, 10分] + ジッター | 同じスケジュール                    |
| メッセージ分割 | 段落境界での分割               | 同じ + コードフェンス対応             |
| リカバリ       | 起動時にキューディレクトリをスキャン | 同じスキャン + 孤立ファイルのクリーンアップ |

## シリーズまとめ

8つの章を通じて、エージェントフレームワークの中核メカニズム:

```
    第01章: while True + stop_reason        (ループ)
    第02章: TOOLS + TOOL_HANDLERS           (手)
    第03章: JSONL + ContextGuard            (記憶)
    第04章: Channel ABC + InboundMessage    (口)
    第05章: BindingTable + session key      (ルーター)
    第06章: 8層プロンプト + TF-IDF          (脳)
    第07章: Heartbeat + Cron                (自発性)
    第08章: DeliveryQueue + backoff         (配信保証)
```

第01章のエージェントループは、第08章の核心部分でもなお認識できる。AIエージェントとは、ディスパッチテーブルを備えた `while True` ループであり、永続化、ルーティング、インテリジェンス、スケジューリング、信頼性のレイヤーで包まれたものである。
