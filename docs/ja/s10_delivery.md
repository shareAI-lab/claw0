# s10: Delivery Queue & Reliable Messaging (配信キューと信頼性のあるメッセージング)

> "Write before you send, retry until you succeed" -- 先にディスクに書き込み、送信が成功するまでリトライする。

## At a Glance

```
  Agent Response
       |
       v
  enqueue (disk write)   <-- 先書き後送信、メッセージ紛失を防止
       |
       v
  attempt delivery
       |
       +---- success ----> ack (delete .json)
       |
       +---- failure ----> fail (update retry_count + backoff)
                |
                +-- retry_count <= MAX_RETRIES --> wait backoff, retry
                |     backoff: [5s, 25s, 2m, 10m, 10m]
                |
                +-- retry_count > MAX_RETRIES ---> move to failed/

  Gateway restart --> recovery_scan() --> resume pending entries
```

- **What we build**: Write-Ahead Queue パターン -- エージェントの応答を先にディスクに永続化し、それから配信を試みる。成功すれば削除 (ack)、失敗すればエクスポネンシャルバックオフでリトライする。
- **Core mechanism**: アトミック書き込み (tmp + rename) + ack/fail 状態マシン + バックオフテーブル [5s, 25s, 2m, 10m, 10m] + 起動時リカバリスキャン。
- **Design pattern**: Outbox パターン -- メッセージ生成と配信を分離し、それぞれが独立して失敗・リトライできる。

## The Problem

1. **送信失敗 = メッセージ紛失**: s09 までのエージェントは応答を直接出力する。Telegram API が 2 分間ダウンしたら、ユーザーはその期間の応答を受け取れない。
2. **プロセス再起動で未送信が消える**: エージェントが応答を生成したが送信完了前にゲートウェイがクラッシュした場合、その応答はメモリとともに消える。
3. **即座のリトライで状況が悪化する**: ネットワーク障害時に即座にリトライを繰り返すと、回復中のサービスに負荷をかけ、障害を長引かせる。バックオフ間隔がなければ「リトライストーム」になる。

## How It Works

### 1. QueuedDelivery -- 配信レコード

一つの配信レコードが「どのチャネルの誰にどんなテキストを送るか」を記録する:

```python
@dataclass
class QueuedDelivery:
    id: str              # UUID (16 文字の hex)
    channel: str         # 配信チャネル (例: "telegram")
    to: str              # 送信先 (例: "user123")
    text: str            # メッセージ本文
    retry_count: int = 0
    last_error: str | None = None
    enqueued_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0
```

**各レコードは独立した JSON ファイルとして永続化される。プロセスが再起動してもファイルシステム上に残り、リカバリ可能。**

### 2. DeliveryQueue -- ディスク永続化キュー

ディレクトリ構造:

```
delivery-queue/
  {uuid}.json          -- 配信待ち
  failed/
    {uuid}.json        -- 最大リトライ回数超過
```

四つのコア操作:

```python
class DeliveryQueue:
    def enqueue(self, channel: str, to: str, text: str) -> str:
        """メッセージをキューに書き込み、配信 ID を返す。"""
        delivery_id = uuid.uuid4().hex[:16]
        entry = QueuedDelivery(
            id=delivery_id, channel=channel, to=to, text=text,
            enqueued_at=time.time(), next_retry_at=0.0,
        )
        self._write_entry(entry)
        return delivery_id

    def ack(self, delivery_id: str) -> None:
        """配信成功を確認し、キューファイルを削除する。"""
        self._entry_path(delivery_id).unlink()

    def fail(self, delivery_id: str, error: str) -> None:
        """配信失敗を記録する。retry_count, last_error, next_retry_at を更新。
        MAX_RETRIES 超過時は move_to_failed を呼ぶ。"""
        entry = self._read_entry(delivery_id)
        entry.retry_count += 1
        entry.last_error = error
        if entry.retry_count > MAX_RETRIES:
            self.move_to_failed(delivery_id)
            return
        backoff_ms = self.compute_backoff_ms(entry.retry_count)
        entry.next_retry_at = time.time() + (backoff_ms / 1000.0)
        self._write_entry(entry)

    def move_to_failed(self, delivery_id: str) -> None:
        """failed/ ディレクトリに移動する。"""
        os.replace(
            str(self._entry_path(delivery_id)),
            str(self.failed_dir / f"{delivery_id}.json"),
        )
```

アトミック書き込み: tmp ファイルに書いてから `os.replace` する:

```python
def _write_entry(self, entry: QueuedDelivery) -> None:
    file_path = self._entry_path(entry.id)
    tmp_path = file_path.parent / f".tmp.{os.getpid()}.{entry.id}.json"
    tmp_path.write_text(
        json.dumps(entry.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(str(tmp_path), str(file_path))
```

**enqueue は書き込みのみ。配信は別のコンポーネント (DeliveryRunner) が担当する。生成と配信が完全に分離されており、それぞれが独立して失敗・リトライできる。**

### 3. バックオフテーブル -- リトライ間隔の計算

バックオフ間隔は固定テーブルに従い、リトライ回数に応じて段階的に延長する:

```python
BACKOFF_MS = [5_000, 25_000, 120_000, 600_000]
MAX_RETRIES = 5

@staticmethod
def compute_backoff_ms(retry_count: int) -> int:
    if retry_count <= 0:
        return 0
    return BACKOFF_MS[min(retry_count - 1, len(BACKOFF_MS) - 1)]
```

| リトライ回数 | バックオフ | 説明 |
|-------|---------|------|
| 1 | 5 秒 | 一時的なネットワーク瞬断 |
| 2 | 25 秒 | 短い障害 |
| 3 | 2 分 | サービスが回復中 |
| 4 | 10 分 | 長期的な障害 |
| 5 | 10 分 | 最後の試行 |
| > 5 | -- | failed/ に移動、手動リカバリ |

**なぜ 2 の冪乗ではなくテーブルなのか: 最初のリトライを素早く (5 秒)、中盤は適度に (25 秒〜2 分)、後半は上限を設ける (10 分)。テーブルは上限を明示的に制御でき、設定変更も容易だ。**

### 4. DeliveryRunner -- バックグラウンド配信スレッド

1 秒ごとにキューをスキャンし、`next_retry_at` が到来したエントリを配信する:

```python
class DeliveryRunner:
    def _process_pending(self) -> None:
        now = time.time()
        for entry in self.queue.load_pending():
            if self._stop_event.is_set():
                break
            if entry.next_retry_at > now:
                continue  # まだバックオフ中
            self._total_attempted += 1
            if self._attempt_delivery(entry):
                self.queue.ack(entry.id)
                self._total_succeeded += 1
            else:
                self._total_failed += 1
                self.queue.fail(entry.id, error_msg)

    def _recovery_scan(self) -> None:
        """起動時にキューをスキャンし、未完了のエントリを報告する。"""
        pending = self.queue.load_pending()
        failed = self.queue.load_failed()
        if pending:
            print_delivery(f"recovery: {len(pending)} pending entries, resuming")
        if failed:
            print_delivery(f"recovery: {len(failed)} entries in failed/")
```

**リカバリスキャン: ゲートウェイが再起動すると、DeliveryRunner は最初にディスク上のキューファイルを走査する。クラッシュ前に送信できなかったメッセージがそのまま残っているため、時間順にソートして配信を再開する。**

### 5. MockDeliveryChannel -- シミュレーション配信チャネル

教育用に配信チャネルをシミュレートし、故障率を調整可能にする:

```python
class MockDeliveryChannel:
    def __init__(self, name: str, fail_rate: float = 0.0):
        self.name = name
        self.fail_rate = fail_rate

    def send(self, to: str, text: str) -> None:
        if random.random() < self.fail_rate:
            raise ConnectionError(
                f"channel={self.name}: connection refused (simulated)"
            )
        print(f"\n[{self.name} -> {to}] {text[:80]}...\n")

    def set_fail_rate(self, rate: float) -> None:
        self.fail_rate = max(0.0, min(1.0, rate))
```

`/simulate-failure` コマンドで故障率を 0% と 50% で切り替え、バックオフとリトライの動作を直接観察できる。

**本番では `mock_channel.send` を `telegram.send_message` や `discord.send_message` に差し替えるだけでよい。インターフェースは同じだ。**

### 6. エージェントループとの統合

s10 の核心的な変更点: エージェントの応答を直接出力せず、配信キューを通じて送信する:

```python
# s09 以前: 直接出力
print_assistant(text)

# s10: 配信キュー経由
did = delivery_queue.enqueue(mock_channel.name, "user", text)
print_info(f"  enqueued -> delivery queue (id={did[:8]}..)")
```

ハートビートの出力も同様にキュー経由で送信する:

```python
for msg in heartbeat.drain_output():
    did = delivery_queue.enqueue(mock_channel.name, "user", msg)
    print_heartbeat(f"enqueued heartbeat message (id={did[:8]}..)")
```

**すべてのアウトバウンドメッセージが同一の配信キューを通過する。配信保証のロジックを一箇所に集約できる。**

## What Changed from s09

| コンポーネント | s09 | s10 |
|-----------|-----|-----|
| 応答の出力先 | 直接 stdout に出力 | 配信キューに enqueue |
| 配信保証 | なし (失敗 = 紛失) | Write-Ahead + リトライ + リカバリ |
| 障害処理 | Cron の自動無効化のみ | バックオフテーブル [5s, 25s, 2m, 10m, 10m] + failed/ |
| 永続化 | Cron Store (jobs.json) | 配信キュー (ファイル/エントリ) を追加 |
| リカバリ | なし | 起動時リカバリスキャン |
| チャネル | なし (ローカル出力のみ) | MockDeliveryChannel (故障シミュレーション対応) |

**Key shift**: 「メッセージを生成して直接出力」から「先にディスクに書き込み、成功するまで配信を試み続ける」へ。配信キューはエージェントの応答と実際の配信を分離し、ネットワーク障害やプロセスクラッシュを透過的に処理する。

## Design Decisions

**なぜインメモリキューではなくディスク永続化なのか?**

インメモリキューはプロセスクラッシュですべて失われる。ディスクに書き込むことで、ゲートウェイが再起動しても未配信のメッセージがリカバリ可能になる。これが「Write-Ahead」の核心であり、データベースの WAL (Write-Ahead Log) と同じ考え方だ。

**なぜ enqueue と delivery を分離するのか (Outbox パターン)?**

分離により、エージェントの応答生成が配信の成否に影響されない。Telegram API がダウンしていてもエージェントは正常に応答を生成し続け、API が回復した時点でバックグラウンドスレッドが自動的に配信する。両者のライフサイクルが独立している。

**なぜ failed/ ディレクトリに移動させるのか?**

MAX_RETRIES (5 回) 超過は通常、一時的な障害ではなく構成エラーやチャネル停止を示す。無限リトライは CPU リソースを浪費し、ログを汚染する。failed/ に移動させて人間のレビューを待ち、`/retry` コマンドで手動リカバリできる。

**In production OpenClaw:** 配信キューは `src/infra/outbound/delivery-queue.ts` に実装されている。本番版はチャネルごとの並行配信、メッセージの優先度付け、配信先のレート制限、メディアファイル (画像・音声) の配信、そして Webhook コールバック通知をサポートしている。バックオフテーブルもカスタマイズ可能だ。`recoverPendingDeliveries()` は起動時だけでなく、チャネル再接続時にも実行される。

## Try It

```sh
cd claw0
python agents/s10_delivery.py
```

操作コマンド:

```
/queue              -- 配信キューの状態を確認
/failed             -- 失敗したエントリを確認
/retry              -- failed/ のエントリをキューに戻す
/simulate-failure   -- 故障率を 0%/50% で切り替え
/heartbeat          -- ハートビートの状態を確認
/trigger            -- 手動でハートビートをトリガー
```

推奨する実験手順:

```
You > Hello, how are you?
  enqueued -> delivery queue (id=a1b2c3d4..)
  [delivery] delivered a1b2c3d4.. to telegram:user
[telegram -> user] Hello! I'm doing well...

You > /simulate-failure
Fail rate -> 50%

You > Tell me about Python.
  enqueued -> delivery queue (id=e5f6g7h8..)
  [delivery] failed e5f6g7h8.. (retry 1/5, next in 5s)
  [delivery] failed e5f6g7h8.. (retry 2/5, next in 25s)
  [delivery] delivered e5f6g7h8.. to telegram:user
```

`/queue` で配信の進行状況を確認し、バックオフ中のエントリの残り待ち時間が表示される。
