[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# claw0

**ゼロから構築する OpenClaw 風 AI エージェントゲートウェイ**

> 10 の段階的セクション、各セクションで 1 つのコアメカニズムを導入.
> 全セクションがそのまま実行可能な Python ファイル.

---

## これは何か?

[OpenClaw](https://github.com/openclaw/openclaw) アーキテクチャに着想を得た、最小構成の AI エージェントゲートウェイをゼロから構築する教材リポジトリ. 各セクションはコアループを変更せず、メカニズムを 1 つだけ追加する.

```
s01: Agent Loop        -- 基礎: while + stop_reason
s02: Tool Use          -- モデルに手を与える: dispatch map
s03: Sessions          -- 再起動しても消えない会話
s04: Multi-Channel     -- 同じ頭脳、複数の出口
s05: Gateway Server    -- 交換機: WebSocket + JSON-RPC
s06: Routing           -- 全メッセージが正しい宛先へ
s07: Soul & Memory     -- 魂を与え、記憶させる
s08: Heartbeat         -- 受動だけでなく能動的に
s09: Cron Scheduler    -- 正しい時に正しいことを
s10: Delivery Queue    -- メッセージは決して失われない
```

## アーキテクチャ概観

```
+--------- claw0 architecture ---------+
|                                           |
|  s10: Delivery Queue (reliable delivery)  |
|  s09: Cron Scheduler (timed tasks)        |
|  s08: Heartbeat (proactive behavior)      |
|  s07: Soul & Memory (personality + recall)|
|  s06: Routing (multi-agent binding)       |
|  s05: Gateway (WebSocket/HTTP server)     |
|  s04: Multi-Channel (channel plugins)     |
|  s03: Sessions (persistent state)         |
|  s02: Tools (bash/read/write/edit)        |
|  s01: Agent Loop (while + stop_reason)    |
|                                           |
+-------------------------------------------+
```

## クイックスタート

```sh
# 1. クローンしてディレクトリに入る
git clone https://github.com/shareAI-lab/claw0.git && cd claw0

# 2. 依存関係をインストール
pip install -r requirements.txt

# 3. 設定
cp .env.example .env
# .env を編集し、API キーとモデル名を記入

# 4. 任意のセクションを実行
python agents/s01_agent_loop.py
python agents/s02_tool_use.py
# ... 以下同様
```

## 学習パス

```
Phase 1: THE LOOP       Phase 2: STATE        Phase 3: GATEWAY      Phase 4: INTELLIGENCE  Phase 5: OPERATIONS
+----------------+      +----------------+    +----------------+    +----------------+     +----------------+
| s01: Agent Loop|      | s03: Sessions  |    | s05: Gateway   |    | s07: Soul/Mem  |     | s09: Cron      |
| s02: Tool Use  | ---> | s04: Multi-Ch  | -> | s06: Routing   | -> | s08: Heartbeat | --> | s10: Delivery  |
| (0 -> 1 tools) |      | (state+channel)|    | (server+route) |    | (persona+auto) |     | (schedule+rely)|
+----------------+      +----------------+    +----------------+    +----------------+     +----------------+
    2 tools                2 mechanisms           2 mechanisms          2 mechanisms           2 mechanisms
```

## セクション詳細

| # | Section | 標語 | コアメカニズム | 新規コンセプト |
|---|---------|------|----------------|----------------|
| 01 | Agent Loop | "一つのループで全てを支配" | while + stop_reason | LLM API, メッセージ履歴 |
| 02 | Tool Use | "モデルに手を与える" | TOOL_HANDLERS dispatch | ツールスキーマ, 安全な実行 |
| 03 | Sessions | "再起動しても消えない会話" | SessionStore + JSONL | 永続化, セッションキー |
| 04 | Multi-Channel | "同じ頭脳、複数の出口" | Channel プラグインインターフェース | 抽象化, メッセージ正規化 |
| 05 | Gateway Server | "交換機" | WebSocket + JSON-RPC | サーバーアーキテクチャ, RPC |
| 06 | Routing | "全メッセージが正しい宛先へ" | Binding resolution | マルチエージェント, ルーティング優先度 |
| 07 | Soul & Memory | "魂を与え、記憶させる" | SOUL.md + MemoryStore | パーソナリティ, ベクトル検索 |
| 08 | Heartbeat | "受動だけでなく能動的に" | HeartbeatRunner | 自律的動作 |
| 09 | Cron Scheduler | "正しい時に正しいことを" | CronService + 3 種のスケジュール | at/every/cron, 自動無効化 |
| 10 | Delivery Queue | "メッセージは決して失われない" | DeliveryQueue + backoff | at-least-once 配信, ディスクバッキング |

## OpenClaw との比較

| コンセプト | claw0 (教材版) | OpenClaw (プロダクション版) |
|------------|----------------|----------------------------|
| Agent Loop | シンプルな while ループ | Lane ベースの並行処理, リトライオニオン |
| Tools | 基本ツール 4 種 | 50 以上のツール + セキュリティポリシー |
| Sessions | JSON ファイル | JSONL トランスクリプト + sessions.json メタデータ |
| Channels | CLI + ファイルモック | Telegram, Discord, Slack, Signal, WhatsApp 他 15 以上のチャネル |
| Gateway | websockets ライブラリ | ネイティブ http + ws, プラグイン HTTP ルート |
| Routing | 優先度バインディング | マルチレベル: peer/guild/team/account/channel + アイデンティティリンク |
| Memory | キーワード検索 | SQLite-vec + FTS5 + エンベディングキャッシュ |
| Heartbeat | Thread + timer | 6 ステップチェックチェーン, Lane 排他制御, 24h 重複排除 |
| Cron | 3 種のスケジュール (at/every/cron) | フル cron パーサー, タイムゾーン対応, SQLite 実行ログ |
| Delivery | ファイルベースキュー + backoff | SQLite キュー, ジッター, 優先度, バッチ配信 |

## ドキュメント構成

```
docs/
  en/    -- English documentation
  zh/    -- Chinese documentation
  ja/    -- Japanese documentation
```

## 前提条件

- Python 3.11+
- Anthropic (または互換プロバイダー) の API キー

## ライセンス

MIT - 学習および教育目的で自由に利用可能.
