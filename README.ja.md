[English](README.md) | [中文](README.zh.md) | [日本語](README.ja.md)

# claw0

**ゼロからイチへ: AI エージェントゲートウェイを構築する**

> 10 の段階的セクション、各セクションが実行可能な Python ファイル.
> 3 言語 (英語, 中国語, 日本語) -- コード + ドキュメント同一ディレクトリ.

---

## これは何か?

多くのエージェントチュートリアルは「APIを1回呼ぶ」だけで終わる. このリポジトリはその while ループから出発し、本番レベルのゲートウェイまで到達する.

ゼロから最小構成の AI エージェントゲートウェイをセクションごとに構築する. 10 セクション、10 の核心コンセプト、約 7,000 行の Python. 各セクションは新しい概念を1つだけ導入し、前のセクションのコードはそのまま維持する. 全 10 セクション完了後には、OpenClaw の本番コードベースを自信を持って読めるようになる.

```sh
s01: Agent Loop           -- 基礎: while + stop_reason
s02: Tool Use             -- モデルにツールを持たせる: dispatch table
s03: Sessions & Context   -- 会話の永続化、オーバーフロー処理
s04: Channels             -- Telegram + Feishu: 完全なチャネルパイプライン
s05: Gateway & Routing    -- 5 段階バインド、セッション分離
s06: Intelligence         -- 魂、記憶、スキル、プロンプト組立
s07: Heartbeat & Cron     -- 能動的エージェント + スケジュールタスク
s08: Delivery             -- 信頼性のあるメッセージキュー + バックオフ
s09: Resilience           -- 3層リトライオニオン + 認証プロファイル輪換
s10: Concurrency          -- 名前付きレーンが混沌を直列化
```

## アーキテクチャ

```
+------------------- claw0 layers -------------------+
|                                                     |
|  s10: Concurrency  (名前付きレーン, generation追跡) |
|  s09: Resilience   (認証輪換, オーバーフロー圧縮)   |
|  s08: Delivery     (先行書込キュー, バックオフ)     |
|  s07: Heartbeat    (レーン排他, cron スケジューラ)   |
|  s06: Intelligence (8層プロンプト, ハイブリッド記憶) |
|  s05: Gateway      (WebSocket, 5段階ルーティング)   |
|  s04: Channels     (Telegram パイプライン, Feishu)   |
|  s03: Sessions     (JSONL 永続化, 3段階リトライ)    |
|  s02: Tools        (dispatch table, 4 ツール)       |
|  s01: Agent Loop   (while True + stop_reason)       |
|                                                     |
+-----------------------------------------------------+
```

## セクション依存関係

```
s01 --> s02 --> s03 --> s04 --> s05
                 |               |
                 v               v
                s06 ----------> s07 --> s08
                 |               |
                 v               v
                s09 ----------> s10
```

- s01-s02: 基礎 (依存なし)
- s03: s02 上に構築 (ツールループに永続化を追加)
- s04: s03 上に構築 (チャネルが InboundMessage をセッションに供給)
- s05: s04 上に構築 (チャネルメッセージをエージェントにルーティング)
- s06: s03 上に構築 (セッションをコンテキストに使用、プロンプト層を追加)
- s07: s06 上に構築 (ハートビートが魂/記憶でプロンプトを構築)
- s08: s07 上に構築 (ハートビート出力がデリバリーキューを経由)
- s09: s03+s06 上に構築 (ContextGuard をオーバーフロー層に再利用、モデル設定)
- s10: s07 上に構築 (単一 Lock を名前付きレーンシステムに置換)

## クイックスタート

```sh
# 1. クローンしてディレクトリに入る
git clone https://github.com/anthropics/claw0.git && cd claw0

# 2. 依存関係をインストール
pip install -r requirements.txt

# 3. 設定
cp .env.example .env
# .env を編集: ANTHROPIC_API_KEY と MODEL_ID を設定

# 4. 任意のセクションを実行 (言語を選択)
python sessions/ja/s01_agent_loop.py    # 日本語
python sessions/en/s01_agent_loop.py    # English
python sessions/zh/s01_agent_loop.py    # 中文
```

## 学習パス

各セクションは新しい概念を1つだけ追加し、前のコードはそのまま維持する:

```
Phase 1: 基礎         Phase 2: 接続            Phase 3: 知能            Phase 4: 自律           Phase 5: 本番
+----------------+    +-------------------+    +-----------------+     +-----------------+    +-----------------+
| s01: Loop      |    | s03: Sessions     |    | s06: Intelligence|    | s07: Heartbeat  |    | s09: Resilience |
| s02: Tools     | -> | s04: Channels     | -> |   魂, 記憶,     | -> |     & Cron       | -> |   & Concurrency |
|                |    | s05: Gateway      |    |   スキル,       |    | s08: Delivery   |    | s10: Lanes      |
|                |    |                   |    |   プロンプト    |    |                 |    |                 |
+----------------+    +-------------------+    +-----------------+     +-----------------+    +-----------------+
 ループ + dispatch     永続化 + ルーティング      人格 + 回想             能動行動 + 信頼性配信    リトライ + 直列化
```

## セクション詳細

| # | セクション | 核心コンセプト | 行数 |
|---|-----------|-------------|------|
| 01 | Agent Loop | `while True` + `stop_reason` -- これがエージェント | ~175 |
| 02 | Tool Use | ツール = schema dict + handler map. モデルが名前を選び、コードが実行 | ~445 |
| 03 | Sessions | JSONL: 書込は追記、読込は再生. 大きくなったら古い部分を要約 | ~890 |
| 04 | Channels | プラットフォームは違えど、全て同じ `InboundMessage` を生成 | ~780 |
| 05 | Gateway | バインドテーブルが (channel, peer) を agent に対応付け. 最も具体的な一致が勝つ | ~625 |
| 06 | Intelligence | システムプロンプト = ディスク上のファイル. ファイルを換えれば人格が変わる | ~750 |
| 07 | Heartbeat & Cron | タイマースレッド: 「実行すべき?」+ ユーザーメッセージと同じパイプライン | ~660 |
| 08 | Delivery | まずディスクに書き、それから送信. クラッシュしてもメッセージは失われない | ~870 |
| 09 | Resilience | 3層リトライオニオン: 認証輪換、オーバーフロー圧縮、ツールループ | ~1130 |
| 10 | Concurrency | 名前付きレーン + FIFOキュー、generation追跡、Future返却 | ~900 |

## リポジトリ構造

```
claw0/
  README.md              English README
  README.zh.md           Chinese README
  README.ja.md           Japanese README
  .env.example           設定テンプレート
  requirements.txt       Python 依存関係
  sessions/              全教学セッション (コード + ドキュメント)
    en/                  English
      s01_agent_loop.py  s01_agent_loop.md
      s02_tool_use.py    s02_tool_use.md
      ...                (10 .py + 10 .md)
    zh/                  Chinese
      s01_agent_loop.py  s01_agent_loop.md
      ...                (10 .py + 10 .md)
    ja/                  日本語
      s01_agent_loop.py  s01_agent_loop.md
      ...                (10 .py + 10 .md)
  workspace/             共有ワークスペースサンプル
    SOUL.md  IDENTITY.md  TOOLS.md  USER.md
    HEARTBEAT.md  BOOTSTRAP.md  AGENTS.md  MEMORY.md
    CRON.json
    skills/example-skill/SKILL.md
```

各言語フォルダは自己完結型: 実行可能な Python コード + ドキュメントが並置. コードロジックは全言語で同一、コメントとドキュメントのみ異なる.

## 前提条件

- Python 3.11+
- Anthropic (または互換プロバイダー) の API キー

## 依存関係

```
anthropic>=0.39.0
python-dotenv>=1.0.0
websockets>=12.0
croniter>=2.0.0
python-telegram-bot>=21.0
httpx>=0.27.0
```

## 関連プロジェクト

- **[learn-claude-code](https://github.com/shareAI-lab/learn-claude-code)** -- 12 の段階的セッションでエージェント**フレームワーク** (nano Claude Code) をゼロから構築する姉妹教材リポジトリ。claw0 がゲートウェイルーティング、チャネル、能動的行動に焦点を当てるのに対し、learn-claude-code はエージェントの内部設計を深掘りする: 構造化計画 (TodoManager + nag)、コンテキスト圧縮 (3層 compact)、ファイルベースのタスク永続化と依存グラフ、チーム連携 (JSONL メールボックス、シャットダウン/プラン承認 FSM)、自律的自己組織化、git worktree 分離による並行実行。本番グレードのユニットエージェントの内部動作を理解したい場合はそちらから。

## ライセンス

MIT
