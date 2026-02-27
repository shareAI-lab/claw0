# 第05章: ゲートウェイとルーティング

> バインディングテーブルが (channel, peer) を agent_id にマッピングする。最も具体的なものが優先。

## アーキテクチャ

```
    Inbound Message (channel, account_id, peer_id, text)
           |
    +------v------+     +----------+
    |   Gateway    | <-- | WS/REPL  |  JSON-RPC 2.0
    +------+------+     +----------+
           |
    +------v------+
    | BindingTable |  5段階の解決:
    +------+------+    T1: peer_id     (最も具体的)
           |           T2: guild_id
           |           T3: account_id
           |           T4: channel
           |           T5: default     (最も一般的)
           |
     (agent_id, binding)
           |
    +------v---------+
    | build_session_key() |  dm_scope がセッション分離を制御
    +------+---------+
           |
    +------v------+
    | AgentManager |  エージェントごとの設定 / 性格 / セッション
    +------+------+
           |
        LLM API
```

## 本章のポイント

- **BindingTable**: ルートバインディングのソート済みリスト。ティア1-5を順に走査し、最初にマッチしたものが勝つ。
- **build_session_key()**: `dm_scope` がセッション分離を制御 (peer単位、channel単位 等)。
- **AgentManager**: マルチエージェントレジストリ -- 各エージェントが独自の性格とモデルを持つ。
- **GatewayServer**: JSON-RPC 2.0を話すオプションのWebSocketサーバー。
- **共有イベントループ**: デーモンスレッド内のasyncioループ、セマフォで同時実行数を4に制限。

## コードウォークスルー

### 1. BindingTable.resolve() -- ルーティングの核心

バインディングは `(tier, -priority)` でソートされる。解決は線形走査で、最初のマッチが勝つ。

```python
@dataclass
class Binding:
    agent_id: str
    tier: int           # 1-5, 小さいほど具体的
    match_key: str      # "peer_id" | "guild_id" | "account_id" | "channel" | "default"
    match_value: str    # 例: "telegram:12345", "discord", "*"
    priority: int = 0   # 同ティア内では大きいほど優先

class BindingTable:
    def resolve(self, channel="", account_id="",
                guild_id="", peer_id="") -> tuple[str | None, Binding | None]:
        for b in self._bindings:
            if b.tier == 1 and b.match_key == "peer_id":
                if ":" in b.match_value:
                    if b.match_value == f"{channel}:{peer_id}":
                        return b.agent_id, b
                elif b.match_value == peer_id:
                    return b.agent_id, b
            elif b.tier == 2 and b.match_key == "guild_id" and b.match_value == guild_id:
                return b.agent_id, b
            elif b.tier == 3 and b.match_key == "account_id" and b.match_value == account_id:
                return b.agent_id, b
            elif b.tier == 4 and b.match_key == "channel" and b.match_value == channel:
                return b.agent_id, b
            elif b.tier == 5 and b.match_key == "default":
                return b.agent_id, b
        return None, None
```

以下のデモバインディングの場合:

```python
bt.add(Binding(agent_id="luna", tier=5, match_key="default", match_value="*"))
bt.add(Binding(agent_id="sage", tier=4, match_key="channel", match_value="telegram"))
bt.add(Binding(agent_id="sage", tier=1, match_key="peer_id",
               match_value="discord:admin-001", priority=10))
```

| 入力                                | ティア | エージェント |
|-------------------------------------|--------|-------------|
| `channel=cli, peer=user1`           | 5      | Luna        |
| `channel=telegram, peer=user2`      | 4      | Sage        |
| `channel=discord, peer=admin-001`   | 1      | Sage        |
| `channel=discord, peer=user3`       | 5      | Luna        |

### 2. dm_scope によるセッションキー

エージェントが解決された後、エージェント設定の `dm_scope` がセッション分離を制御する:

```python
def build_session_key(agent_id, channel="", account_id="",
                      peer_id="", dm_scope="per-peer"):
    aid = normalize_agent_id(agent_id)
    if dm_scope == "per-account-channel-peer" and peer_id:
        return f"agent:{aid}:{channel}:{account_id}:direct:{peer_id}"
    if dm_scope == "per-channel-peer" and peer_id:
        return f"agent:{aid}:{channel}:direct:{peer_id}"
    if dm_scope == "per-peer" and peer_id:
        return f"agent:{aid}:direct:{peer_id}"
    return f"agent:{aid}:main"
```

| dm_scope                   | キーのフォーマット                       | 効果                                |
|----------------------------|------------------------------------------|-------------------------------------|
| `main`                     | `agent:{id}:main`                        | 全員が1つのセッションを共有        |
| `per-peer`                 | `agent:{id}:direct:{peer}`               | ユーザーごとに分離                  |
| `per-channel-peer`         | `agent:{id}:{ch}:direct:{peer}`          | プラットフォームごとに別セッション  |
| `per-account-channel-peer` | `agent:{id}:{ch}:{acc}:direct:{peer}`    | 最も分離度が高い                    |

### 3. AgentConfig -- エージェントごとの性格

各エージェントが独自の設定を持つ。システムプロンプトはここから生成される:

```python
@dataclass
class AgentConfig:
    id: str
    name: str
    personality: str = ""
    model: str = ""              # 空 = グローバルの MODEL_ID を使用
    dm_scope: str = "per-peer"

    def system_prompt(self) -> str:
        parts = [f"You are {self.name}."]
        if self.personality:
            parts.append(f"Your personality: {self.personality}")
        parts.append("Answer questions helpfully and stay in character.")
        return " ".join(parts)
```

## 試してみる

```sh
python ja/s05_gateway_routing.py

# ルーティングのテスト
# You > /bindings                      (全ルートバインディングを表示)
# You > /route cli user1               (defaultでLunaに解決)
# You > /route telegram user2           (channelバインディングでSageに解決)

# 特定のエージェントを強制指定
# You > /switch sage
# You > Hello!                          (ルートに関係なくSageと会話)
# You > /switch off                     (通常ルーティングに復帰)

# WebSocketゲートウェイの起動
# You > /gateway
# Gateway running on ws://localhost:8765
```

## OpenClaw での実装

| 観点             | claw0 (本ファイル)             | OpenClaw 本番環境                      |
|------------------|--------------------------------|----------------------------------------|
| ルート解決       | 5ティアの線形走査              | 同じティアシステム + 設定ファイル      |
| セッションキー   | `dm_scope` パラメータ          | 同じdm_scope + 永続セッション         |
| マルチエージェント | メモリ上の AgentConfig        | エージェントごとのワークスペースディレクトリ |
| ゲートウェイ     | WebSocket + JSON-RPC 2.0      | 同じプロトコル + HTTP API              |
| 並行性           | `asyncio.Semaphore(4)`         | 同じセマフォパターン                   |
