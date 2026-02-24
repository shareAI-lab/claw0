# s06: Message Routing & Bindings (メッセージルーティングとバインディング)

> "Every message finds its home" -- ルーターがメッセージを正しいエージェントへ届ける。

## At a Glance

```
  Inbound Message
  {channel:"telegram", sender:"user123", kind:"direct"}
       |
       v
  +--- MessageRouter.resolve() ----------------+
  |                                             |
  |  Bindings (priority desc):                  |
  |  [P4] peer_id:user123           -> alice    |
  |  [P3] guild_id:server1          -> bob      |
  |  [P2] account_id:bot-account    -> main     |
  |  [P1] channel:telegram          -> main     |
  |  [P0] default                   -> main     |
  |                                             |
  +---------------------------------------------+
       |
       v
  AgentConfig(id="alice") + session_key
       |
       v
  run_agent(config, session_key, message)
```

- **What we build**: 一つのゲートウェイで複数のエージェントを同時に稼働させ、バインディングルールに基づいてメッセージを自動振り分けする。
- **Core mechanism**: 5 段階の優先度マッチング -- peer > guild > account > channel > default。
- **Design pattern**: 宣言的バインディング + dm_scope によるセッション分離粒度の制御。

## The Problem

1. **複数エージェントが共存できない**: s05 のゲートウェイは一つのエージェントしか扱えない。クリエイティブライティングのエージェントと技術Q&Aのエージェントを同時に稼働させるには、独立したゲートウェイを二つ運用する必要があり、運用コストが線形に増加する。
2. **メッセージの振り分け先が不明**: Telegram からのメッセージも Discord からのメッセージも、特定ユーザーからも一般ユーザーからも、すべて同一のエージェントに流れ込み、送信元に基づく分流ができない。
3. **セッション分離が制御不能**: 異なるユーザーの会話が互いに干渉し、同一ユーザーが異なるチャネルで持つコンテキストを独立管理することも共有することもできない。

## How It Works

### 1. AgentConfig -- マルチエージェント設定

各エージェントは独立した model と system_prompt を持ち、異なるプロンプトで異なる「性格」を表現する:

```python
@dataclass
class AgentConfig:
    id: str
    model: str
    system_prompt: str
    tools: list[dict] = field(default_factory=list)
```

設定例 -- 三つのエージェントが一つのゲートウェイを共有する:

```python
DEFAULT_CONFIG = {
    "agents": [
        {"id": "main",  "system_prompt": "You are a helpful assistant."},
        {"id": "alice", "system_prompt": "You are Alice, a creative writing assistant..."},
        {"id": "bob",   "system_prompt": "You are Bob, a technical assistant..."},
    ],
    "bindings": [
        {"peer_id": "user-alice-fan", "agent_id": "alice", "priority": 40},
        {"guild_id": "dev-server",    "agent_id": "bob",   "priority": 30},
        {"channel": "telegram",       "agent_id": "main",  "priority": 10},
    ],
    "default_agent": "main",
    "dm_scope": "per-peer",
}
```

**一つの JSON 設定ですべてのエージェントとルーティングルールを定義でき、コードの変更は不要。**

### 2. Binding -- 5 段階の優先度マッチング

Binding は「どの条件でどのエージェントにマッチするか」を定義する。フィールドが `None` の場合はワイルドカードとして扱われ、priority が高いほど先にマッチする:

```python
@dataclass
class Binding:
    channel: str | None = None       # P1: チャネルレベル
    account_id: str | None = None    # P2: アカウントレベル
    guild_id: str | None = None      # P3: グループレベル
    peer_id: str | None = None       # P4: ユーザーレベル (最も具体的)
    peer_kind: str | None = None
    agent_id: str = "main"
    priority: int = 0                # 大きいほど優先
```

ルーターは priority の降順で各バインディングを順に評価し、最初にマッチしたものを採用する:

```python
class MessageRouter:
    def resolve(self, channel, sender, peer_kind="direct",
                guild_id=None, account_id=None) -> tuple[AgentConfig, str]:
        matched_agent_id = self.default_agent
        for binding in self.bindings:  # priority 降順でソート済み
            if self._matches(binding, channel, sender, peer_kind, guild_id, account_id):
                matched_agent_id = binding.agent_id
                break
        agent = self.agents.get(matched_agent_id) or self.agents[self.default_agent]
        session_key = build_session_key(
            agent_id=agent.id, channel=channel,
            peer_kind=peer_kind,
            peer_id=sender if peer_kind == "direct" else (guild_id or sender),
            dm_scope=self.dm_scope,
        )
        return agent, session_key
```

マッチングロジック: すべての非空フィールドがマッチする必要があり、すべて一致して初めてヒットとみなす:

```python
def _matches(self, binding, channel, sender, peer_kind, guild_id, account_id) -> bool:
    if binding.channel and binding.channel.lower() != channel.lower():
        return False
    if binding.peer_id and binding.peer_id.lower() != sender.lower():
        return False
    # guild_id, account_id, peer_kind も同様...
    return True
```

**優先度モデルによりルールが一目瞭然: 最も具体的な (peer) が最高優先度、フォールバック (default) が最低優先度。**

### 3. Session Key -- セッション分離

session key はエージェント + チャネル + ピア情報をエンコードし、`dm_scope` が DM の分離粒度を制御する:

```python
def build_session_key(agent_id, channel, peer_kind, peer_id, dm_scope="per-peer") -> str:
    if peer_kind != "direct":
        return f"agent:{agent_id}:{channel}:{peer_kind}:{peer_id}"
    if dm_scope == "main":
        return f"agent:{agent_id}:main"
    elif dm_scope == "per-peer":
        return f"agent:{agent_id}:direct:{peer_id}"
    elif dm_scope == "per-channel-peer":
        return f"agent:{agent_id}:{channel}:direct:{peer_id}"
```

| dm_scope | session key | ユースケース |
|----------|-------------|------|
| `main` | `agent:alice:main` | パーソナルアシスタント -- すべての DM が一つのセッションを共有 |
| `per-peer` | `agent:alice:direct:user123` | マルチユーザーボット -- ユーザーごとに独立 |
| `per-channel-peer` | `agent:alice:telegram:direct:user123` | 同一ユーザーがチャネルごとに独立 |

**session key は自動的に構築され、クライアント側で手動指定する必要はない。**

### 4. RoutingGateway -- ルーティング対応ゲートウェイ

s05 をベースに `identify` (身元宣言) とルーティング診断メソッドを追加:

```python
class RoutingGateway:
    def __init__(self, host, port, router, sessions, token=""):
        self._methods = {
            "health": ...,
            "chat.send": self._handle_chat_send,
            "identify": self._handle_identify,
            "routing.resolve": self._handle_routing_resolve,
            "routing.bindings": self._handle_routing_bindings,
            "sessions.list": self._handle_sessions_list,
        }
```

`chat.send` はルーティング解決を自動的に通過する:

```python
async def _handle_chat_send(self, client, params):
    channel = params.get("channel", client.channel)
    sender = params.get("sender", client.sender)
    agent_config, session_key = self.router.resolve(
        channel=channel, sender=sender, ...
    )
    session = self.sessions.get_or_create(session_key, agent_id=agent_config.id)
    return run_agent(agent_config, session, text)
```

**クライアントはメッセージを送るだけで、ルーターが誰が処理するか、どのセッションを使うかを自動的に決定する。**

## What Changed from s05

| コンポーネント | s05 | s06 |
|-----------|-----|-----|
| エージェント | 単一、固定 system_prompt | マルチエージェント、個別設定 |
| ルーティング | なし、すべてのメッセージが同一エージェント | Binding 優先度マッチング、自動振り分け |
| セッション key | クライアントが手動指定 | dm_scope に基づき自動構築 |
| クライアント身元 | なし | `identify` でチャネル/送信者を宣言 |
| 設定 | ハードコード | JSON ファイルまたはデフォルト設定 |
| RPC メソッド | 4 個 | 7 個 (routing/sessions/identify を追加) |

**Key shift**: 「一つのゲートウェイに一つのエージェント」から「一つのゲートウェイに複数エージェント、ルーティングで自動振り分け」へ。

## Design Decisions

**なぜルールチェーンではなく優先度順なのか?**

優先度モデルはシンプルで直感的であり、運用担当者が一目でどのルールが適用されるか分かる。条件の組み合わせロジックや決定木の走査は不要だ。OpenClaw の本番版も同じ優先度ソートを採用している。

**なぜ送信者 ID をそのまま session key として使わないのか?**

同一ユーザーが Telegram と Discord で同時に対話したり、異なるエージェントとやり取りしたりする可能性がある。session key は agent_id + channel + peer の三つの次元をエンコードする必要があり、これらのシナリオを正しく分離できる。`dm_scope` により管理者は必要に応じて粒度を選択できる。

**In production OpenClaw:** Binding の定義は `src/routing/bindings.ts`、session key の構築は `src/routing/session-key.ts` にある。本番版は identity links (クロスチャネルで同一ユーザーを関連付け)、チームレベルバインディング (一つのバインディングがエージェントグループを指定)、`per-account-channel-peer` などのより細かい dm_scope を追加サポートしている。ルーティング解決ではさらに許可リスト、ミュートリスト、コマンドゲーティングなどの次元も考慮される。

## Try It

```sh
cd claw0
python agents/s06_routing.py
```

ゲートウェイを起動した後、別のターミナルでテストクライアントを実行する:

```sh
python agents/s06_routing.py --test-client
```

出力例:

```
--- Routing Diagnostics ---
   telegram | sender=random-user     | kind=direct -> agent=main   session=agent:main:direct:random-user
   telegram | sender=user-alice-fan  | kind=direct -> agent=alice  session=agent:alice:direct:user-alice-fan
    discord | sender=dev-person      | kind=group  -> agent=bob    session=agent:bob:discord:group:dev-server
      slack | sender=someone         | kind=direct -> agent=main   session=agent:main:direct:someone
```

REPL モードでローカルにルーティングをデバッグすることもできる (ゲートウェイの起動不要、LLM 呼び出し不要):

```sh
python agents/s06_routing.py --repl
```

```
route> telegram user-alice-fan
  Agent:       alice
  Session Key: agent:alice:direct:user-alice-fan

route> discord dev-person group dev-server
  Agent:       bob
  Session Key: agent:bob:discord:group:dev-server
```
