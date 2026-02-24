# s05: Gateway Server (ゲートウェイサーバー)

> "The switchboard" -- WebSocket サービスが JSON-RPC 2.0 ですべての通信を統一する。

## At a Glance

```
Browser     Mobile     CLI Client
   |           |           |
   v           v           v
+--------- WebSocket ---------+
|        GatewayServer         |
| +---------------------------+|
| | JSON-RPC 2.0 Router       ||
| |  health      -> status    ||
| |  chat.send   -> run_agent ||
| |  chat.history-> history   ||
| |  channels.status -> list  ||
| +---------------------------+|
|       |             |        |
| SessionStore   broadcast()   |
+------------------------------+
```

- **What we build**: WebSocket ゲートウェイサーバー。JSON-RPC 2.0 プロトコルにより、任意のクライアントがエージェントと対話できる。
- **Core mechanism**: WebSocket 全二重通信 + JSON-RPC メソッドルーティングテーブル + イベントブロードキャスト。
- **Design pattern**: 接続 -> 認証 -> メッセージループ -> メソッドディスパッチ -> 結果/イベントプッシュ。

## The Problem

1. **ターミナルからしか使えない。** s01-s04 のエージェントは `input()` で対話する。ブラウザ、スマートフォンアプリ、リモート CLI は接続できない。

2. **構造化されたプロトコルがない。** メッセージはプレーンテキストであり、型の区別も、リクエスト ID も、エラーコードもない。クライアントは「応答」と「イベント」を区別できず、リクエストとレスポンスのマッチングもできない。

3. **サーバーからのプッシュができない。** HTTP はリクエスト-レスポンスモデルであり、サーバーからクライアントへ「考え中です」や「他のクライアントがメッセージを送りました」と能動的に通知できない。AI 対話にはタイピングイベント、ストリーミング出力、マルチクライアント同期が必要だ。

## How It Works

### 1. JSON-RPC 2.0 プロトコル

すべての WebSocket 通信は JSON-RPC 2.0 に準拠し、三種類のメッセージがある:

```python
# リクエスト -- クライアントが送信、id 付き
{"jsonrpc": "2.0", "id": "req-1", "method": "chat.send", "params": {"text": "hello"}}

# レスポンス -- サーバーが返信、id でリクエストと対応
{"jsonrpc": "2.0", "id": "req-1", "result": {"text": "...", "session_key": "..."}}

# イベント -- サーバーが能動的にプッシュ、id なし
{"jsonrpc": "2.0", "method": "event", "params": {"type": "chat.typing", "session_key": "..."}}
```

判別ルール: `id` + `method` があればリクエスト、`id` + `result`/`error` があればレスポンス、`method` のみで `id` がなければイベント。

**三つのヘルパー関数がこれら三種のメッセージを構築する:**

```python
def make_result(req_id, result):
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})

def make_error(req_id, code, message, data=None):
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return json.dumps({"jsonrpc": "2.0", "id": req_id, "error": err})

def make_event(event_type, payload):
    return json.dumps({"jsonrpc": "2.0", "method": "event",
                        "params": {"type": event_type, **payload}})
```

### 2. 接続のライフサイクル

クライアント接続後の流れ: 認証 -> 登録 -> ウェルカムイベント -> メッセージループ -> 切断時のクリーンアップ。

```python
async def _handle_connection(self, ws: ServerConnection) -> None:
    client_id = str(uuid.uuid4())[:8]

    authenticated = self._authenticate(ws.request.headers if ws.request else {})
    if not authenticated:
        await ws.send(make_error(None, AUTH_ERROR, "Authentication failed"))
        await ws.close(4001, "Unauthorized")
        return

    client = ConnectedClient(ws=ws, client_id=client_id, authenticated=True)
    self.clients[client_id] = client
    await ws.send(make_event("connect.welcome", {
        "client_id": client_id, "server_time": time.time(),
    }))

    try:
        async for raw_message in ws:
            await self._dispatch(client, raw_message)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        del self.clients[client_id]
```

**認証失敗時は直ちに接続を閉じ、メッセージループには入らない。**

### 3. メッセージディスパッチ

各メッセージは四つのステップを経る: JSON パース -> フォーマット検証 -> ルーティングテーブル検索 -> 実行して返却。

```python
async def _dispatch(self, client: ConnectedClient, raw: str) -> None:
    try:
        msg = json.loads(raw)
    except json.JSONDecodeError:
        await client.ws.send(make_error(None, PARSE_ERROR, "invalid JSON"))
        return

    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        await client.ws.send(make_error(msg.get("id"), INVALID_REQUEST, "Invalid request"))
        return

    handler = self._methods.get(msg.get("method", ""))
    if handler is None:
        await client.ws.send(make_error(msg.get("id"), METHOD_NOT_FOUND, "Not found"))
        return

    try:
        result = await handler(client, msg.get("params", {}))
        await client.ws.send(make_result(msg.get("id"), result))
    except Exception as exc:
        await client.ws.send(make_error(msg.get("id"), INTERNAL_ERROR, str(exc)))
```

**各ステップの失敗に対応する JSON-RPC エラーコードがある: -32700 (パース)、-32600 (フォーマット)、-32601 (メソッド)、-32603 (内部)。**

### 4. chat.send -- 中核の RPC メソッド

エージェントにメッセージを送信し、途中で typing イベントをプッシュし、完了後に done イベントをブロードキャストする:

```python
async def _handle_chat_send(self, client: ConnectedClient, params: dict) -> dict:
    text = params.get("text", "").strip()
    session_key = params.get("session_key", "default")

    await client.ws.send(make_event("chat.typing", {"session_key": session_key}))

    session = self.sessions.get_or_create(session_key)
    assistant_text = run_agent(session, text)

    await self._broadcast(make_event("chat.done", {
        "session_key": session_key, "text": assistant_text,
    }))

    return {
        "text": assistant_text,
        "session_key": session_key,
        "message_count": len(session.messages),
    }
```

**シーケンス: テキスト受信 -> typing イベント -> LLM 呼び出し -> done イベント (ブロードキャスト) -> result レスポンス。**

### 5. メソッドルーティングテーブルとブロードキャスト

ゲートウェイの本質は RPC ディスパッチャであり、全クライアントへのブロードキャスト機能を備える:

```python
self._methods = {
    "health": self._handle_health,
    "chat.send": self._handle_chat_send,
    "chat.history": self._handle_chat_history,
    "channels.status": self._handle_channels_status,
}

async def _broadcast(self, message: str) -> None:
    tasks = [c.ws.send(message) for c in self.clients.values()]
    await asyncio.gather(*tasks, return_exceptions=True)
```

**asyncio.gather で並行送信し、個別のクライアントの失敗が他のクライアントに影響しない。**

## What Changed from s04

| コンポーネント | s04 | s05 |
|-----------|-----|-----|
| エントリポイント | `input()` でターミナル読み取り | WebSocket ネットワーク接続 |
| プロトコル | プレーンテキスト | JSON-RPC 2.0 (リクエスト/レスポンス/イベント) |
| クライアント | 単一ターミナル | 複数クライアント並行 (ConnectedClient) |
| 認証 | なし | Bearer Token |
| メッセージプッシュ | なし | broadcast + typing/done イベント |
| エラー処理 | Python 例外 | JSON-RPC エラーコード |

**Key shift**: 「プロセス内の関数呼び出し」から「ネットワークプロトコル通信」へ。エージェントはユーザーと直接対話するのではなく、ゲートウェイを介して構造化された RPC プロトコルで任意のクライアントと通信する。

## Design Decisions

**なぜ HTTP ではなく WebSocket なのか?**

HTTP はリクエスト-レスポンスモデルであり、サーバーから能動的にプッシュできない。AI 対話にはタイピングイベント、ストリーミング出力、マルチクライアント同期が必要であり、いずれもサーバーからクライアントへの能動的なメッセージ送信を要する。WebSocket は全二重通信を提供する。

**なぜ JSON-RPC 2.0 なのか?**

仕様が極めてシンプル (リクエスト/レスポンス/通知の三種)、標準化されたエラーコード体系があり、バッチリクエストにも対応し、どの言語でもクライアントを容易に実装できる。

**In production OpenClaw:** `wss://` (TLS 暗号化) を使用し、デバイスペアリング (QR コード/ペアリングコード) をサポートし、session_key とクライアントロールに基づいてブロードキャスト対象をフィルタリングし、クライアントとサーバー間でプロトコルバージョンをネゴシエートする。メソッドルーティングテーブルは教育版の 4 個から約 100 個に拡張されている。

## Try It

ゲートウェイを起動する:

```sh
cd claw0
python agents/s05_gateway.py
```

別のターミナルでテストクライアントを実行する:

```sh
python agents/s05_gateway.py --test-client
```

テストクライアントは順に: welcome イベントを受信 -> health を呼び出し -> chat.send を呼び出し (typing/done イベントを観察) -> chat.history を呼び出し -> channels.status を呼び出し -> 未知のメソッドを送信 (エラーを観察)。

wscat を使って手動テストすることもできる:

```sh
wscat -c ws://127.0.0.1:18789
> {"jsonrpc":"2.0","id":"1","method":"health","params":{}}
```

`GATEWAY_TOKEN` を設定している場合、接続時に認証ヘッダーが必要:

```sh
wscat -c ws://127.0.0.1:18789 -H "Authorization: Bearer your-token"
```
