# 第09章: レジリエンス

> 1つの呼び出しが失敗したら、ローテーションしてリトライ。

## アーキテクチャ

```
    Profiles: [main-key, backup-key, emergency-key]
         |
    for each non-cooldown profile:          LAYER 1: Auth Rotation
         |
    create client(profile.api_key)
         |
    for compact_attempt in 0..2:            LAYER 2: Overflow Recovery
         |
    _run_attempt(client, model, ...)        LAYER 3: Tool-Use Loop
         |              |
       success       exception
         |              |
    mark_success    classify_failure()
    return result       |
                   overflow? --> compact, retry Layer 2
                   auth/rate? -> mark_failure, break to Layer 1
                   timeout?  --> mark_failure(60s), break to Layer 1
                        |
                   all profiles exhausted?
                        |
                   try fallback models
                        |
                   all fallbacks failed?
                        |
                   raise RuntimeError
```

## 本章のポイント

- **FailoverReason**: 全ての例外を6つのカテゴリ (rate_limit, auth, timeout, billing, overflow, unknown) に分類する列挙型。カテゴリがどのリトライ層で処理するかを決定する。
- **AuthProfile**: 1つの API キーとクールダウン状態を保持するデータクラス。`cooldown_until`、`failure_reason`、`last_good_at` を追跡する。
- **ProfileManager**: 最初のクールダウンしていないプロファイルを選択し、障害をマーク (クールダウンを設定) し、成功をマーク (障害状態をクリア) する。
- **ContextGuard**: 軽量なコンテキストオーバーフロー保護。サイズ超過のツール結果を切り詰め、まだオーバーフローする場合は LLM 要約で履歴を圧縮する。
- **ResilienceRunner**: 3層リトライオニオン。Layer 1 はプロファイルをローテーション、Layer 2 はオーバーフロー圧縮を処理、Layer 3 は標準的なツール使用ループ。
- **リトライ制限**: `BASE_RETRY=24`、`PER_PROFILE=8`、上限は `min(max(base + per_profile * N, 32), 160)`。
- **SimulatedFailure**: 次の API 呼び出し用に合成エラーを準備し、実際の障害なしに各障害クラスの動作を観察できる。

## コードウォークスルー

### 1. classify_failure() -- 例外を適切な層にルーティング

全ての例外はリトライオニオンが次の動作を決定する前に分類を通過する。分類器はエラー文字列を既知のパターンで検査する:

```python
class FailoverReason(Enum):
    rate_limit = "rate_limit"
    auth = "auth"
    timeout = "timeout"
    billing = "billing"
    overflow = "overflow"
    unknown = "unknown"

def classify_failure(exc: Exception) -> FailoverReason:
    msg = str(exc).lower()
    if "rate" in msg or "429" in msg:
        return FailoverReason.rate_limit
    if "auth" in msg or "401" in msg or "key" in msg:
        return FailoverReason.auth
    if "timeout" in msg or "timed out" in msg:
        return FailoverReason.timeout
    if "billing" in msg or "quota" in msg or "402" in msg:
        return FailoverReason.billing
    if "context" in msg or "token" in msg or "overflow" in msg:
        return FailoverReason.overflow
    return FailoverReason.unknown
```

分類結果が異なるクールダウン期間を駆動する:
- `auth` / `billing`: 300秒 (不正なキー、すぐには自己回復しない)
- `rate_limit`: 120秒 (レートリミットウィンドウのリセットを待つ)
- `timeout`: 60秒 (一過性、短いクールダウン)
- `overflow`: プロファイルにクールダウンなし -- 代わりにメッセージを圧縮

### 2. ProfileManager -- クールダウン対応のキーローテーション

プロファイルは順番にチェックされる。クールダウンが切れたプロファイルが利用可能。障害時にはプロファイルがクールダウンに入り、成功時には障害状態がクリアされる。

```python
class ProfileManager:
    def select_profile(self) -> AuthProfile | None:
        now = time.time()
        for profile in self.profiles:
            if now >= profile.cooldown_until:
                return profile
        return None

    def mark_failure(self, profile, reason, cooldown_seconds=300.0):
        profile.cooldown_until = time.time() + cooldown_seconds
        profile.failure_reason = reason.value

    def mark_success(self, profile):
        profile.failure_reason = None
        profile.last_good_at = time.time()
```

### 3. ResilienceRunner.run() -- 3層オニオン

外側のループがプロファイルを順次試行 (Layer 1)。中間のループが圧縮後のオーバーフローをリトライ (Layer 2)。内側の呼び出しがツール使用ループを実行 (Layer 3)。

```python
def run(self, system, messages, tools):
    # LAYER 1: Auth Rotation
    for _rotation in range(len(self.profile_manager.profiles)):
        profile = self.profile_manager.select_profile()
        if profile is None:
            break

        api_client = Anthropic(api_key=profile.api_key)

        # LAYER 2: Overflow Recovery
        layer2_messages = list(messages)
        for compact_attempt in range(MAX_OVERFLOW_COMPACTION):
            try:
                # LAYER 3: Tool-Use Loop
                result, layer2_messages = self._run_attempt(
                    api_client, self.model_id, system,
                    layer2_messages, tools,
                )
                self.profile_manager.mark_success(profile)
                return result, layer2_messages

            except Exception as exc:
                reason = classify_failure(exc)

                if reason == FailoverReason.overflow:
                    # Compact and retry Layer 2
                    layer2_messages = self.guard.truncate_tool_results(layer2_messages)
                    layer2_messages = self.guard.compact_history(
                        layer2_messages, api_client, self.model_id)
                    continue

                elif reason in (FailoverReason.auth, FailoverReason.rate_limit):
                    self.profile_manager.mark_failure(profile, reason)
                    break  # try next profile (Layer 1)

                elif reason == FailoverReason.timeout:
                    self.profile_manager.mark_failure(profile, reason, 60)
                    break  # try next profile (Layer 1)

    # All profiles exhausted -- try fallback models
    for fallback_model in self.fallback_models:
        # ... try with first available profile ...

    raise RuntimeError("all profiles and fallbacks exhausted")
```

### 4. _run_attempt() -- Layer 3 ツール使用ループ

最内層は Section 01/02 と同じ `while True` + `stop_reason` ディスパッチ。モデルが `end_turn` を返すか例外が外側の層に伝播するまでツール呼び出しをループ実行する。

```python
def _run_attempt(self, api_client, model, system, messages, tools):
    current_messages = list(messages)
    iteration = 0

    while iteration < self.max_iterations:
        iteration += 1
        response = api_client.messages.create(
            model=model, max_tokens=8096,
            system=system, tools=tools,
            messages=current_messages,
        )
        current_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return response, current_messages

        elif response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result = process_tool_call(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })
            current_messages.append({"role": "user", "content": tool_results})
            continue

    raise RuntimeError("Tool-use loop exceeded max iterations")
```

## 試してみる

```sh
python ja/s09_resilience.py

# 通常の会話 -- 単一プロファイルでの成功を観察
# You > Hello!

# プロファイルステータスを表示
# You > /profiles

# レートリミット障害をシミュレート -- プロファイルローテーションを観察
# You > /simulate-failure rate_limit
# You > Tell me a joke

# 認証障害をシミュレート
# You > /simulate-failure auth
# You > What time is it?

# 障害後のクールダウンを確認
# You > /cooldowns

# フォールバックチェーンを確認
# You > /fallback

# レジリエンス統計を表示
# You > /stats
```

## OpenClaw での実装

| 観点                | claw0 (本ファイル)                       | OpenClaw 本番環境                            |
|---------------------|------------------------------------------|----------------------------------------------|
| プロファイルローテーション | デモ用3プロファイル、同一キー        | 複数プロバイダーにまたがる実際のキー         |
| 障害分類器          | 例外テキストの文字列マッチング           | 同じパターン + HTTP ステータスコードチェック  |
| オーバーフロー回復  | ツール結果の切り詰め + LLM 要約          | 同じ2段階圧縮                                |
| クールダウン追跡    | メモリ内の float タイムスタンプ           | 同じプロファイル単位のメモリ内追跡           |
| フォールバックモデル | 設定可能なフォールバックチェーン          | 同じチェーン、通常はより小型/安価なモデル    |
| リトライ制限        | BASE_RETRY=24, PER_PROFILE=8, 上限=160   | 同じ計算式                                   |
| シミュレート障害    | テスト用 /simulate-failure コマンド       | 障害注入付き統合テストハーネス               |
