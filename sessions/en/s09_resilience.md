# Section 09: Resilience

> When one call fails, rotate and retry.

## Architecture

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

## Key Concepts

- **FailoverReason**: enum that classifies every exception into one of six categories (rate_limit, auth, timeout, billing, overflow, unknown). The category determines which retry layer handles it.
- **AuthProfile**: a dataclass holding one API key plus cooldown state. Tracks `cooldown_until`, `failure_reason`, and `last_good_at`.
- **ProfileManager**: selects the first non-cooldown profile, marks failures (sets cooldown), marks successes (clears failure state).
- **ContextGuard**: lightweight context overflow protection. Truncates oversized tool results, then compacts history via LLM summary if still overflowing.
- **ResilienceRunner**: the 3-layer retry onion. Layer 1 rotates profiles, Layer 2 handles overflow compaction, Layer 3 is the standard tool-use loop.
- **Retry limits**: `BASE_RETRY=24`, `PER_PROFILE=8`, capped at `min(max(base + per_profile * N, 32), 160)`.
- **SimulatedFailure**: arms a synthetic error for the next API call, letting you observe each failure class in action without real failures.

## Key Code Walkthrough

### 1. classify_failure() -- route exceptions to the right layer

Every exception passes through classification before the retry onion decides
what to do. The classifier examines the error string for known patterns:

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

The classification drives different cooldown durations:
- `auth` / `billing`: 300s (bad key, won't self-heal quickly)
- `rate_limit`: 120s (wait for rate limit window to reset)
- `timeout`: 60s (transient, short cooldown)
- `overflow`: no cooldown on profile -- compact messages instead

### 2. ProfileManager -- cooldown-aware key rotation

Profiles are checked in order. A profile is available when its cooldown has
expired. On failure, the profile goes on cooldown; on success, failure state
is cleared.

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

### 3. ResilienceRunner.run() -- the 3-layer onion

The outer loop iterates profiles (Layer 1). The middle loop retries on
overflow after compaction (Layer 2). The inner call runs the tool-use
loop (Layer 3).

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

### 4. _run_attempt() -- Layer 3 tool-use loop

The innermost layer is the same `while True` + `stop_reason` dispatch from
Sections 01/02. It runs tool calls in a loop until the model returns
`end_turn` or an exception propagates to the outer layers.

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

## Try It

```sh
python en/s09_resilience.py

# Normal conversation -- observe single-profile success
# You > Hello!

# View profile status
# You > /profiles

# Simulate a rate limit failure -- watch profile rotation
# You > /simulate-failure rate_limit
# You > Tell me a joke

# Simulate an auth failure
# You > /simulate-failure auth
# You > What time is it?

# Check cooldowns after failures
# You > /cooldowns

# Check fallback chain
# You > /fallback

# View resilience statistics
# You > /stats
```

## How OpenClaw Does It

| Aspect              | claw0 (this file)                        | OpenClaw production                          |
|---------------------|------------------------------------------|----------------------------------------------|
| Profile rotation    | 3 demo profiles, same key                | Multiple real keys across providers          |
| Failure classifier  | String matching on exception text         | Same pattern, plus HTTP status code checks   |
| Overflow recovery   | Truncate tool results + LLM summary      | Same 2-stage compaction                      |
| Cooldown tracking   | In-memory float timestamps               | Same in-memory tracking per profile          |
| Fallback models     | Configurable fallback chain               | Same chain, typically smaller/cheaper models |
| Retry limits        | BASE_RETRY=24, PER_PROFILE=8, cap=160    | Same formula                                 |
| Simulated failures  | /simulate-failure command for testing     | Integration test harness with fault injection|
