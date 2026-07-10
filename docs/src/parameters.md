# Completion Parameters

Two layers of parameters:

- **`CompletionParameters`**: normalized generation *intent* (temperature, max
  tokens, stop, response format, ...). NOT a wire shape: each provider's
  `build_request` maps it to its own request body, so the same params drive any
  provider identically.
- **`NodeCompletionParameters`**: per-request behavior around the call (system
  prompt override, JSON repair, retry, cost tracking, caching, the wrapped
  `CompletionParameters`).

You pass `NodeCompletionParameters` to `complete`; `None` means defaults.

## CompletionParameters

```rust,no_run
use minillmlib::CompletionParameters;

let params = CompletionParameters::new()
    .with_max_tokens(512)
    .with_temperature(0.7)
    .with_stop(vec!["END".to_string()]);
```

| Field | Meaning |
|---|---|
| `max_tokens` | Provider emits its own key (`max_completion_tokens`, `max_tokens`, Anthropic's required `max_tokens`) |
| `temperature`, `top_p`, `top_k` | Sampling |
| `frequency_penalty`, `presence_penalty`, `repetition_penalty` | Penalties |
| `stop` | Stop sequences (Anthropic `stop_sequences`) |
| `seed` | Reproducibility |
| `response_format` | Force JSON output (`with_json_response()`) |
| `reasoning` | Extended-thinking effort/budget |
| `tools`, `tool_choice`, `parallel_tool_calls` | Normalized tool calling; the provider emits its wire shape (see [Tool Calling](./tool-calling.md)) |
| `extra` | Provider-specific keys (the honest escape hatch, e.g. OpenRouter routing) |

`CompletionParameters` is also a serde type: camelCase keys, every field
optional (missing ones take the defaults above), unknown keys ignored. A flat
JSON settings object (`{"maxTokens": 1024, "temperature": 0.2}`) deserializes
directly, which is handy when parameters arrive as user-facing config.

## NodeCompletionParameters

```rust,no_run
use minillmlib::{CompletionParameters, NodeCompletionParameters};

let params = NodeCompletionParameters::new()
    .with_params(CompletionParameters::new().with_max_tokens(200))
    .with_system_prompt("You are concise.")  // prepend if the thread has no system message
    .expecting_json()                         // parse + repair the response as JSON
    .with_force_prepend("Answer: ")           // make the model continue from this prefix
    .with_cost_tracking(true);                // request usage and fire the cost callback
```

| Builder | Meaning |
|---|---|
| `with_params(..)` | The wrapped `CompletionParameters` |
| `with_system_prompt(..)` | Prepend a system message if absent |
| `with_format_kwargs(..)` / `with_format_kwarg(k, v)` | Fill `{placeholder}`s thread-wide at call time |
| `with_parse_json(true)` / `expecting_json()` | Repair the response as JSON |
| `with_force_prepend(..)` | Prime the assistant turn so the model continues it |
| `with_cache(true)` | Auto-mark the whole prefix for caching (see [Caching](./caching.md)) |
| `with_cost_tracking(true)` | Request and report usage/cost |
| `with_token_price(..)` | Per-request price override |
| `retry`, `exp_back_off`, `back_off_time`, `max_back_off` | Retry policy |
| `crash_on_refusal`, `crash_on_empty_response` | Reject empty / no-JSON responses |
| `timeout_secs` | Total deadline (non-streaming) or idle timeout (streaming) |
