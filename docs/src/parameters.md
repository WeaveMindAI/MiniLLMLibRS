# Completion Parameters

Two layers of parameters:

- **`CompletionParameters`**: what you want from the model (temperature, output
  cap, stop sequences, reasoning, JSON output, ...). You set them once, and each
  provider writes them in its own request format.
- **`NodeCompletionParameters`**: what happens around the call (a system
  message, JSON repair, retries, cost tracking, caching, and the
  `CompletionParameters` themselves).

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
| `max_tokens` | The most tokens the model may write, sent under the provider's own key (`max_completion_tokens`, `max_tokens`). See [Output cap](#output-cap) |
| `temperature`, `top_p`, `top_k` | Sampling |
| `frequency_penalty`, `presence_penalty`, `repetition_penalty` | Penalties (not on Anthropic) |
| `stop` | Stop sequences (Anthropic `stop_sequences`) |
| `seed` | Reproducibility (not on Anthropic) |
| `response_format` | Force JSON output (`with_json_response()`; not on Anthropic) |
| `reasoning` | Whether the model reasons, and how hard. See [Reasoning](#reasoning) |
| `tools`, `tool_choice`, `parallel_tool_calls` | Tools the model may call; each provider writes them in its own format (see [Tool Calling](./tool-calling.md)) |
| `extra` | Any other key, sent as written (OpenRouter routing goes here) |

On the Anthropic preset, a penalty, `seed` or `response_format` fails before the
request is sent, with `MiniLLMError::InvalidParameter`.

Every field starts unset. If neither you nor the generator's
`with_default_params` set a field, it is not sent and the provider's own default
applies. The library adds no other setting, except the cap under
[Output cap](#output-cap) and, when you track cost, the key that asks the
provider for usage.

If your parameters arrive as JSON (a settings object a user edits), they
deserialize straight into `CompletionParameters`: camelCase keys, every field
optional, unknown keys ignored. Reasoning is one word: `"off"`, `"on"`, or the
effort, so `{"maxTokens": 1024, "temperature": 0.2, "reasoning": "high"}` works
as it is.

### Output cap

`max_tokens` counts everything the model writes, its reasoning included. That
is how OpenAI, Anthropic and OpenRouter all count their cap, so your number goes
to each of them as it is.

If you leave it unset, nothing is sent and the provider's own default applies.
Anthropic's API refuses a request without a cap, so there the request carries
the model's largest output, or 4096 if that is unknown. The largest output is
looked up when a request needs it, and the answer is shared by your whole
program for an hour (see
[Cost Estimation](./cost-estimation.md#what-the-library-knows-about-a-model)).
With `with_timeout` set, the request waits at most half of it for the lookup
and then goes out with 4096, keeping the rest of the timeout for itself;
without, it can wait up to 30 seconds. If you want requests never to wait on a
lookup, set `max_tokens`, or give `max_output_tokens` with
`GeneratorInfo::with_model_info`.

### Reasoning

```rust,no_run
use minillmlib::{CompletionParameters, Reasoning};

let unset = CompletionParameters::new();                       // the API's default
let off = CompletionParameters::new().with_reasoning(Reasoning::Off);
let on = CompletionParameters::new().with_reasoning(Reasoning::On(None));
let high = CompletionParameters::new().with_reasoning(Reasoning::On(Some("high".into())));
```

Each provider sends its own spelling of what you asked:

| You set | OpenAI | OpenRouter | Anthropic |
|---|---|---|---|
| nothing | nothing | nothing | nothing |
| `Off` | `reasoning_effort: "none"` | `reasoning: {effort: "none"}` | `thinking: {type: "disabled"}` |
| `On(None)` | `reasoning_effort: "low"` | `reasoning: {effort: "minimal"}` | adaptive thinking, `output_config.effort: "low"` |
| `On(Some("high"))` | `reasoning_effort: "high"` | `reasoning: {effort: "high"}` | adaptive thinking, `output_config.effort: "high"` |

A custom server (`GeneratorInfo::custom`) gets OpenAI's spelling.

`On(None)` sends `low` on OpenAI and Anthropic (the lowest effort every model
there takes), and `minimal` on OpenRouter, which rounds it to the nearest level
the model has. If you want GPT-5's `minimal` on OpenAI, ask for it by name.

The effort word is sent exactly as you wrote it, so use the provider's own
words. Nothing is checked on the way: if the model cannot do what you asked, its
API refuses, and that refusal is the request's error. Some cases you will meet:

- `Off` on a model that cannot switch thinking off is refused.
- On a model that does not reason (`gpt-4o`), OpenAI refuses a reasoning
  setting, so leave `reasoning` unset there.
- `On` on Claude 4.5 and earlier is refused, since they think only within a
  token budget and adaptive thinking came later. Send the budget yourself
  through `extra`, and leave `reasoning` unset, since both write `thinking`:
  `with_extra("thinking", json!({"type": "enabled", "budget_tokens": 2048}))`.
- An effort a model does not have (`"xhigh"` on Claude Sonnet 4.6) is refused.

## NodeCompletionParameters

```rust,no_run
use minillmlib::{CompletionParameters, NodeCompletionParameters};

let params = NodeCompletionParameters::new()
    .with_params(CompletionParameters::new().with_max_tokens(200))
    .with_system_prompt("You are concise.")  // added when the thread has no system message
    .expecting_json()                         // parse + repair the response as JSON
    .with_force_prepend("Answer: ")           // make the model continue from this prefix
    .with_cost_tracking(true);                // request usage, so cost can be reported
```

| Builder | Meaning |
|---|---|
| `with_params(..)` | The wrapped `CompletionParameters` |
| `with_system_prompt(..)` | System message to add when the thread has none |
| `with_format_kwargs(..)` / `with_format_kwarg(k, v)` | Fill `{placeholder}`s thread-wide at call time |
| `with_parse_json(true)` / `expecting_json()` | Repair the response as JSON |
| `with_force_prepend(..)` | Prime the assistant turn so the model continues it |
| `with_cache(true)` | Auto-mark the whole prefix for caching (see [Caching](./caching.md)) |
| `with_cost_tracking(true)` | Ask the provider for token counts (stored in the reply's `usage` metadata). To get the cost, add `with_cost_callback` or call `complete_costed` |
| `with_cost_callback(..)` | Called with the cost of every billed request, a rejected one included. `complete_streaming` and the `_tracked` methods (which report to their `CompletionContext`) refuse it; if you stream, use `complete_streaming_collect` |
| `with_add_child(false)` | Return the reply without attaching it to the tree |
| `with_retry(n)`, `with_exp_back_off`, `with_back_off_time`, `with_max_back_off` | Retries after a 408, 429, 5xx, timeout, dropped connection, or a reply rejected by `with_crash_on_*`. Off by default (`retry` is 0); each retry is a new paid request |
| `with_crash_on_empty`, `with_crash_on_refusal` | Fail on an empty reply, or, with JSON parsing, on JSON that is empty or null; retried when retries are on |
| `with_timeout(secs)` | Total deadline (non-streaming) or idle timeout (streaming); a lookup for Anthropic's cap takes at most half of it |
