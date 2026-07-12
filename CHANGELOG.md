# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.4] - 2026-07-11

### Added

- `GeneratorInfo::with_base_url`: point a generator at a different address (a
  gateway, a proxy, a self-hosted endpoint). EVERY request the crate makes
  for that generator goes there.

### Fixed

- The OpenRouter cost lookup (`/generation`) hardcoded `openrouter.ai`, so a
  generator pointed at a gateway resolved its costs past the gateway (and
  with a credential that gateway had not substituted). It now uses the
  generator's own address, like every other request.
- `PostStreamCtx` (the context a provider gets for an out-of-band cost query)
  carries `base_url` instead of assuming the provider's host. Affects only
  code implementing the `Provider` trait itself.

## [0.5.3] - 2026-07-11

### Added

- `TrackedStream::collect_or_cancel(interrupt)`: drain until the stream
  finishes or the given future fires, with the cost reported on every
  outcome (`CollectOutcome::{Finished, Interrupted, Failed}`): a finished
  stream books from usage, an interrupted one is cancelled and resolves the
  actual out-of-band, a transport-errored one books nothing. The one-call
  shape for consumers with a kill switch; hand-rolling the race and
  forgetting `cancel().await` would silently lose the interrupted call's
  cost.

## [0.5.2] - 2026-07-11

### Fixed

- `/generation` record resolution booked $0 for BYOK routes: the record's
  `total_cost` is OpenRouter's credits charge only, and on BYOK it is 0 with
  the real upstream charge (billed on the user's own provider key) in
  `upstream_inference_cost`. The parsed cost is now their sum. The parse is
  extracted into a pure function pinned by tests built from live records.
- The out-of-band `/generation` resolver gave up before records become
  readable. Measured live: a completed generation's record appears ~9s after
  it finishes, and a CANCELLED call's only after the upstream generation runs
  to its own end (client aborts do not stop these routes) plus the same ~9s.
  The resolver now polls every second for 25s (the endpoint is free; backoff
  only added latency).

### Added

- `TrackedStream::id()`: the provider's generation/response id (empty until
  the first chunk), so callers can correlate a stream with the provider's
  ledger.

## [0.5.1] - 2026-07-10

### Added

- `ChatNode::complete_costed(&generator, params)`: complete and get the
  [`CostInfo`] back WITH the result, no `CompletionContext`/callback ceremony.
  Same accounting as `complete_tracked` (usage from the response, the
  provider's out-of-band resolution as backstop, never a fake $0); an errored
  completion carries no cost info.
- `CompletionParameters` (and `ResponseFormat`) are now serde types: camelCase
  keys, every field optional (missing ones take the defaults), unknown keys
  ignored, so a flat JSON settings object deserializes directly.

## [0.5.0] - 2026-07-10

### Added

- **Pre-send cost estimation** (opt-in feature `estimate`). Ask a
  `GeneratorInfo` what a completion will cost BEFORE sending it:
  `generator.estimate_cost_usd(&messages, &params).await` returns a
  deliberately high USD figure to reserve against (assumes no caching, the
  largest completion the request permits including any reasoning budget, and a
  one-minute clip for media of unstated length). Text is counted by a built-in
  `o200k_base` BPE tokenizer (no external tokenizer dependency; 1.6 MB
  embedded vocabulary) corrected by a measured safety multiplier; images,
  video frames, and audio seconds are priced at each model's published media
  rates. A prompt larger than the model accepts is priced as the largest input
  it does accept, so an estimate always exists; the one error is a model the
  catalog does not know.
- **Published price lookup on the generator** (no feature needed):
  `generator.model_rates()` fetches OpenRouter's per-provider price sheet (the
  only public one; it lists first-party vendors at their own standard rates,
  so it also prices direct OpenAI/Anthropic calls). A price is a property of
  (model, provider): rates are the serving provider's own when known
  (`Provider::openrouter_slug`), else the dearest rate any provider of the
  model charges, the only ceiling that holds wherever routing lands. Prices
  are cached on the generator for an hour and clones share the cache; pool
  long-lived generators keyed by `generator.pricing_key()`.
  `generator.model_rates_served_by(Some(slug))` prices a per-request routing
  pin (see `ProviderSettings::billing_provider()`).
- `GeneratorInfo::with_openrouter_name` sets the model's OpenRouter catalog id
  when the generator's own model id differs (e.g. Anthropic's dated ids); this
  is what unlocks estimation for direct-vendor generators.
- `AudioData::with_duration` / `VideoData::with_duration` declare a clip's
  length so media estimates stop guessing. The duration is estimation
  metadata: saved conversation trees keep it, and it is stripped from request
  payloads so provider schemas never see an unknown key.
- `TokenPrice` gained `audio_per_mtok` / `image_per_mtok` with honest
  fallbacks (an unpublished image rate bills as text; an unpublished audio
  rate at a measured premium multiple).
- Opt-in `live` feature gates the network/billed integration tests.

### Changed

- **Breaking:** `GeneratorInfo` gained a private field (its price cache), so
  struct-literal construction outside the crate no longer compiles; use the
  constructors and builders.
- **Breaking:** `ProviderSettings::billing_provider()` now returns
  `Option<String>` (the pinned provider slug, or `None` when routing is free).
- `ModelRates` is re-exported from the crate root and now lives beside the
  generator rather than the provider internals.

## [0.4.6] - 2026-07-07

### Security

- Refreshed `Cargo.lock` to pull transitively-depended crates to patched
  versions, clearing all open advisories: `openssl` 0.10.75 -> 0.10.81
  (GHSA-xp3w-r5p5-63rr and others), `rustls-webpki` 0.103.8 -> 0.103.13
  (GHSA-82j2-j2ch-gfr8), and `bytes` 1.11.0 -> 1.12.0 (GHSA-434x-w66g-qw3r),
  plus routine bumps across the rest of the locked graph. No direct dependency
  version ranges changed; the manifest floors stay loose for downstream
  compatibility. `reqwest` is held at 0.12 (its 0.13 line has no compatible
  `reqwest-eventsource` release yet).
- **MSRV raised to 1.86** (was 1.83). The refreshed graph pulls `indexmap`
  2.14 (edition 2024, needs 1.85) and the `icu_*` crates 2.2 (need 1.86), so
  building on the latest transitive versions requires 1.86. Holding those
  crates back to keep an older MSRV would reintroduce the stale-pin problem
  the dep refresh was meant to remove.

## [0.4.5] - 2026-07-07

### Added

- **Prompt caching now works through OpenRouter for Claude models.** A
  `Message::cache_breakpoint` was only translated on the native Anthropic wire;
  OpenRouter (an OpenAI-dialect wire) silently dropped every mark, so the
  documented "switch the provider and the same code works" caching story did not
  actually hold there. The `OpenRouterProvider` now renders breakpoints as
  Anthropic-style `cache_control` blocks that OpenRouter passes through, gated to
  Claude models (other backends OpenRouter fronts either auto-cache or would lose
  routing candidates to the supporting-endpoints-only filter). Consumer code is
  unchanged: the same `cache_breakpoint()` / `with_cache(true)` marks now take
  effect on Anthropic, OpenRouter-Claude, and OpenAI alike.
- **`Provider::max_cache_breakpoints`**: the per-request breakpoint cap is now a
  provider property (default unlimited; Anthropic and OpenRouter override it to
  4) instead of a hardcoded constant, so the "keep the last N marks" logic is
  driven by each wire's own limit.

### Changed

- **`Provider::openai_messages_value`**: a new dialect hook (alongside
  `openai_tools_value`, `openai_request_usage`, ...) letting an OpenAI-envelope
  provider customize how the `messages` array is serialized. Defaults to the
  plain payload; OpenRouter overrides it to emit `cache_control`. Providers with
  their own request builder (Anthropic) are unaffected.

## [0.4.4] - 2026-07-07

### Added

- **`ArgumentStream` + `FieldHandle`: per-field live decoding of streamed tool
  arguments.** Every field of a call is a uniform handle: consume it with
  `wait().await` (the complete parsed value) or `delta().await` (the DECODED
  text chunk by chunk as the model generates it, escapes undone), your choice
  per field; any number of fields can stream (a `patch(old_code, new_code)`
  tool works). Fields without a handle are parsed into `fields()` as they
  complete, so non-streaming consumers still get everything extracted.
  Fragments may split at arbitrary positions (mid-escape included) without
  changing the output.
  - **Lenient mode** (opt-in) covers every top-level string value with a
    deterministic rule: an unescaped `"` closes a string only when followed by
    a `, "key":` field boundary (key parsed as a full JSON string, whitespace
    optional) or by `}` at the true end of the call; everything else is
    literal content, and a stream that just stops still delivers what arrived.
    On `finish()`, lenient mode additionally repair-parses the raw arguments
    (via the crate's JSON repair) and fills anything the incremental parse
    missed into `fields()`.
- **`ToolCall::arguments_json_repaired()`**: the non-streaming counterpart,
  parsing a completed call's arguments through the crate's JSON repair
  (trailing commas, unclosed braces, single quotes).

### Changed

- **BREAKING:** `PayloadExtractor` (introduced in 0.4.3) is replaced by
  `ArgumentStream`: the "one designated payload field" concept is gone;
  streaming vs buffering is now the consumer's per-field choice.

## [0.4.3] - 2026-07-07

### Added

- **`PayloadExtractor`: live decoded extraction of a single-string tool
  argument.** For the `{"content": "<big payload>"}` pattern, it turns the
  streaming argument fragments into the payload's decoded text as the model
  generates it (JSON escapes undone), fed fragments split at arbitrary
  positions (mid-escape included). Strict mode fails loudly on malformed
  input; opt-in lenient mode deterministically tolerates sloppy escaping
  (an unescaped `"` not at the true end is literal, raw newlines are
  themselves, `\` before a non-escape is a literal backslash, a stream that
  just stops still yields the full payload). Used by
  `examples/agent_loop.rs`; documented in the Tool Calling guide chapter.

## [0.4.2] - 2026-07-07

### Added

- **Normalized tool / function calling** across every provider (`src/tools.rs`):
  - `ToolDefinition` (name, description, JSON-Schema parameters, `strict`),
    `ToolChoice` (`Auto`/`None`/`Required`/`Tool(name)`), and
    `CompletionParameters::{with_tools, with_tool, with_tool_choice,
    with_parallel_tool_calls}`. Each provider emits its own wire: OpenAI-wire
    `tools`/`tool_choice`/`parallel_tool_calls`; Anthropic `input_schema` tools,
    `{"type":"auto"|"none"|"any"|"tool"}` choice with
    `disable_parallel_tool_use` folded in.
  - Typed `ToolCall { id, name, arguments }` (arguments as raw JSON text, with
    `arguments_json()` parsing loudly) replaces the raw `serde_json::Value`
    tool-call passthrough in `Message`, `CompletionResponse`, and node trees.
  - Streaming tool calls: typed `ToolCallDelta` fragments assembled by
    `ToolCallAccumulator` (sparse-index safe). Anthropic streaming now parses
    `content_block_start` (tool_use) and `input_json_delta` events; previously
    streamed Anthropic tool calls were silently dropped.
  - The Anthropic provider now translates the full loop: assistant `tool_calls`
    become `tool_use` blocks, `Role::Tool` messages become `tool_result` blocks
    (consecutive user-side messages merge into one alternating-role turn, so
    parallel results are grouped as the wire requires).
  - `ChatNode::tool_calls()` and `ChatNode::add_tool_result(call_id, content)`
    complete the agent loop on the tree API; `ChatNode::append_response(..)`
    appends the assistant node after a hand-driven streaming loop (e.g.
    piping tool-argument fragments into a tool as the model generates them).
  - Custom OpenAI-envelope providers can override just the tool wire via new
    `Provider::openai_tools_value` / `openai_tool_choice_value` hooks.
  - Docs: a new "Tool Calling" guide chapter; live round-trip + streaming
    integration tests for both wires; a runnable full agent loop
    (`examples/agent_loop.rs`) mixing a streaming tool (fed argument bytes as
    the model generates them) with a buffered one.

### Changed

- **BREAKING:** `CompletionParameters::tools` is now `Option<Vec<ToolDefinition>>`
  and `tool_choice` is `Option<ToolChoice>` (previously raw `serde_json::Value`
  passthrough). `Message::tool_calls`, `CompletionResponse::tool_calls`, and
  `StreamChunk::tool_calls` are typed likewise.

## [0.4.1] - 2026-06-16

### Changed

- Moved the repository to the `WeaveMindAI` organization and updated the
  `repository` / `homepage` / CI-badge / docs links accordingly. No code changes.

## [0.4.0] - 2026-06-16

A breaking release that generalizes the provider layer so one library API drives
every provider identically, adds native Anthropic + Claude subscription support,
provider-agnostic prompt caching, and honest per-provider cost accounting.

### Added

- **Provider trait owns the full wire dialect.** A `Provider` now owns the
  endpoint, auth headers, request body, response envelope, streaming chunks, usage
  parsing, and cost accounting. The rest of the crate deals only in normalized
  types, so switching providers is a one-line change.
  - Ships `OpenAiProvider`, `OpenRouterProvider`, `GenericProvider` (OpenAI-compatible
    default for self-hosted servers, with a `legacy_token_limit` switch), and
    `AnthropicProvider` (native `/v1/messages`, `content[]` envelope).
  - A custom/enterprise API is a small hand-written `impl Provider`.
- **`Auth` strategy on `GeneratorInfo`** (`ApiKey` / `BearerToken` / `None`), mapped
  to concrete headers by the provider. New builders: `with_api_key`,
  `with_api_key_from_env`, `with_bearer_token`, `with_bearer_token_from_env`,
  `with_auth`, `with_provider`, `with_token_price`, `with_app_attribution`.
- **Native Anthropic provider**: `GeneratorInfo::anthropic(model)` over
  `/v1/messages` with `x-api-key` auth.
- **Claude subscription**: `GeneratorInfo::claude_subscription(model)` uses a Pro/Max
  OAuth token (env `ANTHROPIC_AUTH_TOKEN`, else the live Claude Code credential) so
  usage draws on the subscription quota; cost is a token-count estimate via
  `TokenPrice`. New `resolve_claude_subscription_auth()`.
- **Provider-agnostic prompt caching.** Mark breakpoints on the tree
  (`ChatNode::cache_breakpoint`, `clear_cache_breakpoint`, `clear_all_cache_breakpoints`),
  or auto-mark the prefix per request (`NodeCompletionParameters::with_cache`).
  Anthropic enforces `cache_control` (4-breakpoint cap); OpenAI/OpenRouter auto-cache.
  `ChatNode::ensure_cached` warms the cache and returns its cost.
- **Honest cost accounting.** `Usage` split into disjoint `uncached_input_tokens` /
  `cache_read_tokens` / `cache_write_tokens` buckets; `TokenPrice` with distinct
  cache read/write rates (`with_cache_rates`); `CostResolution`
  (`Resolved` / `Unpriced` / `Unknown`) so a cost is never silently reported as a
  fake `$0`. In-band streaming provider errors now surface loudly instead of
  booking a phantom cost.
- **Live integration tests** behind a `live` Cargo feature (off by default, so
  `cargo test` is free and offline). Mock-server contract tests for the custom /
  self-hosted provider path (OpenAI-compatible and a non-OpenAI wire).
- **Documentation site**: an mdBook guide (`docs/src/`) deployed to GitHub Pages,
  alongside the auto-generated API reference on docs.rs.

### Changed

- **BREAKING:** `ChatNode` is now a cheap, cloneable handle into a shared arena
  (was `Arc<ChatNode>` with `Weak` parents). Methods take `&self`; holding any
  handle keeps its tree alive.
- **BREAKING:** `GeneratorInfo` replaces `api_key` / `organization_id` with the
  `auth: Auth` field and a `provider`.
- **BREAKING:** `CompletionParameters` is normalized intent, not a wire shape (no
  longer (de)serialized directly); the `provider` and `stream` fields are gone
  (OpenRouter routing now goes through `with_openrouter_routing`).
- **BREAKING:** `NodeCompletionParameters` replaces `cost_tracking: CostTrackingType`
  with `track_cost: bool` (+ `token_price`); `with_openrouter_cost_tracking()` is
  removed in favor of `with_cost_tracking(true)`.
- **BREAKING:** `CostInfo` replaces `cached_tokens` with `cache_read_tokens` /
  `cache_write_tokens` and adds `resolution`.
- **BREAKING:** `MediaData` uses an explicit `is_url` flag instead of a magic
  `format == "url"` sentinel; `from_file` fails loudly when the format can't be
  determined.
- Streaming uses an idle timeout (max silence between chunks) rather than a total
  deadline, so a long live generation isn't killed but a dead connection fails fast.

### Removed

- **BREAKING:** `CostTrackingType` enum, `LLMClient::with_timeout`,
  `validate_json_response`, `configure_logging_with_filter`, and the
  `MissingConfig` / `Url` / `NodeNotFound` / `Other` error variants.

### Fixed

- OpenAI/OpenRouter cache-write tokens are read from the correct field
  (`prompt_tokens_details.cache_write_tokens`) and treated as additive (not
  subtracted from `prompt_tokens`), fixing cost underestimation on cache-heavy
  requests.
- JSON repair: depth guard against stack overflow on pathological input, correct
  duplicate-key collapsing, float-overflow handling, smart-quote delimiters.

## [0.3.0] - 2026-02-12

### Added

- **CompletionContext**: Enforced cost tracking wrapper for LLM completions. Wraps a `GeneratorInfo` and guarantees every completion reports cost via an async callback. This is the mechanism WeaveMind uses to track AI usage costs.
  - `CompletionContext::new()`, `report_cost()`, `is_byok()`
  - `CompletionMeta` struct with userId, workflowId, executionId, nodeId, isByok
  - `AsyncCostCallback` type for async cost ingestion (database writes, HTTP, etc.)
- **TrackedStream**: Streaming completion wrapper that automatically reports cost when the stream finishes or is cancelled (dropped).
  - `next_chunk()`, `collect_and_report()`, `accumulated()`, `is_finished()`
  - Drop impl spawns background cost query for cancelled streams
- **Tracked completion methods on ChatNode**:
  - `complete_tracked()`: non-streaming with enforced cost reporting
  - `complete_streaming_tracked()`: returns a `TrackedStream`
  - `complete_streaming_collect_tracked()`: streaming collect with cost reporting
- **OpenRouter generation cost fallback**: When usage data is missing (cancelled streams, some providers), queries OpenRouter's `/api/v1/generation` endpoint with retry backoff

### Changed

- `CompletionMeta` derives `Serialize` and `Deserialize` for downstream use
- Drop-based cost reporting uses `Handle::try_current()` guard (no panic if dropped outside tokio runtime)
- Generation cost query URL-encodes the generation ID parameter
- Cancelled stream cost query retries 3 times (1s, 2s, 4s backoff) instead of a single 2s wait

## [0.2.0] - 2025-12-14

### Added

- **Tree Navigation**: `get_root()` to navigate to root from any node
- **Tree Manipulation**: `detach()` to remove a node from its parent, `merge()` to combine trees
- **Tree Iteration**: `iter_depth_first()`, `iter_breadth_first()`, `iter_leaves()`, `node_count()`
- **Format Kwargs**: Template substitution with `{placeholders}` in message content
  - `set_format_kwarg()`, `get_format_kwarg()`, `formatted_thread()`
  - Supports null placeholders in JSON (filled at runtime)
- **Thread Serialization**: Save and load conversation threads to/from JSON
  - `save_thread()`, `from_thread_file()`, `from_thread_json()`, `from_messages()`
  - `ThreadData` and `ThreadMessage` structs for serialization
- **Cost Tracking**: OpenRouter usage accounting with callbacks
  - `CostInfo`, `CostTrackingType`, `CostCallback` types
  - `with_openrouter_cost_tracking()`, `with_cost_callback()` on `NodeCompletionParameters`
  - Works with both streaming and non-streaming completions
- **Role Helper**: `Role::as_str()` method for string conversion

### Changed

- `Usage` struct now includes `cost`, `cached_tokens`, and `reasoning_tokens` fields
- `pretty_messages()` and `format_conversation()` now apply format_kwargs
- Streaming now waits for usage chunk after finish_reason (OpenRouter sends usage last)

### Fixed

- Streaming completions now correctly receive usage data from OpenRouter

## [0.1.1] - 2025-12-13

### Fixed
- Fixed rustdoc warnings for bare URLs
- Fixed clippy warning: use `is_none_or` instead of `map_or`
- Updated MSRV to 1.83 (required by `icu_properties_data` dependency)

## [0.1.0] - 2025-12-13

### Added

- Initial release
- **ChatNode**: Tree-based conversation structure with branching support
- **Streaming**: SSE-based streaming completions via `reqwest-eventsource`
- **Multimodal**: Support for images (`ImageData`) and audio (`AudioData`)
- **JSON Repair**: Robust JSON repair for malformed LLM outputs
- **Retry Logic**: Exponential backoff with configurable retry attempts
- **Provider Settings**: OpenRouter provider routing (sort, ignore, data_collection)
- **Force Prepend**: Constrained generation with forced response prefixes
- **Pretty Print**: Conversation formatting utilities
- **Pre-configured Providers**: OpenRouter, OpenAI, Anthropic presets
- **Custom Parameters**: Pass arbitrary extra parameters to APIs
- **CLI Tool**: JSON repair command-line utility

### Supported Providers

- OpenRouter (primary target)
- OpenAI
- Any OpenAI-compatible API

### Dependencies

- `tokio` for async runtime
- `reqwest` for HTTP client
- `reqwest-eventsource` for SSE streaming
- `serde` / `serde_json` for serialization
- `tracing` for logging
