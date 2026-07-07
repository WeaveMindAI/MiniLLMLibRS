# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
