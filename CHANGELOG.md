# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
