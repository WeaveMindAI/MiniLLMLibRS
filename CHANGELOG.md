# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
