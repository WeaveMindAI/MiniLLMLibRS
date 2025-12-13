# MiniLLMLib-RS

A minimalist, async-first Rust library for LLM interactions with streaming support.

[![Crates.io](https://img.shields.io/crates/v/minillmlib.svg)](https://crates.io/crates/minillmlib)
[![Documentation](https://docs.rs/minillmlib/badge.svg)](https://docs.rs/minillmlib)
[![CI](https://github.com/qfeuilla/MiniLLMLibRS/actions/workflows/ci.yml/badge.svg)](https://github.com/qfeuilla/MiniLLMLibRS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Async-first**: Built on Tokio for high-performance async operations
- **Streaming Support**: First-class SSE streaming for real-time responses
- **Conversation Trees**: `ChatNode` provides tree-based conversation structure with branching
- **Multimodal**: Support for images and audio in messages
- **JSON Repair**: Robust handling of malformed JSON from LLM outputs
- **OpenRouter Compatible**: Works with OpenRouter, OpenAI, and any OpenAI-compatible API
- **Retry with Backoff**: Built-in exponential backoff and retry logic
- **Provider Routing**: OpenRouter provider settings (sort, ignore, data collection)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
minillmlib = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Quick Start

```rust
use minillmlib::{ChatNode, GeneratorInfo};

#[tokio::main]
async fn main() -> minillmlib::Result<()> {
    // Load .env and configure logging
    minillmlib::init();

    // Create a generator for OpenRouter
    let generator = GeneratorInfo::openrouter("google/gemini-2.0-flash-lite-001");

    // Start a conversation
    let root = ChatNode::root("You are a helpful assistant.");
    let response = root.chat("Hello!", &generator).await?;

    println!("Assistant: {}", response.text().unwrap_or_default());
    Ok(())
}
```

## Environment Variables

Set your API key in a `.env` file or environment:

```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
# Or for direct OpenAI:
OPENAI_API_KEY=sk-your-key-here
```

## Usage Examples

### Basic Completion

```rust
use minillmlib::{ChatNode, GeneratorInfo, CompletionParameters, NodeCompletionParameters};

let generator = GeneratorInfo::openrouter("anthropic/claude-3.5-sonnet");
let root = ChatNode::root("You are helpful.");
let user = root.add_user("What is 2+2?");

// With custom parameters
let params = NodeCompletionParameters::new()
    .with_params(
        CompletionParameters::new()
            .with_temperature(0.0)
            .with_max_tokens(100)
    );

let response = user.complete(&generator, Some(&params)).await?;
println!("{}", response.text().unwrap());
```

### Streaming

```rust
let root = ChatNode::root("You are helpful.");
let user = root.add_user("Tell me a story.");

let mut stream = user.complete_streaming(&generator, None).await?;

while let Some(chunk) = stream.next_chunk().await {
    print!("{}", chunk?.delta);
}
```

### Multi-turn Conversation

```rust
let root = ChatNode::root("You are helpful.");

// First turn
let response1 = root.chat("My name is Alice.", &generator).await?;

// Second turn - context is preserved
let response2 = response1.chat("What's my name?", &generator).await?;
// Response will mention "Alice"
```

### Image Input

```rust
use minillmlib::{ChatNode, GeneratorInfo, ImageData, MessageContent, Message, Role};

let generator = GeneratorInfo::openrouter("google/gemini-2.0-flash-lite-001");
let image = ImageData::from_file("./image.jpg")?;

let content = MessageContent::with_images("Describe this image.", &[image]);
let root = ChatNode::root("You are helpful.");
let user = root.add_child(ChatNode::new(Message {
    role: Role::User,
    content,
    name: None,
    tool_call_id: None,
    tool_calls: None,
}));

let response = user.complete(&generator, None).await?;
```

### Audio Input

```rust
use minillmlib::{AudioData, MessageContent};

let audio = AudioData::from_file("./audio.mp3")?;
let content = MessageContent::with_audio("Transcribe this audio.", &[audio]);
```

### JSON Response with Repair

```rust
let params = NodeCompletionParameters::new()
    .with_parse_json(true)           // Enable JSON repair
    .with_crash_on_refusal(true)     // Retry if no valid JSON
    .with_retry(3);                  // Number of retries

let response = user.complete(&generator, Some(&params)).await?;
// response.text() will contain valid, repaired JSON
```

### Retry with Exponential Backoff

```rust
let params = NodeCompletionParameters::new()
    .with_retry(5)
    .with_exp_back_off(true)
    .with_back_off_time(1.0)    // Start with 1 second
    .with_max_back_off(30.0)    // Max 30 seconds
    .with_crash_on_empty(true); // Retry on empty responses
```

### Force Prepend (Constrained Generation)

```rust
// Force the model to start its response with specific text
let params = NodeCompletionParameters::new()
    .with_force_prepend("Score: ");

// Response will start with "Score: " followed by the model's completion
```

### OpenRouter Provider Settings

```rust
use minillmlib::{CompletionParameters, ProviderSettings};

let provider = ProviderSettings::new()
    .sort_by_throughput()                              // or .sort_by_price()
    .deny_data_collection()
    .with_ignore(vec!["SambaNova".to_string()]);       // Exclude providers

let params = CompletionParameters::new()
    .with_provider(provider);
```

### Custom/Extra Parameters

```rust
// Pass arbitrary parameters to the API
let params = CompletionParameters::new()
    .with_extra("custom_param", serde_json::json!(42))
    .with_extra("another", serde_json::json!({"nested": "value"}));
```

### Pretty Print Conversations

```rust
use minillmlib::{pretty_messages, format_conversation, PrettyPrintConfig};

let root = ChatNode::root("You are helpful.");
let user = root.add_user("Hello");
let assistant = user.add_assistant("Hi there!");

// Default formatting
let pretty = format_conversation(&assistant);
// Output: "SYSTEM: You are helpful.\n\nUSER: Hello\n\nASSISTANT: Hi there!"

// Custom formatting
let config = PrettyPrintConfig::new("[SYS] ", "\n[USR] ", "\n[AST] ");
let pretty = pretty_messages(&assistant, Some(&config));
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `ChatNode` | A node in the conversation tree |
| `GeneratorInfo` | LLM provider configuration |
| `CompletionParameters` | Generation parameters (temperature, max_tokens, etc.) |
| `NodeCompletionParameters` | Per-request settings (retry, JSON parsing, etc.) |
| `Message` | A single message with role and content |
| `MessageContent` | Text or multimodal content |

### GeneratorInfo Methods

```rust
// Pre-configured providers
GeneratorInfo::openrouter(model)    // OpenRouter API
GeneratorInfo::openai(model)        // OpenAI API
GeneratorInfo::anthropic(model)     // Anthropic API
GeneratorInfo::custom(name, url, model)  // Custom endpoint

// Builder methods
.with_api_key(key)
.with_api_key_from_env("ENV_VAR")
.with_header(name, value)
.with_vision()
.with_audio()
.with_max_context(length)
.with_default_params(params)
```

### CompletionParameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `Option<u32>` | `4096` | Maximum tokens to generate |
| `temperature` | `Option<f32>` | `0.7` | Sampling temperature |
| `top_p` | `Option<f32>` | `None` | Nucleus sampling |
| `top_k` | `Option<u32>` | `None` | Top-k sampling |
| `stop` | `Option<Vec<String>>` | `None` | Stop sequences |
| `seed` | `Option<u64>` | `None` | Random seed |
| `provider` | `Option<ProviderSettings>` | `None` | OpenRouter provider routing |
| `extra` | `Option<HashMap>` | `None` | Custom parameters |

### NodeCompletionParameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | `Option<String>` | `None` | Override system prompt |
| `parse_json` | `bool` | `false` | Parse/repair JSON response |
| `force_prepend` | `Option<String>` | `None` | Force response prefix |
| `retry` | `u32` | `4` | Retry attempts |
| `exp_back_off` | `bool` | `false` | Exponential backoff |
| `back_off_time` | `f64` | `1.0` | Initial backoff (seconds) |
| `max_back_off` | `f64` | `15.0` | Max backoff (seconds) |
| `crash_on_refusal` | `bool` | `false` | Error if no JSON |
| `crash_on_empty_response` | `bool` | `false` | Error if empty |

### ProviderSettings (OpenRouter)

| Parameter | Description |
|-----------|-------------|
| `order` | Ordered list of providers to try |
| `sort` | Sort by: "price", "throughput", "latency" |
| `ignore` | Providers to exclude |
| `data_collection` | "allow" or "deny" |
| `allow_fallbacks` | Allow fallback providers |

## CLI Tool

The library includes a CLI for JSON repair:

```bash
# Repair JSON from file
minillmlib-cli input.json

# Repair JSON from stdin
echo '{"key": "value",}' | minillmlib-cli
```

## Running Tests

```bash
# Run all tests (unit + integration)
cargo test

# Run only unit tests (fast, no API calls)
cargo test --lib

# Run integration tests (requires API key)
cargo test --test integration_tests

# Run with output
cargo test -- --nocapture
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
