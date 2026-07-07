# MiniLLMLib-RS

A minimalist, async-first Rust library for LLM interactions with streaming support.

[![Crates.io](https://img.shields.io/crates/v/minillmlib.svg)](https://crates.io/crates/minillmlib)
[![Documentation](https://docs.rs/minillmlib/badge.svg)](https://docs.rs/minillmlib)
[![CI](https://github.com/WeaveMindAI/MiniLLMLibRS/actions/workflows/ci.yml/badge.svg)](https://github.com/WeaveMindAI/MiniLLMLibRS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Documentation

- **Guide** (tutorials, patterns, custom providers): <https://weavemindai.github.io/MiniLLMLibRS/>
- **API reference** (every type and method): <https://docs.rs/minillmlib>

## Features

- **Async-first**: Built on Tokio for high-performance async operations
- **Streaming Support**: First-class SSE streaming for real-time responses
- **Conversation Trees**: `ChatNode` provides tree-based conversation structure with branching
- **Tree Manipulation**: `detach()`, `merge()`, tree iterators (depth-first, breadth-first, leaves)
- **Template Substitution**: Format kwargs with `{placeholders}` in messages
- **Thread Serialization**: Save/load conversation threads to/from JSON files
- **Cost Tracking**: OpenRouter usage accounting with callbacks
- **Tool Calling**: Normalized `ToolDefinition`/`ToolChoice`/`ToolCall` types; each provider emits its own wire (OpenAI `tools`, Anthropic `tool_use`), streaming included
- **Multimodal**: Support for images and audio in messages
- **JSON Repair**: Robust handling of malformed JSON from LLM outputs
- **OpenRouter Compatible**: Works with OpenRouter, OpenAI, and any OpenAI-compatible API
- **Retry with Backoff**: Built-in exponential backoff and retry logic
- **Provider Routing**: OpenRouter provider settings (sort, ignore, data collection)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
minillmlib = "0.2"
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
    let generator = GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");

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
use minillmlib::{ChatNode, GeneratorInfo, ImageData, MessageContent};

let generator = GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");
let image = ImageData::from_file("./image.jpg")?;

let content = MessageContent::with_images("Describe this image.", &[image]);
let root = ChatNode::root("You are helpful.");
let user = root.add_user(content);

let response = user.complete(&generator, None).await?;
```

### Audio Input

```rust
use minillmlib::{AudioData, MessageContent};

let audio = AudioData::from_file("./audio.mp3")?;
let content = MessageContent::with_audio("Transcribe this audio.", &[audio]);
```

### Tool / Function Calling

Tools are normalized: define them once, and each provider emits its own wire
shape (OpenAI-wire `tools`/`tool_calls`, Anthropic `tool_use`/`tool_result`).
See the guide's Tool Calling chapter for the full loop.

```rust
use minillmlib::{CompletionParameters, NodeCompletionParameters, ToolChoice, ToolDefinition};

let params = NodeCompletionParameters::new().with_params(
    CompletionParameters::new()
        .with_tool(ToolDefinition::new(
            "get_weather",
            "Get the current weather for a city",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        ))
        .with_tool_choice(ToolChoice::Auto),
);

let node = user.complete(&generator, Some(&params)).await?;
if let Some(calls) = node.tool_calls() {
    let mut current = node.clone();
    for call in &calls {
        let args = call.arguments_json()?;                 // typed arguments
        let result = run_my_tool(&call.name, &args);       // your code
        current = current.add_tool_result(&call.id, result);
    }
    let answer = current.complete(&generator, Some(&params)).await?;
}
```

For a complete multi-turn agent loop (streaming prose live, one streaming tool
fed argument bytes as the model generates them, one buffered tool), run
`cargo run --example agent_loop` ([examples/agent_loop.rs](examples/agent_loop.rs)).

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

OpenRouter routing is provider-specific, so it's attached via
`with_openrouter_routing` (which carries it under the request's `provider` key);
non-OpenRouter providers simply ignore it.

```rust
use minillmlib::{CompletionParameters, ProviderSettings};

let routing = ProviderSettings::new()
    .sort_by_throughput()                              // or .sort_by_price()
    .deny_data_collection()
    .with_ignore(vec!["SambaNova".to_string()]);       // Exclude providers

let params = CompletionParameters::new()
    .with_openrouter_routing(routing);
```

### Prompt Caching (provider-agnostic)

Mark what to cache on the tree; the provider decides the wire (Anthropic emits
`cache_control`, OpenAI auto-caches). Switch the provider and the same code works.

```rust
let root = ChatNode::root(big_system_prompt);
root.cache_breakpoint();                 // cache just the system prompt
// ...or NodeCompletionParameters::new().with_cache(true) to cache the whole prefix

// Warm the cache before an agent run (cheap to call repeatedly):
let warm_cost = some_node.ensure_cached(&generator, None).await?;

// Clear marks:
root.clear_cache_breakpoint();           // one node
root.clear_all_cache_breakpoints();      // whole tree
```

Cache tokens are priced with distinct read/write rates (cache reads are ~0.1×
input; cache writes a ~1.25× premium):

```rust
let price = TokenPrice::new(1.0, 5.0)        // $/Mtok input, output
    .with_cache_rates(0.1, 1.25);            // $/Mtok cache-read, cache-write
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

### Template Substitution (Format Kwargs)

```rust
use minillmlib::ChatNode;

// Create a reusable prompt template
let root = ChatNode::root("You are {bot_name}, a {style} assistant.");
root.set_format_kwarg("bot_name", "Claude");
root.set_format_kwarg("style", "helpful");

let user = root.add_user("Hi {bot_name}!");

// Get formatted messages with placeholders replaced
let formatted = user.formatted_thread();
// Messages now contain "You are Claude, a helpful assistant." etc.
```

### Save and Load Conversation Threads

```rust
use minillmlib::ChatNode;

// Build a conversation
let root = ChatNode::root("You are helpful.");
root.set_format_kwarg("name", "Alice");
let user = root.add_user("Hello {name}!");
let assistant = user.add_assistant("Hi there!");

// Save to JSON file
assistant.save_thread("conversation.json")?;

// Load from JSON file (returns root and leaf)
let (loaded_root, loaded_leaf) = ChatNode::from_thread_file("conversation.json")?;

// Or load from JSON string
let json = r#"{"prompts": [{"role": "system", "content": "Hello"}], "required_kwargs": {}}"#;
let (root, leaf) = ChatNode::from_thread_json(json)?;
```

### Tree Manipulation

```rust
use minillmlib::ChatNode;

// Navigate to root from any node
let root = some_deep_node.get_root();

// Detach a subtree
let subtree = node.detach();  // node is now a new root

// Merge trees
let merged = tree1_leaf.merge(&tree2_leaf);  // tree2's root becomes child of tree1_leaf

// Iterate over tree
for node in root.iter_depth_first() {
    println!("{}", node.text().unwrap_or_default());
}

// Get all leaves
let leaves = root.iter_leaves();

// Count nodes
let count = root.node_count();
```

### Cost Tracking (OpenRouter)

```rust
use minillmlib::{ChatNode, GeneratorInfo, NodeCompletionParameters, CostInfo};
use std::sync::{Arc, Mutex};

let generator = GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");

// Track costs across multiple requests
let total_cost = Arc::new(Mutex::new(0.0));
let cost_tracker = total_cost.clone();

let params = NodeCompletionParameters::new()
    .with_cost_tracking(true)
    .with_cost_callback(move |info: CostInfo| {
        *cost_tracker.lock().unwrap() += info.cost;
        println!("Request cost: {} credits", info.cost);
        println!("Tokens: {} prompt, {} completion", 
            info.prompt_tokens, info.completion_tokens);
    });

let root = ChatNode::root("You are helpful.");
let user = root.add_user("Hello!");
let response = user.complete(&generator, Some(&params)).await?;

println!("Total spent: {} credits", *total_cost.lock().unwrap());
```

## API Reference

### Core Types

| Type | Description |
|------|-------------|
| `ChatNode` | A node in the conversation tree |
| `GeneratorInfo` | LLM provider configuration |
| `CompletionParameters` | Generation parameters (temperature, max_tokens, etc.) |
| `NodeCompletionParameters` | Per-request settings (retry, JSON parsing, cost tracking, etc.) |
| `Message` | A single message with role and content |
| `MessageContent` | Text or multimodal content |
| `ThreadData` | Serializable conversation thread with format kwargs |
| `CostInfo` | Cost and token usage information from completions |
| `CostResolution` | Whether a reported cost is `Resolved`, `Unpriced`, or `Unknown` |

### GeneratorInfo Methods

```rust
// Pre-configured providers
GeneratorInfo::openrouter(model)         // OpenRouter (OpenAI wire, native USD cost)
GeneratorInfo::openai(model)             // OpenAI (token-only; price via with_token_price)
GeneratorInfo::anthropic(model)          // Native Anthropic /v1/messages, x-api-key auth
GeneratorInfo::claude_subscription(model)// Anthropic wire, Claude Pro/Max OAuth token
GeneratorInfo::custom(name, url, model)  // Custom OpenAI-compatible endpoint

// Auth builder methods
.with_api_key(key)                       // provider chooses header (Bearer / x-api-key)
.with_api_key_from_env("ENV_VAR")
.with_bearer_token(token)                // OAuth / subscription bearer token
.with_bearer_token_from_env("ENV_VAR")

// Other builder methods
.with_token_price(TokenPrice::new(in_per_mtok, out_per_mtok)) // cost estimate for token-only providers
.with_provider(Arc::new(MyProvider))     // swap the wire dialect
.with_header(name, value)
.with_vision()
.with_audio()
.with_max_context(length)
.with_default_params(params)
```

### Claude Subscription (use your Pro/Max plan)

A Claude **Pro/Max subscription** OAuth token authenticates against the native
Anthropic API the same way an API key does, but draws on your **subscription's
rolling quota** (the 5-hour / 7-day window) instead of pay-as-you-go API billing.

`claude_subscription` resolves the token in this order:

1. the `ANTHROPIC_AUTH_TOKEN` env var, if set (explicit override, e.g. from
   `claude setup-token`; you keep it fresh);
2. otherwise the live Claude Code credential at `~/.claude/.credentials.json`
   (`claudeAiOauth.accessToken`), which Claude Code keeps refreshed, so if you're
   logged into Claude Code with your subscription, it just works.

```rust
use minillmlib::{ChatNode, GeneratorInfo, TokenPrice};

// Anthropic returns token counts but no dollar cost, so set a price for a
// resolved cost ESTIMATE (otherwise tracking reports `Unpriced`).
let generator = GeneratorInfo::claude_subscription("claude-haiku-4-5")
    .with_token_price(TokenPrice::new(1.0, 5.0)); // $/Mtok in, $/Mtok out

let root = ChatNode::root("You are helpful.");
let response = root.chat("Hello!", &generator).await?;
```

> **Subscription vs Console.** A subscription token (from Claude Code) bills your
> Pro/Max plan. A Console/API OAuth token (e.g. from the `ant` CLI) bills your
> **API account**, not the subscription; for Console use an API key via
> `GeneratorInfo::anthropic(model)`. Verify which bucket you're hitting by the
> response's rate-limit headers: subscription returns `anthropic-ratelimit-unified-5h-*`;
> the API tier returns `anthropic-ratelimit-input-tokens-limit`.

### CompletionParameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `Option<u32>` | `4096` | Maximum tokens to generate |
| `temperature` | `Option<f32>` | `0.7` | Sampling temperature |
| `top_p` | `Option<f32>` | `None` | Nucleus sampling |
| `top_k` | `Option<u32>` | `None` | Top-k sampling |
| `stop` | `Option<Vec<String>>` | `None` | Stop sequences |
| `seed` | `Option<u64>` | `None` | Random seed |
| `response_format` | `Option<ResponseFormat>` | `None` | Force JSON output |
| `reasoning` | `Option<ReasoningConfig>` | `None` | Extended-thinking effort/budget |
| `extra` | `Option<HashMap>` | `None` | Provider-specific keys (incl. OpenRouter routing) |

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
| `track_cost` | `bool` | `false` | Request and report usage/cost |
| `token_price` | `Option<TokenPrice>` | `None` | Per-request price override (token-only providers) |
| `cost_callback` | `Option<CostCallback>` | `None` | Callback for cost info |

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
# Default: all offline tests (unit + offline integration). No API calls, free.
cargo test

# Unit tests only (fast)
cargo test --lib

# Live integration tests (REAL, billed API calls): opt in with the `live` feature.
# Reads OPENROUTER_API_KEY, ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN from the env
# (or a .env); each live test skips gracefully if its key is absent.
cargo test --features live

# Run with output
cargo test -- --nocapture
```

Without `--features live`, every network test skips, so `cargo test` is free,
offline, and deterministic even when real keys are present in your environment.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
