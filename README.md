# MiniLLMLib-RS

A small async Rust library for calling LLMs, with the same code whatever the provider.

[![Crates.io](https://img.shields.io/crates/v/minillmlib.svg)](https://crates.io/crates/minillmlib)
[![Documentation](https://docs.rs/minillmlib/badge.svg)](https://docs.rs/minillmlib)
[![CI](https://github.com/WeaveMindAI/MiniLLMLibRS/actions/workflows/ci.yml/badge.svg)](https://github.com/WeaveMindAI/MiniLLMLibRS/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Documentation

- **Guide** (tutorials, patterns, custom providers): <https://weavemindai.github.io/MiniLLMLibRS/>
- **API reference** (every type and method): <https://docs.rs/minillmlib>

## Features

- **Async**: built on Tokio
- **Streaming**: replies arrive token by token over SSE
- **Conversation Trees**: `ChatNode` provides tree-based conversation structure with branching
- **Tree Manipulation**: `detach()`, `merge()`, tree iterators (depth-first, breadth-first, leaves)
- **Template Substitution**: Format kwargs with `{placeholders}` in messages
- **Thread Serialization**: Save/load conversation threads to/from JSON files
- **Reasoning**: off, on, or on at an effort you name, written in each provider's own spelling
- **Only what you set is sent**: an unset parameter stays off the wire, except the output cap Anthropic requires
- **Cost Tracking**: what every call cost, through callbacks or returned with the reply, for every provider
- **Cost Estimation**: what a call will cost *before* it is sent, so you can decide whether to allow it (opt-in `estimate` feature)
- **Tool Calling**: Normalized `ToolDefinition`/`ToolChoice`/`ToolCall` types; each provider emits its own wire (OpenAI `tools`, Anthropic `tool_use`), streaming included
- **Multimodal**: Support for images, audio, video and documents (PDF) in messages (Anthropic's own API takes images and PDFs, but no audio or video)
- **JSON Repair**: fixes the broken JSON models often return
- **Providers**: OpenRouter, OpenAI, Anthropic, and any OpenAI-compatible server
- **Retry with Backoff**: Exponential backoff and retries, off unless you ask (each retry is a new paid request)
- **OpenRouter routing**: `OpenRouterRouting` (sort, ignore, data collection)

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
minillmlib = "0.6"
tokio = { version = "1", features = ["full"] }
```

### Optional features

| Feature | Default | What it adds |
| --- | --- | --- |
| `estimate` | off | [Cost estimation](#estimating-a-calls-cost) before a call is sent. Embeds a 1.6 MB token vocabulary, which a plain completion has no use for. |
| `live` | off | Integration tests that make real, billed API calls. |

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

Each preset reads its key from the environment: `OPENROUTER_API_KEY`,
`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`. If you keep them in a `.env` file, call
`minillmlib::init()` first, which loads it.

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

// Second turn: context is preserved
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

### Document Input (PDF)

The model reads the PDF natively (layout, tables, figures, scanned pages).
Declare the page count when you know it: cost estimation prices PDFs per page.

```rust
use minillmlib::{DocumentData, MessageContent};

let document = DocumentData::from_file("./report.pdf")?.with_page_count(12);
let content = MessageContent::with_documents("Summarize this report.", &[document]);
```

### Generated Media (image output)

If the model returns media (an image-generation model, say), it lands on
`CompletionResponse::media` as the same `Media` type you send. If you want to
add the reply, text and media, to a conversation, `to_assistant_message()`
builds that message:

```rust
let (node, response) = user.complete_collect(&generator, None).await?;
for media in &response.media {
    // e.g. Media::Image(img) where img.base64_data is a data: URL or an https URL
}
let assistant_message = response.to_assistant_message();
```

### Tool / Function Calling

Tools are normalized: define them once, and each provider emits its own wire
shape (OpenAI-wire `tools`/`tool_calls`, Anthropic `tool_use`/`tool_result`).
For the full loop, go and read the guide's
[Tool Calling chapter](https://weavemindai.github.io/MiniLLMLibRS/tool-calling.html).

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
        let args = call.arguments_json()?;                 // arguments parsed as JSON
        let result = run_my_tool(&call.name, &args);       // your code
        current = current.add_tool_result(&call.id, result);
    }
    let answer = current.complete(&generator, Some(&params)).await?;
}
```

If you want a tool call's arguments while the model is still writing them,
`ArgumentStream` gives you a `FieldHandle` per argument: `wait().await` for the
finished value, or `delta().await` for its text as it arrives, escapes undone.
A lenient mode handles models that write sloppy JSON. For the details, go and
read the guide's Tool Calling chapter.

For a complete multi-turn agent loop (streaming prose live, one streaming tool
consuming its decoded payload as the model generates it, one buffered tool),
run `OPENROUTER_API_KEY=... cargo run --example agent_loop`
([examples/agent_loop.rs](examples/agent_loop.rs)); it makes real, billed calls.

### JSON Response with Repair

```rust
let params = NodeCompletionParameters::new()
    .with_parse_json(true)           // Enable JSON repair
    .with_crash_on_refusal(true)     // Reject empty or null JSON (retried below)
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

### Force Prepend (start the reply with your text)

```rust
// Force the model to start its response with specific text
let params = NodeCompletionParameters::new()
    .with_force_prepend("Score: ");

// Response will start with "Score: " followed by the model's completion
```

### OpenRouter routing

If you use OpenRouter, `with_openrouter_routing` puts its routing settings under
the request's `provider` key. Set it only on an OpenRouter generator: OpenAI
and Anthropic refuse a request that carries that key.

```rust
use minillmlib::{CompletionParameters, OpenRouterRouting};

let routing = OpenRouterRouting::new()
    .sort_by_throughput()                              // or .sort_by_price()
    .deny_data_collection()
    .with_ignore(vec!["SambaNova".to_string()]);       // Exclude providers

let params = CompletionParameters::new()
    .with_openrouter_routing(routing);
```

### Prompt Caching (provider-agnostic)

Mark what to cache on the tree; the provider decides the wire (Anthropic and
OpenRouter-fronted Claude models emit `cache_control` breakpoints, OpenAI
auto-caches).

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

If you give a price yourself, add the cache rates too, or cache reads and
writes are billed at the plain input rate:

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
user.set_format_kwarg("bot_name", "Claude"); // a node's values fill only its own message

// Get formatted messages with placeholders replaced
let formatted = user.formatted_thread();
// "You are Claude, a helpful assistant." and "Hi Claude!"
```

### Save and Load Conversation Threads

```rust
use minillmlib::ChatNode;

// Build a conversation
let root = ChatNode::root("You are helpful.");
let user = root.add_user("Hello {name}!");
user.set_format_kwarg("name", "Alice");
let assistant = user.add_assistant("Hi there!");

// Save to JSON file
assistant.save_thread("conversation.json")?;

// Load from JSON file (returns root and leaf)
let (loaded_root, loaded_leaf) = ChatNode::from_thread_file("conversation.json")?;

// Or load from JSON string
let json = r#"{"prompts": [{"message": {"role": "system", "content": "Hello"}}]}"#;
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
let merged = tree1_leaf.merge(&tree2_leaf)?;  // a copy of tree2 goes under tree1_leaf; returns the copy's leaf

// Iterate over tree
for node in root.iter_depth_first() {
    println!("{}", node.text().unwrap_or_default());
}

// Get all leaves
let leaves = root.iter_leaves();

// Count nodes
let count = root.node_count();
```

### Cost Tracking

```rust
use minillmlib::{ChatNode, GeneratorInfo, NodeCompletionParameters, CostInfo};
use std::sync::{Arc, Mutex};

let generator = GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");

// Track costs across multiple requests
let total_cost = Arc::new(Mutex::new(0.0));
let cost_tracker = total_cost.clone();

let params = NodeCompletionParameters::new()
    .with_cost_callback(move |info: CostInfo| {
        *cost_tracker.lock().unwrap() += info.cost;
        println!("Request cost: ${}", info.cost);
        println!("Tokens: {} prompt, {} completion", 
            info.prompt_tokens, info.completion_tokens);
    });

let root = ChatNode::root("You are helpful.");
let user = root.add_user("Hello!");
let response = user.complete(&generator, Some(&params)).await?;

println!("Total spent: ${}", *total_cost.lock().unwrap());
```

### Estimating a call's cost

If you want to know what a call will cost before you send it, turn on the
`estimate` feature and ask the generator. You get a deliberately high figure to
reserve against, or `None` when it cannot be worked out:

```rust
let params = CompletionParameters::new().with_max_tokens(1024);
let usd = generator.estimate_cost_usd(&user.thread(), &params).await;
```

For where each provider finds a model's price, what the figure assumes, and how
to make it tighter, go and read
[the cost estimation chapter](https://weavemindai.github.io/MiniLLMLibRS/cost-estimation.html).

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
| `CompletionResponse` | A completion: text, usage, tool calls, and returned media |
| `CostInfo` | Cost and token usage information from completions |
| `CostResolution` | Whether a reported cost is `Resolved`, `Unpriced`, or `Unknown` |

### GeneratorInfo Methods

```rust
// Pre-configured providers
GeneratorInfo::openrouter(model)         // OpenRouter (OpenAI wire, native USD cost)
GeneratorInfo::openai(model)             // OpenAI (token counts; priced at your price, else the catalog's)
GeneratorInfo::anthropic(model)          // Anthropic's /v1/messages, x-api-key auth (token counts; priced at your price, else the catalog's, which usually needs with_openrouter_name)
GeneratorInfo::custom(name, url, model)  // Custom OpenAI-compatible endpoint

// Auth builder methods
.with_api_key(key)                       // provider chooses header (Bearer / x-api-key)
.with_api_key_from_env("ENV_VAR")
.with_bearer_token(token)                // always sent as Authorization: Bearer
.with_bearer_token_from_env("ENV_VAR")

// Other builder methods
.with_token_price(TokenPrice::new(in_per_mtok, out_per_mtok)) // the model's price: bills token-only calls and prices estimates
.with_openrouter_name("anthropic/claude-haiku-4.5") // the model's catalog id, so its price can be looked up
.with_provider(Arc::new(MyProvider))     // swap the wire dialect
.with_header(name, value)
.with_model_info(info)                   // largest output, context window (and price); unset fields are looked up
.with_default_params(params)
```

### CompletionParameters

These are the common ones; for every field, go and read
[the parameters chapter](https://weavemindai.github.io/MiniLLMLibRS/parameters.html).
Every field defaults to unset, and an unset field is not sent.

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_tokens` | `Option<u32>` | The most tokens the model may write, reasoning included. Unset, Anthropic (which requires one) gets the model's largest, or 4096 |
| `temperature` | `Option<f64>` | Sampling temperature |
| `top_p` | `Option<f64>` | Nucleus sampling |
| `top_k` | `Option<u32>` | Top-k sampling |
| `stop` | `Option<Vec<String>>` | Stop sequences |
| `seed` | `Option<u64>` | Random seed |
| `response_format` | `Option<ResponseFormat>` | Force JSON output |
| `reasoning` | `Option<Reasoning>` | `Off`, `On(None)` (a low effort), or `On(Some("high"))`, sent in each provider's own spelling |
| `extra` | `Option<HashMap>` | Provider-specific keys (incl. OpenRouter routing) |

### NodeCompletionParameters

These are the common ones; for every field, go and read
[the parameters chapter](https://weavemindai.github.io/MiniLLMLibRS/parameters.html).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `system_prompt` | `Option<String>` | `None` | System message to add when the thread has none |
| `parse_json` | `bool` | `false` | Parse/repair JSON response |
| `force_prepend` | `Option<String>` | `None` | Force response prefix |
| `retry` | `u32` | `0` | Retries after a failed attempt (each one is a new paid request) |
| `exp_back_off` | `bool` | `false` | Exponential backoff |
| `back_off_time` | `f64` | `1.0` | Initial backoff (seconds) |
| `max_back_off` | `f64` | `15.0` | Max backoff (seconds) |
| `crash_on_refusal` | `bool` | `false` | With `parse_json`, error if the JSON is empty or null |
| `crash_on_empty_response` | `bool` | `false` | Error if empty |
| `track_cost` | `bool` | `false` | Ask for token counts (stored on the reply); for the cost, add a `cost_callback` or call `complete_costed` |
| `cost_callback` | `Option<CostCallback>` | `None` | Callback for cost info |

### OpenRouterRouting

| Parameter | Description |
|-----------|-------------|
| `order` | Ordered list of providers to try |
| `sort` | Sort by: "price", "throughput", "latency" |
| `ignore` | Providers to exclude |
| `data_collection` | "allow" or "deny" |
| `allow_fallbacks` | Allow fallback providers |

## CLI Tool

If you want to repair JSON from the shell, install the bundled CLI with
`cargo install minillmlib`:

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
# Reads OPENROUTER_API_KEY and ANTHROPIC_API_KEY from the env (or a .env);
# each live test skips if its key is absent.
cargo test --features live

# Run with output
cargo test -- --nocapture
```

## License

MIT License. For details, see [LICENSE](LICENSE).

## Contributing

If you want to help, send a pull request.
