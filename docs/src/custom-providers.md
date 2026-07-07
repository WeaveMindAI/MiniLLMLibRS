# Custom Providers

Connecting your own server is one of two cases.

## Case A: your server speaks OpenAI's `/chat/completions`

vLLM, llama.cpp's server, LM Studio, TGI, Ollama's OpenAI endpoint, or your own
OpenAI-compatible wrapper. Nothing custom to write: point `custom()` at it. The
default `GenericProvider` handles the wire.

```rust,no_run
use minillmlib::{ChatNode, GeneratorInfo, TokenPrice};

# async fn run() -> minillmlib::Result<()> {
// base_url is everything BEFORE /chat/completions; the provider appends the path.
let gen = GeneratorInfo::custom("my-server", "http://localhost:8000/v1", "my-model")
    .with_api_key_from_env("MY_SERVER_KEY")    // omit entirely if unauthenticated
    .with_header("X-Tenant", "acme")           // any extra gateway headers
    .with_token_price(TokenPrice::new(0.0, 0.0)); // $/Mtok; 0/0 for a free local model

let answer = ChatNode::root("You are helpful.")
    .chat("hello", &gen).await?;
println!("{}", answer.message.text().unwrap_or(""));
# Ok(()) }
```

For an older server that only accepts `max_tokens` (not `max_completion_tokens`):

```rust,no_run
use minillmlib::{GeneratorInfo, GenericProvider};
use std::sync::Arc;

let gen = GeneratorInfo::custom("old", "http://localhost:8000/v1", "m")
    .with_provider(Arc::new(GenericProvider { legacy_token_limit: true }));
```

If your server speaks the OpenAI envelope but its *tool* shape deviates,
override just the two tool hooks in your `impl Provider`
(`openai_tools_value`, `openai_tool_choice_value`); the rest of the default
request builder stays. See [Tool Calling](./tool-calling.md).

## Case B: your server has a different wire

Different endpoint, auth header, request/response shape: implement the `Provider`
trait once and pass it via `with_provider`. The user-facing API
(`root.chat(...)`) stays identical.

Below is a complete adapter for a made-up "EchoAI" server with a genuinely
different wire: endpoint `/api/generate`, auth header `X-Echo-Key`, request
`{model, prompt, settings}`, response `{output:{text}, meta}`. This mirrors the
tested example in `tests/integration_tests.rs`.

```rust,no_run
use minillmlib::{
    Auth, ChatNode, CompletionParameters, CompletionResponse, CostOutcome, GeneratorInfo,
    Message, MessageContent, Provider, StreamChunk, TokenPrice, Usage,
};
use secrecy::ExposeSecret;
use std::sync::Arc;

#[derive(Debug, Clone)]
struct EchoAi;

impl Provider for EchoAi {
    fn endpoint_url(&self, base: &str) -> String {
        format!("{}/api/generate", base.trim_end_matches('/'))
    }

    fn auth_headers(&self, auth: &Auth) -> minillmlib::Result<Vec<(String, String)>> {
        Ok(match auth.secret() {
            Some(s) => vec![("X-Echo-Key".into(), s.expose_secret().to_string())],
            None => vec![],
        })
    }

    fn build_request(
        &self, model: &str, messages: &[Message], params: &CompletionParameters,
        _stream: bool, _include_usage: bool,
    ) -> minillmlib::Result<serde_json::Value> {
        // Flatten the conversation into one prompt. Fail loudly on multimodal
        // (this wire is text-only) instead of silently dropping the attachment.
        let mut lines = Vec::new();
        for m in messages {
            if let MessageContent::Parts(parts) = &m.content {
                if parts.iter().any(|p| p.as_text().is_none()) {
                    return Err(minillmlib::MiniLLMError::InvalidParameter(
                        "EchoAI is text-only".into(),
                    ));
                }
            }
            lines.push(format!("{}: {}", m.role.as_str(), m.content.all_text()));
        }
        Ok(serde_json::json!({
            "model": model,
            "prompt": lines.join("\n"),
            "settings": { "max_output_tokens": params.max_tokens.unwrap_or(256) },
        }))
    }

    fn parse_response(&self, raw: serde_json::Value) -> minillmlib::Result<CompletionResponse> {
        let text = raw["output"]["text"].as_str()
            .ok_or_else(|| minillmlib::MiniLLMError::MalformedResponse(raw.to_string()))?
            .to_string();
        Ok(CompletionResponse {
            id: raw["meta"]["id"].as_str().unwrap_or("").into(),
            model: raw["meta"]["model"].as_str().unwrap_or("").into(),
            content: text,
            finish_reason: raw["stop"].as_str().map(String::from),
            usage: self.parse_usage(&raw),
            tool_calls: None,
            raw_response: Some(raw),
        })
    }

    fn parse_usage(&self, raw: &serde_json::Value) -> Option<Usage> {
        let meta = raw.get("meta")?;
        Some(Usage {
            uncached_input_tokens: meta["tokens_in"].as_u64().unwrap_or(0) as u32,
            completion_tokens: meta["tokens_out"].as_u64().unwrap_or(0) as u32,
            ..Default::default()
        })
    }

    fn parse_chunk(&self, _data: &str) -> Option<minillmlib::Result<StreamChunk>> {
        None // non-streaming
    }

    fn emits_stream_usage(&self, _requested: bool) -> bool {
        false // never sends a trailing usage chunk; don't wait for one
    }

    fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
        match price {
            Some(p) => CostOutcome::resolved(p.cost_of(&usage), usage),
            None => CostOutcome::unpriced(usage),
        }
    }
}

# async fn run() -> minillmlib::Result<()> {
let gen = GeneratorInfo::custom("echoai", "https://my.host", "echo-1")
    .with_provider(Arc::new(EchoAi))
    .with_api_key("my-secret")
    .with_token_price(TokenPrice::new(1.0, 5.0));

let answer = ChatNode::root("You are EchoAI.")
    .chat("hello", &gen).await?;
# let _ = answer;
# Ok(()) }
```

### What to override

The trait defaults to the OpenAI dialect, so you override only what differs:

| Method | Override when |
|---|---|
| `endpoint_url` | the path isn't `/chat/completions` |
| `auth_headers` | auth isn't `Authorization: Bearer` |
| `build_request` | the request body isn't the OpenAI shape |
| `parse_response` | the response envelope isn't `choices[]` |
| `parse_chunk` | streaming chunks aren't OpenAI deltas (return `None` if non-streaming) |
| `parse_usage` | usage fields differ |
| `emits_stream_usage` | the server may never send a trailing usage chunk (return `false`, or the stream waits for one that never comes) |
| `cost_of` | cost is derived differently |
| `resolve_post_stream` | there's an out-of-band cost endpoint |

### Two rules to copy from the example

- **Fail loudly on anything you can't represent.** EchoAI rejects multimodal
  rather than silently flattening it away.
- **Override `emits_stream_usage` to `false`** if your server never sends a
  trailing usage chunk, or a streaming call will wait for it until the idle
  timeout.
