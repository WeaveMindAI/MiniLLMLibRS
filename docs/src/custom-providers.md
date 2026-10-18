# Custom Providers

How you connect your own server depends on what it speaks.

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

If your server's tool shape differs from OpenAI's, write your own
`impl Provider` (as in Case B below) that overrides `openai_tools_value` and
`openai_tool_choice_value`, plus `cost_of`, which has no default (see
[What to override](#what-to-override)). If your server never sends a usage
chunk when streaming, also make `emits_stream_usage` return `false`, as
`GenericProvider` does. For the tool shapes themselves, go and read
[Tool Calling](./tool-calling.md).

## Case B: your server has a different wire

If your server differs in its endpoint, its auth header or its request and
response shapes, implement the `Provider` trait and pass it with
`with_provider`. Your calling code (`root.chat(...)`) does not change.

Below is a complete adapter for a made-up "EchoAI" server with a different
wire: endpoint `/api/generate`, auth header `X-Echo-Key`, request
`{model, prompt, settings}`, response `{output:{text}, meta}`. This mirrors the
tested example in `tests/integration_tests.rs`. The example reads the key with
the `secrecy` crate and builds JSON with `serde_json`, so add
`secrecy = "0.10"` and `serde_json = "1"` to your `[dependencies]`.

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
        // EchoAI has a field for the output cap and nothing else: any other
        // setting the caller made fails loudly instead of vanishing.
        let only_the_cap = CompletionParameters {
            max_tokens: params.max_tokens,
            ..Default::default()
        };
        if serde_json::to_value(params)? != serde_json::to_value(&only_the_cap)? {
            return Err(minillmlib::MiniLLMError::InvalidParameter(
                "EchoAI takes only max_tokens".into(),
            ));
        }
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
        let mut body = serde_json::json!({
            "model": model,
            "prompt": lines.join("\n"),
        });
        // Send only what the caller set; EchoAI's own default covers the rest.
        if let Some(cap) = params.max_tokens {
            body["settings"] = serde_json::json!({ "max_output_tokens": cap });
        }
        Ok(body)
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
            media: Vec::new(),
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
| `model_info` | you can find a model's price, largest output or context window somewhere (your API, a price sheet): return what you find as a `ModelInfo` (see below) |
| `requires_max_tokens` | your API refuses a request without an output cap: return `true`, and a request that sets none carries the model's largest, or 4096 |
| `build_request` | the request body isn't the OpenAI shape |
| `parse_response` | the response envelope isn't `choices[]` |
| `parse_chunk` | streaming chunks aren't OpenAI deltas (return `None` if non-streaming) |
| `parse_usage` | usage fields differ |
| `emits_stream_usage` | the server may never send a trailing usage chunk (return `false`, or the stream waits for one that never comes) |
| `parse_response_media` | the wire returns model-generated media somewhere other than `message.images` |
| `usage_out_of_band` | your API keeps a record of a generation's usage you can ask for after a stream ended without it |
| `attribution_headers` | your API credits usage to an app through headers |
| `max_cache_breakpoints` | your own `build_request` writes cache markers and your API caps them: return the cap, and mark only the messages `minillmlib::provider::kept_cache_breakpoints(messages, self.max_cache_breakpoints())` picks |
| `wire_keeps_estimation_metadata` | something between you and your server prices media from the request (a gateway, an estimator), and your server ignores unknown keys: return `true` and each media part keeps its width, height, duration and page count |

You always write `cost_of`: it turns token counts into dollars, and has no
default. For a server that reports token counts only, copy EchoAI's four lines.

If your server speaks the OpenAI body with a twist, override the hook for the
twist instead of `build_request`:

| Hook | Override when |
|---|---|
| `openai_token_limit_field` | the cap key isn't `max_completion_tokens` (older servers take `max_tokens`) |
| `openai_reasoning_field` | reasoning isn't `reasoning_effort` (OpenRouter writes its `reasoning` object) |
| `openai_request_usage` | usage has to be asked for in the body |
| `openai_messages_value`, `openai_tools_value`, `openai_tool_choice_value` | those parts of the body differ |

### Telling the library about your models

Cost estimation, the price a call is reported at, and the cap a
`requires_max_tokens` wire fills in all come from the `ModelInfo` your
`model_info` returns. Fill what you can find and leave the rest `None`.

`ModelLookup` gives you `lookup.call` (the generator's client, address and
headers), `lookup.model` (the id on your wire), `lookup.openrouter_name` (its
OpenRouter catalog id, when the caller set one) and `lookup.wanted` (whether
the caller needs the price, the limits, or both, so you can skip a source that
answers only what is not wanted).
Make your requests with `lookup.call.get_cached(path, timeout, parse)`. `path`
is joined onto the generator's base URL, and `parse` gets the status and the
body text. Callers anywhere in the program share each request: one goes out per
full URL, header set and type `parse` returns, whichever client makes it. `parse` runs once per reply
(so what it logs is logged once), and a value parsed from a success or a 404 is
kept for an hour. Any other reply, or a `parse` that returns an
error, is not kept. A request that gets no reply at all (a timeout, a refused
connection) fails the same way for a minute, then is made again. If
`get_cached` returns an `Err`, the failure is already logged, so just leave
that source's fields `None`.

If your models are in OpenRouter's catalog, the whole job is one call:

```rust,ignore
fn model_info(&self, lookup: ModelLookup) -> Option<InfoFuture> {
    let model = lookup.openrouter_name.unwrap_or(lookup.model);
    let catalog = minillmlib::provider::catalog_call(lookup.call.client);
    // Groq's price; with `None`, or if Groq does not serve the model, the
    // dearest provider's.
    Some(Box::pin(async move {
        minillmlib::provider::catalog_info(&catalog, &model, Some("groq")).await
    }))
}
```

If your API has no way to tell, skip `model_info`: callers give the facts with
`GeneratorInfo::with_model_info` and `with_token_price`.

### Reusing the OpenAI dialect piecewise

If you override a method but still want part of OpenAI's behaviour inside it,
call the library's own pieces from `minillmlib::provider::openai_wire`: the
request builder, the response, usage and chunk parsers, and the converters for
messages and tools (`messages_to_payload`, `tool_definition_value`, ...).
