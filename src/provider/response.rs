//! Response types from LLM APIs

use crate::tools::{ToolCall, ToolCallDelta};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Normalized token usage + cost, the common currency every provider's
/// [`Provider`](super::Provider) parses its native wire shape into.
///
/// Input tokens are split into THREE DISJOINT, ADDITIVE buckets so caching is
/// priced correctly across every provider's differing wire conventions:
/// - `uncached_input_tokens`: full-price prompt tokens (no cache involved),
/// - `cache_read_tokens`: served from a warm cache (cheap, ~0.1× input),
/// - `cache_write_tokens`: written to the cache this request (a premium, ~1.25×).
///
/// They never overlap, so total input = the sum of the three, and cost is a clean
/// weighted sum with no subtraction (the old single `cached_tokens` field forced a
/// subtract that was correct for OpenAI's "cached is a subset of prompt_tokens"
/// wire but WRONG for Anthropic's "input_tokens already excludes cached" wire).
/// Each provider's parser maps its native fields into these disjoint buckets.
///
/// Built by the provider (the nested per-provider wire shapes don't match these
/// flat fields), and serialized into node metadata for diagnostics. Deliberately
/// NOT `Deserialize`: a derived flat-field deserializer would silently produce
/// all-zero/`None` fields against the real nested payloads.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Usage {
    /// Full-price input tokens (NOT read from nor written to cache this request).
    pub uncached_input_tokens: u32,

    /// Input tokens served from a warm cache (priced at the cache-read rate).
    pub cache_read_tokens: u32,

    /// Input tokens written to the cache this request (priced at the cache-write
    /// premium). Non-zero only on the request that creates/refreshes a cache entry.
    pub cache_write_tokens: u32,

    /// Number of tokens in the completion (output).
    pub completion_tokens: u32,

    /// Cost in USD (for OpenRouter, the fee; may be 0 on a BYOK free tier or when
    /// the provider returns no native cost). `None` if the wire carried no cost.
    pub cost: Option<f64>,

    /// Upstream inference cost (only for BYOK requests, the actual
    /// cost charged by the provider like Google Vertex or Bedrock)
    pub upstream_inference_cost: Option<f64>,

    /// Reasoning tokens (for models that support it)
    pub reasoning_tokens: Option<u32>,
}

impl Usage {
    /// Total input tokens processed = the three disjoint input buckets summed.
    pub fn prompt_tokens(&self) -> u32 {
        self.uncached_input_tokens + self.cache_read_tokens + self.cache_write_tokens
    }

    /// Total tokens (input + output).
    pub fn total_tokens(&self) -> u32 {
        self.prompt_tokens() + self.completion_tokens
    }

    /// Fold a later usage report into this one, keeping the non-zero/`Some` value
    /// of each field. Needed for providers that split usage across streaming
    /// events (Anthropic sends input tokens in `message_start` and output tokens
    /// in `message_delta`); for single-usage-chunk providers (OpenAI) this is a
    /// plain overwrite since the prior usage is all-zero/`None`.
    pub(crate) fn merge_from(&mut self, other: &Usage) {
        if other.uncached_input_tokens != 0 {
            self.uncached_input_tokens = other.uncached_input_tokens;
        }
        if other.cache_read_tokens != 0 {
            self.cache_read_tokens = other.cache_read_tokens;
        }
        if other.cache_write_tokens != 0 {
            self.cache_write_tokens = other.cache_write_tokens;
        }
        if other.completion_tokens != 0 {
            self.completion_tokens = other.completion_tokens;
        }
        self.cost = other.cost.or(self.cost);
        self.upstream_inference_cost = other
            .upstream_inference_cost
            .or(self.upstream_inference_cost);
        self.reasoning_tokens = other.reasoning_tokens.or(self.reasoning_tokens);
    }
}

/// Whether the cost in a `CostInfo` was actually determined. Consumers must
/// check this before treating the reported amount as truth: only `Resolved`
/// carries an authoritative USD cost. Neither `Unpriced` nor `Unknown` may be
/// silently counted as a real zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CostResolution {
    /// The USD cost is authoritative (returned natively by the provider, or
    /// derived from real token counts and a configured `TokenPrice`).
    #[default]
    Resolved,
    /// Token counts are real, but the provider returns no native cost and no
    /// `TokenPrice` was configured for this generator/request, so the USD amount
    /// is unknown. Set a `TokenPrice` (on the generator or per-request) to resolve
    /// it. The `cost` field is 0.0 and must NOT be treated as a free request.
    Unpriced,
    /// Cost could not be determined at all (no usage was returned and any
    /// out-of-band query failed). Numeric fields are best-effort.
    Unknown,
}

/// Detailed cost information from a completion
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostInfo {
    /// Total cost in credits charged to your account
    pub cost: f64,

    /// Total prompt (input) tokens = uncached + cache-read + cache-write.
    pub prompt_tokens: u32,

    /// Number of completion (output) tokens
    pub completion_tokens: u32,

    /// Total tokens (input + output)
    pub total_tokens: u32,

    /// Input tokens served from a warm cache (priced at the cache-read rate).
    pub cache_read_tokens: u32,

    /// Input tokens written to the cache this request (priced at the cache-write
    /// premium, a one-time cost when the cache entry is created/refreshed).
    pub cache_write_tokens: u32,

    /// Reasoning tokens (if any)
    pub reasoning_tokens: Option<u32>,

    /// The model used
    pub model: String,

    /// Response ID for tracking
    pub response_id: String,

    /// Whether `cost` was actually determined or could not be resolved.
    pub resolution: CostResolution,
}

/// Callback function type for cost ingestion
/// Called with CostInfo after each successful completion
pub type CostCallback = Arc<dyn Fn(CostInfo) + Send + Sync>;

/// A complete response from an LLM API.
///
/// Serialize-only: it is built in-code from parsed responses, never deserialized
/// (and it embeds `Usage`, which is not `Deserialize` by design).
#[derive(Debug, Clone, Serialize)]
pub struct CompletionResponse {
    /// Unique identifier for this completion
    pub id: String,

    /// The model that generated this response
    pub model: String,

    /// The generated text content
    pub content: String,

    /// Finish reason (e.g., "stop", "length", "tool_calls")
    pub finish_reason: Option<String>,

    /// Token usage statistics
    pub usage: Option<Usage>,

    /// Tool calls made by the model (if any), normalized across providers.
    pub tool_calls: Option<Vec<ToolCall>>,

    /// Raw response for debugging
    #[serde(skip)]
    pub raw_response: Option<serde_json::Value>,
}

impl CompletionResponse {
    /// Create a new completion response
    pub fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        content: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            content: content.into(),
            finish_reason: None,
            usage: None,
            tool_calls: None,
            raw_response: None,
        }
    }

    /// Check if the completion finished normally
    pub fn is_complete(&self) -> bool {
        self.finish_reason.as_deref() == Some("stop")
    }

    /// Check if the completion was truncated due to length
    pub fn is_truncated(&self) -> bool {
        self.finish_reason.as_deref() == Some("length")
    }

    /// Check if the model made tool calls
    pub fn has_tool_calls(&self) -> bool {
        self.tool_calls.as_ref().is_some_and(|tc| !tc.is_empty())
    }
}

/// A chunk from a streaming response
#[derive(Debug, Clone, Default)]
pub struct StreamChunk {
    /// The provider's generation id, if this chunk carried one (every OpenAI-wire
    /// chunk does). Threaded into the stream so out-of-band cost resolution can
    /// query the REAL generation, not a locally-minted placeholder.
    pub id: Option<String>,

    /// The delta content in this chunk
    pub delta: String,

    /// Finish reason (only present in final chunk)
    pub finish_reason: Option<String>,

    /// Usage info (only present in final chunk for some providers)
    pub usage: Option<Usage>,

    /// Tool call fragments carried by this chunk (assembled by
    /// [`ToolCallAccumulator`](crate::tools::ToolCallAccumulator)).
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

impl StreamChunk {
    /// Create a new stream chunk with content
    pub fn content(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            ..Default::default()
        }
    }

    /// Create a final chunk with finish reason
    pub fn finished(finish_reason: impl Into<String>) -> Self {
        Self {
            finish_reason: Some(finish_reason.into()),
            ..Default::default()
        }
    }

    /// Check if this is the final chunk
    pub fn is_final(&self) -> bool {
        self.finish_reason.is_some()
    }
}

/// Truncate a response body for inclusion in an error/log, on a char boundary,
/// so error strings can't balloon with a huge (and possibly prompt-bearing) body.
pub(crate) fn preview_str(body: &str) -> String {
    const MAX: usize = 200;
    match body.char_indices().nth(MAX) {
        Some((cut, _)) => format!("{}…", &body[..cut]),
        None => body.to_string(),
    }
}

/// A real error envelope: a NON-EMPTY object under `error`. A `null`, `{}`, or a
/// falsy scalar (`false`/`0`/`""`) is not a failure and must not trip error
/// handling (which would fail an otherwise-good response/stream and silently lose
/// an accepted generation's cost). Every provider emits a populated object on a
/// genuine error and omits the field entirely on success.
fn error_object(raw: &serde_json::Value) -> Option<&serde_json::Value> {
    raw.get("error")
        .filter(|e| e.as_object().is_some_and(|o| !o.is_empty()))
}

/// If `raw` carries an OpenAI-wire `error` object, map it to a typed `Api` error.
/// The single place the OpenAI-wire error envelope is decoded, so a 200-with-error
/// body is surfaced identically whether it arrives as a full response or as an
/// in-band streaming chunk. `None` when there is no error object.
fn openai_error_in(raw: &serde_json::Value) -> Option<crate::error::MiniLLMError> {
    let error = error_object(raw)?;
    let message = error["message"]
        .as_str()
        .map(String::from)
        .unwrap_or_else(|| preview_str(&error.to_string()));
    // Use the error code only when it's a genuine numeric HTTP status in range;
    // providers also send string codes (e.g. "rate_limit_exceeded") or values
    // outside u16. Anything else is an upstream failure -> 502 (retryable), so a
    // transient overload is never misclassified as a non-retryable success.
    let status = error["code"]
        .as_u64()
        .filter(|&c| (100..=599).contains(&c))
        .map(|c| c as u16)
        .unwrap_or(502);
    Some(crate::error::MiniLLMError::Api { status, message })
}

/// If `raw` carries an Anthropic `error` object, map it to a typed `Api` error
/// (status 502: an upstream Anthropic failure is treated as retryable). The single
/// place the Anthropic error envelope is decoded, so a 200-with-error body is
/// surfaced identically from a full response and an in-band stream `error` event.
fn anthropic_error_in(raw: &serde_json::Value) -> Option<crate::error::MiniLLMError> {
    let error = error_object(raw)?;
    let message = error["message"]
        .as_str()
        .map(String::from)
        .unwrap_or_else(|| preview_str(&error.to_string()));
    Some(crate::error::MiniLLMError::Api {
        status: 502,
        message,
    })
}

/// Parse a raw OpenAI-wire response into a CompletionResponse (the default
/// `Provider::parse_response`). `provider.parse_usage` extracts the usage so a
/// provider with native cost fields (OpenRouter) reads them.
///
/// Many OpenAI-compatible providers (OpenRouter included) return HTTP 200 with
/// an error body and no `choices`. We must surface that as a loud error instead
/// of silently producing an empty completion, so callers never mistake an error
/// for a successful empty response.
pub fn parse_openai_response<P: super::Provider + ?Sized>(
    raw: serde_json::Value,
    provider: &P,
) -> crate::error::Result<CompletionResponse> {
    // A 200 response carrying an `error` object is a failure, not a completion.
    if let Some(err) = openai_error_in(&raw) {
        return Err(err);
    }

    let id = raw["id"].as_str().unwrap_or("").to_string();
    let model = raw["model"].as_str().unwrap_or("").to_string();

    // A well-formed completion must carry a first choice with a message. If it
    // does not (and there was no error object above), the response is malformed.
    let choice = raw["choices"]
        .get(0)
        .filter(|c| c.get("message").is_some())
        .ok_or_else(|| {
            crate::error::MiniLLMError::MalformedResponse(preview_str(&raw.to_string()))
        })?;
    let message = &choice["message"];

    // `content` may legitimately be null/absent for a tool-call-only response.
    let content = message["content"].as_str().unwrap_or("").to_string();
    let tool_calls = message["tool_calls"]
        .as_array()
        .map(|entries| parse_openai_tool_calls(entries))
        .transpose()?;
    let finish_reason = choice["finish_reason"].as_str().map(String::from);

    // Usage parsing is provider-specific (field names, native cost fields).
    let usage = provider.parse_usage(&raw);

    Ok(CompletionResponse {
        id,
        model,
        content,
        finish_reason,
        usage,
        tool_calls,
        raw_response: Some(raw),
    })
}

/// Parse the OpenAI-wire `tool_calls` array of a COMPLETE message into typed
/// [`ToolCall`]s. On this wire every entry carries `id`, `function.name`, and
/// `function.arguments` (a JSON string); an entry missing any of them is a
/// malformed response and fails loudly rather than yielding a call the caller
/// cannot answer.
fn parse_openai_tool_calls(entries: &[serde_json::Value]) -> crate::error::Result<Vec<ToolCall>> {
    entries
        .iter()
        .map(|entry| {
            let id = entry["id"].as_str();
            let name = entry["function"]["name"].as_str();
            let arguments = entry["function"]["arguments"].as_str();
            match (id, name) {
                (Some(id), Some(name)) => {
                    Ok(ToolCall::new(id, name, arguments.unwrap_or_default()))
                }
                _ => Err(crate::error::MiniLLMError::MalformedResponse(format!(
                    "tool_calls entry missing id or function.name: {}",
                    preview_str(&entry.to_string())
                ))),
            }
        })
        .collect()
}

/// Parse the OpenAI-wire streaming `delta.tool_calls` entries into normalized
/// [`ToolCallDelta`]s. `index` de-multiplexes parallel calls and is structurally
/// required; a delta without a numeric index cannot be routed to a slot and is
/// skipped loudly.
fn parse_openai_tool_call_deltas(entries: &[serde_json::Value]) -> Vec<ToolCallDelta> {
    entries
        .iter()
        .filter_map(|entry| {
            let Some(index) = entry["index"].as_u64() else {
                tracing::warn!("tool_call delta missing numeric index, skipping");
                return None;
            };
            Some(ToolCallDelta {
                index,
                id: entry["id"].as_str().map(String::from),
                name: entry["function"]["name"].as_str().map(String::from),
                arguments_fragment: entry["function"]["arguments"].as_str().map(String::from),
            })
        })
        .collect()
}

/// Parse an OpenAI-wire streaming chunk from SSE data (the default
/// `Provider::parse_chunk`). `provider.parse_usage` reads provider-specific usage
/// out of the chunk.
pub fn parse_openai_chunk<P: super::Provider + ?Sized>(
    data: &str,
    provider: &P,
) -> Option<crate::error::Result<StreamChunk>> {
    // Handle [DONE] marker
    if data.trim() == "[DONE]" {
        return Some(Ok(StreamChunk::finished("stop")));
    }

    // Parse JSON
    let json: serde_json::Value = serde_json::from_str(data).ok()?;

    // An in-band error frame on a 200 stream is a FAILURE, surfaced loudly through
    // the channel (same path as a transport error) so it is never billed as an
    // accepted generation. Mirrors `parse_openai_response`'s error handling.
    if let Some(err) = openai_error_in(&json) {
        return Some(Err(err));
    }

    // The provider's real generation id (every OpenAI-wire chunk carries it);
    // threaded so out-of-band cost resolution targets the actual generation.
    let id = json["id"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(String::from);

    // Provider-specific usage (OpenRouter/OpenAI send it in the last chunk).
    let usage = provider.parse_usage(&json);

    // Try to get choice (may not be present in usage-only chunks)
    let choice = json["choices"].get(0);

    let delta = choice
        .and_then(|c| c["delta"]["content"].as_str())
        .unwrap_or("")
        .to_string();

    let finish_reason = choice
        .and_then(|c| c["finish_reason"].as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);

    let tool_calls = choice
        .and_then(|c| c["delta"]["tool_calls"].as_array())
        .map(|entries| parse_openai_tool_call_deltas(entries))
        .filter(|deltas| !deltas.is_empty());

    // Return a chunk if it carries anything we track (id alone is not enough to
    // surface, but it rides along with whatever else is present).
    if delta.is_empty() && finish_reason.is_none() && usage.is_none() && tool_calls.is_none() {
        return None;
    }

    Some(Ok(StreamChunk {
        id,
        delta,
        finish_reason,
        usage,
        tool_calls,
    }))
}

// =============================================================================
// Anthropic `/v1/messages` envelope (`content[]`)
// =============================================================================

/// Parse Anthropic's usage object into the normalized DISJOINT buckets.
///
/// Anthropic's wire is ALREADY disjoint: `input_tokens` is the non-cached input
/// only (tokens after the last cache breakpoint), and `cache_read_input_tokens` /
/// `cache_creation_input_tokens` are SEPARATE additive counts. So the mapping is
/// direct, no subtraction. Anthropic returns NO dollar cost, only token counts.
/// Streaming `message_delta` carries only `output_tokens` (input folded from the
/// earlier `message_start` via [`Usage::merge_from`]).
fn parse_anthropic_usage(u: &serde_json::Value) -> Option<Usage> {
    if u.is_null() {
        return None;
    }
    Some(Usage {
        uncached_input_tokens: u["input_tokens"].as_u64().unwrap_or(0) as u32,
        cache_read_tokens: u["cache_read_input_tokens"].as_u64().unwrap_or(0) as u32,
        cache_write_tokens: u["cache_creation_input_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: u["output_tokens"].as_u64().unwrap_or(0) as u32,
        cost: None,
        upstream_inference_cost: None,
        reasoning_tokens: None,
    })
}

/// Parse a completed Anthropic `/v1/messages` response. The envelope is
/// `content[]` blocks (text + optional tool_use), a top-level `stop_reason`, and
/// a token-only `usage`. A 200 carrying an `error` object is surfaced loudly.
pub fn parse_anthropic_response(
    raw: serde_json::Value,
) -> crate::error::Result<CompletionResponse> {
    if let Some(err) = anthropic_error_in(&raw) {
        return Err(err);
    }

    let content_blocks = raw["content"].as_array().ok_or_else(|| {
        crate::error::MiniLLMError::MalformedResponse(preview_str(&raw.to_string()))
    })?;

    // Join every text block; collect tool_use blocks into normalized ToolCalls
    // (the `input` object is serialized to raw JSON text, the normalized form).
    let mut text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    for block in content_blocks {
        match block["type"].as_str() {
            Some("text") => text.push_str(block["text"].as_str().unwrap_or("")),
            Some("tool_use") => {
                let (id, name) = (block["id"].as_str(), block["name"].as_str());
                let (Some(id), Some(name)) = (id, name) else {
                    return Err(crate::error::MiniLLMError::MalformedResponse(format!(
                        "tool_use block missing id or name: {}",
                        preview_str(&block.to_string())
                    )));
                };
                tool_calls.push(ToolCall::new(id, name, block["input"].to_string()));
            }
            _ => {}
        }
    }

    Ok(CompletionResponse {
        id: raw["id"].as_str().unwrap_or("").to_string(),
        model: raw["model"].as_str().unwrap_or("").to_string(),
        content: text,
        finish_reason: raw["stop_reason"].as_str().map(String::from),
        usage: parse_anthropic_usage(&raw["usage"]),
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        raw_response: Some(raw),
    })
}

/// Parse one Anthropic SSE event payload into a [`StreamChunk`]. Anthropic streams
/// a sequence of typed events; each maps to at most one chunk:
/// - `message_start` → carries the message `id` + initial usage (input tokens),
/// - `content_block_start` (`content_block.tool_use`) → a tool call's id + name,
/// - `content_block_delta` (`delta.text_delta`) → the text delta,
/// - `content_block_delta` (`delta.input_json_delta`) → a tool-argument fragment,
/// - `message_delta` → final usage (output tokens) + `stop_reason`,
/// - `message_stop` → terminal marker.
///
/// Tool events reuse the block `index` as the [`ToolCallDelta`] index; that index
/// space is shared with text blocks (so it may be sparse), which the accumulator
/// handles. Other events (text `content_block_start`, `content_block_stop`,
/// `ping`) carry nothing trackable.
pub fn parse_anthropic_chunk(data: &str) -> Option<crate::error::Result<StreamChunk>> {
    let json: serde_json::Value = serde_json::from_str(data).ok()?;
    match json["type"].as_str()? {
        // An in-band `error` event on a 200 stream is a FAILURE (e.g.
        // `overloaded_error` mid-generation). Surface it loudly through the channel,
        // same as the non-streaming `parse_anthropic_response`, so a failed stream
        // is never billed as an accepted generation.
        "error" => Some(Err(anthropic_error_in(&json).unwrap_or_else(|| {
            crate::error::MiniLLMError::Api {
                status: 502,
                message: preview_str(&json.to_string()),
            }
        }))),
        "message_start" => {
            let msg = &json["message"];
            let id = msg["id"]
                .as_str()
                .filter(|s| !s.is_empty())
                .map(String::from);
            let usage = parse_anthropic_usage(&msg["usage"]);
            (id.is_some() || usage.is_some()).then(|| {
                Ok(StreamChunk {
                    id,
                    usage,
                    ..Default::default()
                })
            })
        }
        "content_block_start" => {
            // Only a tool_use block start carries anything trackable (the call's
            // id + name); a text block start is ignorable.
            let block = &json["content_block"];
            if block["type"].as_str() != Some("tool_use") {
                return None;
            }
            let index = json["index"].as_u64()?;
            Some(Ok(StreamChunk {
                tool_calls: Some(vec![ToolCallDelta {
                    index,
                    id: block["id"].as_str().map(String::from),
                    name: block["name"].as_str().map(String::from),
                    arguments_fragment: None,
                }]),
                ..Default::default()
            }))
        }
        "content_block_delta" => match json["delta"]["type"].as_str() {
            Some("input_json_delta") => {
                let index = json["index"].as_u64()?;
                let frag = json["delta"]["partial_json"].as_str().unwrap_or("");
                (!frag.is_empty()).then(|| {
                    Ok(StreamChunk {
                        tool_calls: Some(vec![ToolCallDelta {
                            index,
                            arguments_fragment: Some(frag.to_string()),
                            ..Default::default()
                        }]),
                        ..Default::default()
                    })
                })
            }
            _ => {
                let delta = json["delta"]["text"].as_str().unwrap_or("").to_string();
                (!delta.is_empty()).then(|| {
                    Ok(StreamChunk {
                        delta,
                        ..Default::default()
                    })
                })
            }
        },
        "message_delta" => {
            let finish_reason = json["delta"]["stop_reason"].as_str().map(String::from);
            let usage = parse_anthropic_usage(&json["usage"]);
            (finish_reason.is_some() || usage.is_some()).then(|| {
                Ok(StreamChunk {
                    finish_reason,
                    usage,
                    ..Default::default()
                })
            })
        }
        "message_stop" => Some(Ok(StreamChunk::finished("stop"))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{OpenRouterProvider, Provider, TokenPrice};

    /// The accounting used to parse usage in these tests (OpenAI-wire shape).
    fn acct() -> OpenRouterProvider {
        OpenRouterProvider
    }

    #[test]
    fn parse_response_threads_tool_calls_and_finish_reason() {
        let raw = serde_json::json!({
            "id": "gen-1",
            "model": "test-model",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"}}]
                }
            }]
        });
        let resp = acct().parse_response(raw).unwrap();
        assert_eq!(resp.id, "gen-1");
        assert_eq!(resp.content, "");
        assert_eq!(resp.finish_reason.as_deref(), Some("tool_calls"));
        let tc = resp.tool_calls.expect("tool_calls threaded through");
        assert_eq!(tc[0].id, "call_1");
        assert_eq!(tc[0].name, "get_weather");
        assert_eq!(tc[0].arguments, "{}");
    }

    #[test]
    fn parse_response_rejects_malformed_tool_call_entry() {
        // An entry without id or function.name is unusable (no way to answer the
        // call); it must fail loudly, not produce a fabricated ToolCall.
        let raw = serde_json::json!({
            "id": "gen-1", "model": "m",
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "content": null,
                    "tool_calls": [{"type": "function", "function": {"arguments": "{}"}}]
                }
            }]
        });
        assert!(acct().parse_response(raw).is_err());
    }

    #[test]
    fn parse_response_surfaces_200_error_body_loudly() {
        // OpenRouter/OpenAI 200-with-error-body must become an Api error, not an
        // empty success.
        let raw = serde_json::json!({
            "error": {"message": "model overloaded", "code": 503}
        });
        let err = acct().parse_response(raw).unwrap_err();
        match err {
            crate::error::MiniLLMError::Api { status, message } => {
                assert_eq!(status, 503);
                assert_eq!(message, "model overloaded");
            }
            other => panic!("expected Api error, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_error_with_string_code_defaults_to_retryable_502() {
        // A non-numeric error code (e.g. "rate_limit_exceeded") must NOT collapse
        // to 200 (a fake non-retryable success); it becomes 502 (retryable).
        let raw = serde_json::json!({
            "error": {"message": "slow down", "code": "rate_limit_exceeded"}
        });
        match acct().parse_response(raw).unwrap_err() {
            crate::error::MiniLLMError::Api { status, .. } => assert_eq!(status, 502),
            other => panic!("expected Api error, got {other:?}"),
        }
        // An out-of-range numeric code also defaults to 502 (no u16 truncation).
        let raw = serde_json::json!({ "error": {"message": "x", "code": 999_999} });
        match acct().parse_response(raw).unwrap_err() {
            crate::error::MiniLLMError::Api { status, .. } => assert_eq!(status, 502),
            other => panic!("expected Api error, got {other:?}"),
        }
    }

    #[test]
    fn parse_response_rejects_malformed_missing_choices() {
        let raw = serde_json::json!({ "id": "gen-1", "model": "m" });
        assert!(acct().parse_response(raw).is_err());
    }

    #[test]
    fn openrouter_parses_usage_and_aggregates_byok_cost() {
        // The OpenRouter accounting parses its nested usage shape and sums the
        // fee + BYOK upstream charge in its own cost_of (the aggregation that must
        // stay provider-specific).
        let raw = serde_json::json!({
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                "cost": 0.001,
                "cost_details": {"upstream_inference_cost": 0.009},
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 2}
            }
        });
        let usage = acct().parse_usage(&raw).expect("usage parsed");
        assert_eq!(usage.prompt_tokens(), 10, "total input = sum of buckets");
        assert_eq!(
            usage.cache_read_tokens, 4,
            "cached_tokens → cache_read bucket"
        );
        assert_eq!(
            usage.uncached_input_tokens, 6,
            "10 total − 4 cached = 6 uncached"
        );
        assert_eq!(usage.upstream_inference_cost, Some(0.009));
        assert_eq!(usage.reasoning_tokens, Some(2));

        let outcome = acct().cost_of(usage, None);
        assert_eq!(outcome.resolution, CostResolution::Resolved);
        assert!((outcome.usd - 0.010).abs() < 1e-9);
    }

    #[test]
    fn openai_wire_splits_cache_read_as_subset_and_cache_write_as_additive() {
        // The two cache buckets sit DIFFERENTLY relative to prompt_tokens, and
        // getting it wrong mis-bills cache-heavy requests:
        //   - cached_tokens (READ) is a SUBSET of prompt_tokens → subtract it.
        //   - cache_write_tokens (WRITE) is ADDITIVE (billed on top, NOT in
        //     prompt_tokens) → do NOT subtract it.
        // prompt_tokens = 10000 (= 8000 uncached + 2000 cache-read); writes = 5000.
        let raw = serde_json::json!({
            "usage": {
                "prompt_tokens": 10000,
                "completion_tokens": 100,
                "prompt_tokens_details": {
                    "cached_tokens": 2000,
                    "cache_write_tokens": 5000
                }
            }
        });
        let usage = acct().parse_usage(&raw).expect("usage parsed");
        assert_eq!(usage.cache_read_tokens, 2000);
        assert_eq!(
            usage.cache_write_tokens, 5000,
            "write read from cache_write_tokens"
        );
        assert_eq!(
            usage.uncached_input_tokens, 8000,
            "subtract only the cache-read subset (10000 − 2000), NOT the write"
        );
        // Total input = the three disjoint buckets: 8000 + 2000 + 5000 = 15000.
        assert_eq!(
            usage.prompt_tokens(),
            15000,
            "writes are additive, so total input exceeds prompt_tokens"
        );

        // Pricing must reflect all four buckets at their own rates ($/Mtok): input
        // 3, read 0.3, write 3.75, output 15. 8000×3 + 2000×0.3 + 5000×3.75 +
        // 100×15 = 24000+600+18750+1500 = 44850 micro-$ ⇒ $0.04485. The buggy
        // `−(read+write)` split would yield uncached=3000 and undercharge the input.
        let price = TokenPrice::new(3.0, 15.0).with_cache_rates(0.3, 3.75);
        let usd = price.cost_of(&usage);
        assert!((usd - 0.04485).abs() < 1e-9, "got {usd}");
    }

    #[test]
    fn openai_wire_cached_exceeding_prompt_reports_unknown_not_a_fabricated_split() {
        // The disjoint split assumes cache READS are a subset of prompt_tokens. If a
        // wire violates that (cached > prompt), the split would be a silently-wrong
        // cost, so parse_usage must FAIL LOUDLY (return None → Unknown), not clamp.
        let raw = serde_json::json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 15}
            }
        });
        assert!(
            acct().parse_usage(&raw).is_none(),
            "cached > prompt must yield no usage (Unknown cost), not a clamped split"
        );

        // Boundary: cached == prompt is a valid subset (all input was a cache hit) →
        // uncached 0, not rejected.
        let raw = serde_json::json!({
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 10}
            }
        });
        let usage = acct().parse_usage(&raw).expect("cached == prompt is valid");
        assert_eq!(usage.uncached_input_tokens, 0);
        assert_eq!(usage.cache_read_tokens, 10);
    }

    #[test]
    fn error_object_ignores_benign_falsy_error_fields() {
        // A clean response/chunk with a benign falsy `error` (null, {}, false, 0, "")
        // must NOT be treated as a failure (which would fail a good stream and lose
        // an accepted generation's cost). Only a non-empty error OBJECT is an error.
        for benign in [
            serde_json::json!({"error": null}),
            serde_json::json!({"error": {}}),
            serde_json::json!({"error": false}),
            serde_json::json!({"error": 0}),
            serde_json::json!({"error": ""}),
            // A string/array `error` is not the real error envelope (every provider
            // sends a non-empty OBJECT); treat it as non-error, matching the wire.
            serde_json::json!({"error": "some string"}),
            serde_json::json!({"error": ["a", "b"]}),
            serde_json::json!({"id": "gen-1"}),
        ] {
            assert!(
                openai_error_in(&benign).is_none(),
                "benign error field must not be an error: {benign}"
            );
            assert!(anthropic_error_in(&benign).is_none());
        }
        // A real (non-empty) error object IS detected.
        let real = serde_json::json!({"error": {"message": "boom"}});
        assert!(openai_error_in(&real).is_some());
        assert!(anthropic_error_in(&real).is_some());
    }

    #[test]
    fn parse_stream_chunk_extracts_typed_tool_call_deltas() {
        // First delta carries index/id/name + an argument fragment; the second
        // continues the arguments. A delta without an index is skipped loudly.
        let c = acct()
            .parse_chunk(
                r#"{"id":"gen-1","choices":[{"delta":{"tool_calls":[
                    {"index":0,"id":"c0","type":"function",
                     "function":{"name":"search","arguments":"{\"q\":"}},
                    {"function":{"arguments":"ignored, no index"}}
                ]}}]}"#,
            )
            .unwrap()
            .unwrap();
        let deltas = c.tool_calls.expect("tool call deltas parsed");
        assert_eq!(deltas.len(), 1, "index-less delta skipped");
        assert_eq!(deltas[0].index, 0);
        assert_eq!(deltas[0].id.as_deref(), Some("c0"));
        assert_eq!(deltas[0].name.as_deref(), Some("search"));
        assert_eq!(deltas[0].arguments_fragment.as_deref(), Some("{\"q\":"));
    }

    #[test]
    fn parse_stream_chunk_done_marker() {
        let chunk = acct().parse_chunk("[DONE]").unwrap().unwrap();
        assert_eq!(chunk.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn parse_stream_chunk_extracts_real_generation_id() {
        // The chunk's top-level `id` must be threaded so cancellation cost
        // resolution targets the real generation, not a placeholder.
        let chunk = acct()
            .parse_chunk(r#"{"id":"gen-abc","choices":[{"delta":{"content":"hi"}}]}"#)
            .unwrap()
            .unwrap();
        assert_eq!(chunk.id.as_deref(), Some("gen-abc"));
        assert_eq!(chunk.delta, "hi");
    }

    #[test]
    fn openai_in_band_error_chunk_surfaces_as_err() {
        // A 200 stream that emits a top-level `{"error":...}` frame must become a
        // loud Err on the chunk path (not silently swallowed as None), so a failed
        // generation is never billed as accepted.
        let out = acct()
            .parse_chunk(r#"{"error":{"message":"overloaded","code":503}}"#)
            .expect("error frame must produce Some(Err), not None");
        match out {
            Err(crate::error::MiniLLMError::Api { status, message }) => {
                assert_eq!(status, 503);
                assert_eq!(message, "overloaded");
            }
            other => panic!("expected Some(Err(Api)), got {other:?}"),
        }
    }

    // ---- Anthropic envelope ---------------------------------------------------

    #[test]
    fn anthropic_response_joins_text_blocks_and_parses_usage() {
        let raw = serde_json::json!({
            "id": "msg_1",
            "model": "claude-haiku-4-5",
            "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 9, "output_tokens": 4, "cache_read_input_tokens": 2}
        });
        let resp = parse_anthropic_response(raw).unwrap();
        assert_eq!(resp.id, "msg_1");
        assert_eq!(resp.content, "Hello world");
        assert_eq!(resp.finish_reason.as_deref(), Some("end_turn"));
        let u = resp.usage.expect("usage parsed");
        // Anthropic's input_tokens (9) EXCLUDES cached; cache_read (2) is additive.
        assert_eq!(u.uncached_input_tokens, 9);
        assert_eq!(u.cache_read_tokens, 2);
        assert_eq!(u.cache_write_tokens, 0);
        assert_eq!(
            u.prompt_tokens(),
            11,
            "total input = 9 uncached + 2 cache-read"
        );
        assert_eq!(u.completion_tokens, 4);
        assert_eq!(u.total_tokens(), 15);
        assert!(u.cost.is_none(), "Anthropic never returns a dollar cost");
    }

    #[test]
    fn anthropic_response_threads_tool_use_blocks() {
        let raw = serde_json::json!({
            "id": "msg_2", "model": "m",
            "content": [
                {"type": "text", "text": "calling"},
                {"type": "tool_use", "id": "tu_1", "name": "get_weather",
                 "input": {"city": "Paris"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 2}
        });
        let resp = parse_anthropic_response(raw).unwrap();
        assert_eq!(resp.content, "calling");
        let tc = resp.tool_calls.expect("tool_use threaded");
        assert_eq!(tc[0].id, "tu_1");
        assert_eq!(tc[0].name, "get_weather");
        // input is serialized to raw JSON text (the normalized argument form).
        assert_eq!(tc[0].arguments, r#"{"city":"Paris"}"#);
    }

    #[test]
    fn anthropic_chunk_tool_use_block_start_and_json_deltas() {
        // content_block_start (tool_use) carries the call's id/name at its block
        // index; input_json_delta events carry argument fragments at that index.
        let start = parse_anthropic_chunk(
            r#"{"type":"content_block_start","index":1,
                "content_block":{"type":"tool_use","id":"tu_1","name":"get_weather","input":{}}}"#,
        )
        .unwrap()
        .unwrap();
        let d = start.tool_calls.expect("tool_use start mapped");
        assert_eq!(d[0].index, 1);
        assert_eq!(d[0].id.as_deref(), Some("tu_1"));
        assert_eq!(d[0].name.as_deref(), Some("get_weather"));

        let frag = parse_anthropic_chunk(
            r#"{"type":"content_block_delta","index":1,
                "delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}"#,
        )
        .unwrap()
        .unwrap();
        let d = frag.tool_calls.expect("json delta mapped");
        assert_eq!(d[0].index, 1);
        assert_eq!(d[0].arguments_fragment.as_deref(), Some("{\"city\":"));

        // A TEXT content_block_start stays ignorable.
        assert!(parse_anthropic_chunk(
            r#"{"type":"content_block_start","index":0,
                "content_block":{"type":"text","text":""}}"#
        )
        .is_none());
    }

    #[test]
    fn anthropic_response_surfaces_error_body_loudly() {
        let raw = serde_json::json!({"type": "error",
            "error": {"type": "overloaded_error", "message": "overloaded"}});
        match parse_anthropic_response(raw).unwrap_err() {
            crate::error::MiniLLMError::Api { message, .. } => assert_eq!(message, "overloaded"),
            other => panic!("expected Api error, got {other:?}"),
        }
    }

    #[test]
    fn anthropic_response_rejects_missing_content() {
        let raw = serde_json::json!({"id": "x", "model": "m"});
        assert!(parse_anthropic_response(raw).is_err());
    }

    #[test]
    fn anthropic_chunk_message_start_carries_id_and_input_usage() {
        let c = parse_anthropic_chunk(
            r#"{"type":"message_start","message":{"id":"msg_9","usage":{"input_tokens":15,"output_tokens":1}}}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(c.id.as_deref(), Some("msg_9"));
        assert_eq!(c.usage.as_ref().unwrap().uncached_input_tokens, 15);
    }

    #[test]
    fn anthropic_chunk_content_delta_carries_text() {
        let c = parse_anthropic_chunk(
            r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(c.delta, "hi");
        // Non-text events produce nothing.
        assert!(parse_anthropic_chunk(r#"{"type":"content_block_start"}"#).is_none());
        assert!(parse_anthropic_chunk(r#"{"type":"ping"}"#).is_none());
    }

    #[test]
    fn anthropic_chunk_message_delta_carries_stop_and_output_usage() {
        let c = parse_anthropic_chunk(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":9}}"#,
        )
        .unwrap()
        .unwrap();
        assert_eq!(c.finish_reason.as_deref(), Some("end_turn"));
        assert_eq!(c.usage.as_ref().unwrap().completion_tokens, 9);
        // message_stop terminates.
        let stop = parse_anthropic_chunk(r#"{"type":"message_stop"}"#)
            .unwrap()
            .unwrap();
        assert_eq!(stop.finish_reason.as_deref(), Some("stop"));
    }

    #[test]
    fn anthropic_in_band_error_event_surfaces_as_err() {
        // The exact production failure: a 200 stream emitting an `error` event must
        // become a loud Err (not the old silent `_ => None`), so cost accounting
        // sees the failure and books nothing. Mirrors parse_anthropic_response.
        let out = parse_anthropic_chunk(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"overloaded"}}"#,
        )
        .expect("error event must produce Some(Err), not None");
        match out {
            Err(crate::error::MiniLLMError::Api { message, .. }) => {
                assert_eq!(message, "overloaded")
            }
            other => panic!("expected Some(Err(Api)), got {other:?}"),
        }
    }

    #[test]
    fn usage_merge_accumulates_split_input_and_output() {
        // Anthropic splits usage: input in message_start, output in message_delta.
        // merge_from must keep both and recompute the total.
        let mut acc = Usage {
            uncached_input_tokens: 15,
            completion_tokens: 1,
            ..Default::default()
        };
        let delta = Usage {
            uncached_input_tokens: 0,
            completion_tokens: 9,
            ..Default::default()
        };
        acc.merge_from(&delta);
        assert_eq!(
            acc.uncached_input_tokens, 15,
            "input from message_start preserved"
        );
        assert_eq!(
            acc.completion_tokens, 9,
            "output from message_delta applied"
        );
        assert_eq!(
            acc.total_tokens(),
            24,
            "total recomputed from merged buckets"
        );
    }
}
