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

    /// Media the model RETURNED (an image-generation model's output),
    /// normalized to the same typed media the request side uses, so a
    /// caller appends them to a conversation (see
    /// [`Self::to_assistant_message`]) or stores them without touching
    /// the raw envelope. Parsed from the OpenAI-wire `message.images`
    /// entries (OpenRouter's normalized field for image output); empty
    /// for text-only completions and wires that return no media.
    /// Serde-skipped like `raw_response`: `Media` is transient in-code
    /// material, not a persistence shape.
    #[serde(skip)]
    pub media: Vec<crate::message::Media>,

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
            media: Vec::new(),
            raw_response: None,
        }
    }

    /// Build the ASSISTANT [`Message`](crate::message::Message) this
    /// completion appends to a conversation: plain text content when the
    /// model returned no media, multimodal parts (text first, then each
    /// returned media) otherwise, with the completion's tool calls
    /// carried over. The one canonical response-to-history conversion,
    /// so callers never assemble it by hand.
    pub fn to_assistant_message(&self) -> crate::message::Message {
        let mut message = if self.media.is_empty() {
            crate::message::Message::assistant(self.content.clone())
        } else {
            let mut parts = Vec::new();
            if !self.content.is_empty() {
                parts.push(crate::message::ContentPart::text(self.content.clone()));
            }
            parts.extend(
                self.media
                    .iter()
                    .map(crate::message::ContentPart::from_media),
            );
            crate::message::Message::assistant(crate::message::MessageContent::parts(parts))
        };
        message.tool_calls = self.tool_calls.clone();
        message
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
pub(crate) fn error_object(raw: &serde_json::Value) -> Option<&serde_json::Value> {
    raw.get("error")
        .filter(|e| e.as_object().is_some_and(|o| !o.is_empty()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_response_becomes_a_text_message() {
        let resp = CompletionResponse::new("id", "m", "plain answer");
        let message = resp.to_assistant_message();
        assert_eq!(message.text(), Some("plain answer"));
        assert!(matches!(
            message.content,
            crate::message::MessageContent::Text(_)
        ));
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
                error_object(&benign).is_none(),
                "benign error field must not be an error: {benign}"
            );
        }
        // A real (non-empty) error object IS detected.
        let real = serde_json::json!({"error": {"message": "boom"}});
        assert!(error_object(&real).is_some());
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
