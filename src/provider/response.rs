//! Response types from LLM APIs

use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Token usage information
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    #[serde(default)]
    pub prompt_tokens: u32,

    /// Number of tokens in the completion
    #[serde(default)]
    pub completion_tokens: u32,

    /// Total tokens used
    #[serde(default)]
    pub total_tokens: u32,

    /// Cost in credits (OpenRouter specific)
    #[serde(default)]
    pub cost: Option<f64>,

    /// Cached tokens (tokens read from cache)
    #[serde(default)]
    pub cached_tokens: Option<u32>,

    /// Reasoning tokens (for models that support it)
    #[serde(default)]
    pub reasoning_tokens: Option<u32>,
}

/// Detailed cost information from a completion
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostInfo {
    /// Total cost in credits charged to your account
    pub cost: f64,

    /// Number of prompt tokens
    pub prompt_tokens: u32,

    /// Number of completion tokens
    pub completion_tokens: u32,

    /// Total tokens
    pub total_tokens: u32,

    /// Cached tokens (if any)
    pub cached_tokens: Option<u32>,

    /// Reasoning tokens (if any)
    pub reasoning_tokens: Option<u32>,

    /// The model used
    pub model: String,

    /// Response ID for tracking
    pub response_id: String,
}

/// Type of cost tracking to use
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CostTrackingType {
    /// No cost tracking
    #[default]
    None,
    /// OpenRouter's usage accounting (adds `usage: { include: true }` to request)
    OpenRouter,
}

/// Callback function type for cost ingestion
/// Called with CostInfo after each successful completion
pub type CostCallback = Arc<dyn Fn(CostInfo) + Send + Sync>;

/// A complete response from an LLM API
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Tool calls made by the model (if any)
    pub tool_calls: Option<Vec<serde_json::Value>>,

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
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// The delta content in this chunk
    pub delta: String,

    /// Finish reason (only present in final chunk)
    pub finish_reason: Option<String>,

    /// Usage info (only present in final chunk for some providers)
    pub usage: Option<Usage>,

    /// Tool call deltas
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

impl StreamChunk {
    /// Create a new stream chunk with content
    pub fn content(delta: impl Into<String>) -> Self {
        Self {
            delta: delta.into(),
            finish_reason: None,
            usage: None,
            tool_calls: None,
        }
    }

    /// Create a final chunk with finish reason
    pub fn finished(finish_reason: impl Into<String>) -> Self {
        Self {
            delta: String::new(),
            finish_reason: Some(finish_reason.into()),
            usage: None,
            tool_calls: None,
        }
    }

    /// Check if this is the final chunk
    pub fn is_final(&self) -> bool {
        self.finish_reason.is_some()
    }
}

/// Parse a raw API response into a CompletionResponse
pub fn parse_completion_response(
    raw: serde_json::Value,
) -> crate::error::Result<CompletionResponse> {
    let id = raw["id"].as_str().unwrap_or("").to_string();
    let model = raw["model"].as_str().unwrap_or("").to_string();

    // Extract content from choices
    let content = raw["choices"]
        .get(0)
        .and_then(|c| c["message"]["content"].as_str())
        .unwrap_or("")
        .to_string();

    let finish_reason = raw["choices"]
        .get(0)
        .and_then(|c| c["finish_reason"].as_str())
        .map(String::from);

    // Parse usage
    let usage = raw.get("usage").map(|u| Usage {
        prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
        total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
        cost: u["cost"].as_f64(),
        cached_tokens: u["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .map(|v| v as u32),
        reasoning_tokens: u["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .map(|v| v as u32),
    });

    // Extract tool calls
    let tool_calls = raw["choices"]
        .get(0)
        .and_then(|c| c["message"]["tool_calls"].as_array())
        .cloned();

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

/// Parse a streaming chunk from SSE data
pub fn parse_stream_chunk(data: &str) -> Option<StreamChunk> {
    // Handle [DONE] marker
    if data.trim() == "[DONE]" {
        return Some(StreamChunk::finished("stop"));
    }

    // Parse JSON
    let json: serde_json::Value = serde_json::from_str(data).ok()?;

    // Parse usage if present (OpenRouter sends this in the last chunk, possibly without choices)
    let usage = json.get("usage").and_then(|u| {
        if u.is_null() {
            None
        } else {
            Some(Usage {
                prompt_tokens: u["prompt_tokens"].as_u64().unwrap_or(0) as u32,
                completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
                total_tokens: u["total_tokens"].as_u64().unwrap_or(0) as u32,
                cost: u["cost"].as_f64(),
                cached_tokens: u["prompt_tokens_details"]["cached_tokens"]
                    .as_u64()
                    .map(|v| v as u32),
                reasoning_tokens: u["completion_tokens_details"]["reasoning_tokens"]
                    .as_u64()
                    .map(|v| v as u32),
            })
        }
    });

    // Try to get choice (may not be present in usage-only chunks)
    let choice = json["choices"].get(0);

    // Get delta content
    let delta = choice
        .and_then(|c| c["delta"]["content"].as_str())
        .unwrap_or("")
        .to_string();

    let finish_reason = choice
        .and_then(|c| c["finish_reason"].as_str())
        .filter(|s| !s.is_empty())
        .map(String::from);

    // Extract tool call deltas
    let tool_calls = choice.and_then(|c| c["delta"]["tool_calls"].as_array().cloned());

    // Return chunk if we have any content, finish reason, or usage
    if delta.is_empty() && finish_reason.is_none() && usage.is_none() && tool_calls.is_none() {
        return None;
    }

    Some(StreamChunk {
        delta,
        finish_reason,
        usage,
        tool_calls,
    })
}
