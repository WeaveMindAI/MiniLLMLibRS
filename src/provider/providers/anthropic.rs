//! Anthropic: every piece of Anthropic-specific wire knowledge, in one
//! place: the native `/v1/messages` request (system hoisted to a
//! top-level field, alternating merged turns, required `max_tokens`,
//! tool_use/tool_result blocks, `cache_control` markers), the
//! `content[]` response envelope, the SSE event stream, the disjoint
//! usage mapping, and the error envelope.

use super::super::auth::Auth;
use super::super::response::{error_object, preview_str, CompletionResponse, StreamChunk, Usage};
use super::super::wire::{
    kept_cache_breakpoints, price_or_unpriced, CostOutcome, Provider, TokenPrice,
};
use crate::error::{MiniLLMError, Result};
use crate::generator::CompletionParameters;
use crate::message::Message;
use crate::tools::{ToolCall, ToolCallDelta};
use secrecy::ExposeSecret;

/// A tool definition's `/v1/messages` shape: `{name, description, input_schema}`.
fn tool_definition_value(def: &crate::tools::ToolDefinition) -> serde_json::Value {
    let mut tool = serde_json::json!({
        "name": def.name,
        "input_schema": def.parameters,
    });
    if let Some(desc) = &def.description {
        tool["description"] = serde_json::json!(desc);
    }
    if let Some(strict) = def.strict {
        tool["strict"] = serde_json::json!(strict);
    }
    tool
}

/// A tool choice's `/v1/messages` value. `disable_parallel_tool_use` is folded
/// in by the request builder (it lives inside this object on this wire), not
/// here.
fn tool_choice_value(choice: &crate::tools::ToolChoice) -> serde_json::Value {
    use crate::tools::ToolChoice;
    match choice {
        ToolChoice::Auto => serde_json::json!({ "type": "auto" }),
        ToolChoice::None => serde_json::json!({ "type": "none" }),
        ToolChoice::Required => serde_json::json!({ "type": "any" }),
        ToolChoice::Tool(name) => serde_json::json!({ "type": "tool", "name": name }),
    }
}

/// An assistant tool call as a `tool_use` content block. Parses the raw
/// argument text (this wire's `input` is a JSON object, not a string),
/// failing loudly on invalid JSON.
fn tool_use_block(call: &ToolCall) -> Result<serde_json::Value> {
    Ok(serde_json::json!({
        "type": "tool_use",
        "id": call.id,
        "name": call.name,
        "input": call.arguments_json()?,
    }))
}

/// Anthropic's native Messages API. A DIFFERENT wire envelope from OpenAI:
/// `/v1/messages` (not `/chat/completions`), `system` is a top-level field (not a
/// role=system message), `max_tokens` is required, the response is `content[]`
/// blocks (not `choices[]`), and usage is `input_tokens`/`output_tokens` (no
/// dollar cost, price via `TokenPrice`, like OpenAI). Auth is `x-api-key` for an
/// API key, or `Authorization: Bearer` for a subscription OAuth token.
#[derive(Debug, Clone, Default)]
pub struct AnthropicProvider;

/// The Messages API version pin (a date string Anthropic requires on every call).
const ANTHROPIC_VERSION: &str = "2023-06-01";

impl AnthropicProvider {
    /// The message's full text, FAILING LOUDLY on multimodal content (image/audio/
    /// video) which has no Anthropic mapping wired yet. `all_text()` joins every
    /// text part, so a multi-text message never silently drops its later parts the
    /// way `get_text()` (first part only) would. Shared by the turn and system paths.
    ///
    /// Multimodal content (image/audio/video parts) has no Anthropic mapping wired
    /// yet, and Anthropic's block shape differs from the OpenAI-shaped normalized
    /// parts, so a multimodal message FAILS LOUDLY rather than silently shipping a
    /// text-only request that drops the attachment. (Wiring Anthropic image/document
    /// blocks is a clean future extension.)
    fn text_only(msg: &Message) -> Result<String> {
        use crate::message::MessageContent;
        if let MessageContent::Parts(parts) = &msg.content {
            if parts.iter().any(|p| p.as_text().is_none()) {
                return Err(MiniLLMError::InvalidParameter(
                    "the Anthropic provider does not yet support multimodal content (image/audio/video); send text-only messages or use an OpenAI-wire provider".to_string(),
                ));
            }
        }
        Ok(msg.content.all_text())
    }

    /// Map one non-system message to its Anthropic turn: the wire role
    /// (`user`/`assistant`; a tool RESULT is a `user` turn on this wire) plus its
    /// content blocks:
    /// - assistant `tool_calls` become `tool_use` blocks after any text,
    /// - a `Role::Tool` message becomes a `tool_result` block (requiring its
    ///   `tool_call_id`, failing loudly without one),
    /// - `cached` puts a `cache_control` marker on the message's last block.
    fn turn_blocks(msg: &Message, cached: bool) -> Result<(&'static str, Vec<serde_json::Value>)> {
        use crate::message::Role;
        let text = Self::text_only(msg)?;
        let mut blocks: Vec<serde_json::Value> = Vec::new();

        let role = match msg.role {
            Role::Tool => {
                let Some(call_id) = &msg.tool_call_id else {
                    return Err(MiniLLMError::InvalidParameter(
                        "a tool-result message needs a tool_call_id (build it via Message::tool)"
                            .to_string(),
                    ));
                };
                blocks.push(serde_json::json!({
                    "type": "tool_result",
                    "tool_use_id": call_id,
                    "content": text,
                }));
                "user"
            }
            role => {
                // A text block only when there is text OR nothing else to say
                // (Anthropic rejects empty text blocks, but an all-empty message
                // still needs a body).
                if !text.is_empty() || msg.tool_calls.is_none() {
                    blocks.push(serde_json::json!({ "type": "text", "text": text }));
                }
                if let Some(calls) = &msg.tool_calls {
                    for call in calls {
                        blocks.push(tool_use_block(call)?);
                    }
                }
                if role == Role::Assistant {
                    "assistant"
                } else {
                    "user"
                }
            }
        };

        if cached {
            let last = blocks
                .last_mut()
                .expect("every turn has at least one block");
            last["cache_control"] = serde_json::json!({ "type": "ephemeral" });
        }
        Ok((role, blocks))
    }
}

impl Provider for AnthropicProvider {
    fn openrouter_slug(&self) -> Option<&'static str> {
        Some("anthropic")
    }

    fn endpoint_url(&self, base_url: &str) -> String {
        format!("{}/v1/messages", base_url.trim_end_matches('/'))
    }

    /// Anthropic allows at most 4 `cache_control` breakpoints per request.
    fn max_cache_breakpoints(&self) -> usize {
        4
    }

    /// `x-api-key` for an API key; `Authorization: Bearer` for a subscription
    /// token. `anthropic-version` is always sent; the `oauth-2025-04-20` beta is
    /// added on the bearer path (harmless, and future-proofs the OAuth route).
    fn auth_headers(&self, auth: &Auth) -> Result<Vec<(String, String)>> {
        let mut headers = vec![(
            "anthropic-version".to_string(),
            ANTHROPIC_VERSION.to_string(),
        )];
        match auth {
            Auth::ApiKey(k) => {
                headers.push(("x-api-key".to_string(), k.expose_secret().to_string()));
            }
            Auth::BearerToken(t) => {
                headers.push((
                    "Authorization".to_string(),
                    format!("Bearer {}", t.expose_secret()),
                ));
                headers.push(("anthropic-beta".to_string(), "oauth-2025-04-20".to_string()));
            }
            Auth::None => {}
        }
        Ok(headers)
    }

    /// Build the `/v1/messages` body: hoist system message(s) to the top-level
    /// `system` field, map the rest to user/assistant turns, require `max_tokens`
    /// (Anthropic rejects a request without it, so fall back to the params' default),
    /// and carry the sampling params Anthropic accepts plus merged `extra`.
    ///
    /// Prompt caching: a [`Message::cache_breakpoint`] becomes a `cache_control`
    /// marker on that block. Anthropic allows at most 4 breakpoints per request, so
    /// if more are marked we keep the LAST 4 (the most-recent prefix, the biggest
    /// reusable span) and warn. A marked block is emitted in the block-array form
    /// (a plain string can't carry the marker).
    fn build_request(
        &self,
        model: &str,
        messages: &[Message],
        params: &CompletionParameters,
        stream: bool,
        _include_usage: bool,
    ) -> Result<serde_json::Value> {
        use crate::message::Role;

        // Fail loudly on normalized fields with no Anthropic mapping wired yet,
        // rather than silently dropping them. Each is a clean future translation,
        // not a silent "for now" omission.
        for (present, field) in [
            (params.response_format.is_some(), "response_format"),
            (params.reasoning.is_some(), "reasoning"),
        ] {
            if present {
                return Err(MiniLLMError::InvalidParameter(format!(
                    "the Anthropic provider does not yet translate `{field}`; omit it or use an OpenAI-wire provider"
                )));
            }
        }

        // Enforce the wire's breakpoint cap: of all marked messages, only
        // the last `max_cache_breakpoints` actually get a marker.
        let kept = kept_cache_breakpoints(messages, self.max_cache_breakpoints());

        // System turns are hoisted. Track whether any hoisted system message is a
        // (kept) breakpoint so the system block carries the marker.
        //
        // Non-system messages become (role, blocks) turns; consecutive turns with
        // the same wire role are merged (Anthropic requires alternating roles, and
        // parallel tool results MUST share one user turn with their tool_result
        // blocks together, immediately after the assistant's tool_use turn).
        let mut system = String::new();
        let mut system_cached = false;
        let mut turns: Vec<(&'static str, Vec<serde_json::Value>)> = Vec::new();
        for (i, msg) in messages.iter().enumerate() {
            let cached = kept.contains(&i);
            if msg.role == Role::System {
                // Run the system text through the same all-text + multimodal guard
                // as a turn, so a multi-text or multimodal system message can't
                // silently drop content here either.
                let text = Self::text_only(msg)?;
                if !system.is_empty() {
                    system.push('\n');
                }
                system.push_str(&text);
                system_cached |= cached;
            } else {
                let (role, blocks) = Self::turn_blocks(msg, cached)?;
                match turns.last_mut() {
                    Some((last_role, last_blocks)) if *last_role == role => {
                        last_blocks.extend(blocks)
                    }
                    _ => turns.push((role, blocks)),
                }
            }
        }
        // A single plain text block collapses to the string form (the common
        // no-tools wire); anything richer keeps the block array.
        let turns: Vec<serde_json::Value> = turns
            .into_iter()
            .map(|(role, blocks)| {
                let content = match blocks.as_slice() {
                    [only] if only["type"] == "text" && only.get("cache_control").is_none() => {
                        only["text"].clone()
                    }
                    _ => serde_json::json!(blocks),
                };
                serde_json::json!({ "role": role, "content": content })
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model,
            "messages": turns,
            "stream": stream,
            // Anthropic REQUIRES max_tokens. The params default (4096) is the floor
            // when the caller leaves it unset.
            "max_tokens": params.max_tokens.unwrap_or(4096),
        });
        if !system.is_empty() {
            // A cached system uses the block-array form so it can carry the marker.
            body["system"] = if system_cached {
                serde_json::json!([{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }])
            } else {
                serde_json::json!(system)
            };
        }
        // Sampling params Anthropic accepts (it ignores OpenAI-only ones).
        if let Some(t) = params.temperature {
            body["temperature"] = serde_json::json!(t);
        }
        if let Some(p) = params.top_p {
            body["top_p"] = serde_json::json!(p);
        }
        if let Some(k) = params.top_k {
            body["top_k"] = serde_json::json!(k);
        }
        if let Some(stop) = &params.stop {
            body["stop_sequences"] = serde_json::json!(stop);
        }

        // Tools: normalized definitions → Anthropic's {name, description,
        // input_schema} shape.
        if let Some(tools) = &params.tools {
            body["tools"] =
                serde_json::Value::Array(tools.iter().map(tool_definition_value).collect());
        }
        // tool_choice: Anthropic carries the parallel-calls setting INSIDE this
        // object (`disable_parallel_tool_use`), so `parallel_tool_calls: false`
        // forces a tool_choice (defaulting to auto) to have somewhere to live.
        // `Some(true)` is the wire default and emits nothing extra; on a `None`
        // choice (tool calling forbidden) the flag is meaningless and omitted.
        let choice = match (&params.tool_choice, params.parallel_tool_calls) {
            (Some(c), _) => Some(c.clone()),
            (None, Some(false)) => Some(crate::tools::ToolChoice::Auto),
            (None, _) => None,
        };
        if let Some(choice) = choice {
            let mut value = tool_choice_value(&choice);
            if params.parallel_tool_calls == Some(false) && choice != crate::tools::ToolChoice::None
            {
                value["disable_parallel_tool_use"] = serde_json::json!(true);
            }
            body["tool_choice"] = value;
        }

        // Merge `extra`, rejecting collisions with a reserved key loudly.
        if let (Some(extra), Some(obj)) = (params.extra.clone(), body.as_object_mut()) {
            for (key, value) in extra {
                if obj.contains_key(&key) {
                    return Err(MiniLLMError::InvalidParameter(format!(
                        "extra param '{}' collides with a built-in Anthropic request key",
                        key
                    )));
                }
                obj.insert(key, value);
            }
        }

        Ok(body)
    }

    /// Parse the `content[]` envelope: join text blocks, map `stop_reason`, parse
    /// `usage` (token counts only). Surfaces an `error` object loudly.
    fn parse_response(&self, raw: serde_json::Value) -> Result<CompletionResponse> {
        parse_anthropic_response(raw)
    }

    /// Parse one Anthropic SSE event into a `StreamChunk` (or surface an in-band
    /// `error` event as a loud `Err`).
    fn parse_chunk(&self, data: &str) -> Option<Result<StreamChunk>> {
        parse_anthropic_chunk(data)
    }

    /// Anthropic always sends a trailing `message_delta` carrying final usage, so
    /// when tracking we WAIT for it (unlike a bare OpenAI server).
    fn emits_stream_usage(&self, requested: bool) -> bool {
        requested
    }

    /// Token-only, like OpenAI: derive cost from `TokenPrice` or report `Unpriced`
    /// (Anthropic returns no dollar amount, on either API-key or subscription auth).
    fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
        price_or_unpriced(usage, price)
    }
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
        // Anthropic's chat wire returns no media blocks today.
        media: Vec::new(),
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

// ── Claude subscription credentials ─────────────────────────────────

/// Resolve a Claude **subscription** bearer token, env superseding the on-disk
/// Claude Code credential.
///
/// 1. `ANTHROPIC_AUTH_TOKEN` if set and non-empty (explicit override; the caller
///    is responsible for keeping it fresh, e.g. from `claude setup-token`);
/// 2. else the Claude Code credential at `~/.claude/.credentials.json`
///    (`claudeAiOauth.accessToken`), the live Pro/Max subscription token, which
///    Claude Code refreshes on disk, so reading it each call always gets a current
///    token without the library having to manage OAuth refresh.
///
/// Returns [`Auth::None`] when neither source yields a token (the request then
/// fails loudly as unauthenticated, rather than silently using the wrong account).
pub fn resolve_claude_subscription_auth() -> Auth {
    if let Ok(token) = std::env::var("ANTHROPIC_AUTH_TOKEN") {
        if !token.trim().is_empty() {
            return Auth::BearerToken(secrecy::SecretString::from(token));
        }
    }
    match dirs_home().map(|h| h.join(".claude/.credentials.json")) {
        Some(path) => match read_claude_code_token(&path) {
            Some(token) => Auth::BearerToken(secrecy::SecretString::from(token)),
            None => Auth::None,
        },
        None => Auth::None,
    }
}

/// The user's home directory (`$HOME`), or `None` if unset.
fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Extract `claudeAiOauth.accessToken` from a Claude Code credentials file.
/// Pure over the file contents (the path is read here, parsing is in
/// [`parse_claude_code_token`]) so the parse is unit-testable without a real file.
fn read_claude_code_token(path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    parse_claude_code_token(&contents)
}

/// Parse the subscription access token out of a Claude Code credentials JSON body.
fn parse_claude_code_token(contents: &str) -> Option<String> {
    let json: serde_json::Value = serde_json::from_str(contents).ok()?;
    json["claudeAiOauth"]["accessToken"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::CostResolution;

    #[test]
    fn definition_anthropic_wire_shape() {
        let v = tool_definition_value(&weather_tool());
        assert_eq!(v["name"], "get_weather");
        assert_eq!(v["input_schema"]["type"], "object");
        assert!(v.get("strict").is_none(), "strict omitted when unset");
        // OpenAI-only keys must not leak.
        assert!(v.get("type").is_none());
        assert!(v.get("parameters").is_none());
    }

    #[test]
    fn choice_anthropic_wire_values() {
        assert_eq!(tool_choice_value(&ToolChoice::Auto)["type"], "auto");
        assert_eq!(tool_choice_value(&ToolChoice::None)["type"], "none");
        // OpenAI "required" is Anthropic "any".
        assert_eq!(tool_choice_value(&ToolChoice::Required)["type"], "any");
        let forced = tool_choice_value(&ToolChoice::Tool("get_weather".into()));
        assert_eq!(forced["type"], "tool");
        assert_eq!(forced["name"], "get_weather");
    }

    #[test]
    fn call_anthropic_block_parses_arguments_to_object() {
        let b = tool_use_block(&ToolCall::new("c1", "get_weather", r#"{"city":"Paris"}"#)).unwrap();
        assert_eq!(b["type"], "tool_use");
        assert_eq!(b["id"], "c1");
        assert_eq!(b["name"], "get_weather");
        assert_eq!(b["input"]["city"], "Paris");
        assert!(b["input"].is_object(), "input is an object, not a string");
        // Invalid argument text fails loudly instead of shipping garbage.
        assert!(tool_use_block(&ToolCall::new("c2", "t", "{bad")).is_err());
    }
    /// A fully-uncached usage (all input in the full-price bucket).
    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            uncached_input_tokens: prompt,
            completion_tokens: completion,
            ..Default::default()
        }
    }

    #[test]
    fn anthropic_endpoint_is_v1_messages() {
        let p = AnthropicProvider;
        assert_eq!(
            p.endpoint_url("https://api.anthropic.com"),
            "https://api.anthropic.com/v1/messages"
        );
        // Trailing slash normalized.
        assert_eq!(
            p.endpoint_url("https://api.anthropic.com/"),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn anthropic_auth_headers_api_key_vs_bearer() {
        let p = AnthropicProvider;
        // API key → x-api-key (+ version), NOT Authorization.
        let h = p.auth_headers(&Auth::ApiKey("sk-ant-key".into())).unwrap();
        assert!(h.iter().any(|(k, v)| k == "x-api-key" && v == "sk-ant-key"));
        assert!(h.iter().any(|(k, _)| k == "anthropic-version"));
        assert!(!h.iter().any(|(k, _)| k == "Authorization"));

        // Subscription bearer → Authorization: Bearer (+ version + oauth beta).
        let h = p
            .auth_headers(&Auth::BearerToken("sk-ant-oat01-tok".into()))
            .unwrap();
        assert!(h
            .iter()
            .any(|(k, v)| k == "Authorization" && v == "Bearer sk-ant-oat01-tok"));
        assert!(h.iter().any(|(k, _)| k == "anthropic-version"));
        assert!(h
            .iter()
            .any(|(k, v)| k == "anthropic-beta" && v == "oauth-2025-04-20"));
        assert!(!h.iter().any(|(k, _)| k == "x-api-key"));
    }

    #[test]
    fn anthropic_build_request_hoists_system_and_requires_max_tokens() {
        let p = AnthropicProvider;
        let messages = vec![
            Message::system("You are terse."),
            Message::user("Hi"),
            Message::assistant("Hello."),
            Message::user("Bye"),
        ];
        let params = CompletionParameters::new().with_temperature(0.5);
        let body = p
            .build_request("claude-haiku-4-5", &messages, &params, false, true)
            .unwrap();

        // System hoisted to top-level, NOT a message.
        assert_eq!(body["system"], "You are terse.");
        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3, "system turn is hoisted out of messages");
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "Hi");
        assert_eq!(msgs[1]["role"], "assistant");
        // max_tokens present (Anthropic requires it); defaults to params default.
        assert_eq!(body["max_tokens"], 4096);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["model"], "claude-haiku-4-5");
        assert_eq!(body["stream"], false);
    }

    #[test]
    fn anthropic_build_request_respects_explicit_max_tokens_and_stop() {
        let p = AnthropicProvider;
        let messages = vec![Message::user("Hi")];
        let params = CompletionParameters::new()
            .with_max_tokens(64)
            .with_stop(vec!["END".to_string()]);
        let body = p
            .build_request("m", &messages, &params, true, true)
            .unwrap();
        assert_eq!(body["max_tokens"], 64);
        // OpenAI `stop` maps to Anthropic `stop_sequences`.
        assert_eq!(body["stop_sequences"][0], "END");
        assert_eq!(body["stream"], true);
        // No system field when there's no system message.
        assert!(body.get("system").is_none());
    }

    #[test]
    fn anthropic_build_request_rejects_extra_collision() {
        let p = AnthropicProvider;
        let messages = vec![Message::user("Hi")];
        // `model` collides with a reserved key → loud error.
        let params = CompletionParameters::new().with_extra("model", serde_json::json!("x"));
        assert!(p
            .build_request("m", &messages, &params, false, true)
            .is_err());
        // A genuinely-extra key is fine.
        let params =
            CompletionParameters::new().with_extra("metadata", serde_json::json!({"user": "u1"}));
        assert!(p
            .build_request("m", &messages, &params, false, true)
            .is_ok());
    }

    #[test]
    fn anthropic_build_request_fails_loudly_on_every_untranslated_field() {
        // EVERY field the rejection loop guards must fail loudly, never be silently
        // dropped. One assertion per field, so removing any entry from the
        // production list fails this test.
        use crate::generator::ReasoningConfig;
        let p = AnthropicProvider;
        let messages = vec![Message::user("Hi")];

        let cases: Vec<(&str, CompletionParameters)> = vec![
            (
                "response_format",
                CompletionParameters::new().with_json_response(),
            ),
            (
                "reasoning",
                CompletionParameters::new().with_reasoning(ReasoningConfig {
                    effort: Some("high".into()),
                    max_tokens: None,
                    exclude: None,
                }),
            ),
        ];
        for (field, params) in cases {
            assert!(
                p.build_request("m", &messages, &params, false, true)
                    .is_err(),
                "{field} must fail loudly, not vanish"
            );
        }
    }

    #[test]
    fn anthropic_build_request_fails_loudly_on_multimodal_content() {
        // A message with an image attachment must error rather than ship a
        // text-only request that silently drops the image.
        use crate::message::{ImageData, MessageContent};
        let p = AnthropicProvider;
        let img = ImageData::from_url("https://example.com/x.png");
        let mut msg = Message::user("look at this");
        msg.content = MessageContent::with_images("look at this", &[img]);
        assert!(p
            .build_request("m", &[msg], &CompletionParameters::new(), false, true)
            .is_err());
    }

    #[test]
    fn anthropic_build_request_keeps_all_text_parts_of_a_multitext_message() {
        // A message stored as multiple TEXT parts (e.g. built via merge) must send
        // ALL its text, not just the first part. get_text() would drop the rest.
        use crate::message::{ContentPart, MessageContent, Role};
        let p = AnthropicProvider;
        let mut user = Message::user("");
        user.content = MessageContent::Parts(vec![
            ContentPart::text("first"),
            ContentPart::text("second"),
        ]);
        // Same for a multi-text system message (hoisted via the system path).
        let mut system = Message {
            role: Role::System,
            ..Message::user("")
        };
        system.content =
            MessageContent::Parts(vec![ContentPart::text("sysA"), ContentPart::text("sysB")]);

        let body = p
            .build_request(
                "m",
                &[system, user],
                &CompletionParameters::new(),
                false,
                true,
            )
            .unwrap();
        // all_text() newline-joins the parts; both parts must survive.
        assert_eq!(body["messages"][0]["content"], "first\nsecond");
        assert_eq!(body["system"], "sysA\nsysB");
    }

    // ---- Anthropic tool calling -----------------------------------------------

    use crate::tools::{ToolCall, ToolChoice, ToolDefinition};

    fn weather_tool() -> ToolDefinition {
        ToolDefinition::new(
            "get_weather",
            "Get the weather",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        )
    }

    #[test]
    fn anthropic_build_request_emits_tools_and_tool_choice() {
        let p = AnthropicProvider;
        let messages = vec![Message::user("weather in Paris?")];
        let params = CompletionParameters::new()
            .with_tool(weather_tool().with_strict(true))
            .with_tool_choice(ToolChoice::Required)
            .with_parallel_tool_calls(false);
        let body = p
            .build_request("m", &messages, &params, false, true)
            .unwrap();

        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(body["tools"][0]["strict"], true);
        // Required → Anthropic "any"; parallel=false folds into tool_choice.
        assert_eq!(body["tool_choice"]["type"], "any");
        assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
    }

    #[test]
    fn anthropic_parallel_false_without_choice_forces_auto_choice() {
        // Anthropic has no top-level parallel flag; with no explicit choice, the
        // flag needs an auto tool_choice object to live in.
        let p = AnthropicProvider;
        let params = CompletionParameters::new()
            .with_tool(weather_tool())
            .with_parallel_tool_calls(false);
        let body = p
            .build_request("m", &[Message::user("hi")], &params, false, true)
            .unwrap();
        assert_eq!(body["tool_choice"]["type"], "auto");
        assert_eq!(body["tool_choice"]["disable_parallel_tool_use"], true);
        // parallel=true is the wire default: nothing emitted.
        let params = CompletionParameters::new()
            .with_tool(weather_tool())
            .with_parallel_tool_calls(true);
        let body = p
            .build_request("m", &[Message::user("hi")], &params, false, true)
            .unwrap();
        assert!(body.get("tool_choice").is_none());
    }

    #[test]
    fn anthropic_assistant_tool_calls_become_tool_use_blocks() {
        let p = AnthropicProvider;
        let mut assistant = Message::assistant("checking");
        assistant.tool_calls = Some(vec![ToolCall::new(
            "tu_1",
            "get_weather",
            r#"{"city":"Paris"}"#,
        )]);
        let messages = vec![Message::user("weather?"), assistant];
        let body = p
            .build_request("m", &messages, &CompletionParameters::new(), false, true)
            .unwrap();

        let blocks = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(blocks[0]["type"], "text");
        assert_eq!(blocks[0]["text"], "checking");
        assert_eq!(blocks[1]["type"], "tool_use");
        assert_eq!(blocks[1]["id"], "tu_1");
        assert_eq!(blocks[1]["name"], "get_weather");
        assert_eq!(blocks[1]["input"]["city"], "Paris", "input is an object");
    }

    #[test]
    fn anthropic_assistant_tool_call_without_text_has_no_empty_text_block() {
        // Anthropic rejects empty text blocks: a tool-call-only assistant turn
        // must emit ONLY the tool_use block.
        let p = AnthropicProvider;
        let mut assistant = Message::assistant("");
        assistant.tool_calls = Some(vec![ToolCall::new("tu_1", "get_weather", "{}")]);
        let body = p
            .build_request(
                "m",
                &[Message::user("weather?"), assistant],
                &CompletionParameters::new(),
                false,
                true,
            )
            .unwrap();
        let blocks = body["messages"][1]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0]["type"], "tool_use");
    }

    #[test]
    fn anthropic_tool_results_become_one_user_turn_with_tool_result_blocks() {
        // Parallel tool results (consecutive Role::Tool messages) must share ONE
        // user turn, tool_result blocks first; trailing user text joins that turn
        // AFTER the results (Anthropic's required ordering).
        let p = AnthropicProvider;
        let mut assistant = Message::assistant("");
        assistant.tool_calls = Some(vec![
            ToolCall::new("tu_1", "get_weather", r#"{"city":"Paris"}"#),
            ToolCall::new("tu_2", "get_weather", r#"{"city":"Lyon"}"#),
        ]);
        let messages = vec![
            Message::user("weather?"),
            assistant,
            Message::tool("tu_1", "15 degrees"),
            Message::tool("tu_2", "18 degrees"),
            Message::user("thanks, summarize"),
        ];
        let body = p
            .build_request("m", &messages, &CompletionParameters::new(), false, true)
            .unwrap();

        let turns = body["messages"].as_array().unwrap();
        assert_eq!(
            turns.len(),
            3,
            "user / assistant / merged results+text user"
        );
        assert_eq!(turns[2]["role"], "user");
        let blocks = turns[2]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(blocks[0]["type"], "tool_result");
        assert_eq!(blocks[0]["tool_use_id"], "tu_1");
        assert_eq!(blocks[0]["content"], "15 degrees");
        assert_eq!(blocks[1]["type"], "tool_result");
        assert_eq!(blocks[1]["tool_use_id"], "tu_2");
        assert_eq!(blocks[2]["type"], "text");
        assert_eq!(blocks[2]["text"], "thanks, summarize");
    }

    #[test]
    fn anthropic_tool_result_without_call_id_fails_loudly() {
        let p = AnthropicProvider;
        let mut orphan = Message::tool("x", "result");
        orphan.tool_call_id = None;
        assert!(p
            .build_request(
                "m",
                &[Message::user("hi"), orphan],
                &CompletionParameters::new(),
                false,
                true
            )
            .is_err());
    }

    #[test]
    fn anthropic_invalid_tool_call_arguments_fail_loudly() {
        // An assistant tool call whose stored arguments are not valid JSON cannot
        // be expressed as an Anthropic `input` object; it must error, not ship "{}".
        let p = AnthropicProvider;
        let mut assistant = Message::assistant("");
        assistant.tool_calls = Some(vec![ToolCall::new("tu_1", "t", "{not json")]);
        assert!(p
            .build_request(
                "m",
                &[Message::user("hi"), assistant],
                &CompletionParameters::new(),
                false,
                true
            )
            .is_err());
    }

    // ---- Anthropic cache breakpoints -----------------------------------------

    /// A message with the cache breakpoint flag set.
    fn cached_msg(m: Message) -> Message {
        Message {
            cache_breakpoint: true,
            ..m
        }
    }

    #[test]
    fn anthropic_no_breakpoint_uses_plain_string_content() {
        let p = AnthropicProvider;
        let messages = vec![Message::system("sys"), Message::user("hi")];
        let body = p
            .build_request("m", &messages, &CompletionParameters::new(), false, true)
            .unwrap();
        // No marks → system is a plain string, user content is a plain string.
        assert!(body["system"].is_string());
        assert!(body["messages"][0]["content"].is_string());
    }

    #[test]
    fn anthropic_breakpoint_on_system_emits_block_with_cache_control() {
        let p = AnthropicProvider;
        let messages = vec![
            cached_msg(Message::system("big system")),
            Message::user("hi"),
        ];
        let body = p
            .build_request("m", &messages, &CompletionParameters::new(), false, true)
            .unwrap();
        // Marked system → block-array form carrying cache_control.
        assert_eq!(body["system"][0]["type"], "text");
        assert_eq!(body["system"][0]["text"], "big system");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn anthropic_breakpoint_on_turn_emits_block_with_cache_control() {
        let p = AnthropicProvider;
        let messages = vec![
            Message::system("sys"),
            cached_msg(Message::user("cache me")),
            Message::user("new"),
        ];
        let body = p
            .build_request("m", &messages, &CompletionParameters::new(), false, true)
            .unwrap();
        // Consecutive user messages merge into ONE user turn (Anthropic wants
        // alternating roles); the marked message's block carries cache_control,
        // the unmarked one's block does not.
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0]["text"], "cache me");
        assert_eq!(blocks[0]["cache_control"]["type"], "ephemeral");
        assert_eq!(blocks[1]["text"], "new");
        assert!(blocks[1].get("cache_control").is_none());
    }

    #[test]
    fn anthropic_caps_breakpoints_at_four_keeping_the_last() {
        let p = AnthropicProvider;
        // 5 marked user turns; only the LAST 4 should carry cache_control.
        let messages: Vec<Message> = (0..5)
            .map(|i| cached_msg(Message::user(format!("turn{i}"))))
            .collect();
        let body = p
            .build_request("m", &messages, &CompletionParameters::new(), false, true)
            .unwrap();
        // The 5 consecutive user messages merge into one user turn of 5 text
        // blocks; only the LAST 4 blocks carry cache_control.
        let blocks = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(blocks.len(), 5);
        assert!(
            blocks[0].get("cache_control").is_none(),
            "oldest mark dropped"
        );
        for b in &blocks[1..5] {
            assert_eq!(b["cache_control"]["type"], "ephemeral");
        }
    }

    #[test]
    fn anthropic_cost_is_token_priced_or_unpriced() {
        let p = AnthropicProvider;
        // No price → Unpriced (never a fake $0), tokens survive.
        let unpriced = p.cost_of(usage(100, 50), None);
        assert_eq!(unpriced.resolution, CostResolution::Unpriced);
        assert_eq!(unpriced.usage.prompt_tokens(), 100);
        // With price → Resolved estimate.
        let price = TokenPrice::new(1.0, 5.0); // $1/Mtok in, $5/Mtok out
        let resolved = p.cost_of(usage(1_000_000, 1_000_000), Some(&price));
        assert_eq!(resolved.resolution, CostResolution::Resolved);
        assert!((resolved.usd - 6.0).abs() < 1e-9);
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

    // ── Claude subscription credentials ──────────────────────────────

    #[test]
    fn parses_claude_code_subscription_token() {
        let body = r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-abc",
            "refreshToken":"sk-ant-ort01-x","subscriptionType":"max"}}"#;
        assert_eq!(
            parse_claude_code_token(body).as_deref(),
            Some("sk-ant-oat01-abc")
        );
    }

    #[test]
    fn missing_or_empty_token_is_none() {
        assert!(parse_claude_code_token(r#"{"claudeAiOauth":{}}"#).is_none());
        assert!(parse_claude_code_token(r#"{"claudeAiOauth":{"accessToken":""}}"#).is_none());
        assert!(parse_claude_code_token("not json").is_none());
        assert!(parse_claude_code_token(r#"{"other":"shape"}"#).is_none());
    }
}
