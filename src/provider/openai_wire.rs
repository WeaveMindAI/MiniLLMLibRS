//! The shared OpenAI `/chat/completions` DIALECT: request building,
//! response/chunk/usage parsing, and the error envelope, for every
//! provider speaking an OpenAI-compatible wire.
//!
//! This is the one concrete wire the [`Provider`] trait's DEFAULT
//! methods delegate to (most services speak it); a provider on a
//! different envelope overrides the shape methods and never touches
//! this module. The `openai_*` hooks on the trait are the dialect's
//! parameterization points (token-limit key, usage opt-in, tool
//! shapes), consulted only from here.

use super::auth::Auth;
use super::response::{error_object, preview_str, CompletionResponse, StreamChunk, Usage};
use super::wire::Provider;
use crate::error::{MiniLLMError, Result};
use crate::generator::CompletionParameters;
use crate::message::Message;
use crate::tools::{ToolCall, ToolCallDelta};
use secrecy::ExposeSecret;

/// Attach a `cache_control` marker to an OpenAI-wire message's content: a
/// plain string becomes the one-block array form (a string can't carry the
/// marker); an existing parts array gets the marker on its last text part.
/// A message with no markable text (an assistant turn that is pure
/// `tool_calls`) keeps no marker; the OpenAI wire has nowhere to put one, and
/// a dropped breakpoint only shortens the cached prefix to the previous mark.
pub(crate) fn mark_openai_message(msg: &mut serde_json::Value) {
    let marker = serde_json::json!({ "type": "ephemeral" });
    match &mut msg["content"] {
        serde_json::Value::String(s) if !s.is_empty() => {
            let text = s.clone();
            msg["content"] =
                serde_json::json!([{ "type": "text", "text": text, "cache_control": marker }]);
        }
        serde_json::Value::Array(parts) => {
            match parts.iter_mut().rev().find(|p| p["type"] == "text") {
                Some(part) => part["cache_control"] = marker,
                None => tracing::warn!(
                    "cache breakpoint on a message with no text part; marker dropped"
                ),
            }
        }
        _ => tracing::warn!(
            "cache breakpoint on a message with no markable text content; marker dropped"
        ),
    }
}

/// OpenAI-wire auth headers: a key or token both become `Authorization: Bearer`.
pub(crate) fn openai_auth_headers(auth: &Auth) -> Result<Vec<(String, String)>> {
    match auth {
        Auth::ApiKey(s) | Auth::BearerToken(s) => Ok(vec![(
            "Authorization".to_string(),
            format!("Bearer {}", s.expose_secret()),
        )]),
        Auth::None => Ok(Vec::new()),
    }
}

/// Build the OpenAI `/chat/completions` request body by EXPLICITLY mapping each
/// normalized [`CompletionParameters`] field to its OpenAI wire key (the params
/// struct is normalized intent, not a wire shape). The request-owned keys
/// (`model`/`messages`/`stream`, the provider token-limit key, usage opt-in) are
/// overlaid, then `extra` is merged, failing loudly on a collision with any key
/// already set. The provider's `openai_*` hooks supply the dialect points that
/// vary across OpenAI-compatible wires (token-limit key, usage opt-in, tool
/// shapes).
pub(crate) fn openai_build_request<P: Provider + ?Sized>(
    model: &str,
    messages: &[Message],
    params: &CompletionParameters,
    stream: bool,
    include_usage: bool,
    provider: &P,
) -> Result<serde_json::Value> {
    let mut body = serde_json::json!({
        "model": model,
        "messages": provider.openai_messages_value(model, messages),
        "stream": stream,
    });
    let obj = body.as_object_mut().expect("json object");

    // Normalized sampling/intent fields → OpenAI keys.
    if let Some(v) = params.max_tokens {
        obj.insert(
            provider.openai_token_limit_field().to_string(),
            serde_json::json!(v),
        );
    }
    if let Some(v) = params.temperature {
        obj.insert("temperature".into(), serde_json::json!(v));
    }
    if let Some(v) = params.top_p {
        obj.insert("top_p".into(), serde_json::json!(v));
    }
    if let Some(v) = params.top_k {
        obj.insert("top_k".into(), serde_json::json!(v));
    }
    if let Some(v) = params.frequency_penalty {
        obj.insert("frequency_penalty".into(), serde_json::json!(v));
    }
    if let Some(v) = params.presence_penalty {
        obj.insert("presence_penalty".into(), serde_json::json!(v));
    }
    if let Some(v) = params.repetition_penalty {
        obj.insert("repetition_penalty".into(), serde_json::json!(v));
    }
    if let Some(v) = &params.stop {
        obj.insert("stop".into(), serde_json::json!(v));
    }
    if let Some(v) = params.seed {
        obj.insert("seed".into(), serde_json::json!(v));
    }
    if let Some(v) = &params.response_format {
        obj.insert("response_format".into(), response_format_value(v));
    }
    if let Some(v) = &params.tools {
        obj.insert("tools".into(), provider.openai_tools_value(v));
    }
    if let Some(v) = &params.tool_choice {
        obj.insert("tool_choice".into(), provider.openai_tool_choice_value(v));
    }
    if let Some(v) = params.parallel_tool_calls {
        obj.insert("parallel_tool_calls".into(), serde_json::json!(v));
    }
    if let Some(v) = &params.reasoning {
        obj.insert("reasoning".into(), serde_json::to_value(v)?);
    }

    if include_usage {
        provider.openai_request_usage(&mut body, stream);
    }

    // Merge `extra`, failing loudly on a collision with any key already present.
    if let (Some(extra), Some(obj)) = (params.extra.clone(), body.as_object_mut()) {
        for (key, value) in extra {
            if obj.contains_key(&key) {
                return Err(MiniLLMError::InvalidParameter(format!(
                    "extra param '{}' collides with a built-in request key; set it via the typed builder instead of with_extra",
                    key
                )));
            }
            obj.insert(key, value);
        }
    }

    Ok(body)
}

/// Parse the OpenAI-wire usage object into the normalized DISJOINT buckets.
///
/// The two cache buckets sit DIFFERENTLY relative to `prompt_tokens`, and getting
/// this wrong mis-bills cache-heavy requests (verified against OpenRouter's wire,
/// 2026-06):
/// - `prompt_tokens` is the TOTAL input charged at full+read rates.
/// - `prompt_tokens_details.cached_tokens` (cache READS) is a SUBSET of
///   `prompt_tokens`, so the disjoint full-price remainder is
///   `uncached = prompt_tokens − cache_read`.
/// - `prompt_tokens_details.cache_write_tokens` (cache WRITES) is ADDITIVE: it is
///   billed at a premium ON TOP of `prompt_tokens` and is NOT included in it, so
///   it must NOT be subtracted (OpenRouter started returning this field natively
///   in early 2026; plain OpenAI has no separate write charge and omits it).
///
/// Shared by every OpenAI-wire provider; cost fields are read separately by the
/// providers that report them.
pub(crate) fn parse_openai_usage(u: &serde_json::Value) -> Option<Usage> {
    if u.is_null() {
        return None;
    }
    let total_input = u["prompt_tokens"].as_u64().unwrap_or(0) as u32;
    let cache_read = u["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap_or(0) as u32;
    // OpenRouter surfaces cache writes here; plain OpenAI does not (no write charge).
    let cache_write = u["prompt_tokens_details"]["cache_write_tokens"]
        .as_u64()
        .unwrap_or(0) as u32;
    // The whole disjoint split assumes cache READS are a SUBSET of prompt_tokens.
    // If a wire reports more cached reads than prompt_tokens, that assumption is
    // violated and any split we compute would be a silently-wrong cost. Fail loudly
    // (report no usage → Unknown cost) rather than clamp to a fabricated number.
    if cache_read > total_input {
        tracing::error!(
            prompt_tokens = total_input,
            cached_tokens = cache_read,
            "OpenAI-wire usage reports cached_tokens > prompt_tokens; cached is not a subset on this wire, cost would be wrong, reporting Unknown"
        );
        return None;
    }
    Some(Usage {
        // Cache READS are a subset of prompt_tokens → subtract them to get the
        // full-price remainder. Cache WRITES are additive (separate from
        // prompt_tokens) → do NOT subtract.
        uncached_input_tokens: total_input - cache_read,
        cache_read_tokens: cache_read,
        cache_write_tokens: cache_write,
        completion_tokens: u["completion_tokens"].as_u64().unwrap_or(0) as u32,
        cost: None,
        upstream_inference_cost: None,
        reasoning_tokens: u["completion_tokens_details"]["reasoning_tokens"]
            .as_u64()
            .map(|v| v as u32),
    })
}

/// Locate the `usage` object on a non-streaming response or a streaming chunk
/// (both OpenAI-wire put it under a top-level `usage` key).
pub(crate) fn usage_field(value: &serde_json::Value) -> Option<&serde_json::Value> {
    value.get("usage").filter(|u| !u.is_null())
}

/// Parse the OpenAI-wire usage out of a raw response/chunk (finds the `usage`
/// field, then parses it). Backs the default `Provider::parse_usage`.
pub(crate) fn parse_openai_usage_field(raw: &serde_json::Value) -> Option<Usage> {
    parse_openai_usage(usage_field(raw)?)
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
    let media = provider.parse_response_media(message)?;
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
        media,
        raw_response: Some(raw),
    })
}

/// Parse the OpenAI-wire `message.images` array (image-generation output,
/// OpenRouter's normalized field) into typed media; an absent field is a
/// text-only completion (empty). Each entry is an `image_url` part whose
/// `url` carries a `data:` URL or an https URL; an entry with no url is a
/// malformed response and fails loudly rather than silently dropping a
/// generated (and billed) image. The default body of
/// [`Provider::parse_response_media`](super::Provider::parse_response_media);
/// a provider with a different media wire overrides the hook instead.
pub(crate) fn parse_openai_response_images(
    message: &serde_json::Value,
) -> crate::error::Result<Vec<crate::message::Media>> {
    let Some(entries) = message["images"].as_array() else {
        return Ok(Vec::new());
    };
    entries
        .iter()
        .map(|entry| {
            let url = entry["image_url"]["url"].as_str().ok_or_else(|| {
                crate::error::MiniLLMError::MalformedResponse(format!(
                    "response image entry has no image_url.url: {}",
                    preview_str(&entry.to_string())
                ))
            })?;
            Ok(crate::message::Media::Image(
                crate::message::ImageData::from_url(url),
            ))
        })
        .collect()
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
// Wire projections of the normalized types
// =============================================================================

/// Project a message CONTENT onto the wire: a plain string, or the parts
/// array.
///
/// `keep_estimation_metadata` decides whether the media parts' estimation
/// metadata (`duration_secs`, `width`, `height`) rides the wire. A strict
/// provider schema would reject the unknown keys, so the provider impl
/// decides ([`Provider::wire_keeps_estimation_metadata`]): a wire that
/// tolerates them keeps them, so anything metering the request in flight
/// (a client-side estimator, a billing gateway) can price the media
/// exactly; everyone else sheds them here. Serde round trips (saved
/// trees) always keep the metadata regardless.
///
/// [`Provider::wire_keeps_estimation_metadata`]: crate::Provider::wire_keeps_estimation_metadata
pub fn content_value(
    content: &crate::message::MessageContent,
    keep_estimation_metadata: bool,
) -> serde_json::Value {
    use crate::message::MessageContent as MC;
    match content {
        MC::Text(text) => serde_json::json!(text),
        MC::Parts(parts) => {
            let mut value = serde_json::json!(parts);
            for part in value.as_array_mut().expect("parts serialize to an array") {
                if !keep_estimation_metadata {
                    for media_key in ["input_audio", "video_url", "image_url", "file"] {
                        if let Some(media) = part.get_mut(media_key).and_then(|v| v.as_object_mut())
                        {
                            media.remove("duration_secs");
                            media.remove("width");
                            media.remove("height");
                            media.remove("page_count");
                        }
                    }
                }
                // Audio wire normalization: `data` may arrive as a
                // `data:<mime>;base64,<payload>` URL (a caller that
                // holds media as data URLs). The audio wire wants
                // raw base64 + `format`, so split it here; images
                // and video take data URLs verbatim and need none.
                if let Some(audio) = part.get_mut("input_audio").and_then(|v| v.as_object_mut()) {
                    let split = audio
                        .get("data")
                        .and_then(|v| v.as_str())
                        .and_then(|d| d.strip_prefix("data:"))
                        .and_then(|rest| rest.split_once(";base64,"))
                        .map(|(mime, payload)| {
                            (
                                mime.rsplit('/').next().unwrap_or("").to_string(),
                                payload.to_string(),
                            )
                        });
                    if let Some((format, payload)) = split {
                        audio.insert("data".into(), serde_json::json!(payload));
                        if !format.is_empty() && !audio.contains_key("format") {
                            audio.insert("format".into(), serde_json::json!(format));
                        }
                    }
                }
            }
            value
        }
    }
}

/// The messages payload (assistant `tool_calls` as function entries, tool
/// results as `role: tool` messages). Non-OpenAI wires build their own
/// payload in `build_request`. `keep_estimation_metadata` follows the
/// provider's wire tolerance (see
/// [`Provider::wire_keeps_estimation_metadata`]).
pub fn messages_to_payload(
    messages: &[Message],
    keep_estimation_metadata: bool,
) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|msg| {
            let mut obj = serde_json::json!({
                "role": msg.role,
                "content": content_value(&msg.content, keep_estimation_metadata),
            });

            if let Some(name) = &msg.name {
                obj["name"] = serde_json::json!(name);
            }
            if let Some(tool_call_id) = &msg.tool_call_id {
                obj["tool_call_id"] = serde_json::json!(tool_call_id);
            }
            if let Some(tool_calls) = &msg.tool_calls {
                obj["tool_calls"] =
                    serde_json::Value::Array(tool_calls.iter().map(tool_call_value).collect());
            }

            obj
        })
        .collect()
}

/// A tool definition's wire shape: `{"type":"function","function":{...}}`.
pub fn tool_definition_value(def: &crate::tools::ToolDefinition) -> serde_json::Value {
    let mut function = serde_json::json!({
        "name": def.name,
        "parameters": def.parameters,
    });
    if let Some(desc) = &def.description {
        function["description"] = serde_json::json!(desc);
    }
    if let Some(strict) = def.strict {
        function["strict"] = serde_json::json!(strict);
    }
    serde_json::json!({ "type": "function", "function": function })
}

/// A tool choice's wire value.
pub fn tool_choice_value(choice: &crate::tools::ToolChoice) -> serde_json::Value {
    use crate::tools::ToolChoice;
    match choice {
        ToolChoice::Auto => serde_json::json!("auto"),
        ToolChoice::None => serde_json::json!("none"),
        ToolChoice::Required => serde_json::json!("required"),
        ToolChoice::Tool(name) => serde_json::json!({
            "type": "function",
            "function": { "name": name },
        }),
    }
}

/// An assistant tool call's wire entry:
/// `{"id","type":"function","function":{"name","arguments"}}` (arguments as
/// a JSON string, which is what this wire expects).
pub fn tool_call_value(call: &crate::tools::ToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments,
        },
    })
}

/// A response format's wire value (`{"type": "json_object"}`).
pub fn response_format_value(format: &crate::generator::ResponseFormat) -> serde_json::Value {
    match format {
        crate::generator::ResponseFormat::JsonObject => {
            serde_json::json!({"type": "json_object"})
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::wire::TokenPrice;
    use super::*;
    use crate::provider::OpenRouterProvider;

    fn weather_tool() -> crate::tools::ToolDefinition {
        crate::tools::ToolDefinition::new(
            "get_weather",
            "Get the current weather for a city",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        )
    }

    use crate::message::{AudioData, ContentPart, Message, MessageContent};
    use crate::tools::{ToolCall, ToolChoice};

    #[test]
    fn definition_openai_wire_shape() {
        let v = tool_definition_value(&weather_tool().with_strict(true));
        assert_eq!(v["type"], "function");
        assert_eq!(v["function"]["name"], "get_weather");
        assert_eq!(
            v["function"]["description"],
            "Get the current weather for a city"
        );
        assert_eq!(v["function"]["parameters"]["type"], "object");
        assert_eq!(v["function"]["strict"], true);
    }

    #[test]
    fn choice_openai_wire_values() {
        assert_eq!(tool_choice_value(&ToolChoice::Auto), "auto");
        assert_eq!(tool_choice_value(&ToolChoice::None), "none");
        assert_eq!(tool_choice_value(&ToolChoice::Required), "required");
        let forced = tool_choice_value(&ToolChoice::Tool("get_weather".into()));
        assert_eq!(forced["type"], "function");
        assert_eq!(forced["function"]["name"], "get_weather");
    }

    #[test]
    fn call_openai_wire_keeps_arguments_as_string() {
        let v = tool_call_value(&ToolCall::new("c1", "get_weather", r#"{"city":"Paris"}"#));
        assert_eq!(v["id"], "c1");
        assert_eq!(v["type"], "function");
        assert_eq!(v["function"]["name"], "get_weather");
        assert_eq!(v["function"]["arguments"], r#"{"city":"Paris"}"#);
        assert!(v["function"]["arguments"].is_string());
    }

    #[test]
    fn payload_emits_openai_tool_wire_shapes() {
        // Assistant tool_calls → OpenAI function entries (arguments as a JSON
        // string); a tool result → role=tool with tool_call_id.
        let mut assistant = Message::assistant("checking");
        assistant.tool_calls = Some(vec![crate::tools::ToolCall::new(
            "c1",
            "get_weather",
            r#"{"city":"Paris"}"#,
        )]);
        let payload = messages_to_payload(&[assistant, Message::tool("c1", "15 degrees")], false);

        assert_eq!(payload[0]["tool_calls"][0]["id"], "c1");
        assert_eq!(payload[0]["tool_calls"][0]["type"], "function");
        assert_eq!(
            payload[0]["tool_calls"][0]["function"]["name"],
            "get_weather"
        );
        assert!(
            payload[0]["tool_calls"][0]["function"]["arguments"].is_string(),
            "OpenAI wire wants arguments as a JSON string"
        );
        assert_eq!(payload[1]["role"], "tool");
        assert_eq!(payload[1]["tool_call_id"], "c1");
        assert_eq!(payload[1]["content"], "15 degrees");
    }

    /// Estimation metadata (durations, dimensions) follows the provider's
    /// wire tolerance: a strict schema sheds it (the default), a tolerant
    /// wire keeps it so an in-flight meter can price the media exactly from
    /// the request bytes. The rest of the part survives either way.
    #[test]
    fn estimation_metadata_follows_the_wires_tolerance() {
        use crate::message::{DocumentData, ImageData, MessageContent, VideoData};

        let content = MessageContent::parts(vec![
            ContentPart::text("what is in this?"),
            ContentPart::audio(&AudioData::from_bytes(&[0u8; 4], "mp3").with_duration(3.5)),
            ContentPart::video(&VideoData::from_url("https://x/y.mp4").with_duration(12.0)),
            ContentPart::image(&ImageData::from_url("https://x/y.png").with_dimensions(800, 600)),
            ContentPart::document(
                &DocumentData::from_bytes(b"%PDF", "application/pdf").with_page_count(7),
            ),
        ]);

        // Strict wire (the default): every metadata key is shed.
        let strict = content_value(&content, false);
        let parts = strict.as_array().expect("parts stay an array");
        assert_eq!(parts[0]["text"], "what is in this?");
        assert!(
            parts[1]["input_audio"].get("duration_secs").is_none(),
            "{strict}"
        );
        assert_eq!(
            parts[1]["input_audio"]["format"], "mp3",
            "only the metadata is shed"
        );
        assert!(
            parts[2]["video_url"].get("duration_secs").is_none(),
            "{strict}"
        );
        assert_eq!(parts[2]["video_url"]["url"], "https://x/y.mp4");
        assert!(parts[3]["image_url"].get("width").is_none(), "{strict}");
        // Pin the part's identity first: without it the page_count assertion
        // would also pass vacuously if the file part went missing or moved.
        assert_eq!(parts[4]["type"], "file", "{strict}");
        assert!(parts[4]["file"].get("page_count").is_none(), "{strict}");
        assert_eq!(
            parts[4]["file"]["filename"], "document.pdf",
            "only the metadata is shed"
        );

        // Tolerant wire: the metadata rides the payload.
        let tolerant = content_value(&content, true);
        let parts = tolerant.as_array().expect("parts stay an array");
        assert_eq!(parts[1]["input_audio"]["duration_secs"], 3.5);
        assert_eq!(parts[2]["video_url"]["duration_secs"], 12.0);
        assert_eq!(parts[3]["image_url"]["width"], 800);
        assert_eq!(parts[3]["image_url"]["height"], 600);
        assert_eq!(parts[4]["file"]["page_count"], 7);
    }

    /// A `data:` URL in an audio part's `data` splits into raw base64 +
    /// `format` on the wire (the audio wire's shape); image and video
    /// parts take data URLs verbatim.
    #[test]
    fn audio_data_url_normalizes_to_base64_plus_format_on_the_wire() {
        let audio = AudioData::from_url("data:audio/mp3;base64,aGk=");
        let content = MessageContent::parts(vec![ContentPart::audio(&audio)]);
        let wire = content_value(&content, false);
        assert_eq!(wire[0]["input_audio"]["data"], "aGk=");
        assert_eq!(wire[0]["input_audio"]["format"], "mp3");
        // A plain https URL still rides verbatim with no format.
        let remote = AudioData::from_url("https://x.example/clip.mp3");
        let wire = content_value(
            &MessageContent::parts(vec![ContentPart::audio(&remote)]),
            false,
        );
        assert_eq!(wire[0]["input_audio"]["data"], "https://x.example/clip.mp3");
        assert!(wire[0]["input_audio"].get("format").is_none());
    }

    /// The accounting used to parse usage in these tests (OpenAI-wire shape).
    fn acct() -> OpenRouterProvider {
        OpenRouterProvider
    }

    /// Image-generation output (`message.images`, OpenRouter's
    /// normalized field) lands as typed media, and the canonical
    /// response-to-history conversion carries text + media as parts.
    #[test]
    fn parse_response_surfaces_returned_images_as_media() {
        let raw = serde_json::json!({
            "id": "gen-1", "model": "img-model",
            "choices": [{
                "message": {
                    "content": "here you go",
                    "images": [
                        { "type": "image_url",
                          "image_url": { "url": "data:image/png;base64,aGk=" } },
                    ],
                },
                "finish_reason": "stop",
            }],
        });
        let resp = parse_openai_response(raw, &acct()).unwrap();
        assert_eq!(resp.media.len(), 1);
        let crate::message::Media::Image(img) = &resp.media[0] else {
            panic!("expected an image");
        };
        assert!(
            img.is_url(),
            "a data: URL rides verbatim as a URL reference"
        );

        let message = resp.to_assistant_message();
        let crate::message::MessageContent::Parts(parts) = &message.content else {
            panic!("media response must produce parts");
        };
        assert_eq!(parts[0].as_text(), Some("here you go"));
        assert!(matches!(
            parts[1],
            crate::message::ContentPart::Image { .. }
        ));

        // A malformed image entry (no url) fails loudly: a generated
        // (billed) image must never be silently dropped.
        let bad = serde_json::json!({
            "id": "gen-2", "model": "img-model",
            "choices": [{ "message": { "content": "", "images": [{ "type": "image_url" }] } }],
        });
        assert!(parse_openai_response(bad, &acct()).is_err());
    }

    /// The media-extraction hook is per-provider: a wire that returns
    /// media somewhere other than `message.images` overrides
    /// `parse_response_media` alone, and the normalized result still
    /// lands in `CompletionResponse.media`, without re-implementing the
    /// whole response parse.
    #[test]
    fn a_provider_overrides_where_returned_media_lives_on_its_wire() {
        use super::super::wire::{CostOutcome, TokenPrice};

        #[derive(Debug)]
        struct SpokenProvider;
        impl Provider for SpokenProvider {
            fn parse_response_media(
                &self,
                message: &serde_json::Value,
            ) -> crate::error::Result<Vec<crate::message::Media>> {
                // This wire returns spoken audio under `message.audio`.
                match message["audio"]["data"].as_str() {
                    Some(data) => Ok(vec![crate::message::Media::Audio(
                        crate::message::AudioData::from_base64(data, "wav"),
                    )]),
                    None => Ok(Vec::new()),
                }
            }
            fn cost_of(&self, _usage: Usage, _price: Option<&TokenPrice>) -> CostOutcome {
                CostOutcome::unknown()
            }
        }

        let raw = serde_json::json!({
            "id": "gen-1", "model": "speaks",
            "choices": [{ "message": { "content": "said aloud", "audio": { "data": "aGk=" } } }],
        });
        let resp = parse_openai_response(raw, &SpokenProvider).unwrap();
        assert_eq!(resp.media.len(), 1);
        assert!(matches!(resp.media[0], crate::message::Media::Audio(_)));
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
}
