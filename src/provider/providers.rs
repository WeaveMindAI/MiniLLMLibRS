//! Built-in [`Provider`] implementations for the providers this crate
//! ships with. Each owns exactly one provider's wire knowledge.
//!
//! Shared OpenAI-wire helpers (`openai_*`, `parse_openai_usage_field`) back the
//! default [`Provider`] methods, so OpenAI-envelope providers stay tiny and a
//! different-envelope provider (Anthropic) overrides only the shape methods.

use super::auth::Auth;
use super::response::Usage;
use super::wire::{AppIdentity, CostFuture, CostOutcome, PostStreamCtx, Provider, TokenPrice};
use crate::error::{MiniLLMError, Result};
use crate::generator::CompletionParameters;
use crate::message::{messages_to_payload, Message};
use secrecy::ExposeSecret;

// =============================================================================
// Shared OpenAI-wire helpers (back the default Provider methods)
// =============================================================================

/// Indices of the messages that KEEP their cache breakpoint under the
/// provider's [`Provider::max_cache_breakpoints`] cap: of all marked
/// messages, the LAST `max` (the most-recent prefixes are the largest
/// reusable spans). Warns when marks are dropped.
pub(crate) fn kept_cache_breakpoints(
    messages: &[Message],
    max: usize,
) -> std::collections::HashSet<usize> {
    let marked: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.cache_breakpoint)
        .map(|(i, _)| i)
        .collect();
    if marked.len() > max {
        tracing::warn!(
            "this provider allows at most {} cache breakpoints per request; {} were marked, keeping the last {}",
            max,
            marked.len(),
            max
        );
    }
    marked.iter().rev().take(max).copied().collect()
}

/// Attach a `cache_control` marker to an OpenAI-wire message's content: a
/// plain string becomes the one-block array form (a string can't carry the
/// marker); an existing parts array gets the marker on its last text part.
/// A message with no markable text (an assistant turn that is pure
/// `tool_calls`) keeps no marker; the OpenAI wire has nowhere to put one, and
/// a dropped breakpoint only shortens the cached prefix to the previous mark.
fn mark_openai_message(msg: &mut serde_json::Value) {
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
        obj.insert("response_format".into(), v.to_openai_value());
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
fn parse_openai_usage(u: &serde_json::Value) -> Option<Usage> {
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
fn usage_field(value: &serde_json::Value) -> Option<&serde_json::Value> {
    value.get("usage").filter(|u| !u.is_null())
}

/// Parse the OpenAI-wire usage out of a raw response/chunk (finds the `usage`
/// field, then parses it). Backs the default `Provider::parse_usage`.
pub(crate) fn parse_openai_usage_field(raw: &serde_json::Value) -> Option<Usage> {
    parse_openai_usage(usage_field(raw)?)
}

/// Cost for a provider that returns no native USD: derive it from a configured
/// `TokenPrice`, otherwise report `Unpriced` (real tokens, unknown price, never a
/// fake $0). Shared by every token-only provider.
fn price_or_unpriced(usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
    match price {
        Some(p) => CostOutcome::resolved(p.cost_of(&usage), usage),
        None => CostOutcome::unpriced(usage),
    }
}

// =============================================================================
// OpenRouter
// =============================================================================

/// OpenRouter: OpenAI-wire request/response, plus native USD cost
/// (`usage.cost` + BYOK `usage.cost_details.upstream_inference_cost`), usage
/// opt-in via `usage:{include:true}`, and an out-of-band `/generation` endpoint.
#[derive(Debug, Clone, Default)]
pub struct OpenRouterProvider;

impl OpenRouterProvider {
    /// Read OpenRouter's native cost fields onto a base usage parsed from the
    /// shared OpenAI-wire shape.
    fn with_or_cost(mut usage: Usage, u: &serde_json::Value) -> Usage {
        usage.cost = u["cost"].as_f64();
        usage.upstream_inference_cost = u["cost_details"]["upstream_inference_cost"].as_f64();
        usage
    }
}

impl Provider for OpenRouterProvider {
    fn openai_request_usage(&self, body: &mut serde_json::Value, _stream: bool) {
        body["usage"] = serde_json::json!({ "include": true });
    }

    /// OpenRouter fronts Anthropic endpoints, whose wire caps the markers.
    fn max_cache_breakpoints(&self) -> usize {
        4
    }

    /// OpenRouter passes Anthropic-style `cache_control` markers through to
    /// Claude endpoints (and routes only to endpoints that support them), so a
    /// [`Message::cache_breakpoint`] becomes a marker on the message's content,
    /// capped at the last [`Provider::max_cache_breakpoints`]. Emission is
    /// gated to Claude models: the other providers OpenRouter fronts either
    /// auto-cache (OpenAI, Gemini, DeepSeek) or would lose routing candidates
    /// to the supporting-endpoints-only filter.
    fn openai_messages_value(&self, model: &str, messages: &[Message]) -> Vec<serde_json::Value> {
        let mut payload = messages_to_payload(messages);
        let lower = model.to_ascii_lowercase();
        if !lower.contains("claude") && !lower.contains("anthropic") {
            return payload;
        }
        for i in kept_cache_breakpoints(messages, self.max_cache_breakpoints()) {
            mark_openai_message(&mut payload[i]);
        }
        payload
    }

    /// OpenRouter attributes usage to an app via `HTTP-Referer` (the app URL) and
    /// `X-Title` (the app name) for its rankings.
    fn attribution_headers(&self, app: Option<&AppIdentity>) -> Vec<(String, String)> {
        match app {
            Some(app) => vec![
                ("HTTP-Referer".to_string(), app.url.clone()),
                ("X-Title".to_string(), app.title.clone()),
            ],
            None => Vec::new(),
        }
    }

    fn parse_usage(&self, response: &serde_json::Value) -> Option<Usage> {
        let u = usage_field(response)?;
        Some(Self::with_or_cost(parse_openai_usage(u)?, u))
    }

    /// OpenRouter aggregates its native fee plus the BYOK upstream charge. This
    /// sum is the provider-specific cost aggregation that must stay here, not in a
    /// shared helper. When OpenRouter returned no `cost` at all, fall back to the
    /// shared token-pricing path.
    fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
        match usage.cost {
            Some(or_fee) => {
                let usd = or_fee + usage.upstream_inference_cost.unwrap_or(0.0);
                CostOutcome::resolved(usd, usage)
            }
            None => price_or_unpriced(usage, price),
        }
    }

    fn resolve_post_stream<'a>(&'a self, ctx: PostStreamCtx<'a>) -> CostFuture<'a> {
        Box::pin(async move {
            if ctx.generation_id.is_empty() {
                return CostOutcome::unknown();
            }
            // OpenRouter may not finalize the generation record immediately; poll
            // every second before giving up to an honest Unknown. Plain 1s polls,
            // no backoff: the endpoint is free and the caller is waiting, so the
            // only cost of polling fast is nothing and the cost of polling slow
            // is user-visible latency. Measured: a completed generation's record
            // appears ~9s after it finishes, and a CANCELLED call's only after
            // the upstream generation runs to its own end anyway (client aborts
            // do not stop these routes) plus the same ~9s, i.e. ~18s for a short
            // generation. We poll for 25s.
            for _ in 0..25 {
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                if let Some(usage) =
                    query_generation(ctx.client, ctx.base_url, ctx.generation_id, ctx.auth).await
                {
                    return self.cost_of(usage, ctx.price);
                }
                tracing::debug!("OpenRouter generation {} not found yet", ctx.generation_id);
            }
            CostOutcome::unknown()
        })
    }
}

/// Query OpenRouter's `/api/v1/generation` for a finished generation's usage.
/// `None` on any failure or when the record carries no usable cost.
async fn query_generation(
    client: &reqwest::Client,
    base_url: &str,
    generation_id: &str,
    auth: &Auth,
) -> Option<Usage> {
    let api_key = auth.secret()?;
    let encoded =
        url::form_urlencoded::byte_serialize(generation_id.as_bytes()).collect::<String>();
    // The generator's own address, never a hardcoded host: a generator
    // pointed at a gateway resolves its costs through that gateway too.
    let url = format!("{}/generation?id={}", base_url.trim_end_matches('/'), encoded);

    let response = match client
        .get(&url)
        .header("Authorization", format!("Bearer {}", api_key.expose_secret()))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Generation cost query for {} failed: {}", generation_id, e);
            return None;
        }
    };
    if !response.status().is_success() {
        tracing::warn!(
            "Generation cost query for {} returned {}",
            generation_id,
            response.status()
        );
        return None;
    }
    let json: serde_json::Value = response.json().await.ok()?;
    usage_from_generation_record(json.get("data")?)
}

/// Parse a `/generation` record's `data` object into a `Usage`. Pure.
///
/// The record uses different field names than chat-completions usage. Require a
/// numeric total_cost: a record without it is unresolved, not free. Tokens come
/// from the native_tokens_* fields.
///
/// Same two-part money split as chat-completions usage: `total_cost` is what
/// OpenRouter charged in credits, and on a BYOK route it is 0 with the real
/// upstream charge (billed on the user's own provider key) in
/// `upstream_inference_cost`. The all-in cost is their sum; it goes in `cost`
/// with `upstream_inference_cost: None` so `cost_of` can't re-add it.
fn usage_from_generation_record(data: &serde_json::Value) -> Option<Usage> {
    let cost =
        data["total_cost"].as_f64()? + data["upstream_inference_cost"].as_f64().unwrap_or(0.0);
    let prompt = data["tokens_prompt"].as_u64().unwrap_or(0) as u32;
    let completion = data["tokens_completion"].as_u64().unwrap_or(0) as u32;
    // `tokens_prompt` is total input; `native_tokens_cached` is the cached-read
    // subset. Split into disjoint buckets (no separate write count here). Unlike the
    // streaming-usage path, `total_cost` here is the AUTHORITATIVE USD charge, so a
    // subset-violation only skews the token breakdown, not the money: warn and clamp
    // rather than discard a known-correct cost.
    let cache_read = data["native_tokens_cached"].as_u64().unwrap_or(0) as u32;
    if cache_read > prompt {
        tracing::warn!(
            tokens_prompt = prompt,
            native_tokens_cached = cache_read,
            "/generation reports cached > prompt; token breakdown clamped (cost is authoritative)"
        );
    }
    Some(Usage {
        uncached_input_tokens: prompt.saturating_sub(cache_read),
        cache_read_tokens: cache_read,
        cache_write_tokens: 0,
        completion_tokens: completion,
        cost: Some(cost),
        upstream_inference_cost: None,
        reasoning_tokens: data["native_tokens_reasoning"].as_u64().map(|v| v as u32),
    })
}

// =============================================================================
// OpenAI
// =============================================================================

/// OpenAI: OpenAI-wire, returns token counts but no dollar cost (price them via a
/// configured `TokenPrice`). Streaming usage requires the
/// `stream_options:{include_usage:true}` opt-in.
#[derive(Debug, Clone, Default)]
pub struct OpenAiProvider;

impl Provider for OpenAiProvider {
    fn openrouter_slug(&self) -> Option<&'static str> {
        Some("openai")
    }

    fn openai_request_usage(&self, body: &mut serde_json::Value, stream: bool) {
        // OpenAI only emits a usage chunk on streaming when explicitly asked.
        if stream {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }
    }

    // parse_usage uses the default (`parse_openai_usage_field`): OpenAI reports
    // no native cost fields, so the base OpenAI-wire parse is exactly right.

    /// OpenAI reports no native cost; price tokens via `TokenPrice` or report
    /// `Unpriced`.
    fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
        price_or_unpriced(usage, price)
    }

    // No out-of-band endpoint: a cancelled stream that never delivered usage is
    // genuinely unresolvable, so the default `resolve_post_stream` (Unknown) is
    // correct. A stream that DID deliver usage prices it via `cost_of`.
}

// =============================================================================
// Generic OpenAI-compatible
// =============================================================================

/// A minimal OpenAI-compatible provider: token counts only, no native cost, no
/// usage opt-in flag assumed, no out-of-band endpoint. The default for
/// [`GeneratorInfo::custom`](crate::GeneratorInfo::custom).
#[derive(Debug, Clone, Default)]
pub struct GenericProvider {
    /// Some older OpenAI-compatible servers only accept the legacy `max_tokens`
    /// request key. Set true for those.
    pub legacy_token_limit: bool,
}

impl Provider for GenericProvider {
    fn openai_token_limit_field(&self) -> &'static str {
        if self.legacy_token_limit {
            "max_tokens"
        } else {
            "max_completion_tokens"
        }
    }

    /// A bare OpenAI-compatible server has no usage opt-in (the default
    /// `openai_request_usage` is a no-op) and may never emit a usage chunk, so the
    /// streaming reader must NOT wait for one (it would wedge the stream until the
    /// idle timeout). Cost is still parsed opportunistically if one arrives.
    fn emits_stream_usage(&self, _requested: bool) -> bool {
        false
    }

    // parse_usage uses the default (`parse_openai_usage_field`).

    fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
        price_or_unpriced(usage, price)
    }
}

// =============================================================================
// Anthropic (native `/v1/messages`, `content[]` envelope)
// =============================================================================

use super::response::{CompletionResponse, StreamChunk};

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
                        blocks.push(call.to_anthropic_block()?);
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
            body["tools"] = serde_json::Value::Array(
                tools
                    .iter()
                    .map(crate::tools::ToolDefinition::to_anthropic_value)
                    .collect(),
            );
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
            let mut value = choice.to_anthropic_value();
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
        super::response::parse_anthropic_response(raw)
    }

    /// Parse one Anthropic SSE event into a `StreamChunk` (or surface an in-band
    /// `error` event as a loud `Err`).
    fn parse_chunk(&self, data: &str) -> Option<Result<StreamChunk>> {
        super::response::parse_anthropic_chunk(data)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::CostResolution;

    /// A fully-uncached usage (all input in the full-price bucket).
    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            uncached_input_tokens: prompt,
            completion_tokens: completion,
            ..Default::default()
        }
    }

    #[test]
    fn token_price_costs_prompt_and_completion_per_mtok() {
        let price = TokenPrice::new(3.0, 15.0); // $3/Mtok in, $15/Mtok out
        let u = usage(1_000_000, 1_000_000);
        assert!((price.cost_of(&u) - 18.0).abs() < 1e-9);
    }

    #[test]
    fn token_price_bills_cache_read_and_write_at_their_own_rates() {
        // read 0.3/Mtok, write 3.75/Mtok (1.25× the 3.0 input).
        let price = TokenPrice::new(3.0, 15.0).with_cache_rates(0.3, 3.75);
        // Disjoint: 200k uncached, 800k cache-read, 100k cache-write, 0 output.
        let u = Usage {
            uncached_input_tokens: 200_000,
            cache_read_tokens: 800_000,
            cache_write_tokens: 100_000,
            ..Default::default()
        };
        // 200k×3.0 ($0.6) + 800k×0.3 ($0.24) + 100k×3.75 ($0.375) = $1.215
        assert!(
            (price.cost_of(&u) - 1.215).abs() < 1e-9,
            "got {}",
            price.cost_of(&u)
        );
    }

    #[test]
    fn cache_rates_fall_back_to_input_rate_when_unset() {
        // No cache rates set → read and write both bill at the input rate.
        let price = TokenPrice::new(2.0, 0.0);
        let u = Usage {
            uncached_input_tokens: 0,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 1_000_000,
            ..Default::default()
        };
        // 1M×2.0 + 1M×2.0 = $4.0 (both at input rate)
        assert!((price.cost_of(&u) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn openai_is_unpriced_without_a_price_and_resolved_with_one() {
        let acct = OpenAiProvider;
        let unpriced = acct.cost_of(usage(100, 50), None);
        assert_eq!(unpriced.resolution, CostResolution::Unpriced);
        assert_eq!(unpriced.usd, 0.0);
        // tokens survive so the consumer can price them later
        assert_eq!(unpriced.usage.prompt_tokens(), 100);

        let price = TokenPrice::new(1.0, 1.0); // $1/Mtok both ways
        let resolved = acct.cost_of(usage(1_000_000, 0), Some(&price));
        assert_eq!(resolved.resolution, CostResolution::Resolved);
        assert!((resolved.usd - 1.0).abs() < 1e-9);
    }

    #[test]
    fn openrouter_aggregates_fee_plus_byok_upstream() {
        // chat-completions shape: usage.cost is the fee, upstream is a separate
        // addend that cost_of sums.
        let acct = OpenRouterProvider;
        let mut u = usage(10, 5);
        u.cost = Some(0.001);
        u.upstream_inference_cost = Some(0.009);
        let outcome = acct.cost_of(u, None);
        assert_eq!(outcome.resolution, CostResolution::Resolved);
        assert!((outcome.usd - 0.010).abs() < 1e-9);
    }

    #[test]
    fn openrouter_all_in_generation_cost_is_not_double_counted() {
        // The /generation shape (produced by query_generation): the all-in cost is
        // in `cost` and `upstream_inference_cost` is None, so cost_of must NOT add
        // anything on top, returning exactly the all-in figure.
        let acct = OpenRouterProvider;
        let mut u = usage(10, 5);
        u.cost = Some(0.010); // already includes the BYOK upstream charge
        u.upstream_inference_cost = None;
        let outcome = acct.cost_of(u, None);
        assert!(
            (outcome.usd - 0.010).abs() < 1e-9,
            "must not re-add upstream"
        );
    }

    #[test]
    fn openrouter_no_native_cost_falls_back_to_price_then_unpriced() {
        let acct = OpenRouterProvider;
        // No native cost, no price -> Unpriced (not a fake $0).
        let no_price = acct.cost_of(usage(1_000_000, 0), None);
        assert_eq!(no_price.resolution, CostResolution::Unpriced);
        // No native cost, with price -> Resolved from tokens.
        let price = TokenPrice::new(2.0, 0.0);
        let priced = acct.cost_of(usage(1_000_000, 0), Some(&price));
        assert_eq!(priced.resolution, CostResolution::Resolved);
        assert!((priced.usd - 2.0).abs() < 1e-9);
    }

    // ---- Anthropic provider ---------------------------------------------------

    use crate::generator::CompletionParameters;
    use crate::message::Message;

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

    // ---- OpenRouter cache breakpoints -----------------------------------------

    #[test]
    fn openrouter_claude_marked_messages_carry_cache_control() {
        let p = OpenRouterProvider;
        let messages = vec![
            cached_msg(Message::system("big system")),
            Message::user("hi"),
            cached_msg(Message::user("monitor")),
        ];
        let body = p
            .build_request(
                "anthropic/claude-sonnet-4.5",
                &messages,
                &CompletionParameters::new(),
                false,
                true,
            )
            .unwrap();
        let msgs = body["messages"].as_array().unwrap();
        // Marked messages switch to the block-array form carrying the marker;
        // an unmarked one stays a plain string.
        assert_eq!(msgs[0]["content"][0]["text"], "big system");
        assert_eq!(msgs[0]["content"][0]["cache_control"]["type"], "ephemeral");
        assert_eq!(msgs[1]["content"], "hi");
        assert_eq!(msgs[2]["content"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn openrouter_non_claude_model_emits_no_markers() {
        let p = OpenRouterProvider;
        let messages = vec![cached_msg(Message::user("x"))];
        let payload = p.openai_messages_value("openai/gpt-5", &messages);
        assert_eq!(payload[0]["content"], "x");
    }

    #[test]
    fn openrouter_caps_markers_at_four_keeping_the_last() {
        let p = OpenRouterProvider;
        let messages: Vec<Message> = (0..5)
            .map(|i| cached_msg(Message::user(format!("m{i}"))))
            .collect();
        let payload = p.openai_messages_value("anthropic/claude-opus-4", &messages);
        assert_eq!(payload[0]["content"], "m0", "oldest mark dropped");
        for msg in &payload[1..5] {
            assert_eq!(msg["content"][0]["cache_control"]["type"], "ephemeral");
        }
    }

    #[test]
    fn openrouter_marked_tool_result_carries_cache_control() {
        let p = OpenRouterProvider;
        let messages = vec![cached_msg(Message::tool("c1", "result"))];
        let payload = p.openai_messages_value("anthropic/claude-opus-4", &messages);
        assert_eq!(payload[0]["tool_call_id"], "c1");
        assert_eq!(
            payload[0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
    }

    #[test]
    fn openrouter_marked_pure_tool_call_assistant_drops_marker() {
        let p = OpenRouterProvider;
        let call = crate::tools::ToolCall {
            id: "c1".to_string(),
            name: "f".to_string(),
            arguments: "{}".to_string(),
        };
        let assistant = Message {
            tool_calls: Some(vec![call]),
            ..Message::assistant("")
        };
        let payload = p.openai_messages_value("anthropic/claude-opus-4", &[cached_msg(assistant)]);
        // No markable text content: the marker is dropped (prefix falls back
        // to the previous breakpoint), the tool_calls stay intact.
        assert_eq!(payload[0]["content"], "");
        assert!(payload[0]["tool_calls"].is_array());
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

    /// A real BYOK `/generation` record (captured live): OpenRouter's own
    /// charge is 0 and the upstream provider charge, billed on the user's
    /// key, is in `upstream_inference_cost`. The parsed cost is their sum,
    /// never a fake $0.
    #[test]
    fn a_byok_generation_record_books_the_upstream_charge() {
        let data = serde_json::json!({
            "tokens_prompt": 22, "tokens_completion": 2625,
            "native_tokens_prompt": 22, "native_tokens_completion": 2230,
            "native_tokens_reasoning": 0, "native_tokens_cached": 0,
            "is_byok": true, "total_cost": 0, "upstream_inference_cost": 0.0008942,
        });
        let usage = usage_from_generation_record(&data).expect("parses");
        assert!((usage.cost.unwrap() - 0.0008942).abs() < 1e-12);
        assert_eq!(usage.upstream_inference_cost, None, "already summed; must not re-add");
        assert_eq!(usage.uncached_input_tokens, 22);
        assert_eq!(usage.completion_tokens, 2625);
    }

    #[test]
    fn a_credits_generation_record_books_total_cost() {
        let data = serde_json::json!({
            "tokens_prompt": 30, "tokens_completion": 1800,
            "total_cost": 0.000723, "upstream_inference_cost": null,
        });
        let usage = usage_from_generation_record(&data).expect("parses");
        assert!((usage.cost.unwrap() - 0.000723).abs() < 1e-12);
    }

    #[test]
    fn a_generation_record_without_total_cost_is_unresolved_not_free() {
        let data = serde_json::json!({ "tokens_prompt": 30, "tokens_completion": 1800 });
        assert!(usage_from_generation_record(&data).is_none());
    }
}
