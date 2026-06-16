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
/// already set.
pub(crate) fn openai_build_request(
    model: &str,
    messages: &[Message],
    params: &CompletionParameters,
    stream: bool,
    include_usage: bool,
    token_limit_field: &str,
    request_usage: impl FnOnce(&mut serde_json::Value),
) -> Result<serde_json::Value> {
    let mut body = serde_json::json!({
        "model": model,
        "messages": messages_to_payload(messages),
        "stream": stream,
    });
    let obj = body.as_object_mut().expect("json object");

    // Normalized sampling/intent fields → OpenAI keys.
    if let Some(v) = params.max_tokens {
        obj.insert(token_limit_field.to_string(), serde_json::json!(v));
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
        obj.insert("tools".into(), serde_json::json!(v));
    }
    if let Some(v) = &params.tool_choice {
        obj.insert("tool_choice".into(), v.clone());
    }
    if let Some(v) = &params.reasoning {
        obj.insert("reasoning".into(), serde_json::to_value(v)?);
    }

    if include_usage {
        request_usage(&mut body);
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
            // OpenRouter may not finalize the generation record immediately; retry
            // with backoff before giving up to an honest Unknown.
            for delay_secs in [1u64, 2, 4] {
                tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                if let Some(usage) = query_generation(ctx.client, ctx.generation_id, ctx.auth).await
                {
                    return self.cost_of(usage, ctx.price);
                }
                tracing::debug!(
                    "OpenRouter generation {} not found yet (waited {}s)",
                    ctx.generation_id,
                    delay_secs
                );
            }
            CostOutcome::unknown()
        })
    }
}

/// Query OpenRouter's `/api/v1/generation` for a finished generation's usage.
/// `None` on any failure or when the record carries no usable cost.
async fn query_generation(
    client: &reqwest::Client,
    generation_id: &str,
    auth: &Auth,
) -> Option<Usage> {
    let api_key = auth.secret()?;
    let encoded =
        url::form_urlencoded::byte_serialize(generation_id.as_bytes()).collect::<String>();
    let url = format!("https://openrouter.ai/api/v1/generation?id={}", encoded);

    let response = match client
        .get(&url)
        .header(
            "Authorization",
            format!("Bearer {}", api_key.expose_secret()),
        )
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
    let data = json.get("data")?;

    // The /generation record uses different field names than chat-completions
    // usage. Require a numeric total_cost: a record without it is unresolved, not
    // free. tokens come from the native_tokens_* fields.
    //
    // IMPORTANT: unlike chat-completions `usage.cost` (the OpenRouter fee only,
    // with the BYOK upstream charge in a SEPARATE field that `cost_of` adds),
    // `/generation.total_cost` is the ALL-IN charge (upstream + fee). So we put it
    // in `cost` and leave `upstream_inference_cost: None`, otherwise `cost_of`
    // would re-add the upstream charge and double-count it.
    let cost = data["total_cost"].as_f64()?;
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
    /// Map a normalized message's content to Anthropic's content shape. A plain
    /// string for text-only content (our common case); the block-array form with a
    /// `cache_control` marker when `cached` (a string can't carry the marker).
    ///
    /// Multimodal content (image/audio/video parts) has no Anthropic mapping wired
    /// yet, and Anthropic's block shape differs from the OpenAI-shaped normalized
    /// parts, so a multimodal message FAILS LOUDLY rather than silently shipping a
    /// text-only request that drops the attachment. (Wiring Anthropic image/document
    /// blocks is a clean future extension.)
    /// The message's full text, FAILING LOUDLY on multimodal content (image/audio/
    /// video) which has no Anthropic mapping wired yet. `all_text()` joins every
    /// text part, so a multi-text message never silently drops its later parts the
    /// way `get_text()` (first part only) would. Shared by the turn and system paths.
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

    fn message_content(msg: &Message, cached: bool) -> Result<serde_json::Value> {
        let text = Self::text_only(msg)?;
        Ok(if cached {
            serde_json::json!([{
                "type": "text",
                "text": text,
                "cache_control": {"type": "ephemeral"},
            }])
        } else {
            serde_json::json!(text)
        })
    }
}

impl Provider for AnthropicProvider {
    fn endpoint_url(&self, base_url: &str) -> String {
        format!("{}/v1/messages", base_url.trim_end_matches('/'))
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
        // rather than silently dropping them (which would, e.g., make tools never
        // fire even though the response parser DOES handle Anthropic tool_use).
        // Each is a clean future translation, not a silent "for now" omission.
        for (present, field) in [
            (params.tools.is_some(), "tools"),
            (params.tool_choice.is_some(), "tool_choice"),
            (params.response_format.is_some(), "response_format"),
            (params.reasoning.is_some(), "reasoning"),
        ] {
            if present {
                return Err(MiniLLMError::InvalidParameter(format!(
                    "the Anthropic provider does not yet translate `{field}`; omit it or use an OpenAI-wire provider"
                )));
            }
        }

        // Enforce Anthropic's 4-breakpoint cap: of all marked messages, only the
        // last 4 actually get a marker (most-recent prefix = the largest reusable
        // span). Compute the set of message indices that keep their breakpoint.
        const MAX_BREAKPOINTS: usize = 4;
        let marked: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(|(_, m)| m.cache_breakpoint)
            .map(|(i, _)| i)
            .collect();
        if marked.len() > MAX_BREAKPOINTS {
            tracing::warn!(
                "Anthropic allows at most {} cache breakpoints; {} were marked, keeping the last {}",
                MAX_BREAKPOINTS,
                marked.len(),
                MAX_BREAKPOINTS
            );
        }
        let kept: std::collections::HashSet<usize> =
            marked.iter().rev().take(MAX_BREAKPOINTS).copied().collect();

        // System turns are hoisted. Track whether any hoisted system message is a
        // (kept) breakpoint so the system block carries the marker.
        let mut system = String::new();
        let mut system_cached = false;
        let mut turns: Vec<serde_json::Value> = Vec::new();
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
                turns.push(serde_json::json!({
                    "role": msg.role,
                    "content": Self::message_content(msg, cached)?,
                }));
            }
        }

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
        // dropped (a dropped `tools` would make the model never call a tool even
        // though the response parser handles tool_use). One assertion per field, so
        // removing any entry from the production list fails this test.
        use crate::generator::ReasoningConfig;
        let p = AnthropicProvider;
        let messages = vec![Message::user("Hi")];

        let cases: Vec<(&str, CompletionParameters)> = vec![
            (
                "tools",
                CompletionParameters {
                    tools: Some(vec![serde_json::json!({"type": "function"})]),
                    ..CompletionParameters::new()
                },
            ),
            (
                "tool_choice",
                CompletionParameters {
                    tool_choice: Some(serde_json::json!("auto")),
                    ..CompletionParameters::new()
                },
            ),
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
        // The marked user turn is a block-array with cache_control; the unmarked
        // one stays a plain string.
        assert_eq!(body["messages"][0]["content"][0]["text"], "cache me");
        assert_eq!(
            body["messages"][0]["content"][0]["cache_control"]["type"],
            "ephemeral"
        );
        assert!(body["messages"][1]["content"].is_string());
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
        let turns = body["messages"].as_array().unwrap();
        // turn0 (the oldest, dropped) → plain string; turns 1..=4 → cached blocks.
        assert!(turns[0]["content"].is_string(), "oldest mark dropped");
        for t in &turns[1..5] {
            assert_eq!(t["content"][0]["cache_control"]["type"], "ephemeral");
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
}
