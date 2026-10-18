//! OpenRouter: every piece of OpenRouter-specific wire knowledge.
//!
//! OpenAI-dialect request/response (the trait defaults), plus native USD
//! cost (`usage.cost` + BYOK `usage.cost_details.upstream_inference_cost`),
//! usage opt-in via `usage:{include:true}`, app attribution headers,
//! Claude-model cache markers, and the out-of-band `/generation` endpoint
//! for streams that died before delivering usage.

use super::super::openai_wire::messages_to_payload;
use super::super::openai_wire::{mark_openai_message, parse_openai_usage, usage_field};
use super::super::response::Usage;
use super::super::wire::{
    kept_cache_breakpoints, price_or_unpriced, AppIdentity, CostOutcome, InfoFuture, ModelLookup,
    Provider, SideCall, TokenPrice, UsageFuture, UsageQuery,
};
use crate::message::Message;
use serde::{Deserialize, Serialize};

/// OpenRouter provider routing settings
///
/// See: <https://openrouter.ai/docs/provider-routing>
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OpenRouterRouting {
    /// Ordered list of provider names to try
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<Vec<String>>,

    /// Sort providers by: "price", "throughput", or "latency"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sort: Option<String>,

    /// List of providers to ignore/exclude
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ignore: Option<Vec<String>>,

    /// Data collection preference: "allow" or "deny"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_collection: Option<String>,

    /// Allow fallback to other providers if preferred ones fail
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,

    /// Require specific provider parameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
}

impl OpenRouterRouting {
    /// Create new provider settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialize to the JSON value OpenRouter expects under the request's
    /// `"provider"` key (used by
    /// [`CompletionParameters::with_openrouter_routing`](crate::CompletionParameters::with_openrouter_routing)).
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("routing settings are plain strings and flags")
    }

    /// Set ordered list of providers to try
    pub fn with_order(mut self, providers: Vec<String>) -> Self {
        self.order = Some(providers);
        self
    }

    /// Sort by throughput (fastest)
    pub fn sort_by_throughput(mut self) -> Self {
        self.sort = Some("throughput".to_string());
        self
    }

    /// Sort by price (cheapest)
    pub fn sort_by_price(mut self) -> Self {
        self.sort = Some("price".to_string());
        self
    }

    /// Sort by latency (lowest)
    pub fn sort_by_latency(mut self) -> Self {
        self.sort = Some("latency".to_string());
        self
    }

    /// Set providers to ignore/exclude
    pub fn with_ignore(mut self, providers: Vec<String>) -> Self {
        self.ignore = Some(providers);
        self
    }

    /// Deny data collection
    pub fn deny_data_collection(mut self) -> Self {
        self.data_collection = Some("deny".to_string());
        self
    }

    /// Allow data collection
    pub fn allow_data_collection(mut self) -> Self {
        self.data_collection = Some("allow".to_string());
        self
    }

    /// Set whether to allow fallbacks
    pub fn with_fallbacks(mut self, allow: bool) -> Self {
        self.allow_fallbacks = Some(allow);
        self
    }
}

impl crate::generator::CompletionParameters {
    /// Set OpenRouter provider-routing settings (order/sort/ignore/data_collection).
    ///
    /// OpenRouter-specific: it goes through [`extra`](Self::extra) under the
    /// `"provider"` key, the home for a provider-specific knob. Every provider
    /// sends that key, and only OpenRouter understands it: OpenAI and Anthropic
    /// refuse a request carrying it, so set it only for OpenRouter.
    pub fn with_openrouter_routing(self, routing: OpenRouterRouting) -> Self {
        self.with_extra("provider", routing.to_value())
    }
}

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

    /// OpenRouter's own `reasoning` object, which it translates per upstream
    /// model itself: off is `effort: "none"`, on with no effort is
    /// `"minimal"` (OpenRouter's lowest, which it rounds to the nearest level
    /// a model has), and an effort is the caller's word as written.
    fn openai_reasoning_field(
        &self,
        reasoning: &crate::generator::Reasoning,
    ) -> (&'static str, serde_json::Value) {
        use crate::generator::Reasoning;
        let effort = match reasoning {
            Reasoning::Off => "none",
            Reasoning::On(None) => "minimal",
            Reasoning::On(Some(effort)) => effort,
        };
        ("reasoning", serde_json::json!({ "effort": effort }))
    }

    /// The model from OpenRouter's catalog, read through the generator's own
    /// address (a gateway in front of OpenRouter sees it), at the dearest
    /// provider that serves it, since routing may land on any of them.
    fn model_info(&self, lookup: ModelLookup) -> Option<InfoFuture> {
        let model = lookup.openrouter_name.unwrap_or(lookup.model);
        Some(Box::pin(async move {
            super::catalog_info(&lookup.call, &model, None).await
        }))
    }

    /// OpenRouter normalizes requests before forwarding, so unknown keys in
    /// message parts never reach a strict upstream (verified empirically:
    /// extra keys inside text/image parts, inside `image_url`, and at the
    /// top level all pass, on OpenAI- and Anthropic-served models alike).
    /// Keeping the metadata is what lets an in-flight meter price media
    /// exactly from the request bytes.
    fn wire_keeps_estimation_metadata(&self) -> bool {
        true
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
        let mut payload = messages_to_payload(messages, self.wire_keeps_estimation_metadata());
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

    fn usage_out_of_band<'a>(&'a self, ctx: UsageQuery<'a>) -> UsageFuture<'a> {
        Box::pin(async move {
            if ctx.generation_id.is_empty() {
                return None;
            }
            // OpenRouter may not finalize the generation record immediately; poll
            // every second before giving up (the cost is then Unknown). Plain 1s polls,
            // no backoff: the endpoint is free and the caller is waiting, so the
            // only cost of polling fast is nothing and the cost of polling slow
            // is user-visible latency. Measured: a completed generation's record
            // appears ~9s after it finishes, and a CANCELLED call's only after
            // the upstream generation runs to its own end anyway (client aborts
            // do not stop these routes) plus the same ~9s, i.e. ~18s for a short
            // generation. We poll for 25s in all, slow queries included.
            let poll = async {
                loop {
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                    if let Some(usage) = query_generation(&ctx.call, ctx.generation_id).await {
                        return usage;
                    }
                    tracing::debug!("OpenRouter generation {} not found yet", ctx.generation_id);
                }
            };
            tokio::time::timeout(std::time::Duration::from_secs(25), poll)
                .await
                .ok()
        })
    }
}

/// How long one generation-record query may take, so one hung query does not
/// use up the whole poll.
const GENERATION_QUERY_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

/// Query OpenRouter's `/api/v1/generation` for a finished generation's usage,
/// at the generator's own address with its own headers. `None` on any failure
/// or when the record carries no usable cost.
async fn query_generation(call: &SideCall, generation_id: &str) -> Option<Usage> {
    let encoded =
        url::form_urlencoded::byte_serialize(generation_id.as_bytes()).collect::<String>();
    let (status, body) = match call
        .get(
            &format!("/generation?id={encoded}"),
            GENERATION_QUERY_TIMEOUT,
        )
        .await
    {
        Ok(reply) => reply,
        Err(e) => {
            tracing::error!("Generation cost query for {} failed: {}", generation_id, e);
            return None;
        }
    };
    if !status.is_success() {
        tracing::warn!(
            "Generation cost query for {} returned {}",
            generation_id,
            status
        );
        return None;
    }
    let usage = serde_json::from_str::<serde_json::Value>(&body)
        .ok()
        .and_then(|json| usage_from_generation_record(json.get("data")?));
    if usage.is_none() {
        tracing::warn!(
            "Generation cost query for {} returned a record with no usable cost: {}",
            generation_id,
            super::super::response::preview_str(&body)
        );
    }
    usage
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::CostResolution;

    use crate::generator::CompletionParameters;

    /// OpenRouter takes reasoning as its own `reasoning` object, never
    /// OpenAI's `reasoning_effort`: `"none"` for off, `"minimal"` for on with
    /// no effort, and the caller's word as written otherwise.
    #[test]
    fn openrouter_reasoning_is_its_reasoning_object() {
        use crate::generator::Reasoning;
        let p = OpenRouterProvider;
        let messages = vec![Message::user("Hi")];
        for (reasoning, effort) in [
            (Reasoning::Off, "none"),
            (Reasoning::On(None), "minimal"),
            (Reasoning::On(Some("high".into())), "high"),
        ] {
            let params = CompletionParameters::new().with_reasoning(reasoning);
            let body = p
                .build_request("z-ai/glm-5.3-flash", &messages, &params, false, true)
                .unwrap();
            assert_eq!(body["reasoning"], serde_json::json!({ "effort": effort }));
            assert!(body.get("reasoning_effort").is_none());
        }
    }

    /// A message with its cache breakpoint set (the builder input the
    /// marker tests exercise).
    fn cached_msg(m: Message) -> Message {
        Message {
            cache_breakpoint: true,
            ..m
        }
    }

    /// A fully-uncached usage (all input in the full-price bucket).
    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            uncached_input_tokens: prompt,
            completion_tokens: completion,
            ..Default::default()
        }
    }

    /// The OpenRouter accounting parses its nested usage shape and sums the
    /// fee + BYOK upstream charge in its own cost_of (the aggregation that
    /// must stay provider-specific).
    #[test]
    fn openrouter_parses_usage_and_aggregates_byok_cost() {
        let raw = serde_json::json!({
            "usage": {
                "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
                "cost": 0.001,
                "cost_details": {"upstream_inference_cost": 0.009},
                "prompt_tokens_details": {"cached_tokens": 4},
                "completion_tokens_details": {"reasoning_tokens": 2}
            }
        });
        let p = OpenRouterProvider;
        let usage = p.parse_usage(&raw).expect("usage parsed");
        assert_eq!(usage.prompt_tokens(), 10, "total input = sum of buckets");
        assert_eq!(
            usage.cache_read_tokens, 4,
            "cached_tokens -> cache_read bucket"
        );
        assert_eq!(
            usage.uncached_input_tokens, 6,
            "10 total - 4 cached = 6 uncached"
        );
        assert_eq!(usage.upstream_inference_cost, Some(0.009));
        assert_eq!(usage.reasoning_tokens, Some(2));

        let outcome = p.cost_of(usage, None);
        assert_eq!(outcome.resolution, CostResolution::Resolved);
        assert!((outcome.usd - 0.010).abs() < 1e-9);
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
        assert_eq!(
            usage.upstream_inference_cost, None,
            "already summed; must not re-add"
        );
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
