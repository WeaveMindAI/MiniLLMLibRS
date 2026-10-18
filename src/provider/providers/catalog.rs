//! OpenRouter's model catalog, read into a [`ModelInfo`]. It lists first-party
//! models at the vendors' own standard rates, so it also prices direct OpenAI
//! and Anthropic models (neither vendor publishes prices in its own API): every
//! built-in provider's [`Provider::model_info`](crate::Provider::model_info)
//! reads it, and so can yours.
//!
//! A price is a property of (model, provider), never a model alone: the same
//! model spans a 5x range across providers, so a read names the provider that
//! serves the call, or takes the dearest of them all. Rates arrive as USD
//! **per token** (decimal strings); [`TokenPrice`] wants USD per **million**
//! tokens. The conversion happens here, exactly once.

use std::time::Duration;

use serde::Deserialize;

use crate::error::{MiniLLMError, Result};
use crate::provider::model_id::validate_model_id;
use crate::provider::response::preview_str;
use crate::provider::wire::{ModelInfo, SideCall, TokenPrice};

/// How long to wait on the catalog before giving up.
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// Where OpenRouter's public catalog is read: `openrouter.ai` over `client`,
/// with no credential (the catalog needs none). An OpenRouter generator reads
/// it through its own address instead, so a gateway in front of OpenRouter
/// sees the read.
pub fn catalog_call(client: reqwest_middleware::ClientWithMiddleware) -> SideCall {
    SideCall {
        client,
        base_url: "https://openrouter.ai/api/v1".into(),
        headers: Default::default(),
    }
}

/// What OpenRouter's catalog, read through `call`, says about `model` (its
/// catalog id, such as `openai/gpt-4o`): priced as `provider` serves it (an
/// OpenRouter slug such as `"openai"` or `"groq"`), or at the dearest provider
/// of the model when `provider` is `None` or does not serve it. A model the
/// catalog does not list, or a read that fails, comes back unknown; either is
/// logged once per read. The read is shared across the process and kept for an
/// hour ([`SideCall::get_cached`]).
pub async fn catalog_info(call: &SideCall, model: &str, provider: Option<&str>) -> ModelInfo {
    if let Err(e) = validate_model_id(model) {
        tracing::warn!("cannot read OpenRouter's catalog: {e}");
        return ModelInfo::default();
    }
    let path = format!("/models/{model}/endpoints");
    let url = call.url(&path);
    let id = model.to_string();
    let endpoints = call
        .get_cached(&path, FETCH_TIMEOUT, move |status, body| {
            parse_endpoints(&id, &url, status, body)
        })
        .await;
    match endpoints {
        Ok(Some(endpoints)) => select(&endpoints, provider),
        // Already logged by the read.
        Ok(None) | Err(_) => ModelInfo::default(),
    }
}

/// Every endpoint's prices for `model` from one catalog reply; `None` when
/// the catalog does not list it (logged here, once per read).
fn parse_endpoints(
    model: &str,
    url: &str,
    status: reqwest::StatusCode,
    body: &str,
) -> Result<Option<Vec<PricedEndpoint>>> {
    if status == reqwest::StatusCode::NOT_FOUND {
        tracing::warn!(
            "OpenRouter's catalog does not list `{model}`, so its price is unknown; set the \
             generator's `with_openrouter_name` to its catalog id"
        );
        return Ok(None);
    }
    if !status.is_success() {
        return Err(MiniLLMError::Api {
            status: status.as_u16(),
            message: format!("reading {url} failed: {}", preview_str(body)),
        });
    }
    let body: EndpointsResponse = serde_json::from_str(body)?;
    if body.data.endpoints.is_empty() {
        tracing::warn!(
            "OpenRouter's catalog lists `{model}` with no endpoints, so its price is unknown"
        );
        return Ok(None);
    }
    price_endpoints(body.data.endpoints).map(Some)
}

// ----- Wire shapes ----------------------------------------------------
//
// Only the fields we price from are named; the rest of each entry is ignored.

#[derive(Deserialize)]
struct EndpointsResponse {
    data: EndpointsData,
}

#[derive(Deserialize)]
struct EndpointsData {
    endpoints: Vec<Endpoint>,
}

#[derive(Deserialize)]
struct Endpoint {
    /// The stable provider slug, optionally suffixed with a region or variant
    /// (`anthropic`, `amazon-bedrock/us-east-1`). The part before the slash is
    /// the provider; the suffix distinguishes that provider's own endpoints.
    tag: String,
    pricing: Pricing,
    #[serde(default)]
    max_completion_tokens: Option<u32>,
    #[serde(default)]
    context_length: Option<u32>,
}

impl Endpoint {
    /// The provider slug, without the region or variant suffix.
    fn provider_slug(&self) -> &str {
        self.tag.split('/').next().unwrap_or(&self.tag)
    }
}

/// Rates as published: USD per token, as decimal strings. An endpoint that does
/// not support a bucket simply omits its key.
#[derive(Deserialize)]
struct Pricing {
    prompt: String,
    completion: String,
    #[serde(default)]
    input_cache_read: Option<String>,
    #[serde(default)]
    input_cache_write: Option<String>,
    /// Audio input, per token. Absent on models that bill audio as plain input.
    #[serde(default)]
    audio: Option<String>,
    /// Image input, per token. Absent on models that bill images as plain input.
    #[serde(default)]
    image: Option<String>,
}

/// Parse a per-token rate string into USD per million tokens.
///
/// A rate must be a finite, non-negative number. Anything else is a corrupt
/// catalog, and every other reading of it silently understates the bound:
/// `"NaN"` parses, and because `f64::max` returns its non-NaN operand a NaN rate
/// is DISCARDED by [`select`], handing the bound to a cheaper sibling endpoint.
/// `"inf"` and `"1e400"` parse to infinity, and a negative rate lowers the bound
/// directly. So they are rejected here, at the one place a rate enters the
/// program, rather than defended against everywhere downstream.
fn per_mtok(field: &str, raw: &str) -> Result<f64> {
    let malformed = |why: &str| {
        MiniLLMError::MalformedResponse(format!("model catalog: {field} rate {raw:?} {why}"))
    };
    let parsed = raw
        .parse::<f64>()
        .map_err(|_| malformed("is not a number"))?;
    if !parsed.is_finite() {
        return Err(malformed("is not finite"));
    }
    if parsed < 0.0 {
        return Err(malformed("is negative"));
    }
    Ok(parsed * 1_000_000.0)
}

impl Endpoint {
    fn priced(&self) -> Result<PricedEndpoint> {
        let mut price = TokenPrice::new(
            per_mtok("prompt", &self.pricing.prompt)?,
            per_mtok("completion", &self.pricing.completion)?,
        );
        // Only set cache rates when BOTH are published. `TokenPrice` falls back to
        // the input rate for a missing bucket, which is the correct behaviour for
        // an endpoint that does not price that bucket separately.
        match (
            &self.pricing.input_cache_read,
            &self.pricing.input_cache_write,
        ) {
            (Some(r), Some(w)) => {
                price = price.with_cache_rates(
                    per_mtok("input_cache_read", r)?,
                    per_mtok("input_cache_write", w)?,
                )
            }
            // Read-only caching (OpenAI charges nothing to write): a write costs
            // the plain input rate, which is what leaving it unset yields.
            (Some(r), None) => price.cache_read_per_mtok = Some(per_mtok("input_cache_read", r)?),
            _ => {}
        }
        price = price.with_media_rates(
            self.pricing
                .audio
                .as_deref()
                .map(|r| per_mtok("audio", r))
                .transpose()?,
            self.pricing
                .image
                .as_deref()
                .map(|r| per_mtok("image", r))
                .transpose()?,
        );

        Ok(PricedEndpoint {
            provider_slug: self.provider_slug().to_string(),
            price,
            max_output_tokens: self.max_completion_tokens,
            context_length: self.context_length,
        })
    }
}

/// The dearer of two already-resolved rates. Both are `Some` by construction in
/// [`select`], where every fallback is applied before anything is compared.
fn max_rate(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.max(y)),
        (some, None) | (None, some) => some,
    }
}

/// One endpoint, parsed and validated: who serves it, what it charges, and
/// its limits.
#[derive(Debug, Clone)]
struct PricedEndpoint {
    provider_slug: String,
    price: TokenPrice,
    max_output_tokens: Option<u32>,
    context_length: Option<u32>,
}

/// Parse and validate every endpoint's prices before any is used, so one
/// corrupt rate fails the whole model rather than only the providers that
/// happen to be picked.
fn price_endpoints(endpoints: Vec<Endpoint>) -> Result<Vec<PricedEndpoint>> {
    endpoints.iter().map(Endpoint::priced).collect()
}

/// The rates that bound what this model can cost when `provider` serves it.
///
/// `provider` is an OpenRouter provider slug (`anthropic`, `openai`,
/// `amazon-bedrock`, ...) when the caller knows who will serve the call, `None`
/// when routing is free to pick anyone. A named provider that serves this model
/// through no endpoint falls back to ALL endpoints rather than failing: the
/// caller's knowledge did not match the catalog, so the only honest bound left is
/// the dearest anyone charges.
///
/// The answer is never one endpoint but the dearest RATE from every candidate,
/// taken bucket by bucket: picking one endpoint whole would inherit its cheap
/// buckets alongside its dear one, and a sibling could bill more on exactly
/// those. The token limits are likewise the most permissive, so an output bound
/// is never understated. For a single candidate this is exactly its price.
fn select(endpoints: &[PricedEndpoint], provider: Option<&str>) -> ModelInfo {
    let matches_provider = |e: &&PricedEndpoint| match provider {
        Some(slug) => e.provider_slug.eq_ignore_ascii_case(slug),
        None => true,
    };
    let candidates: Vec<&PricedEndpoint> = match endpoints
        .iter()
        .filter(matches_provider)
        .collect::<Vec<_>>()
    {
        served if !served.is_empty() => served,
        _ => endpoints.iter().collect(),
    };

    let mut worst: Option<(TokenPrice, Option<u32>, Option<u32>)> = None;
    for endpoint in candidates {
        let price = &endpoint.price;
        let context = endpoint.context_length;
        // An absent bucket is not "no charge", it is "charged at this endpoint's
        // input rate". Resolving that fallback BEFORE comparing is what makes the
        // maximum meaningful: an endpoint that omits its cache-read rate really is
        // dearer on cache reads than one that publishes a discount.
        let (input, output) = (price.input_per_mtok, price.output_per_mtok);
        let effective = TokenPrice {
            input_per_mtok: input,
            output_per_mtok: output,
            cache_read_per_mtok: Some(price.cache_read_per_mtok.unwrap_or(input)),
            cache_write_per_mtok: Some(price.cache_write_per_mtok.unwrap_or(input)),
            audio_per_mtok: Some(price.audio_rate()),
            image_per_mtok: Some(price.image_rate()),
        };

        worst = Some(match worst {
            None => (effective, endpoint.max_output_tokens, context),
            Some((w, w_max, w_context)) => (
                TokenPrice {
                    input_per_mtok: w.input_per_mtok.max(input),
                    output_per_mtok: w.output_per_mtok.max(output),
                    cache_read_per_mtok: max_rate(
                        w.cache_read_per_mtok,
                        effective.cache_read_per_mtok,
                    ),
                    cache_write_per_mtok: max_rate(
                        w.cache_write_per_mtok,
                        effective.cache_write_per_mtok,
                    ),
                    audio_per_mtok: max_rate(w.audio_per_mtok, effective.audio_per_mtok),
                    image_per_mtok: max_rate(w.image_per_mtok, effective.image_per_mtok),
                },
                match (w_max, endpoint.max_output_tokens) {
                    // No published cap is the LEAST restrictive, so it wins.
                    (None, _) | (_, None) => None,
                    (Some(a), Some(b)) => Some(a.max(b)),
                },
                // The widest window any candidate publishes.
                w_context.max(context),
            ),
        });
    }

    match worst {
        Some((price, max_output_tokens, context_length)) => ModelInfo {
            price: Some(price),
            max_output_tokens,
            context_length,
        },
        None => ModelInfo::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a catalog response the same way a fetch does, so a test sees the
    /// failure exactly where a real response would produce it.
    fn parse(json: &str) -> Result<Vec<PricedEndpoint>> {
        price_endpoints(
            serde_json::from_str::<EndpointsResponse>(json)
                .unwrap()
                .data
                .endpoints,
        )
    }

    /// The same, for a response known to be well formed.
    fn endpoints(json: &str) -> Vec<PricedEndpoint> {
        parse(json).expect("fixture is well formed")
    }

    /// Rates come from parsing a decimal string and scaling by a million, so they
    /// carry float error and must never be compared with `==`.
    #[track_caller]
    fn assert_rate(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-9,
            "rate {actual} is not {expected}"
        );
    }

    /// Three providers, three prices. Real shape: `glm-5.2` spans 0.54 to 3.00 in
    /// and 1.76 to 10.25 out across 27 endpoints.
    const SPREAD: &str = r#"{"data":{"endpoints":[
        {"tag":"ambient","context_length":128000,"max_completion_tokens":8192,
         "pricing":{"prompt":"0.00000054","completion":"0.00000176"}},
        {"tag":"fireworks","context_length":128000,"max_completion_tokens":4096,
         "pricing":{"prompt":"0.0000021","completion":"0.0000066"}},
        {"tag":"wafer","context_length":64000,"max_completion_tokens":2048,
         "pricing":{"prompt":"0.000003","completion":"0.00001025"}}
    ]}}"#;

    /// With no provider pinned, routing may land anywhere, so every rate must be
    /// the dearest any candidate charges. Taking the advertised or cheapest rate
    /// here under-reserved a real request by 3.9x.
    #[test]
    fn an_unpinned_provider_is_priced_at_the_dearest_rate_of_any_endpoint() {
        let rates = select(&endpoints(SPREAD), None);
        assert_rate(rates.price.as_ref().unwrap().input_per_mtok, 3.0);
        assert_rate(rates.price.as_ref().unwrap().output_per_mtok, 10.25);
        // The limits are the most PERMISSIVE, so the output bound is never
        // understated: 8192 from the cheap endpoint, 64000 context from the dear
        // one. Taking them from whichever endpoint set the price would let a
        // caller's max_tokens be clamped below what routing would actually allow.
        assert_eq!(rates.max_output_tokens, Some(8192));
        assert_eq!(rates.context_length, Some(128_000));
    }

    /// Picking one endpoint whole would inherit its cheap buckets. The bound is
    /// taken bucket by bucket, so a sibling that is dearer on audio alone still
    /// raises the audio rate.
    #[test]
    fn each_rate_is_bounded_independently_of_the_others() {
        let json = r#"{"data":{"endpoints":[
            {"tag":"cheap-text-dear-audio","context_length":1000,
             "pricing":{"prompt":"0.000001","completion":"0.000002","audio":"0.0001"}},
            {"tag":"dear-text-cheap-audio","context_length":1000,
             "pricing":{"prompt":"0.000005","completion":"0.000009","audio":"0.000005"}}
        ]}}"#;
        let rates = select(&endpoints(json), None);
        assert_rate(rates.price.as_ref().unwrap().input_per_mtok, 5.0);
        assert_rate(rates.price.as_ref().unwrap().output_per_mtok, 9.0);
        // The dear-audio endpoint is cheap on text, and vice versa. Both maxima win.
        assert_rate(rates.price.as_ref().unwrap().audio_rate(), 100.0);
    }

    #[test]
    fn a_pinned_provider_is_priced_at_that_providers_endpoint() {
        let rates = select(&endpoints(SPREAD), Some("fireworks"));
        assert_rate(rates.price.as_ref().unwrap().input_per_mtok, 2.1);
        assert_rate(rates.price.as_ref().unwrap().output_per_mtok, 6.6);
        assert_eq!(rates.max_output_tokens, Some(4096));
    }

    /// One provider commonly owns several endpoints (regions, tiers) at different
    /// prices. Pinning the provider does not pin the price, so bound it.
    #[test]
    fn a_provider_owning_several_endpoints_is_priced_at_its_dearest() {
        let json = r#"{"data":{"endpoints":[
            {"tag":"amazon-bedrock/us-east-1","context_length":200000,
             "pricing":{"prompt":"0.0000022","completion":"0.000011"}},
            {"tag":"amazon-bedrock/global","context_length":200000,
             "pricing":{"prompt":"0.000002","completion":"0.00001"}},
            {"tag":"anthropic","context_length":200000,
             "pricing":{"prompt":"0.000002","completion":"0.00001"}}
        ]}}"#;
        let bedrock = select(&endpoints(json), Some("amazon-bedrock"));
        assert_rate(bedrock.price.as_ref().unwrap().output_per_mtok, 11.0);

        let direct = select(&endpoints(json), Some("anthropic"));
        assert_rate(direct.price.as_ref().unwrap().output_per_mtok, 10.0);
    }

    /// A provider that serves the model through no endpoint prices at the dearest
    /// of ALL endpoints: the caller's knowledge did not match the catalog, and the
    /// only bound that still holds wherever the call really lands is the dearest
    /// anyone charges. An error here would break estimation for a name mismatch
    /// that costs nothing to bound over.
    #[test]
    fn a_provider_that_does_not_serve_the_model_prices_at_the_dearest_of_all() {
        let fallback = select(&endpoints(SPREAD), Some("anthropic"));
        let dearest = select(&endpoints(SPREAD), None);
        assert_eq!(fallback, dearest);
    }

    /// Provider slugs are lowercase; a generator's display name ("Anthropic") is
    /// not. The match is case-insensitive so the name works as the fallback slug.
    #[test]
    fn a_provider_name_matches_its_slug_regardless_of_case() {
        let pinned = select(&endpoints(SPREAD), Some("Fireworks"));
        assert_rate(pinned.price.as_ref().unwrap().input_per_mtok, 2.1);
    }

    #[test]
    fn per_token_strings_become_per_million_rates_with_their_cache_buckets() {
        let json = r#"{"data":{"endpoints":[{"tag":"anthropic","context_length":1000000,
            "max_completion_tokens":128000,
            "pricing":{"prompt":"0.000003","completion":"0.000015",
                       "input_cache_read":"0.0000003","input_cache_write":"0.00000375"}}]}}"#;
        let rates = select(&endpoints(json), None);
        assert_rate(rates.price.as_ref().unwrap().input_per_mtok, 3.0);
        assert_rate(rates.price.as_ref().unwrap().output_per_mtok, 15.0);
        assert_rate(
            rates.price.as_ref().unwrap().cache_read_per_mtok.unwrap(),
            0.3,
        );
        assert_rate(
            rates.price.as_ref().unwrap().cache_write_per_mtok.unwrap(),
            3.75,
        );
    }

    /// OpenAI publishes a cache-read discount and no write charge, so a write bills
    /// at the plain input rate. `select` resolves that fallback rather than leaving
    /// it unset: after bounding, every bucket has a real number, because "absent"
    /// meant "input rate" and that is what a sibling endpoint gets compared against.
    #[test]
    fn a_read_only_cache_resolves_its_write_rate_to_the_input_rate() {
        let json = r#"{"data":{"endpoints":[{"tag":"openai","context_length":400000,
            "pricing":{"prompt":"0.000005","completion":"0.00003","input_cache_read":"0.0000005"}}]}}"#;
        let rates = select(&endpoints(json), None);
        assert_rate(
            rates.price.as_ref().unwrap().cache_read_per_mtok.unwrap(),
            0.5,
        );
        assert_rate(
            rates.price.as_ref().unwrap().cache_write_per_mtok.unwrap(),
            5.0,
        );
        // This endpoint prices neither image nor audio. An image bills exactly as
        // text; audio always carries a premium, so it bills at the assumed multiple.
        assert_rate(rates.price.as_ref().unwrap().image_rate(), 5.0);
        assert_rate(
            rates.price.as_ref().unwrap().audio_rate(),
            5.0 * crate::provider::wire::AUDIO_RATE_FALLBACK_MULTIPLE,
        );

        // Which is what a write really costs, resolved or not.
        let usage = crate::provider::Usage {
            cache_write_tokens: 1_000_000,
            ..Default::default()
        };
        assert!((rates.price.as_ref().unwrap().cost_of(&usage) - 5.0).abs() < 1e-9);
    }

    /// A cache-read rate is a DISCOUNT, so an endpoint that omits it (billing reads
    /// at the full input rate) is dearer on reads than one that publishes one. The
    /// bound must take the omission's resolved value, not treat it as free.
    #[test]
    fn an_endpoint_omitting_its_cache_discount_raises_the_bound() {
        let json = r#"{"data":{"endpoints":[
            {"tag":"discounted","context_length":1000,
             "pricing":{"prompt":"0.000005","completion":"0.00001","input_cache_read":"0.0000005"}},
            {"tag":"no-discount","context_length":1000,
             "pricing":{"prompt":"0.000005","completion":"0.00001"}}
        ]}}"#;
        let rates = select(&endpoints(json), None);
        // Not 0.5: the second endpoint charges the full 5.0 for a cache read.
        assert_rate(
            rates.price.as_ref().unwrap().cache_read_per_mtok.unwrap(),
            5.0,
        );
    }

    #[test]
    fn an_endpoint_with_no_completion_cap_keeps_its_context_window() {
        let json = r#"{"data":{"endpoints":[{"tag":"x","context_length":32768,
            "pricing":{"prompt":"0.0000001","completion":"0.0000002"}}]}}"#;
        let rates = select(&endpoints(json), None);
        assert_eq!(rates.max_output_tokens, None);
        assert_eq!(rates.context_length, Some(32_768));
    }

    /// Every rate that is not a finite, non-negative number is a corrupt catalog,
    /// and it is rejected when the response is parsed, not when a price is later
    /// read. So one bad endpoint condemns the whole model, loudly and immediately,
    /// rather than only the providers a selection happens to pick.
    ///
    /// `"NaN"` is the dangerous one: it parses, and `f64::max` returns its non-NaN
    /// operand, so a NaN rate on the DEAREST endpoint would be silently discarded
    /// and a cheaper sibling would win the bound.
    #[test]
    fn a_rate_that_is_not_a_finite_non_negative_number_fails_the_fetch() {
        for bad in [
            "free", "NaN", "nan", "inf", "-inf", "infinity", "1e400", "-1", "-0.5",
        ] {
            let json = format!(
                r#"{{"data":{{"endpoints":[{{"tag":"x","context_length":1000,
                   "pricing":{{"prompt":"{bad}","completion":"0.000001"}}}}]}}}}"#
            );
            let Err(err) = parse(&json) else {
                panic!("rate {bad:?} was accepted; it must fail loudly");
            };
            assert!(
                matches!(err, MiniLLMError::MalformedResponse(_)),
                "{bad:?}: {err:?}"
            );
        }
    }

    /// The concrete disaster the guard above prevents: a NaN rate on the dearest
    /// endpoint would be dropped by `f64::max`, and the cheap sibling would set the
    /// bound. Rejecting the whole response is the only honest answer.
    #[test]
    fn a_nan_rate_never_lets_a_cheaper_sibling_set_the_bound() {
        let json = r#"{"data":{"endpoints":[
            {"tag":"cheap","context_length":1000,
             "pricing":{"prompt":"0.000001","completion":"0.000002"}},
            {"tag":"dear","context_length":1000,
             "pricing":{"prompt":"NaN","completion":"0.00001"}}
        ]}}"#;
        assert!(
            parse(json).is_err(),
            "one poisoned endpoint condemns the model"
        );
    }

    /// A model the catalog does not list, or lists with no endpoints, has an
    /// unknown price rather than a zero one.
    #[test]
    fn a_model_without_endpoints_has_an_unknown_price() {
        let url = "https://x/models/m/endpoints";
        let missing = parse_endpoints("m", url, reqwest::StatusCode::NOT_FOUND, "").unwrap();
        assert!(missing.is_none());
        let empty = parse_endpoints(
            "m",
            url,
            reqwest::StatusCode::OK,
            r#"{"data":{"endpoints":[]}}"#,
        )
        .unwrap();
        assert!(empty.is_none());
        assert!(parse_endpoints("m", url, reqwest::StatusCode::BAD_GATEWAY, "down").is_err());
    }

    #[test]
    fn a_region_suffixed_tag_resolves_to_its_provider() {
        let e = &endpoints(
            r#"{"data":{"endpoints":[{"tag":"google-vertex/europe","context_length":1,
                "pricing":{"prompt":"0.1","completion":"0.2"}}]}}"#,
        )[0];
        assert_eq!(e.provider_slug, "google-vertex");
    }
}
