//! What a generator's model costs: [`GeneratorInfo::model_rates`].
//!
//! Prices come from OpenRouter's catalog, the only published unified price
//! sheet; it lists first-party models at the vendors' own standard rates, so it
//! also prices direct OpenAI/Anthropic calls (neither vendor publishes prices in
//! its own API). A price is a property of (model, provider), never a model
//! alone: the same model spans a 5x range across providers, so lookups name the
//! provider too.
//!
//! Rates arrive as USD **per token** (decimal strings); [`TokenPrice`] wants USD
//! per **million** tokens. The conversion happens here, exactly once.

use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tokio::sync::Mutex;

use crate::error::{MiniLLMError, Result};
use crate::generator::GeneratorInfo;
use crate::provider::wire::TokenPrice;

const BASE_URL: &str = "https://openrouter.ai/api/v1";

/// How long a fetched price stays fresh. Prices move on the order of months; an
/// hour bounds staleness without hammering the endpoint.
const TTL: Duration = Duration::from_secs(3600);

/// How long to wait on OpenRouter before giving up.
///
/// This is a service-to-service call the caller cannot see or cancel, so it needs
/// a bound. Without one a hung connection wedges every price lookup on this
/// generator, forever.
const FETCH_TIMEOUT: Duration = Duration::from_secs(30);

/// One model's billable limits, alongside its price.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRates {
    /// Per-token rates, cache buckets included when the model supports caching.
    pub price: TokenPrice,
    /// The largest completion the endpoint will produce. `None` when OpenRouter
    /// publishes no completion cap for it, in which case the context window is
    /// the only ceiling that exists.
    pub max_completion_tokens: Option<u32>,
    /// Total context window, the hard ceiling on input plus output.
    pub context_length: u32,
}

/// The generator's cached prices. The model is recorded so a generator whose
/// `model` field is later changed refetches rather than serving another model's
/// prices.
#[derive(Debug)]
pub(crate) struct CachedPrices {
    model: String,
    endpoints: Vec<PricedEndpoint>,
    fetched_at: Instant,
}

/// The shareable slot a [`GeneratorInfo`] stores its cache in.
pub(crate) type PriceCache = Arc<Mutex<Option<CachedPrices>>>;

impl GeneratorInfo {
    /// What this generator's model charges, from OpenRouter's published catalog.
    ///
    /// The model is looked up by [`openrouter_name`](Self::openrouter_name), else
    /// by [`model`](Self::model). Rates are the serving provider's own when it
    /// knows its catalog slug ([`Provider::openrouter_slug`](crate::Provider::openrouter_slug));
    /// otherwise the dearest rate any provider of the model charges, the only
    /// figure that holds wherever routing lands. The one failure is a model the
    /// catalog does not know.
    ///
    /// Prices are cached on this generator for an hour; clones share the cache,
    /// so keep generators alive rather than recreating them per call.
    pub async fn model_rates(&self) -> Result<ModelRates> {
        self.model_rates_served_by(Some(&self.catalog_provider())).await
    }

    /// The key that determines this generator's prices: its catalog model id and
    /// serving-provider slug. Two generators with equal keys price identically,
    /// so use it as the map key when pooling generators to keep price caches
    /// warm; the library itself holds no registry.
    pub fn pricing_key(&self) -> String {
        format!("{}:{}", self.catalog_provider(), self.catalog_model())
    }

    /// The id this generator's model has in OpenRouter's catalog.
    fn catalog_model(&self) -> &str {
        self.openrouter_name.as_deref().unwrap_or(&self.model)
    }

    /// The slug of the provider whose rates bill this generator's calls: the
    /// provider implementation's own when it knows it, else the generator's name
    /// (lowercased, since slugs are lowercase and matching ignores case).
    fn catalog_provider(&self) -> String {
        self.provider
            .openrouter_slug()
            .map(str::to_string)
            .unwrap_or_else(|| self.name.to_lowercase())
    }

    /// [`model_rates`](Self::model_rates), priced as served by `provider` (an
    /// OpenRouter slug), e.g. a per-request routing pin from
    /// [`ProviderSettings::billing_provider`](crate::ProviderSettings::billing_provider).
    /// `None` prices at the dearest endpoint of any provider.
    pub async fn model_rates_served_by(&self, provider: Option<&str>) -> Result<ModelRates> {
        // The catalog read rides this generator's client, like every other
        // request made on its behalf (an injected client sees it too).
        self.rates_with(provider, |model| fetch_endpoints(self.client(), model)).await
    }

    /// A deliberately high estimate, in USD, of what one completion will cost:
    /// this generator's rates ([`model_rates`](Self::model_rates)) priced against
    /// the prompt and the largest completion `params` permits. See
    /// [`estimate_cost_usd`](crate::provider::estimate_cost_usd) for what the
    /// figure assumes.
    ///
    /// If the prompt carries audio or video, set each clip's real length
    /// (`with_duration`); a clip of unstated length is assumed to run a minute,
    /// which overshoots short clips several-fold and UNDERSHOOTS anything
    /// longer, the one gap in the high-side guarantee.
    #[cfg(feature = "estimate")]
    pub async fn estimate_cost_usd(&self,
        messages: &[crate::message::Message],
        params: &super::CompletionParameters,
    ) -> Result<f64> {
        let rates = self.model_rates().await?;
        Ok(crate::provider::estimate_cost_usd(messages, params, &rates))
    }

    /// [`model_rates_served_by`](Self::model_rates_served_by), over an injectable
    /// fetch.
    ///
    /// The caching and locking live here, with the network as a parameter, so a
    /// test can drive the coordination without a socket. There is one code path:
    /// the public methods are this function with the real fetch supplied.
    async fn rates_with<F>(
        &self,
        provider: Option<&str>,
        fetch: impl FnOnce(String) -> F,
    ) -> Result<ModelRates>
    where
        F: std::future::Future<Output = Result<Vec<PricedEndpoint>>>,
    {
        let model = self.catalog_model();
        // Checked before the lock is taken, so a bad id costs neither.
        validate_model_id(model)?;

        // Held across the fetch, on purpose: concurrent lookups on one generator
        // (or its clones) share a single request instead of racing. The lock is
        // per generator, so it never blocks a different generator's lookup.
        let mut cached = self.prices.lock().await;
        let stale = cached
            .as_ref()
            .is_none_or(|c| c.model != model || c.fetched_at.elapsed() >= TTL);
        if stale {
            // Fetched before the slot is touched: a failed fetch returns via `?`
            // with the slot unchanged. The surviving entry is re-judged by the
            // staleness predicate on every later call, so a caller it mismatches
            // retries, and a caller it is still fresh and correct for is served.
            let endpoints = fetch(model.to_string()).await?;
            *cached = Some(CachedPrices {
                model: model.to_string(),
                endpoints,
                fetched_at: Instant::now(),
            });
        }

        let prices = cached.as_ref().expect("populated above or returned early");
        select(&prices.endpoints, model, provider)
    }
}

/// Fetch and validate every endpoint's prices for `model`. Parsing here rather
/// than per lookup means a corrupt rate fails once, loudly, instead of lying in
/// the cache until some provider selection happens to read it.
async fn fetch_endpoints(
    client: crate::provider::LLMClient,
    model: String,
) -> Result<Vec<PricedEndpoint>> {
    debug_assert!(validate_model_id(&model).is_ok(), "callers validate first");
    let response = client
        .http()
        .get(format!("{BASE_URL}/models/{model}/endpoints"))
        .timeout(FETCH_TIMEOUT)
        .send()
        .await
        .map_err(|e| match e {
            reqwest_middleware::Error::Reqwest(e) => MiniLLMError::Http(e),
            reqwest_middleware::Error::Middleware(e) => MiniLLMError::InvalidParameter(format!(
                "injected client refused the price-catalog read: {e:#}"
            )),
        })?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Err(MiniLLMError::InvalidParameter(format!(
            "model {model:?} is not in OpenRouter's catalog, so it cannot be priced"
        )));
    }
    let body: EndpointsResponse = response.error_for_status()?.json().await?;
    price_endpoints(body.data.endpoints)
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
    let parsed = raw.parse::<f64>().map_err(|_| malformed("is not a number"))?;
    if !parsed.is_finite() {
        return Err(malformed("is not finite"));
    }
    if parsed < 0.0 {
        return Err(malformed("is negative"));
    }
    Ok(parsed * 1_000_000.0)
}

impl Endpoint {
    fn rates(&self) -> Result<ModelRates> {
        let mut price = TokenPrice::new(
            per_mtok("prompt", &self.pricing.prompt)?,
            per_mtok("completion", &self.pricing.completion)?,
        );
        // Only set cache rates when BOTH are published. `TokenPrice` falls back to
        // the input rate for a missing bucket, which is the correct behaviour for
        // an endpoint that does not price that bucket separately.
        match (&self.pricing.input_cache_read, &self.pricing.input_cache_write) {
            (Some(r), Some(w)) => {
                price = price
                    .with_cache_rates(per_mtok("input_cache_read", r)?, per_mtok("input_cache_write", w)?)
            }
            // Read-only caching (OpenAI charges nothing to write): a write costs
            // the plain input rate, which is what leaving it unset yields.
            (Some(r), None) => price.cache_read_per_mtok = Some(per_mtok("input_cache_read", r)?),
            _ => {}
        }
        price = price.with_media_rates(
            self.pricing.audio.as_deref().map(|r| per_mtok("audio", r)).transpose()?,
            self.pricing.image.as_deref().map(|r| per_mtok("image", r)).transpose()?,
        );

        let context_length = self.context_length.ok_or_else(|| {
            MiniLLMError::MalformedResponse(format!("model catalog: endpoint {} has no context_length", self.tag))
        })?;

        Ok(ModelRates { price, max_completion_tokens: self.max_completion_tokens, context_length })
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

/// One endpoint, parsed and validated: who serves it, and what it charges.
///
/// The cache stores these rather than the raw wire shape, so a corrupt rate
/// fails the fetch loudly, for the whole model, instead of surfacing later and
/// only for whichever provider selection happened to touch it.
#[derive(Debug, Clone)]
struct PricedEndpoint {
    provider_slug: String,
    rates: ModelRates,
}

/// Parse and validate every endpoint's prices, failing on the first corrupt one.
fn price_endpoints(endpoints: Vec<Endpoint>) -> Result<Vec<PricedEndpoint>> {
    endpoints
        .into_iter()
        .map(|endpoint| {
            Ok(PricedEndpoint {
                provider_slug: endpoint.provider_slug().to_string(),
                rates: endpoint.rates()?,
            })
        })
        .collect()
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
fn select(
    endpoints: &[PricedEndpoint],
    model: &str,
    provider: Option<&str>,
) -> Result<ModelRates> {
    let matches_provider = |e: &&PricedEndpoint| match provider {
        Some(slug) => e.provider_slug.eq_ignore_ascii_case(slug),
        None => true,
    };
    let candidates: Vec<&PricedEndpoint> = match endpoints.iter().filter(matches_provider).collect::<Vec<_>>() {
        served if !served.is_empty() => served,
        _ => endpoints.iter().collect(),
    };

    let mut worst: Option<ModelRates> = None;
    for endpoint in candidates {
        let rates = endpoint.rates.clone();
        // An absent bucket is not "no charge", it is "charged at this endpoint's
        // input rate". Resolving that fallback BEFORE comparing is what makes the
        // maximum meaningful: an endpoint that omits its cache-read rate really is
        // dearer on cache reads than one that publishes a discount.
        let price = &rates.price;
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
            None => ModelRates { price: effective, ..rates },
            Some(w) => ModelRates {
                price: TokenPrice {
                    input_per_mtok: w.price.input_per_mtok.max(input),
                    output_per_mtok: w.price.output_per_mtok.max(output),
                    cache_read_per_mtok: max_rate(w.price.cache_read_per_mtok, effective.cache_read_per_mtok),
                    cache_write_per_mtok: max_rate(w.price.cache_write_per_mtok, effective.cache_write_per_mtok),
                    audio_per_mtok: max_rate(w.price.audio_per_mtok, effective.audio_per_mtok),
                    image_per_mtok: max_rate(w.price.image_per_mtok, effective.image_per_mtok),
                },
                max_completion_tokens: match (w.max_completion_tokens, rates.max_completion_tokens) {
                    // No published cap is the LEAST restrictive, so it wins.
                    (None, _) | (_, None) => None,
                    (Some(a), Some(b)) => Some(a.max(b)),
                },
                context_length: w.context_length.max(rates.context_length),
            },
        });
    }

    worst.ok_or_else(|| {
        MiniLLMError::InvalidParameter(format!(
            "model {model:?} has no endpoints, so it cannot be priced"
        ))
    })
}

/// Every character a real model id is made of. Stated positively, so a new
/// separator OpenRouter invents fails loudly here rather than silently reshaping a
/// URL.
fn is_model_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':' | '/')
}

/// Refuse anything that is not a model id.
///
/// A model id is caller-supplied and legitimately contains `/`
/// (`anthropic/claude-sonnet-4.6`). Everything else is refused rather than
/// escaped: `evil?free=1` would push the `/endpoints` suffix into a query string
/// and hit a different route, and `../../../models` would climb out of the API
/// version prefix. Both would return a plausible response for the wrong thing.
/// Escaping does not save either: a URL parser decodes `%2E%2E` and traverses
/// anyway, and the `url` crate's segment writer silently drops dot segments,
/// which would fetch prices for a DIFFERENT model. A wrong price is worse than
/// an error.
fn validate_model_id(model: &str) -> Result<()> {
    let refuse = |why: &str| {
        Err(MiniLLMError::InvalidParameter(format!("model id {model:?} {why}, so it cannot be priced")))
    };

    if model.is_empty() {
        return refuse("is empty");
    }
    if let Some(bad) = model.chars().find(|c| !is_model_id_char(*c)) {
        return refuse(&format!("contains {bad:?}, which no model id contains"));
    }
    // Made of legal characters, yet `.` and `..` mean "here" and "up one" to every
    // URL parser. A model is never named either.
    if model.split('/').any(|segment| segment.is_empty() || segment == "." || segment == "..") {
        return refuse("has an empty or dotted path segment");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a catalog response the same way a fetch does, so a test sees the
    /// failure exactly where a real response would produce it.
    fn parse(json: &str) -> Result<Vec<PricedEndpoint>> {
        price_endpoints(serde_json::from_str::<EndpointsResponse>(json).unwrap().data.endpoints)
    }

    /// The same, for a response known to be well formed.
    fn endpoints(json: &str) -> Vec<PricedEndpoint> {
        parse(json).expect("fixture is well formed")
    }

    /// Rates come from parsing a decimal string and scaling by a million, so they
    /// carry float error and must never be compared with `==`.
    #[track_caller]
    fn assert_rate(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "rate {actual} is not {expected}");
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
        let rates = select(&endpoints(SPREAD), "z-ai/glm-5.2", None).unwrap();
        assert_rate(rates.price.input_per_mtok, 3.0);
        assert_rate(rates.price.output_per_mtok, 10.25);
        // The limits are the most PERMISSIVE, so the output bound is never
        // understated: 8192 from the cheap endpoint, 64000 context from the dear
        // one. Taking them from whichever endpoint set the price would let a
        // caller's max_tokens be clamped below what routing would actually allow.
        assert_eq!(rates.max_completion_tokens, Some(8192));
        assert_eq!(rates.context_length, 128_000);
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
        let rates = select(&endpoints(json), "m", None).unwrap();
        assert_rate(rates.price.input_per_mtok, 5.0);
        assert_rate(rates.price.output_per_mtok, 9.0);
        // The dear-audio endpoint is cheap on text, and vice versa. Both maxima win.
        assert_rate(rates.price.audio_rate(), 100.0);
    }

    #[test]
    fn a_pinned_provider_is_priced_at_that_providers_endpoint() {
        let rates = select(&endpoints(SPREAD), "z-ai/glm-5.2", Some("fireworks")).unwrap();
        assert_rate(rates.price.input_per_mtok, 2.1);
        assert_rate(rates.price.output_per_mtok, 6.6);
        assert_eq!(rates.max_completion_tokens, Some(4096));
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
        let bedrock =
            select(&endpoints(json), "anthropic/claude-sonnet-5", Some("amazon-bedrock")).unwrap();
        assert_rate(bedrock.price.output_per_mtok, 11.0);

        let direct =
            select(&endpoints(json), "anthropic/claude-sonnet-5", Some("anthropic")).unwrap();
        assert_rate(direct.price.output_per_mtok, 10.0);
    }

    /// A provider that serves the model through no endpoint prices at the dearest
    /// of ALL endpoints: the caller's knowledge did not match the catalog, and the
    /// only bound that still holds wherever the call really lands is the dearest
    /// anyone charges. An error here would break estimation for a name mismatch
    /// that costs nothing to bound over.
    #[test]
    fn a_provider_that_does_not_serve_the_model_prices_at_the_dearest_of_all() {
        let fallback = select(&endpoints(SPREAD), "z-ai/glm-5.2", Some("anthropic")).unwrap();
        let dearest = select(&endpoints(SPREAD), "z-ai/glm-5.2", None).unwrap();
        assert_eq!(fallback, dearest);
    }

    /// Provider slugs are lowercase; a generator's display name ("Anthropic") is
    /// not. The match is case-insensitive so the name works as the fallback slug.
    #[test]
    fn a_provider_name_matches_its_slug_regardless_of_case() {
        let pinned = select(&endpoints(SPREAD), "z-ai/glm-5.2", Some("Fireworks")).unwrap();
        assert_rate(pinned.price.input_per_mtok, 2.1);
    }

    #[test]
    fn per_token_strings_become_per_million_rates_with_their_cache_buckets() {
        let json = r#"{"data":{"endpoints":[{"tag":"anthropic","context_length":1000000,
            "max_completion_tokens":128000,
            "pricing":{"prompt":"0.000003","completion":"0.000015",
                       "input_cache_read":"0.0000003","input_cache_write":"0.00000375"}}]}}"#;
        let rates = select(&endpoints(json), "m", None).unwrap();
        assert_rate(rates.price.input_per_mtok, 3.0);
        assert_rate(rates.price.output_per_mtok, 15.0);
        assert_rate(rates.price.cache_read_per_mtok.unwrap(), 0.3);
        assert_rate(rates.price.cache_write_per_mtok.unwrap(), 3.75);
    }

    /// OpenAI publishes a cache-read discount and no write charge, so a write bills
    /// at the plain input rate. `select` resolves that fallback rather than leaving
    /// it unset: after bounding, every bucket has a real number, because "absent"
    /// meant "input rate" and that is what a sibling endpoint gets compared against.
    #[test]
    fn a_read_only_cache_resolves_its_write_rate_to_the_input_rate() {
        let json = r#"{"data":{"endpoints":[{"tag":"openai","context_length":400000,
            "pricing":{"prompt":"0.000005","completion":"0.00003","input_cache_read":"0.0000005"}}]}}"#;
        let rates = select(&endpoints(json), "openai/gpt-5.5", None).unwrap();
        assert_rate(rates.price.cache_read_per_mtok.unwrap(), 0.5);
        assert_rate(rates.price.cache_write_per_mtok.unwrap(), 5.0);
        // This endpoint prices neither image nor audio. An image bills exactly as
        // text; audio always carries a premium, so it bills at the assumed multiple.
        assert_rate(rates.price.image_rate(), 5.0);
        assert_rate(rates.price.audio_rate(), 5.0 * crate::provider::wire::AUDIO_RATE_FALLBACK_MULTIPLE);

        // Which is what a write really costs, resolved or not.
        let usage = crate::provider::Usage { cache_write_tokens: 1_000_000, ..Default::default() };
        assert!((rates.price.cost_of(&usage) - 5.0).abs() < 1e-9);
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
        let rates = select(&endpoints(json), "m", None).unwrap();
        // Not 0.5: the second endpoint charges the full 5.0 for a cache read.
        assert_rate(rates.price.cache_read_per_mtok.unwrap(), 5.0);
    }

    #[test]
    fn an_endpoint_with_no_completion_cap_keeps_its_context_window() {
        let json = r#"{"data":{"endpoints":[{"tag":"x","context_length":32768,
            "pricing":{"prompt":"0.0000001","completion":"0.0000002"}}]}}"#;
        let rates = select(&endpoints(json), "some/model", None).unwrap();
        assert_eq!(rates.max_completion_tokens, None);
        assert_eq!(rates.context_length, 32_768);
    }

    /// Every rate that is not a finite, non-negative number is a corrupt catalog,
    /// and it is rejected when the response is parsed, not when a price is later
    /// read. So one bad endpoint condemns the whole model, loudly and immediately,
    /// rather than lying in the cache until some provider selection touches it.
    ///
    /// `"NaN"` is the dangerous one: it parses, and `f64::max` returns its non-NaN
    /// operand, so a NaN rate on the DEAREST endpoint would be silently discarded
    /// and a cheaper sibling would win the bound.
    #[test]
    fn a_rate_that_is_not_a_finite_non_negative_number_fails_the_fetch() {
        for bad in ["free", "NaN", "nan", "inf", "-inf", "infinity", "1e400", "-1", "-0.5"] {
            let json = format!(
                r#"{{"data":{{"endpoints":[{{"tag":"x","context_length":1000,
                   "pricing":{{"prompt":"{bad}","completion":"0.000001"}}}}]}}}}"#
            );
            let Err(err) = parse(&json) else {
                panic!("rate {bad:?} was accepted; it must fail loudly");
            };
            assert!(matches!(err, MiniLLMError::MalformedResponse(_)), "{bad:?}: {err:?}");
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
        assert!(parse(json).is_err(), "one poisoned endpoint condemns the model");
    }

    /// A fetch that never touches the network: it counts its calls and hands back a
    /// fixed price list.
    fn fake_fetch(calls: &std::sync::atomic::AtomicUsize) -> Result<Vec<PricedEndpoint>> {
        calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        parse(SPREAD)
    }

    fn generator() -> GeneratorInfo {
        GeneratorInfo::openrouter("z-ai/glm-5.2")
    }

    /// The pricing key is exactly what determines the price: the catalog model
    /// id and the provider slug. Generators the catalog would price identically
    /// share a key; generators it would price differently do not.
    #[test]
    fn the_pricing_key_names_the_provider_and_the_catalog_model() {
        let direct = GeneratorInfo::anthropic("claude-haiku-4-5-20251001")
            .with_openrouter_name("anthropic/claude-haiku-4.5");
        assert_eq!(direct.pricing_key(), "anthropic:anthropic/claude-haiku-4.5");

        // The same model through OpenRouter prices at the dearest of ALL its
        // providers, not at Anthropic's own rate, so the keys must differ.
        let routed = GeneratorInfo::openrouter("anthropic/claude-haiku-4.5");
        assert_eq!(routed.pricing_key(), "openrouter:anthropic/claude-haiku-4.5");
        assert_ne!(direct.pricing_key(), routed.pricing_key());

        // Two independently built but identically configured generators share a
        // key: that is what lets a caller pool them in a map.
        assert_eq!(routed.pricing_key(), GeneratorInfo::openrouter("anthropic/claude-haiku-4.5").pricing_key());
    }

    /// A price already fetched is not fetched again while it is fresh.
    #[tokio::test]
    async fn a_fresh_price_is_served_from_the_cache() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let generator = generator();

        for _ in 0..3 {
            generator
                .rates_with(None, |_| async { fake_fetch(&calls) })
                .await
                .expect("the fixture prices");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1, "three lookups, one fetch");
    }

    /// A clone shares the cache. This is what makes a pool of long-lived
    /// generators keep prices warm rather than refetch per call.
    #[tokio::test]
    async fn a_clone_shares_the_cache() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let generator = generator();
        let clone = generator.clone();

        generator
            .rates_with(None, |_| async { fake_fetch(&calls) })
            .await
            .expect("the fixture prices");
        clone
            .rates_with(None, |_| async { fake_fetch(&calls) })
            .await
            .expect("served from the shared cache");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "the clone reused the fetch");
    }

    /// Two provider selections over one generator share the one fetch, and each
    /// still gets its own answer: the prices are parsed once and selected from
    /// many times.
    #[tokio::test]
    async fn two_selections_over_one_generator_share_one_fetch() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let generator = generator();

        let any = generator.rates_with(None, |_| async { fake_fetch(&calls) }).await.unwrap();
        let pinned = generator
            .rates_with(Some("fireworks"), |_| async { fake_fetch(&calls) })
            .await
            .unwrap();

        assert_eq!(calls.load(Ordering::SeqCst), 1, "one fetch serves both selections");
        assert_rate(any.price.output_per_mtok, 10.25);
        assert_rate(pinned.price.output_per_mtok, 6.6);
    }

    /// A failed FIRST fetch leaves the slot empty, so the next caller retries
    /// rather than inheriting the failure or billing against a zero price.
    #[tokio::test]
    async fn a_failed_first_fetch_leaves_the_slot_empty_and_the_next_caller_retries() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let generator = generator();

        let failed = generator
            .rates_with(None, |_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(MiniLLMError::Timeout)
            })
            .await;
        assert!(failed.is_err(), "the error surfaces rather than a zero price");

        generator
            .rates_with(None, |_| async { fake_fetch(&calls) })
            .await
            .expect("the retry succeeds");
        assert_eq!(calls.load(Ordering::SeqCst), 2, "the failure was not cached");
    }

    /// A failed REFETCH over a populated slot keeps the previous entry, and the
    /// staleness predicate re-judges it per call: the model it still matches is
    /// served from it (no wasted refetch), the model whose fetch failed retries,
    /// and nobody is ever served the wrong model's prices.
    #[tokio::test]
    async fn a_failed_refetch_keeps_the_entry_the_cache_is_still_fresh_for() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let mut generator = generator();

        generator.rates_with(None, |_| async { fake_fetch(&calls) }).await.unwrap();

        // Switching the model makes the entry stale; the refetch for it fails.
        generator.model = "openai/gpt-5.5".to_string();
        let failed = generator
            .rates_with(None, |_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(MiniLLMError::Timeout)
            })
            .await;
        assert!(failed.is_err(), "the error surfaces");

        // Back on the first model: served from the surviving entry, no refetch.
        generator.model = "z-ai/glm-5.2".to_string();
        let rates = generator.rates_with(None, |_| async { fake_fetch(&calls) }).await.unwrap();
        assert_rate(rates.price.output_per_mtok, 10.25);
        assert_eq!(calls.load(Ordering::SeqCst), 2, "the survivor was served, not refetched");

        // The other model is still unpriced: it retries rather than reading the
        // survivor's prices. Its fixture prices differently, so the returned rate
        // proves the answer came from the refetch and not the survivor.
        generator.model = "openai/gpt-5.5".to_string();
        let other = generator
            .rates_with(None, |_| async {
                calls.fetch_add(1, Ordering::SeqCst);
                parse(
                    r#"{"data":{"endpoints":[{"tag":"openai","context_length":400000,
                       "pricing":{"prompt":"0.000005","completion":"0.00003"}}]}}"#,
                )
            })
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3, "the unpriced model refetched");
        assert_rate(other.price.output_per_mtok, 30.0);
    }

    /// `model` is a public field, so it can be changed after prices were fetched.
    /// The cache remembers which model it priced and refetches on mismatch, rather
    /// than serving another model's prices.
    #[tokio::test]
    async fn changing_the_model_invalidates_the_cache() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let calls = AtomicUsize::new(0);
        let mut generator = generator();

        generator.rates_with(None, |_| async { fake_fetch(&calls) }).await.unwrap();
        generator.model = "openai/gpt-5.5".to_string();
        generator.rates_with(None, |_| async { fake_fetch(&calls) }).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2, "a different model is a different fetch");
    }

    /// A lookup on one generator must not wait on a fetch for another: the cache
    /// and its lock are per generator, never global.
    ///
    /// Proven by making the first fetch block until the second has completed: a
    /// shared lock would deadlock here.
    #[tokio::test]
    async fn a_slow_fetch_on_one_generator_does_not_block_another() {
        let slow_gen = generator();
        let quick_gen = GeneratorInfo::openrouter("openai/gpt-5.5");
        let (release, released) = tokio::sync::oneshot::channel::<()>();

        let slow = slow_gen.rates_with(None, |_| async move {
            // Blocks until the other generator has been priced.
            released.await.expect("the other lookup finishes first");
            parse(SPREAD)
        });

        let quick = async {
            let rates = quick_gen
                .rates_with(None, |_| async { parse(SPREAD) })
                .await
                .expect("an unrelated generator prices while the slow fetch is in flight");
            release.send(()).expect("the slow fetch is still waiting");
            rates
        };

        // Deadlocks if the lock were shared. The test harness would hang, so the
        // timeout turns that into a failure rather than a stuck suite.
        let both = tokio::time::timeout(Duration::from_secs(5), futures::future::join(slow, quick));
        let (slow, _quick) = both.await.expect("neither lookup blocked the other");
        assert!(slow.is_ok());
    }

    /// An id that is not a model id is refused, never escaped.
    ///
    /// Escaping does not work: a URL parser decodes `%2E%2E` and traverses anyway,
    /// and the `url` crate's segment writer silently drops the dot segments, which
    /// would fetch prices for a DIFFERENT model. Both were verified against
    /// `reqwest::Url`. A wrong price is worse than an error.
    #[test]
    fn an_id_that_is_not_a_model_id_is_refused() {
        let hostile = [
            ("evil?free=1", "a query would swallow the /endpoints suffix"),
            ("x#frag", "a fragment would truncate the path"),
            ("../../../models", "traversal would climb out of the api version"),
            ("a/../b", "traversal, buried mid-path"),
            ("a/./b", "a dot segment resolves away"),
            ("a//b", "an empty segment collapses"),
            ("a b", "a space is not a model id character"),
            ("a\nb", "nor is a newline"),
            ("a%2fb", "nor is a percent escape"),
            ("", "an empty id names nothing"),
        ];
        for (bad, why) in hostile {
            let Err(err) = validate_model_id(bad) else {
                panic!("{bad:?} was accepted, but {why}");
            };
            assert!(matches!(err, MiniLLMError::InvalidParameter(_)), "{bad:?}: {err:?}");
        }
    }

    /// A real model id is accepted, separator and all.
    #[test]
    fn a_real_model_id_is_accepted() {
        for real in ["anthropic/claude-sonnet-4.6", "openai/gpt-5.5", "o3", "z-ai/glm-5.2"] {
            assert!(validate_model_id(real).is_ok(), "{real} is a real id");
        }
    }

    /// A model the catalog lists with no endpoints at all cannot be priced, and
    /// says so rather than returning a zero bound.
    #[test]
    fn a_model_with_no_endpoints_at_all_fails_loudly() {
        let err = select(&endpoints(r#"{"data":{"endpoints":[]}}"#), "m", None).unwrap_err();
        assert!(err.to_string().contains("has no endpoints"), "{err}");
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
