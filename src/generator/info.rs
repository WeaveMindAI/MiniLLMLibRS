//! Generator information - configuration for LLM providers

use crate::error::Result;
use crate::provider::{
    AnthropicProvider, AppIdentity, Auth, GenericProvider, ModelInfo, ModelLookup, OpenAiProvider,
    OpenRouterProvider, Provider, SideCall, TokenPrice, Wanted,
};
use reqwest_middleware::ClientWithMiddleware;
use secrecy::SecretString;
use std::sync::Arc;
use std::time::Duration;

/// The output cap a request carries on a wire that requires one when neither
/// the caller nor the model's info gives one.
const UNKNOWN_MAX_TOKENS: u32 = 4096;

/// Configuration for an LLM provider/generator
#[derive(Debug, Clone)]
pub struct GeneratorInfo {
    /// Display name for this generator
    pub name: String,

    /// Base URL for the API (e.g., `https://openrouter.ai/api/v1`)
    pub base_url: String,

    /// Model identifier (e.g., "anthropic/claude-3.5-sonnet")
    pub model: String,

    /// The model's id in OpenRouter's catalog, when it differs from [`model`].
    /// The built-in providers look a model's price up in that catalog
    /// ([`model_info`](Self::model_info)) by this name, falling back to `model`
    /// itself (an OpenRouter generator's model id already IS the catalog id).
    /// A vendor's own id rarely matches the catalog's (Anthropic's
    /// `claude-haiku-4-5-20251001` is the catalog's `anthropic/claude-haiku-4.5`),
    /// so a direct-vendor generator sets this to have its price found.
    ///
    /// [`model`]: Self::model
    pub openrouter_name: Option<String>,

    /// How this generator authenticates. The provider maps it to concrete headers
    /// (OpenAI-wire `Authorization: Bearer`, Anthropic `x-api-key` or bearer).
    pub auth: Auth,

    /// Custom headers to include in requests
    pub custom_headers: Vec<(String, String)>,

    /// Provider implementation: the wire dialect (token-limit key, usage opt-in,
    /// usage parsing, cost aggregation, out-of-band resolution, attribution
    /// headers). Swap this to target a different provider.
    pub provider: Arc<dyn Provider>,

    /// What you tell the library about the model: its price, its largest
    /// output, its context window. Whatever you leave unset is looked up
    /// ([`model_info`](Self::model_info)).
    pub info: ModelInfo,

    /// Calling-app identity for providers that attribute usage to an app (e.g.
    /// OpenRouter rankings). The provider decides which headers express it.
    pub app_attribution: Option<AppIdentity>,

    /// Default completion parameters for this generator
    pub default_params: super::CompletionParameters,

    /// A caller-supplied HTTP client. When set, every request made for this
    /// generator (completions, streaming, out-of-band cost queries) goes
    /// through it, so the caller's routing and middleware see the whole
    /// conversation. `None` = the crate's shared pooled client.
    pub http_client: Option<reqwest_middleware::ClientWithMiddleware>,
}

impl GeneratorInfo {
    /// Create a new GeneratorInfo with minimal configuration (generic
    /// OpenAI-compatible accounting; set `with_provider` for a specific provider).
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            model: model.into(),
            openrouter_name: None,
            auth: Auth::None,
            custom_headers: Vec::new(),
            provider: Arc::new(GenericProvider::default()),
            info: ModelInfo::default(),
            app_attribution: None,
            default_params: super::CompletionParameters::default(),
            http_client: None,
        }
    }

    /// Point this generator at a different address (a gateway, a proxy, a
    /// self-hosted endpoint). Every request the crate makes for it,
    /// completions and out-of-band cost queries alike, goes here.
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set the model's OpenRouter catalog id, so its price is found when the
    /// generator's own model id is not already a catalog id (see
    /// [`openrouter_name`](Self::openrouter_name)).
    pub fn with_openrouter_name(mut self, name: impl Into<String>) -> Self {
        self.openrouter_name = Some(name.into());
        self
    }

    /// Set the calling-app identity for provider usage attribution.
    pub fn with_app_attribution(
        mut self,
        url: impl Into<String>,
        title: impl Into<String>,
    ) -> Self {
        self.app_attribution = Some(AppIdentity {
            url: url.into(),
            title: title.into(),
        });
        self
    }

    /// Set the provider implementation (wire dialect: token-limit key, usage
    /// opt-in, usage parsing, cost aggregation, out-of-band resolution,
    /// attribution headers).
    pub fn with_provider(mut self, provider: Arc<dyn Provider>) -> Self {
        self.provider = provider;
        self
    }

    /// Set the model's per-token price (USD per million tokens), the
    /// [`info`](Self::info) price, over any looked up.
    pub fn with_token_price(mut self, price: TokenPrice) -> Self {
        self.info.price = Some(price);
        self
    }

    /// Set the API key (provider-issued; the provider chooses the header).
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.auth = Auth::ApiKey(SecretString::from(key.into()));
        self
    }

    /// Set the API key from an environment variable (no-op if unset).
    pub fn with_api_key_from_env(mut self, env_var: &str) -> Self {
        if let Ok(key) = std::env::var(env_var) {
            self.auth = Auth::ApiKey(SecretString::from(key));
        }
        self
    }

    /// Set an OAuth/bearer token. Always sent
    /// as `Authorization: Bearer <token>`.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.auth = Auth::BearerToken(SecretString::from(token.into()));
        self
    }

    /// Set a bearer token from an environment variable (no-op if unset).
    pub fn with_bearer_token_from_env(mut self, env_var: &str) -> Self {
        if let Ok(token) = std::env::var(env_var) {
            self.auth = Auth::BearerToken(SecretString::from(token));
        }
        self
    }

    /// Set the auth strategy directly.
    pub fn with_auth(mut self, auth: Auth) -> Self {
        self.auth = auth;
        self
    }

    /// Add a custom header
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_headers.push((name.into(), value.into()));
        self
    }

    /// Set default completion parameters
    pub fn with_default_params(mut self, params: super::CompletionParameters) -> Self {
        self.default_params = params;
        self
    }

    /// The parameters a call with `params` sends: this generator's
    /// [`default_params`](Self::default_params), with every field the caller
    /// set taking precedence; and on a wire whose API requires an output cap
    /// ([`Provider::requires_max_tokens`](crate::Provider::requires_max_tokens)),
    /// when neither set one, the model's largest
    /// ([`ModelInfo::max_output_tokens`]), or 4096 when that is unknown. Every
    /// request goes through here.
    pub async fn request_params(
        &self,
        params: &super::CompletionParameters,
    ) -> super::CompletionParameters {
        self.request_params_over(self.client().http(), params, None)
            .await
    }

    /// [`request_params`](Self::request_params), with any lookup over
    /// `client` (the one the request rides) and given up after `deadline`, in
    /// which case the cap is 4096: a slow lookup never delays a request past
    /// its own deadline, and it keeps running for the next one.
    pub(crate) async fn request_params_over(
        &self,
        client: &ClientWithMiddleware,
        params: &super::CompletionParameters,
        deadline: Option<Duration>,
    ) -> super::CompletionParameters {
        let mut sent = self.default_params.merge(params);
        if sent.max_tokens.is_none() && self.provider.requires_max_tokens() {
            let largest = match self.info.max_output_tokens {
                Some(largest) => Some(largest),
                None => {
                    let limits = Wanted {
                        price: false,
                        limits: true,
                    };
                    let lookup = self.model_info_for(client, limits);
                    match deadline {
                        None => lookup.await.max_output_tokens,
                        Some(deadline) => match tokio::time::timeout(deadline, lookup).await {
                            Ok(info) => info.max_output_tokens,
                            Err(_) => {
                                tracing::warn!(
                                    "looking `{}` up took longer than the request's deadline, \
                                     so it goes out with max_tokens {UNKNOWN_MAX_TOKENS}",
                                    self.model
                                );
                                None
                            }
                        },
                    }
                }
            };
            sent.max_tokens = Some(largest.unwrap_or(UNKNOWN_MAX_TOKENS));
        }
        sent
    }

    /// Supply the HTTP client every request for this generator rides
    /// (completions, streaming, out-of-band cost queries). Accepts a plain
    /// `reqwest::Client` or a `ClientWithMiddleware`.
    ///
    /// Without one, the generator uses the crate's shared client
    /// ([`global_client`](crate::provider::global_client)). If your program
    /// starts more than one tokio runtime, give each runtime's generators a
    /// client of their own: a connection belongs to the runtime that opened
    /// it, so once that runtime shuts down, a shared client fails with
    /// `MiniLLMError::Http`, whose cause chain ends in "runtime dropped the
    /// dispatch task". A client you pass replaces the shared one completely,
    /// including its 600-second limit on a silent connection.
    pub fn with_http_client(
        mut self,
        client: impl Into<reqwest_middleware::ClientWithMiddleware>,
    ) -> Self {
        self.http_client = Some(client.into());
        self
    }

    /// What the library knows about this generator's model: every field you
    /// gave in [`info`](Self::info), and the rest from the provider's lookup
    /// ([`Provider::model_info`](crate::Provider::model_info)), made over this
    /// generator's client. The lookup's requests are shared across the process
    /// and kept for an hour ([`SideCall::get_cached`](crate::SideCall::get_cached)),
    /// so asking again is free. A field nobody can tell stays unknown.
    pub async fn model_info(&self) -> ModelInfo {
        self.model_info_for(self.client().http(), Wanted::ALL).await
    }

    /// The facts in `wanted` that were not given, looked up over `client`,
    /// with everything given. Nothing is looked up when nothing wanted is
    /// missing: the price is missing when none was given, the limits when no
    /// largest output was.
    pub(crate) async fn model_info_for(
        &self,
        client: &ClientWithMiddleware,
        wanted: Wanted,
    ) -> ModelInfo {
        let missing = Wanted {
            price: wanted.price && self.info.price.is_none(),
            limits: wanted.limits
                && (self.info.max_output_tokens.is_none() || self.info.context_length.is_none()),
        };
        if !missing.price && !missing.limits {
            return self.info.clone();
        }
        let call = match self.side_call(client) {
            Ok(call) => call,
            Err(e) => {
                tracing::warn!("cannot look `{}` up: {e}", self.model);
                return self.info.clone();
            }
        };
        let lookup = ModelLookup {
            call,
            model: self.model.clone(),
            openrouter_name: self.openrouter_name.clone(),
            wanted: missing,
        };
        match self.provider.model_info(lookup) {
            Some(lookup) => self.info.clone().or(lookup.await),
            None => self.info.clone(),
        }
    }

    /// Tell the library about the model (see [`info`](Self::info)): every
    /// field you set is used as given, over anything set before; the rest is
    /// looked up.
    pub fn with_model_info(mut self, info: ModelInfo) -> Self {
        self.info = info.or(self.info);
        self
    }

    /// A deliberately high estimate, in USD, of what one completion will cost,
    /// or `None` when the model's price, or how much it can produce, is
    /// unknown. The caller's parameters over the generator's
    /// [`default_params`](Self::default_params) are priced against
    /// [`model_info`](Self::model_info), with an unset cap read as the model's
    /// largest output, else its context window: never below the cap the
    /// request will carry, whatever a lookup at send time answers. See
    /// [`estimate_cost_usd`](crate::provider::estimate_cost_usd) for what the
    /// figure assumes.
    ///
    /// If the prompt carries audio or video, set each clip's real length
    /// (`with_duration`), and give each PDF its page count
    /// (`with_page_count`): a clip of unstated length is assumed to run a
    /// minute and a PDF twenty pages, which UNDERSHOOTS anything longer. Those
    /// and a request that writes the cache are where the figure can come in
    /// low.
    #[cfg(feature = "estimate")]
    pub async fn estimate_cost_usd(
        &self,
        messages: &[crate::message::Message],
        params: &super::CompletionParameters,
    ) -> Option<f64> {
        let info = self.model_info().await;
        crate::provider::estimate_cost_usd(messages, &self.default_params.merge(params), &info)
    }

    /// Where a side call about this generator's requests goes: over `client`,
    /// to its address, with its headers.
    pub(crate) fn side_call(&self, client: &ClientWithMiddleware) -> Result<SideCall> {
        Ok(SideCall {
            client: client.clone(),
            base_url: self.base_url.clone(),
            headers: crate::provider::client::generator_headers(self)?,
        })
    }

    /// The `LLMClient` requests for this generator go through: the injected
    /// one when set, else the crate's shared pooled client.
    pub(crate) fn client(&self) -> crate::provider::LLMClient {
        match &self.http_client {
            Some(client) => crate::provider::LLMClient::with_client(client.clone()),
            None => crate::provider::client::global_client().clone(),
        }
    }

    /// Get the full completions endpoint URL (the provider owns the path suffix:
    /// OpenAI-wire `/chat/completions`, Anthropic `/v1/messages`).
    pub fn completions_url(&self) -> String {
        self.provider.endpoint_url(&self.base_url)
    }
}

// Pre-configured generators for common providers
impl GeneratorInfo {
    /// Create an OpenRouter generator (native USD cost, `/generation` fallback).
    ///
    /// Attribution defaults to the library's identity; override it with
    /// [`with_app_attribution`](Self::with_app_attribution) (or a
    /// `CompletionContext`) to attribute usage to your app. The OpenRouter
    /// provider turns the attribution into `HTTP-Referer`/`X-Title` headers.
    pub fn openrouter(model: impl Into<String>) -> Self {
        Self::new("OpenRouter", "https://openrouter.ai/api/v1", model)
            .with_provider(Arc::new(OpenRouterProvider))
            .with_api_key_from_env("OPENROUTER_API_KEY")
            .with_app_attribution("https://github.com/minillmlib", "MiniLLMLib")
    }

    /// Create an OpenAI generator.
    ///
    /// OpenAI returns token counts but no dollar cost, so calls are priced at
    /// the model's price: the one given with
    /// [`with_token_price`](Self::with_token_price), else OpenRouter's catalog
    /// rate for `openai/<model>`.
    pub fn openai(model: impl Into<String>) -> Self {
        Self::new("OpenAI", "https://api.openai.com/v1", model)
            .with_provider(Arc::new(OpenAiProvider))
            .with_api_key_from_env("OPENAI_API_KEY")
    }

    /// Create a native Anthropic generator (`/v1/messages`, `content[]` envelope,
    /// `x-api-key` auth from `ANTHROPIC_API_KEY`).
    ///
    /// Anthropic returns token counts but no dollar cost, so calls are priced
    /// at the model's price: the one given with
    /// [`with_token_price`](Self::with_token_price), else OpenRouter's catalog
    /// rate, which needs [`with_openrouter_name`](Self::with_openrouter_name)
    /// since Anthropic's ids are not catalog ids.
    pub fn anthropic(model: impl Into<String>) -> Self {
        Self::new("Anthropic", "https://api.anthropic.com", model)
            .with_provider(Arc::new(AnthropicProvider))
            .with_api_key_from_env("ANTHROPIC_API_KEY")
    }

    /// Create a custom URL-based generator
    pub fn custom(
        name: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self::new(name, base_url, model)
    }
}

#[cfg(test)]
mod tests {
    use super::GeneratorInfo;
    use crate::generator::CompletionParameters;
    use crate::provider::{
        CostOutcome, CostResolution, InfoFuture, ModelInfo, ModelLookup, Provider, TokenPrice,
        Usage, Wanted,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    /// Answers lookups with fixed info, after `delay`, recording what each
    /// lookup was given; optionally requires a cap like Anthropic.
    #[derive(Debug, Default)]
    struct Scripted {
        lookups: AtomicUsize,
        seen: Mutex<Vec<(Option<String>, Wanted)>>,
        requires_cap: bool,
        delay: Duration,
    }

    impl Provider for Scripted {
        fn model_info(&self, lookup: ModelLookup) -> Option<InfoFuture> {
            self.lookups.fetch_add(1, Ordering::SeqCst);
            self.seen
                .lock()
                .unwrap()
                .push((lookup.openrouter_name.clone(), lookup.wanted));
            let delay = self.delay;
            Some(Box::pin(async move {
                tokio::time::sleep(delay).await;
                ModelInfo {
                    price: Some(TokenPrice::new(1.0, 2.0)),
                    max_output_tokens: Some(100),
                    context_length: Some(1000),
                }
            }))
        }

        fn requires_max_tokens(&self) -> bool {
            self.requires_cap
        }

        fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
            crate::provider::wire::price_or_unpriced(usage, price)
        }
    }

    fn scripted(provider: Scripted) -> (GeneratorInfo, Arc<Scripted>) {
        let provider = Arc::new(provider);
        let generator =
            GeneratorInfo::new("t", "http://unused", "a").with_provider(provider.clone());
        (generator, provider)
    }

    fn requiring_cap() -> Scripted {
        Scripted {
            requires_cap: true,
            ..Default::default()
        }
    }

    /// The caller's fields win over the generator's defaults, the defaults fill
    /// what the caller left unset, and what neither set stays unset on a wire
    /// that requires nothing.
    #[tokio::test]
    async fn the_caller_wins_over_the_generator_defaults() {
        let generator = GeneratorInfo::openai("m").with_default_params(
            CompletionParameters::new()
                .with_temperature(0.2)
                .with_seed(3),
        );
        let sent = generator
            .request_params(&CompletionParameters::new().with_temperature(0.9))
            .await;
        assert_eq!(sent.temperature, Some(0.9));
        assert_eq!(sent.seed, Some(3));
        assert_eq!(sent.max_tokens, None);
    }

    /// On a wire that requires a cap: the caller's, else the given largest
    /// output (no lookup), else the looked-up one.
    #[tokio::test]
    async fn a_required_cap_is_the_callers_else_the_models() {
        let (generator, provider) = scripted(requiring_cap());
        let set = CompletionParameters::new().with_max_tokens(7);
        assert_eq!(generator.request_params(&set).await.max_tokens, Some(7));
        let given = generator.clone().with_model_info(ModelInfo {
            max_output_tokens: Some(50),
            ..Default::default()
        });
        let unset = CompletionParameters::new();
        assert_eq!(given.request_params(&unset).await.max_tokens, Some(50));
        assert_eq!(provider.lookups.load(Ordering::SeqCst), 0);
        assert_eq!(generator.request_params(&unset).await.max_tokens, Some(100));
    }

    /// A lookup slower than the request's deadline is given up on: the request
    /// goes out with 4096 at the deadline rather than wait or fail.
    #[tokio::test]
    async fn a_slow_lookup_gives_way_to_4096_at_the_deadline() {
        let (generator, _) = scripted(Scripted {
            delay: Duration::from_secs(60),
            ..requiring_cap()
        });
        let client = generator.client().http().clone();
        let started = std::time::Instant::now();
        let sent = generator
            .request_params_over(
                &client,
                &CompletionParameters::new(),
                Some(Duration::from_millis(20)),
            )
            .await;
        assert_eq!(sent.max_tokens, Some(4096));
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    /// The fields the caller gives win, the rest is looked up, and the lookup
    /// is told what is still missing and the catalog id; with every field
    /// given, nothing is looked up.
    #[tokio::test]
    async fn given_fields_win_and_the_rest_is_looked_up() {
        let (generator, provider) = scripted(Scripted::default());
        let given = generator
            .clone()
            .with_openrouter_name("vendor/a")
            .with_model_info(ModelInfo {
                max_output_tokens: Some(7),
                ..Default::default()
            });
        let info = given.model_info().await;
        assert_eq!(info.max_output_tokens, Some(7));
        assert_eq!(info.price, Some(TokenPrice::new(1.0, 2.0)));
        let seen = provider.seen.lock().unwrap().clone();
        assert_eq!(seen[0].0.as_deref(), Some("vendor/a"));
        assert_eq!(
            seen[0].1,
            Wanted::ALL,
            "the context window is still missing, so the limits are asked for"
        );

        let enough = generator
            .with_token_price(TokenPrice::new(3.0, 4.0))
            .with_model_info(ModelInfo {
                max_output_tokens: Some(9),
                context_length: Some(90),
                ..Default::default()
            });
        let info = enough.model_info().await;
        assert_eq!(
            info.price,
            Some(TokenPrice::new(3.0, 4.0)),
            "with_model_info keeps the price"
        );
        assert_eq!(info.context_length, Some(90));
        assert_eq!(provider.lookups.load(Ordering::SeqCst), 1);
    }

    /// A provider with no way to look a model up tells only what was given.
    #[tokio::test]
    async fn a_provider_without_a_lookup_tells_only_what_was_given() {
        let generator = GeneratorInfo::custom("n", "http://unused", "m")
            .with_token_price(TokenPrice::new(1.0, 1.0));
        let info = generator.model_info().await;
        assert_eq!(info.price, Some(TokenPrice::new(1.0, 1.0)));
        assert_eq!(info.max_output_tokens, None);
    }

    /// A response the provider cannot price on its own is priced at the
    /// looked-up price, with only the price asked for; one the caller priced
    /// asks nothing.
    #[tokio::test]
    async fn an_unpriced_response_is_priced_at_the_looked_up_price() {
        let (generator, provider) = scripted(Scripted::default());
        let response = crate::provider::CompletionResponse {
            id: "r".into(),
            model: "a".into(),
            content: "hi".into(),
            finish_reason: None,
            usage: Some(Usage {
                uncached_input_tokens: 1_000_000,
                completion_tokens: 1_000_000,
                ..Default::default()
            }),
            tool_calls: None,
            media: Vec::new(),
            raw_response: None,
        };
        let cost = crate::tracking::cost_for_response(&generator, &response).await;
        assert_eq!(cost.resolution, CostResolution::Resolved);
        assert!((cost.cost - 3.0).abs() < 1e-9, "{cost:?}");
        assert_eq!(
            provider.seen.lock().unwrap()[0].1,
            Wanted {
                price: true,
                limits: false
            }
        );

        let priced = generator.with_token_price(TokenPrice::new(2.0, 2.0));
        let cost = crate::tracking::cost_for_response(&priced, &response).await;
        assert!((cost.cost - 4.0).abs() < 1e-9, "{cost:?}");
        assert_eq!(provider.lookups.load(Ordering::SeqCst), 1);
    }
}
