//! Generator information - configuration for LLM providers

use crate::provider::{
    AnthropicProvider, AppIdentity, Auth, GenericProvider, OpenAiProvider, OpenRouterProvider,
    Provider, TokenPrice,
};
use secrecy::SecretString;
use std::sync::Arc;

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
    /// This is what unlocks cost estimation: [`model_rates`] looks the model up
    /// by this name, falling back to `model` itself (an OpenRouter generator's
    /// model id already IS the catalog id, so it needs nothing set here). A
    /// vendor's own id rarely matches the catalog's (Anthropic's
    /// `claude-haiku-4-5-20251001` is the catalog's `anthropic/claude-haiku-4.5`),
    /// so a direct-vendor generator sets this to become estimable.
    ///
    /// [`model`]: Self::model
    /// [`model_rates`]: Self::model_rates
    pub openrouter_name: Option<String>,

    /// How this generator authenticates. The provider maps it to concrete headers
    /// (OpenAI-wire `Authorization: Bearer`, Anthropic `x-api-key` or bearer).
    pub auth: Auth,

    /// Custom headers to include in requests
    pub custom_headers: Vec<(String, String)>,

    /// Whether this provider supports streaming
    pub supports_streaming: bool,

    /// Whether this provider supports vision/images
    pub supports_vision: bool,

    /// Whether this provider supports audio input
    pub supports_audio: bool,

    /// Maximum context length (tokens)
    pub max_context_length: Option<usize>,

    /// Provider implementation: the wire dialect (token-limit key, usage opt-in,
    /// usage parsing, cost aggregation, out-of-band resolution, attribution
    /// headers). Swap this to target a different provider.
    pub provider: Arc<dyn Provider>,

    /// Per-token price for this model, used to derive cost when the provider
    /// returns token counts but no dollar amount (OpenAI, Anthropic, ...). When
    /// `None` and the provider has no native cost, cost is reported `Unpriced`.
    pub token_price: Option<TokenPrice>,

    /// Calling-app identity for providers that attribute usage to an app (e.g.
    /// OpenRouter rankings). The provider decides which headers express it.
    pub app_attribution: Option<AppIdentity>,

    /// Default completion parameters for this generator
    pub default_params: super::CompletionParameters,

    /// The generator's cached view of OpenRouter's published prices for its
    /// model, behind [`model_rates`](Self::model_rates). Clones share it, so a
    /// generator kept alive keeps its prices warm instead of refetching per call.
    pub(crate) prices: super::pricing::PriceCache,

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
            supports_streaming: true,
            supports_vision: false,
            supports_audio: false,
            max_context_length: None,
            provider: Arc::new(GenericProvider::default()),
            token_price: None,
            app_attribution: None,
            default_params: super::CompletionParameters::default(),
            prices: super::pricing::PriceCache::default(),
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

    /// Set the model's OpenRouter catalog id, which unlocks cost estimation when
    /// the generator's own model id is not already a catalog id (see
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

    /// Set the per-token price (USD per million tokens) used to derive cost for
    /// providers that return token counts but no dollar amount.
    pub fn with_token_price(mut self, price: TokenPrice) -> Self {
        self.token_price = Some(price);
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

    /// Set an OAuth/bearer token (e.g. a Claude subscription token). Always sent
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

    /// Enable vision support
    pub fn with_vision(mut self) -> Self {
        self.supports_vision = true;
        self
    }

    /// Enable audio support
    pub fn with_audio(mut self) -> Self {
        self.supports_audio = true;
        self
    }

    /// Set max context length
    pub fn with_max_context(mut self, length: usize) -> Self {
        self.max_context_length = Some(length);
        self
    }

    /// Set default completion parameters
    pub fn with_default_params(mut self, params: super::CompletionParameters) -> Self {
        self.default_params = params;
        self
    }

    /// Supply the HTTP client every request for this generator rides
    /// (completions, streaming, out-of-band cost queries). Accepts a plain
    /// `reqwest::Client` or a `ClientWithMiddleware`.
    pub fn with_http_client(
        mut self,
        client: impl Into<reqwest_middleware::ClientWithMiddleware>,
    ) -> Self {
        self.http_client = Some(client.into());
        self
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
    /// OpenAI returns token counts but no dollar cost, so set a
    /// [`with_token_price`](Self::with_token_price) to get a resolved cost;
    /// otherwise cost tracking reports `Unpriced`.
    pub fn openai(model: impl Into<String>) -> Self {
        Self::new("OpenAI", "https://api.openai.com/v1", model)
            .with_provider(Arc::new(OpenAiProvider))
            .with_api_key_from_env("OPENAI_API_KEY")
    }

    /// Create a native Anthropic generator (`/v1/messages`, `content[]` envelope,
    /// `x-api-key` auth from `ANTHROPIC_API_KEY`).
    ///
    /// Anthropic returns token counts but no dollar cost, so set a
    /// [`with_token_price`](Self::with_token_price) for a resolved cost; otherwise
    /// cost tracking reports `Unpriced`.
    pub fn anthropic(model: impl Into<String>) -> Self {
        Self::new("Anthropic", "https://api.anthropic.com", model)
            .with_provider(Arc::new(AnthropicProvider))
            .with_api_key_from_env("ANTHROPIC_API_KEY")
    }

    /// Create a Claude **subscription** generator: native Anthropic wire,
    /// authenticated with a Claude Pro/Max OAuth bearer token so usage draws on
    /// the **subscription's** rolling quota rather than pay-as-you-go API billing.
    ///
    /// The token is resolved by [`crate::resolve_claude_subscription_auth`]: the
    /// `ANTHROPIC_AUTH_TOKEN` env var supersedes, otherwise the live Claude Code
    /// credential at `~/.claude/.credentials.json` (which Claude Code keeps
    /// refreshed) is used. NOTE: a Console/API OAuth token (e.g. from the `ant`
    /// CLI) bills the API account, NOT the subscription; use an API key via
    /// [`anthropic`](Self::anthropic) for Console, and this preset only for the
    /// actual Pro/Max subscription token.
    ///
    /// Cost is an ESTIMATE: Anthropic returns only token counts, so set a
    /// [`with_token_price`](Self::with_token_price) reflecting the model's
    /// published price for a `Resolved` USD estimate; otherwise `Unpriced`.
    pub fn claude_subscription(model: impl Into<String>) -> Self {
        Self::new("Claude (subscription)", "https://api.anthropic.com", model)
            .with_provider(Arc::new(AnthropicProvider))
            .with_auth(crate::provider::resolve_claude_subscription_auth())
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
mod estimation_identity_tests {
    use super::GeneratorInfo;

    /// The names cost estimation resolves from live on concepts that already
    /// exist: the provider impl knows its catalog slug, and the generator knows
    /// its model's catalog id (defaulting to the model itself).
    #[test]
    fn each_vendor_provider_knows_its_own_catalog_slug() {
        assert_eq!(
            GeneratorInfo::anthropic("m").provider.openrouter_slug(),
            Some("anthropic")
        );
        assert_eq!(
            GeneratorInfo::claude_subscription("m")
                .provider
                .openrouter_slug(),
            Some("anthropic"),
            "a subscription call is still served by Anthropic"
        );
        assert_eq!(
            GeneratorInfo::openai("m").provider.openrouter_slug(),
            Some("openai")
        );
    }

    /// A router or a custom API is not a vendor the catalog lists, so it claims
    /// no slug: estimation bounds over every provider serving the model, which is
    /// exactly where such a call may land.
    #[test]
    fn a_router_or_custom_provider_claims_no_slug() {
        assert_eq!(
            GeneratorInfo::openrouter("m").provider.openrouter_slug(),
            None
        );
        assert_eq!(
            GeneratorInfo::custom("n", "u", "m")
                .provider
                .openrouter_slug(),
            None
        );
    }

    /// A vendor's own model id rarely matches the catalog's; setting the catalog
    /// id is what unlocks estimation. Unset, the model id itself is the lookup.
    #[test]
    fn the_openrouter_name_defaults_to_unset_and_is_settable() {
        let plain = GeneratorInfo::anthropic("claude-haiku-4-5-20251001");
        assert_eq!(plain.openrouter_name, None);

        let estimable = plain.with_openrouter_name("anthropic/claude-haiku-4.5");
        assert_eq!(
            estimable.openrouter_name.as_deref(),
            Some("anthropic/claude-haiku-4.5")
        );
    }
}
