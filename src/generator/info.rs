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
        }
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
