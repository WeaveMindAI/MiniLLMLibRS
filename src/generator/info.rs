//! Generator information - configuration for LLM providers

use secrecy::SecretString;

/// Configuration for an LLM provider/generator
#[derive(Debug, Clone)]
pub struct GeneratorInfo {
    /// Display name for this generator
    pub name: String,

    /// Base URL for the API (e.g., "https://openrouter.ai/api/v1")
    pub base_url: String,

    /// Model identifier (e.g., "anthropic/claude-3.5-sonnet")
    pub model: String,

    /// API key (stored securely)
    pub api_key: Option<SecretString>,

    /// Optional organization ID
    pub organization_id: Option<String>,

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

    /// Default completion parameters for this generator
    pub default_params: super::CompletionParameters,
}

impl GeneratorInfo {
    /// Create a new GeneratorInfo with minimal configuration
    pub fn new(
        name: impl Into<String>,
        base_url: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into(),
            model: model.into(),
            api_key: None,
            organization_id: None,
            custom_headers: Vec::new(),
            supports_streaming: true,
            supports_vision: false,
            supports_audio: false,
            max_context_length: None,
            default_params: super::CompletionParameters::default(),
        }
    }

    /// Set the API key
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(SecretString::from(key.into()));
        self
    }

    /// Set API key from environment variable
    pub fn with_api_key_from_env(mut self, env_var: &str) -> Self {
        if let Ok(key) = std::env::var(env_var) {
            self.api_key = Some(SecretString::from(key));
        }
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

    /// Get the full completions endpoint URL
    pub fn completions_url(&self) -> String {
        let base = self.base_url.trim_end_matches('/');
        format!("{}/chat/completions", base)
    }
}

// Pre-configured generators for common providers
impl GeneratorInfo {
    /// Create an OpenRouter generator
    pub fn openrouter(model: impl Into<String>) -> Self {
        Self::new("OpenRouter", "https://openrouter.ai/api/v1", model)
            .with_api_key_from_env("OPENROUTER_API_KEY")
            .with_header("HTTP-Referer", "https://github.com/minillmlib")
            .with_header("X-Title", "MiniLLMLib")
    }

    /// Create an OpenAI generator
    pub fn openai(model: impl Into<String>) -> Self {
        Self::new("OpenAI", "https://api.openai.com/v1", model)
            .with_api_key_from_env("OPENAI_API_KEY")
    }

    /// Create an Anthropic generator (via OpenRouter format)
    pub fn anthropic(model: impl Into<String>) -> Self {
        Self::new("Anthropic", "https://api.anthropic.com/v1", model)
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
