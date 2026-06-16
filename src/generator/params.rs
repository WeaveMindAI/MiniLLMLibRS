//! Completion parameters for LLM requests

use serde::{Deserialize, Serialize};

/// Normalized, provider-agnostic completion parameters.
///
/// This is NOT a wire shape; it is normalized *intent*. Each provider's
/// [`build_request`](crate::Provider::build_request) translates these fields into
/// its own request body (OpenAI's flat keys, Anthropic's `/v1/messages` shape,
/// etc.), so the same parameters drive any provider identically. Provider-specific
/// knobs that have no normalized meaning (e.g. OpenRouter routing) go through
/// [`extra`](Self::extra), the documented escape hatch: they are honestly just
/// extra wire keys, not pretend-universal fields.
#[derive(Debug, Clone)]
pub struct CompletionParameters {
    /// Maximum tokens to generate. The provider emits it under its own key
    /// (`max_completion_tokens`, `max_tokens`, Anthropic's required `max_tokens`).
    pub max_tokens: Option<u32>,

    /// Temperature for sampling (0.0 = deterministic, 2.0 = very random)
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling
    pub top_p: Option<f32>,

    /// Top-k sampling
    pub top_k: Option<u32>,

    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: Option<f32>,

    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: Option<f32>,

    /// Stop sequences (OpenAI `stop`, Anthropic `stop_sequences`).
    pub stop: Option<Vec<String>>,

    /// Seed for reproducibility
    pub seed: Option<u64>,

    /// Normalized response-format intent (e.g. force JSON output). The provider
    /// maps it to its wire (OpenAI `response_format`, Anthropic structured output).
    pub response_format: Option<ResponseFormat>,

    /// Tool/function definitions (currently OpenAI-shaped JSON, passed through).
    pub tools: Option<Vec<serde_json::Value>>,

    /// Tool choice strategy.
    pub tool_choice: Option<serde_json::Value>,

    /// Reasoning configuration (for models that support extended thinking).
    pub reasoning: Option<ReasoningConfig>,

    /// Provider-specific extra keys, merged into the request body as-is. This is
    /// the honest home for anything without a normalized meaning, e.g. OpenRouter
    /// routing via [`with_openrouter_routing`](Self::with_openrouter_routing).
    pub extra: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Reasoning configuration for models that support extended thinking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    /// Effort level: "none", "minimal", "low", "medium", "high", "xhigh"
    /// "none" disables reasoning entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub effort: Option<String>,

    /// Explicit reasoning token budget (used by Anthropic, Gemini).
    /// When set, overrides effort-based calculation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// If true, reasoning is performed but excluded from the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exclude: Option<bool>,
}

/// OpenRouter provider routing settings
///
/// See: <https://openrouter.ai/docs/provider-routing>
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProviderSettings {
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

impl ProviderSettings {
    /// Create new provider settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialize to the JSON value OpenRouter expects under the request's
    /// `"provider"` key (used by [`CompletionParameters::with_openrouter_routing`]).
    pub fn to_value(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
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

impl Default for CompletionParameters {
    fn default() -> Self {
        Self {
            max_tokens: Some(4096),
            temperature: Some(0.7),
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop: None,
            seed: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            extra: None,
        }
    }
}

impl CompletionParameters {
    /// Create new parameters with just max_tokens
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set top_p
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Set stop sequences
    pub fn with_stop(mut self, stop: Vec<String>) -> Self {
        self.stop = Some(stop);
        self
    }

    /// Set JSON response format
    pub fn with_json_response(mut self) -> Self {
        self.response_format = Some(ResponseFormat::JsonObject);
        self
    }

    /// Set seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set reasoning configuration
    pub fn with_reasoning(mut self, reasoning: ReasoningConfig) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Merge with another set of parameters (other takes precedence)
    pub fn merge(&self, other: &CompletionParameters) -> CompletionParameters {
        // Merge extra params - combine both, with other taking precedence
        let merged_extra = match (&self.extra, &other.extra) {
            (Some(base), Some(over)) => {
                let mut merged = base.clone();
                merged.extend(over.clone());
                Some(merged)
            }
            (Some(base), None) => Some(base.clone()),
            (None, Some(over)) => Some(over.clone()),
            (None, None) => None,
        };

        CompletionParameters {
            max_tokens: other.max_tokens.or(self.max_tokens),
            temperature: other.temperature.or(self.temperature),
            top_p: other.top_p.or(self.top_p),
            top_k: other.top_k.or(self.top_k),
            frequency_penalty: other.frequency_penalty.or(self.frequency_penalty),
            presence_penalty: other.presence_penalty.or(self.presence_penalty),
            repetition_penalty: other.repetition_penalty.or(self.repetition_penalty),
            stop: other.stop.clone().or_else(|| self.stop.clone()),
            seed: other.seed.or(self.seed),
            response_format: other
                .response_format
                .clone()
                .or_else(|| self.response_format.clone()),
            tools: other.tools.clone().or_else(|| self.tools.clone()),
            tool_choice: other
                .tool_choice
                .clone()
                .or_else(|| self.tool_choice.clone()),
            reasoning: other.reasoning.clone().or_else(|| self.reasoning.clone()),
            extra: merged_extra,
        }
    }

    /// Set OpenRouter provider-routing settings (order/sort/ignore/data_collection).
    ///
    /// OpenRouter-specific: it goes through [`extra`](Self::extra) under the
    /// `"provider"` key (the honest home for a provider-specific knob), so it
    /// reaches OpenRouter's wire and is simply ignored by providers that don't
    /// understand it, rather than masquerading as a universal parameter.
    pub fn with_openrouter_routing(self, routing: ProviderSettings) -> Self {
        self.with_extra("provider", routing.to_value())
    }

    /// Add extra parameters to pass directly to the API
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        let extra = self
            .extra
            .get_or_insert_with(std::collections::HashMap::new);
        extra.insert(key.into(), value);
        self
    }

    /// Set multiple extra parameters at once
    pub fn with_extras(
        mut self,
        extras: std::collections::HashMap<String, serde_json::Value>,
    ) -> Self {
        self.extra = Some(extras);
        self
    }
}

/// Normalized response-format intent. The provider maps it to its wire.
///
/// Only the constraint the library can actually request is represented: forcing a
/// JSON object. "Plain text" is the absence of a constraint (`response_format:
/// None`), not a variant: a `Text` variant would be unreachable decoration until
/// a builder produced it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResponseFormat {
    /// Force the model to emit a JSON object.
    JsonObject,
}

impl ResponseFormat {
    /// The OpenAI `response_format` value (`{"type": "json_object"}`).
    pub fn to_openai_value(&self) -> serde_json::Value {
        match self {
            ResponseFormat::JsonObject => serde_json::json!({"type": "json_object"}),
        }
    }
}

use crate::provider::{CostCallback, TokenPrice};

/// Per-request completion parameters (used when calling complete on a node)
#[derive(Clone)]
pub struct NodeCompletionParameters {
    /// Override completion parameters
    pub params: Option<CompletionParameters>,

    /// System prompt to prepend
    pub system_prompt: Option<String>,

    /// Completion-level format kwargs: applied at format time as the base layer
    /// for `{placeholder}` substitution. Per-node override kwargs (set via
    /// `ChatNode::set_format_kwarg`) layer on top and win on key collision.
    pub format_kwargs: std::collections::HashMap<String, String>,

    /// Whether the assistant response is appended as a real child of the node.
    /// When false, the returned node is a "phantom": it knows its parent (so
    /// `thread()` works) but the parent does not list it, leaving the tree
    /// untouched. Default: true.
    pub add_child: bool,

    /// Whether to parse/repair JSON response
    pub parse_json: bool,

    /// Auto-cache the stable prompt prefix for this request: marks the whole
    /// prompt (everything up to the turn being generated) as a cache breakpoint,
    /// so the provider caches it (Anthropic) or it's a no-op (OpenAI auto-caches).
    /// A convenience over marking individual nodes; explicit
    /// [`ChatNode::cache_breakpoint`](crate::ChatNode::cache_breakpoint) marks are
    /// always honored in addition. Default: false.
    pub use_cache: bool,

    /// Force text to be prepended to the assistant's response
    /// e.g., force_prepend="Score: " makes the LLM start with "Score: "
    pub force_prepend: Option<String>,

    // Retry and error handling
    /// Number of retry attempts on error (default: 4)
    pub retry: u32,

    /// Use exponential backoff between retries
    pub exp_back_off: bool,

    /// Initial wait time in seconds for backoff (default: 1.0)
    pub back_off_time: f64,

    /// Maximum wait time in seconds for backoff (default: 15.0)
    pub max_back_off: f64,

    /// Raise error if model doesn't return valid JSON (when parse_json=true)
    pub crash_on_refusal: bool,

    /// Raise error if model returns empty output
    pub crash_on_empty_response: bool,

    /// Custom request timeout in seconds. For non-streaming completions this is
    /// the total-response deadline; for streaming completions it is the idle
    /// timeout (max silence between chunks), so a long but live generation is not
    /// killed while a dead/silent connection still fails loudly.
    pub timeout_secs: Option<u64>,

    // Cost tracking
    /// Whether to request and report usage/cost for this completion. The *how*
    /// (usage opt-in flag, parsing, aggregation, out-of-band resolution) is owned
    /// by the generator's provider accounting; this only says whether to track.
    pub track_cost: bool,

    /// Per-request override of the generator's `token_price` (for providers that
    /// price by token). When `None`, the generator's price is used.
    pub token_price: Option<TokenPrice>,

    /// Callback function called with cost info after each completion
    pub cost_callback: Option<CostCallback>,
}

impl Default for NodeCompletionParameters {
    fn default() -> Self {
        Self {
            params: None,
            system_prompt: None,
            format_kwargs: std::collections::HashMap::new(),
            add_child: true,
            parse_json: false,
            use_cache: false,
            force_prepend: None,
            retry: 4,
            exp_back_off: false,
            back_off_time: 1.0,
            max_back_off: 15.0,
            crash_on_refusal: false,
            crash_on_empty_response: false,
            timeout_secs: None,
            track_cost: false,
            token_price: None,
            cost_callback: None,
        }
    }
}

impl std::fmt::Debug for NodeCompletionParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeCompletionParameters")
            .field("params", &self.params)
            .field("system_prompt", &self.system_prompt)
            .field("format_kwargs", &self.format_kwargs)
            .field("add_child", &self.add_child)
            .field("parse_json", &self.parse_json)
            .field("force_prepend", &self.force_prepend)
            .field("retry", &self.retry)
            .field("track_cost", &self.track_cost)
            .field("cost_callback", &self.cost_callback.is_some())
            .finish()
    }
}

impl NodeCompletionParameters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_params(mut self, params: CompletionParameters) -> Self {
        self.params = Some(params);
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the completion-level format kwargs (replaces any already set).
    pub fn with_format_kwargs(mut self, kwargs: std::collections::HashMap<String, String>) -> Self {
        self.format_kwargs = kwargs;
        self
    }

    /// Add a single completion-level format kwarg.
    pub fn with_format_kwarg(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.format_kwargs.insert(key.into(), value.into());
        self
    }

    /// Set whether the assistant response is appended as a real child (true) or
    /// returned as a phantom node leaving the tree untouched (false).
    pub fn with_add_child(mut self, add_child: bool) -> Self {
        self.add_child = add_child;
        self
    }

    /// Enable JSON parsing/repair of the response
    pub fn with_parse_json(mut self, parse: bool) -> Self {
        self.parse_json = parse;
        self
    }

    /// Readable alias for `with_parse_json(true)`.
    pub fn expecting_json(mut self) -> Self {
        self.parse_json = true;
        self
    }

    /// Auto-cache the stable prompt prefix for this request. The provider decides
    /// the wire (Anthropic marks it; OpenAI auto-caches anyway). Explicit per-node
    /// [`cache_breakpoint`](crate::ChatNode::cache_breakpoint) marks still apply.
    pub fn with_cache(mut self, use_cache: bool) -> Self {
        self.use_cache = use_cache;
        self
    }

    /// Force text to be prepended to the assistant's response
    pub fn with_force_prepend(mut self, prepend: impl Into<String>) -> Self {
        self.force_prepend = Some(prepend.into());
        self
    }

    /// Set number of retry attempts
    pub fn with_retry(mut self, retry: u32) -> Self {
        self.retry = retry;
        self
    }

    /// Enable exponential backoff
    pub fn with_exp_back_off(mut self, enabled: bool) -> Self {
        self.exp_back_off = enabled;
        self
    }

    /// Set initial backoff time in seconds
    pub fn with_back_off_time(mut self, secs: f64) -> Self {
        self.back_off_time = secs;
        self
    }

    /// Set maximum backoff time in seconds
    pub fn with_max_back_off(mut self, secs: f64) -> Self {
        self.max_back_off = secs;
        self
    }

    /// Crash if model doesn't return valid JSON (requires parse_json=true)
    pub fn with_crash_on_refusal(mut self, crash: bool) -> Self {
        self.crash_on_refusal = crash;
        self
    }

    /// Crash if model returns empty response
    pub fn with_crash_on_empty(mut self, crash: bool) -> Self {
        self.crash_on_empty_response = crash;
        self
    }

    pub fn with_timeout(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// Enable usage/cost tracking. The provider's accounting (on the generator)
    /// decides how usage is requested, parsed, aggregated, and resolved.
    pub fn with_cost_tracking(mut self, track: bool) -> Self {
        self.track_cost = track;
        self
    }

    /// Override the generator's per-token price for this request (for providers
    /// that price by token).
    pub fn with_token_price(mut self, price: TokenPrice) -> Self {
        self.token_price = Some(price);
        self
    }

    /// Set a callback function to be called with cost info after each completion
    ///
    /// # Example
    /// ```
    /// use std::sync::{Arc, Mutex};
    /// use minillmlib::NodeCompletionParameters;
    ///
    /// let total_cost = Arc::new(Mutex::new(0.0));
    /// let cost_tracker = total_cost.clone();
    ///
    /// let params = NodeCompletionParameters::default()
    ///     .with_cost_tracking(true)
    ///     .with_cost_callback(move |info| {
    ///         let mut cost = cost_tracker.lock().unwrap();
    ///         *cost += info.cost;
    ///         println!("Request cost: {} credits, Total: {}", info.cost, *cost);
    ///     });
    /// ```
    pub fn with_cost_callback<F>(mut self, callback: F) -> Self
    where
        F: Fn(crate::provider::CostInfo) + Send + Sync + 'static,
    {
        self.cost_callback = Some(std::sync::Arc::new(callback));
        self
    }
}
