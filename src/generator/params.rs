//! Completion parameters for LLM requests

use serde::{Deserialize, Serialize};

/// Parameters for LLM completion requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionParameters {
    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Temperature for sampling (0.0 = deterministic, 2.0 = very random)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p (nucleus) sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Top-k sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,

    /// Frequency penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,

    /// Presence penalty (-2.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,

    /// Repetition penalty (1.0 = no penalty)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub repetition_penalty: Option<f32>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,

    /// Seed for reproducibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,

    /// Whether to stream the response
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,

    /// Response format (e.g., "json_object")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<ResponseFormat>,

    /// Tool/function definitions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,

    /// Tool choice strategy
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,

    /// OpenRouter provider settings (order, sort, ignore, data_collection, etc.)
    ///
    /// See: <https://openrouter.ai/docs/provider-routing>
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderSettings>,

    /// Extra parameters to pass directly to the API
    /// These are merged into the request body as-is
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub extra: Option<std::collections::HashMap<String, serde_json::Value>>,
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
            stream: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            provider: None,
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

    /// Enable streaming
    pub fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
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
            stream: other.stream.or(self.stream),
            response_format: other
                .response_format
                .clone()
                .or_else(|| self.response_format.clone()),
            tools: other.tools.clone().or_else(|| self.tools.clone()),
            tool_choice: other
                .tool_choice
                .clone()
                .or_else(|| self.tool_choice.clone()),
            provider: other.provider.clone().or_else(|| self.provider.clone()),
            extra: merged_extra,
        }
    }

    /// Set OpenRouter provider settings
    pub fn with_provider(mut self, provider: ProviderSettings) -> Self {
        self.provider = Some(provider);
        self
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

/// Response format specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseFormat {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
}

use crate::provider::{CostCallback, CostTrackingType};

/// Per-request completion parameters (used when calling complete on a node)
#[derive(Clone)]
pub struct NodeCompletionParameters {
    /// Override the generator for this request
    pub generator: Option<super::GeneratorInfo>,

    /// Override completion parameters
    pub params: Option<CompletionParameters>,

    /// System prompt to prepend
    pub system_prompt: Option<String>,

    /// Whether to use streaming
    pub stream: Option<bool>,

    /// Whether to parse/repair JSON response
    pub parse_json: bool,

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

    /// Custom request timeout in seconds
    pub timeout_secs: Option<u64>,

    // Cost tracking
    /// Type of cost tracking to use (default: None)
    pub cost_tracking: CostTrackingType,

    /// Callback function called with cost info after each completion
    pub cost_callback: Option<CostCallback>,
}

impl Default for NodeCompletionParameters {
    fn default() -> Self {
        Self {
            generator: None,
            params: None,
            system_prompt: None,
            stream: None,
            parse_json: false,
            force_prepend: None,
            retry: 4,
            exp_back_off: false,
            back_off_time: 1.0,
            max_back_off: 15.0,
            crash_on_refusal: false,
            crash_on_empty_response: false,
            timeout_secs: None,
            cost_tracking: CostTrackingType::None,
            cost_callback: None,
        }
    }
}

impl std::fmt::Debug for NodeCompletionParameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeCompletionParameters")
            .field("generator", &self.generator)
            .field("params", &self.params)
            .field("system_prompt", &self.system_prompt)
            .field("stream", &self.stream)
            .field("parse_json", &self.parse_json)
            .field("force_prepend", &self.force_prepend)
            .field("retry", &self.retry)
            .field("cost_tracking", &self.cost_tracking)
            .field("cost_callback", &self.cost_callback.is_some())
            .finish()
    }
}

impl NodeCompletionParameters {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_generator(mut self, generator: super::GeneratorInfo) -> Self {
        self.generator = Some(generator);
        self
    }

    pub fn with_params(mut self, params: CompletionParameters) -> Self {
        self.params = Some(params);
        self
    }

    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    pub fn with_streaming(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }

    /// Enable JSON parsing/repair of the response
    pub fn with_parse_json(mut self, parse: bool) -> Self {
        self.parse_json = parse;
        self
    }

    /// Alias for with_parse_json(true) - for backwards compatibility
    pub fn expecting_json(mut self) -> Self {
        self.parse_json = true;
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

    /// Enable cost tracking with OpenRouter's usage accounting
    pub fn with_openrouter_cost_tracking(mut self) -> Self {
        self.cost_tracking = CostTrackingType::OpenRouter;
        self
    }

    /// Set the cost tracking type
    pub fn with_cost_tracking(mut self, tracking_type: CostTrackingType) -> Self {
        self.cost_tracking = tracking_type;
        self
    }

    /// Set a callback function to be called with cost info after each completion
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::{Arc, Mutex};
    ///
    /// let total_cost = Arc::new(Mutex::new(0.0));
    /// let cost_tracker = total_cost.clone();
    ///
    /// let params = NodeCompletionParameters::default()
    ///     .with_openrouter_cost_tracking()
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
