//! Completion parameters for LLM requests

use crate::tools::{ToolChoice, ToolDefinition};
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
///
/// Every field is unset by default, and an unset field is no preference: the
/// provider's own default applies (see each field for what that means on a
/// wire that requires the key).
///
/// Serde uses camelCase keys with every field optional (missing fields stay
/// unset), so a JSON settings object (`{"maxTokens": 1024,
/// "temperature": 0.2}`) deserializes directly; unknown keys are ignored.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CompletionParameters {
    /// The most tokens the model may produce, reasoning and answer together.
    /// The provider emits it under its own key (`max_completion_tokens`,
    /// `max_tokens`); every supported wire already counts reasoning inside it.
    /// Unset means no limit beyond the model's own: nothing is sent, except on
    /// a wire whose API requires the key (Anthropic), which sends the model's
    /// largest output, or 4096 when that is unknown.
    pub max_tokens: Option<u32>,

    /// Temperature for sampling (0.0 = deterministic, 2.0 = very random).
    /// `f64` on purpose, like every sampling knob here: these values reach
    /// the wire as JSON numbers, and squeezing 0.9 through an f32 emits
    /// `0.8999999761581421` (the shortest text that round-trips the widened
    /// float). Strict providers reject that many decimals (Z.AI caps
    /// temperature at two).
    pub temperature: Option<f64>,

    /// Top-p (nucleus) sampling
    pub top_p: Option<f64>,

    /// Top-k sampling
    pub top_k: Option<u32>,

    /// Frequency penalty (-2.0 to 2.0)
    pub frequency_penalty: Option<f64>,

    /// Presence penalty (-2.0 to 2.0)
    pub presence_penalty: Option<f64>,

    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: Option<f64>,

    /// Stop sequences (OpenAI `stop`, Anthropic `stop_sequences`).
    pub stop: Option<Vec<String>>,

    /// Seed for reproducibility
    pub seed: Option<u64>,

    /// Normalized response-format intent (e.g. force JSON output), sent as
    /// OpenAI's `response_format`. Anthropic has no such field and refuses it.
    pub response_format: Option<ResponseFormat>,

    /// Tools the model may call, as normalized [`ToolDefinition`]s. The provider
    /// emits its wire shape (OpenAI `tools[].function`, Anthropic `input_schema`).
    pub tools: Option<Vec<ToolDefinition>>,

    /// How the model must treat the tools (auto / none / required / a specific
    /// tool). The provider emits its wire shape.
    pub tool_choice: Option<ToolChoice>,

    /// Whether the model may call several tools in one turn. `None` leaves the
    /// provider default (parallel allowed). OpenAI-wire emits
    /// `parallel_tool_calls`; Anthropic folds `disable_parallel_tool_use` into
    /// `tool_choice`.
    pub parallel_tool_calls: Option<bool>,

    /// Whether the model reasons, and how hard (see [`Reasoning`]). Unset
    /// sends nothing, so the API's own default applies.
    pub reasoning: Option<Reasoning>,

    /// Provider-specific extra keys, merged into the request body as-is. This is
    /// the honest home for anything without a normalized meaning, e.g. OpenRouter
    /// routing via [`with_openrouter_routing`](Self::with_openrouter_routing).
    pub extra: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Whether the model reasons before answering. Unset (`None` on
/// [`CompletionParameters::reasoning`]) sends nothing, so the API's own default
/// applies.
///
/// Each provider sends its own spelling, and nothing is checked: a model that
/// cannot do what was asked refuses at the API, and that refusal is the
/// request's error.
///
/// In JSON it is one word: `"off"`, `"on"`, or anything else, which is an
/// effort sent as written (`"high"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reasoning {
    /// What the API calls no reasoning.
    Off,
    /// Reason. `None` is the API's lowest effort; `Some` is an effort word
    /// sent as written, in the API's own vocabulary.
    On(Option<String>),
}

impl Reasoning {
    /// The one word for it: `"off"`, `"on"`, or the effort.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Off => "off",
            Self::On(None) => "on",
            Self::On(Some(effort)) => effort,
        }
    }
}

impl From<&str> for Reasoning {
    fn from(word: &str) -> Self {
        match word {
            "off" => Self::Off,
            "on" => Self::On(None),
            effort => Self::On(Some(effort.to_string())),
        }
    }
}

impl Serialize for Reasoning {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for Reasoning {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        Ok(Self::from(String::deserialize(d)?.as_str()))
    }
}

impl CompletionParameters {
    /// Create new parameters with every field unset.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Set temperature
    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = Some(temp);
        self
    }

    /// Set top_p
    pub fn with_top_p(mut self, top_p: f64) -> Self {
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

    /// Set whether the model reasons, and how hard.
    pub fn with_reasoning(mut self, reasoning: Reasoning) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set the tools the model may call (replaces any already set).
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Add a single tool the model may call.
    pub fn with_tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.get_or_insert_with(Vec::new).push(tool);
        self
    }

    /// Set the tool-choice strategy (auto / none / required / a specific tool).
    pub fn with_tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }

    /// Allow or forbid the model calling several tools in one turn.
    pub fn with_parallel_tool_calls(mut self, allow: bool) -> Self {
        self.parallel_tool_calls = Some(allow);
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
            parallel_tool_calls: other.parallel_tool_calls.or(self.parallel_tool_calls),
            reasoning: other.reasoning.clone().or_else(|| self.reasoning.clone()),
            extra: merged_extra,
        }
    }

    /// Add extra parameters to pass directly to the API
    pub fn with_extra(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        let extra = self
            .extra
            .get_or_insert_with(std::collections::HashMap::new);
        extra.insert(key.into(), value);
        self
    }

    /// Replace every extra parameter at once, dropping any set before
    /// (OpenRouter routing included).
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ResponseFormat {
    /// Force the model to emit a JSON object.
    JsonObject,
}

use crate::provider::CostCallback;

/// Per-request completion parameters (used when calling complete on a node)
#[derive(Clone)]
pub struct NodeCompletionParameters {
    /// The completion parameters for this request, layered over the
    /// generator's `default_params` (see
    /// [`GeneratorInfo::request_params`](crate::GeneratorInfo::request_params)).
    pub params: CompletionParameters,

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
    /// Number of retries after a failed attempt (default: 0, so one attempt).
    /// Each retry is a new paid request, so it is only made when asked for.
    pub retry: u32,

    /// Use exponential backoff between retries
    pub exp_back_off: bool,

    /// Initial wait time in seconds for backoff (default: 1.0)
    pub back_off_time: f64,

    /// Maximum wait time in seconds for backoff (default: 15.0)
    pub max_back_off: f64,

    /// With `parse_json`, reject a response whose repaired JSON is empty or
    /// null (what a model that refused instead of answering produces). A
    /// rejected response is retried when `retry > 0`.
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

    /// Callback function called with cost info after each completion
    pub cost_callback: Option<CostCallback>,
}

impl Default for NodeCompletionParameters {
    fn default() -> Self {
        Self {
            params: CompletionParameters::default(),
            system_prompt: None,
            format_kwargs: std::collections::HashMap::new(),
            add_child: true,
            parse_json: false,
            use_cache: false,
            force_prepend: None,
            retry: 0,
            exp_back_off: false,
            back_off_time: 1.0,
            max_back_off: 15.0,
            crash_on_refusal: false,
            crash_on_empty_response: false,
            timeout_secs: None,
            track_cost: false,
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
            .field("use_cache", &self.use_cache)
            .field("retry", &self.retry)
            .field("exp_back_off", &self.exp_back_off)
            .field("back_off_time", &self.back_off_time)
            .field("max_back_off", &self.max_back_off)
            .field("crash_on_refusal", &self.crash_on_refusal)
            .field("crash_on_empty_response", &self.crash_on_empty_response)
            .field("timeout_secs", &self.timeout_secs)
            .field("track_cost", &self.track_cost)
            .field("cost_callback", &self.cost_callback.is_some())
            .finish()
    }
}

impl NodeCompletionParameters {
    pub fn new() -> Self {
        Self::default()
    }

    /// The request's deadline, if any. 0 means none, not an instantly-firing
    /// 0s deadline.
    pub(crate) fn timeout(&self) -> Option<std::time::Duration> {
        self.timeout_secs
            .filter(|&s| s > 0)
            .map(std::time::Duration::from_secs)
    }

    /// Whether the request asks the provider for usage: when cost is tracked,
    /// and when a cost callback is waiting for it.
    pub(crate) fn tracks_usage(&self) -> bool {
        self.track_cost || self.cost_callback.is_some()
    }

    pub fn with_params(mut self, params: CompletionParameters) -> Self {
        self.params = params;
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

    /// With `parse_json`, reject a response whose repaired JSON is empty or
    /// null (retried when `retry > 0`).
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
    ///         println!("Request cost: ${}, total: ${}", info.cost, *cost);
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

#[cfg(test)]
mod wire_tests {
    use super::CompletionParameters;

    /// A JSON settings object deserializes directly: camelCase keys, every
    /// field optional (missing ones stay unset), unknown keys ignored.
    /// This is what lets a node's flat config map onto the parameters without
    /// a hand-written field-by-field translation.
    #[test]
    fn a_camel_case_settings_object_deserializes_directly() {
        let params: CompletionParameters = serde_json::from_value(serde_json::json!({
            "maxTokens": 1024,
            "temperature": 0.25,
            "topP": 0.9,
            "frequencyPenalty": 0.5,
            "seed": 7,
            "somebodyElsesKey": true,
        }))
        .unwrap();
        assert_eq!(params.max_tokens, Some(1024));
        assert_eq!(params.temperature, Some(0.25));
        assert_eq!(params.top_p, Some(0.9));
        assert_eq!(params.frequency_penalty, Some(0.5));
        assert_eq!(params.seed, Some(7));

        // Missing fields stay unset, same as ::new().
        let defaults: CompletionParameters = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(defaults.max_tokens, None);
        assert_eq!(defaults.temperature, None);
    }

    /// Reasoning is one word in JSON, in both directions, so a settings object
    /// carries it as `"reasoning": "high"`. Any word that is not `off` or `on`
    /// is an effort, kept as written.
    #[test]
    fn reasoning_is_one_word_in_json() {
        use super::Reasoning;
        for (reasoning, word) in [
            (Reasoning::Off, "off"),
            (Reasoning::On(None), "on"),
            (Reasoning::On(Some("high".into())), "high"),
            (
                Reasoning::On(Some("whatever-a-vendor-adds".into())),
                "whatever-a-vendor-adds",
            ),
        ] {
            assert_eq!(serde_json::to_value(&reasoning).unwrap(), word);
            assert_eq!(
                serde_json::from_value::<Reasoning>(serde_json::json!(word)).unwrap(),
                reasoning
            );
        }
        let params: CompletionParameters =
            serde_json::from_value(serde_json::json!({ "reasoning": "high" })).unwrap();
        assert_eq!(params.reasoning, Some(Reasoning::On(Some("high".into()))));
    }

    /// No field has a value nobody chose: a default `CompletionParameters`
    /// serializes with every key null, so nothing it carries can reach a wire.
    #[test]
    fn default_parameters_set_nothing() {
        let value = serde_json::to_value(CompletionParameters::default()).unwrap();
        for (key, field) in value.as_object().unwrap() {
            assert!(
                field.is_null(),
                "`{key}` defaults to {field}, expected unset"
            );
        }
    }

    #[test]
    fn parameters_round_trip_through_json() {
        let params = CompletionParameters::new().with_max_tokens(64).with_seed(3);
        let back: CompletionParameters =
            serde_json::from_str(&serde_json::to_string(&params).unwrap()).unwrap();
        assert_eq!(back.max_tokens, Some(64));
        assert_eq!(back.seed, Some(3));
    }
}
