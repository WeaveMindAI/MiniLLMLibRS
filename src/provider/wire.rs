//! Provider-specific wire knowledge: the [`Provider`] trait.
//!
//! Every provider's wire differs:
//! - the request key for the max-output-tokens limit (`max_completion_tokens` vs
//!   `max_tokens`),
//! - opting into usage reporting (OpenRouter `usage:{include:true}`, OpenAI
//!   streaming `stream_options:{include_usage:true}`, Anthropic always-on),
//! - the usage field names (`prompt_tokens` vs `input_tokens`, etc.),
//! - whether cost is returned natively in USD (OpenRouter) or not at all (OpenAI,
//!   Anthropic return token counts only, you price them via [`TokenPrice`]),
//! - out-of-band cost resolution (OpenRouter has a `/generation` endpoint; most
//!   providers have none),
//! - attribution headers (OpenRouter's `HTTP-Referer`/`X-Title`).
//!
//! All of it lives behind the [`Provider`] trait, owned by
//! [`GeneratorInfo`](crate::GeneratorInfo). The rest of the crate deals only in
//! the normalized [`Usage`] and [`CostOutcome`]; adding a provider that shares the
//! OpenAI request/response *envelope* is one trait impl. (A provider with a
//! different response envelope (Anthropic's `content[]` vs `choices[]`) also
//! needs the envelope parse behind the trait; that is a clean future extension,
//! not yet wired since no such provider ships.)

use super::auth::Auth;
use super::response::{CompletionResponse, StreamChunk, Usage};
use super::{CostInfo, CostResolution};
use crate::generator::CompletionParameters;
use crate::message::Message;
use std::future::Future;
use std::pin::Pin;

/// Per-token pricing, used to derive cost for providers that report token counts
/// but no dollar amount (OpenAI, Anthropic, ...). Rates are USD per **million**
/// tokens (the unit every provider's price sheet quotes), so a number off a
/// pricing page drops straight in.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TokenPrice {
    /// USD per million full-price input/prompt tokens.
    pub input_per_mtok: f64,
    /// USD per million output/completion tokens.
    pub output_per_mtok: f64,
    /// USD per million **cache-read** tokens (a discount, typically ~0.1× input).
    /// Falls back to `input_per_mtok` when `None`.
    pub cache_read_per_mtok: Option<f64>,
    /// USD per million **cache-write** tokens (a premium, typically ~1.25× input
    /// for a 5-minute cache, ~2× for 1-hour). Falls back to `input_per_mtok` when
    /// `None` (e.g. providers with no separate write charge, like OpenAI).
    pub cache_write_per_mtok: Option<f64>,
}

impl TokenPrice {
    /// New price with input/output rates (USD per million tokens). Cache rates
    /// default to the input rate until set via [`with_cache_rates`](Self::with_cache_rates).
    pub fn new(input_per_mtok: f64, output_per_mtok: f64) -> Self {
        Self {
            input_per_mtok,
            output_per_mtok,
            cache_read_per_mtok: None,
            cache_write_per_mtok: None,
        }
    }

    /// Set distinct cache-read and cache-write rates (USD per million tokens).
    /// These come straight off a provider's pricing sheet (e.g. OpenRouter's
    /// "Cached Read" / "Cached Write" columns; Anthropic's 0.1× read, 1.25× write).
    pub fn with_cache_rates(mut self, read_per_mtok: f64, write_per_mtok: f64) -> Self {
        self.cache_read_per_mtok = Some(read_per_mtok);
        self.cache_write_per_mtok = Some(write_per_mtok);
        self
    }

    /// Price a usage record as a clean weighted sum over the DISJOINT input
    /// buckets (no subtraction), so it is correct for every provider regardless of
    /// whether its wire reports cached tokens as a subset of input (OpenAI) or as
    /// separate additive counts (Anthropic). Cache rates fall back to the input
    /// rate when unset.
    pub fn cost_of(&self, usage: &Usage) -> f64 {
        let read_rate = self.cache_read_per_mtok.unwrap_or(self.input_per_mtok);
        let write_rate = self.cache_write_per_mtok.unwrap_or(self.input_per_mtok);
        (usage.uncached_input_tokens as f64 * self.input_per_mtok
            + usage.cache_read_tokens as f64 * read_rate
            + usage.cache_write_tokens as f64 * write_rate
            + usage.completion_tokens as f64 * self.output_per_mtok)
            / 1_000_000.0
    }
}

/// The outcome of pricing a completion: a normalized usage plus a USD cost whose
/// trustworthiness is flagged by [`CostResolution`]. Carries the usage so a
/// consumer can re-price or audit tokens even when the cost itself is `Unpriced`.
#[derive(Debug, Clone)]
pub struct CostOutcome {
    pub resolution: CostResolution,
    pub usd: f64,
    pub usage: Usage,
}

impl CostOutcome {
    /// A resolved cost (trusted USD amount).
    pub fn resolved(usd: f64, usage: Usage) -> Self {
        Self {
            resolution: CostResolution::Resolved,
            usd,
            usage,
        }
    }

    /// Tokens are real but no price is available (token-only provider with no
    /// `TokenPrice` configured). The USD is 0 but flagged `Unpriced` so it is
    /// never mistaken for a free request; set a [`TokenPrice`] to resolve it.
    pub fn unpriced(usage: Usage) -> Self {
        Self {
            resolution: CostResolution::Unpriced,
            usd: 0.0,
            usage,
        }
    }

    /// Cost could not be determined at all (no usage, failed out-of-band query).
    pub fn unknown() -> Self {
        Self {
            resolution: CostResolution::Unknown,
            usd: 0.0,
            usage: Usage::default(),
        }
    }

    /// Project into the public [`CostInfo`] reported to callbacks.
    pub fn into_cost_info(
        self,
        model: impl Into<String>,
        response_id: impl Into<String>,
    ) -> CostInfo {
        CostInfo {
            cost: self.usd,
            prompt_tokens: self.usage.prompt_tokens(),
            completion_tokens: self.usage.completion_tokens,
            total_tokens: self.usage.total_tokens(),
            cache_read_tokens: self.usage.cache_read_tokens,
            cache_write_tokens: self.usage.cache_write_tokens,
            reasoning_tokens: self.usage.reasoning_tokens,
            model: model.into(),
            response_id: response_id.into(),
            resolution: self.resolution,
        }
    }
}

/// Context for an out-of-band post-stream cost query (a cancelled/usage-less
/// stream). Carries what a provider needs to hit its own endpoint, if it has one.
pub struct PostStreamCtx<'a> {
    pub client: &'a reqwest::Client,
    pub generation_id: &'a str,
    pub auth: &'a Auth,
    pub price: Option<&'a TokenPrice>,
}

/// Boxed future returned by [`Provider::resolve_post_stream`] (keeps the
/// trait object-safe since async-fn-in-trait is not yet dyn-compatible).
pub type CostFuture<'a> = Pin<Box<dyn Future<Output = CostOutcome> + Send + 'a>>;

/// The calling application's identity, for providers that attribute usage to an
/// app (e.g. OpenRouter rankings). Set on the [`GeneratorInfo`](crate::GeneratorInfo);
/// the provider decides which headers express it.
#[derive(Debug, Clone)]
pub struct AppIdentity {
    pub url: String,
    pub title: String,
}

/// All provider-specific wire knowledge: the COMPLETE dialect for one provider.
///
/// The trait owns everything that differs on the wire so the rest of the crate
/// deals only in normalized types ([`Message`], [`CompletionParameters`],
/// [`CompletionResponse`], [`StreamChunk`], [`Usage`], [`CostOutcome`]). The five
/// "shape" methods ([`endpoint_url`](Self::endpoint_url),
/// [`auth_headers`](Self::auth_headers), [`build_request`](Self::build_request),
/// [`parse_response`](Self::parse_response), [`parse_chunk`](Self::parse_chunk))
/// default to the OpenAI `/chat/completions` + `choices[]` dialect via shared free
/// functions, so an OpenAI-wire provider overrides only its cost/usage specifics.
/// A provider with a different envelope (Anthropic's `/v1/messages` + `content[]`)
/// overrides the shape methods too.
pub trait Provider: Send + Sync + std::fmt::Debug {
    // ---- wire shape (default = OpenAI `/chat/completions` + `choices[]`) -------

    /// The full completions endpoint URL for `base_url`. Default appends
    /// `/chat/completions`; Anthropic appends `/v1/messages`.
    fn endpoint_url(&self, base_url: &str) -> String {
        format!("{}/chat/completions", base_url.trim_end_matches('/'))
    }

    /// HTTP auth headers for this provider's wire, given the generator's [`Auth`]
    /// strategy. Default OpenAI-wire: a key or token becomes
    /// `Authorization: Bearer <secret>`. Anthropic maps `ApiKey` to `x-api-key`.
    fn auth_headers(&self, auth: &Auth) -> crate::error::Result<Vec<(String, String)>> {
        super::providers::openai_auth_headers(auth)
    }

    /// Build the request body from normalized inputs. `include_usage` asks the
    /// provider to opt into usage reporting if its wire requires a flag. Default =
    /// the OpenAI body shape (typed params + `model`/`messages`/`stream` + the
    /// provider's token-limit key + usage opt-in + merged `extra`).
    fn build_request(
        &self,
        model: &str,
        messages: &[Message],
        params: &CompletionParameters,
        stream: bool,
        include_usage: bool,
    ) -> crate::error::Result<serde_json::Value> {
        super::providers::openai_build_request(model, messages, params, stream, include_usage, self)
    }

    /// (OpenAI-default helper) the request-body key for the max-output-tokens
    /// limit. Only consulted by the default [`build_request`](Self::build_request);
    /// a provider that overrides `build_request` ignores it.
    fn openai_token_limit_field(&self) -> &'static str {
        "max_completion_tokens"
    }

    /// (OpenAI-default helper) mutate the body to opt into usage reporting. Only
    /// consulted by the default [`build_request`](Self::build_request).
    fn openai_request_usage(&self, _body: &mut serde_json::Value, _stream: bool) {}

    /// (OpenAI-default helper) the wire value for the `tools` array. Only
    /// consulted by the default [`build_request`](Self::build_request); override
    /// it for an OpenAI-envelope server whose tool shape deviates.
    fn openai_tools_value(&self, tools: &[crate::tools::ToolDefinition]) -> serde_json::Value {
        serde_json::Value::Array(
            tools
                .iter()
                .map(crate::tools::ToolDefinition::to_openai_value)
                .collect(),
        )
    }

    /// (OpenAI-default helper) the wire value for `tool_choice`. Only consulted
    /// by the default [`build_request`](Self::build_request); override it for an
    /// OpenAI-envelope server whose tool-choice shape deviates.
    fn openai_tool_choice_value(&self, choice: &crate::tools::ToolChoice) -> serde_json::Value {
        choice.to_openai_value()
    }

    /// Parse a completed (non-streaming) raw response into a normalized
    /// [`CompletionResponse`] (content, usage, tool calls, finish reason). Default
    /// parses the OpenAI `choices[]` envelope.
    fn parse_response(&self, raw: serde_json::Value) -> crate::error::Result<CompletionResponse> {
        super::response::parse_openai_response(raw, self)
    }

    /// Parse one streaming SSE `data:` payload:
    /// - `None` for a frame that carries nothing trackable (e.g. `ping`),
    /// - `Some(Err(_))` when the frame is an in-band PROVIDER ERROR (a 200 stream
    ///   that then reports a failure, e.g. Anthropic's `{"type":"error"}` or an
    ///   OpenAI-wire top-level `{"error":{...}}`). This surfaces loudly through the
    ///   same channel-error path as a transport failure, so a failed generation is
    ///   never silently treated as an accepted (and billed) one,
    /// - `Some(Ok(chunk))` for a real content/usage/finish chunk.
    ///
    /// Default parses OpenAI-wire deltas.
    fn parse_chunk(&self, data: &str) -> Option<crate::error::Result<StreamChunk>> {
        super::response::parse_openai_chunk(data, self)
    }

    /// Extract a normalized [`Usage`] from a raw object (a non-streaming response
    /// body OR a streaming chunk; both OpenAI-wire put usage under `usage`).
    /// Consulted by the default `parse_response`/`parse_chunk`; a provider with a
    /// different envelope parses usage inside its own overrides instead.
    fn parse_usage(&self, raw: &serde_json::Value) -> Option<Usage> {
        super::providers::parse_openai_usage_field(raw)
    }

    // ---- cost + cross-cutting wire (no OpenAI envelope assumption) -------------

    /// Whether a *streaming* response from this provider will actually deliver a
    /// trailing usage chunk, given whether usage was `requested`. The streaming
    /// reader uses this to decide whether to wait for a usage chunk before
    /// finishing: waiting for one that never arrives wedges the stream until its
    /// idle timeout. Default: `requested`.
    fn emits_stream_usage(&self, requested: bool) -> bool {
        requested
    }

    /// HTTP headers attributing the request to the calling app, if the provider
    /// supports it (e.g. OpenRouter's `HTTP-Referer`/`X-Title`). Default: none.
    fn attribution_headers(&self, _app: Option<&AppIdentity>) -> Vec<(String, String)> {
        Vec::new()
    }

    /// Compute the USD cost for a usage record. The single place a provider
    /// aggregates its cost fields (OpenRouter sums fee + BYOK upstream) or derives
    /// cost from tokens × `price`. Token-only providers with no price return
    /// [`CostOutcome::unpriced`].
    fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome;

    /// Resolve cost out-of-band for a stream that ended *without* usage. Only
    /// reached when no usage was captured. Default: unresolvable → `Unknown`.
    fn resolve_post_stream<'a>(&'a self, _ctx: PostStreamCtx<'a>) -> CostFuture<'a> {
        Box::pin(async { CostOutcome::unknown() })
    }
}
