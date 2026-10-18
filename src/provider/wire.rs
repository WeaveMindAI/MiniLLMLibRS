//! Provider-specific wire knowledge: the [`Provider`] trait.
//!
//! Every provider's wire differs:
//! - the request key for the max-output-tokens limit (`max_completion_tokens` vs
//!   `max_tokens`),
//! - opting into usage reporting (OpenRouter `usage:{include:true}`, OpenAI
//!   streaming `stream_options:{include_usage:true}`, Anthropic always-on),
//! - the usage field names (`prompt_tokens` vs `input_tokens`, etc.),
//! - whether cost is returned natively in USD (OpenRouter) or not at all (OpenAI,
//!   Anthropic return token counts only, priced at the model's [`TokenPrice`]),
//! - usage reported out of band (OpenRouter has a `/generation` endpoint; most
//!   providers have none),
//! - where the model's price and limits are looked up,
//! - attribution headers (OpenRouter's `HTTP-Referer`/`X-Title`).
//!
//! All of it lives behind the [`Provider`] trait, owned by
//! [`GeneratorInfo`](crate::GeneratorInfo). The rest of the crate deals only in
//! the normalized [`Usage`] and [`CostOutcome`]. The trait itself is
//! wire-agnostic; its DEFAULT method bodies implement the most common concrete
//! wire (the OpenAI `/chat/completions` dialect, in `super::openai_wire`),
//! parameterized by the `openai_*` hooks, so a provider sharing that envelope
//! is a tiny impl. A provider on a different envelope overrides the shape
//! methods wholesale (`AnthropicProvider` is the shipped example: its own
//! request body, response parse, and event stream, all in its own file).

use super::auth::Auth;
use super::response::{CompletionResponse, StreamChunk, Usage};
use super::{CostInfo, CostResolution};
use crate::generator::CompletionParameters;
use crate::message::Message;
use std::future::Future;
use std::pin::Pin;

/// What to charge for audio when a model publishes no audio rate, as a multiple of
/// its text rate.
///
/// Every model that does publish one charges a premium: 2x to 3.3x on Gemini,
/// 12.8x on OpenAI's audio model. This bounds that range, so an unpublished rate
/// is over-charged rather than silently under-charged. Mistral's Voxtral charges a
/// thousand times text, but it publishes that rate, so the fallback never applies.
pub const AUDIO_RATE_FALLBACK_MULTIPLE: f64 = 13.0;

/// Per-token pricing, used to derive cost for providers that report token counts
/// but no dollar amount (OpenAI, Anthropic, ...). Rates are USD per **million**
/// tokens (the unit every provider's price sheet quotes), so a number off a
/// pricing page drops straight in.
/// Grows as providers invent new billing buckets (`#[non_exhaustive]`), so code
/// outside this crate constructs it through [`TokenPrice::new`] and the `with_*`
/// setters, never a struct literal.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
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
    /// USD per million **audio input** tokens, a steep premium: twice the input
    /// rate on most models that price it separately, thirteen times on OpenAI's
    /// audio model, a thousand times on Mistral's Voxtral. `None` when the model
    /// does not price audio apart from text, in which case the input rate applies.
    ///
    /// Not used by [`cost_of`](Self::cost_of): providers fold audio tokens into the
    /// prompt count on the wire, so a completion's real cost cannot separate them.
    /// It exists for estimating a call BEFORE it is sent, where the caller knows
    /// how much audio it is about to send.
    pub audio_per_mtok: Option<f64>,
    /// USD per million **image input** tokens. Every model publishing one today
    /// prices it equal to the input rate, so the fallback is exact rather than a
    /// guess; it is read anyway so a model that starts charging a premium is
    /// billed correctly instead of silently under-charged.
    ///
    /// Not used by [`cost_of`](Self::cost_of), for the same reason as audio.
    pub image_per_mtok: Option<f64>,
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
            audio_per_mtok: None,
            image_per_mtok: None,
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

    /// Set the audio-input and image-input rates (USD per million tokens). Pass
    /// `None` for a modality the model does not price apart from text.
    pub fn with_media_rates(
        mut self,
        audio_per_mtok: Option<f64>,
        image_per_mtok: Option<f64>,
    ) -> Self {
        self.audio_per_mtok = audio_per_mtok;
        self.image_per_mtok = image_per_mtok;
        self
    }

    /// What a million audio-input tokens cost.
    ///
    /// A model that publishes no audio rate still charges a premium for audio, so
    /// falling back to the plain input rate would under-charge. Among models that
    /// do publish one, the premium runs from 2x (Gemini) to 12.8x (OpenAI's audio
    /// model); `AUDIO_RATE_FALLBACK_MULTIPLE` bounds the mainstream range. The
    /// thousand-fold outlier (Mistral's Voxtral) publishes its rate, so the
    /// fallback never applies to it.
    pub fn audio_rate(&self) -> f64 {
        self.audio_per_mtok
            .unwrap_or(self.input_per_mtok * AUDIO_RATE_FALLBACK_MULTIPLE)
    }

    /// What a million image-input tokens cost. Falls back to the plain input rate,
    /// which is exactly what every model publishing an image rate charges.
    pub fn image_rate(&self) -> f64 {
        self.image_per_mtok.unwrap_or(self.input_per_mtok)
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

/// Where a side call about a request goes (a model lookup, a cost query): the
/// client, address and headers (auth, attribution, custom) the request itself
/// rides, so a generator behind a gateway is asked through that gateway exactly
/// like its requests are.
#[derive(Debug, Clone)]
pub struct SideCall {
    pub client: reqwest_middleware::ClientWithMiddleware,
    pub base_url: String,
    pub headers: reqwest::header::HeaderMap,
}

impl SideCall {
    /// The URL of `path` (with its query, if any) under the base URL, joined
    /// with exactly one `/`.
    pub fn url(&self, path: &str) -> String {
        format!(
            "{}/{}",
            self.base_url.trim_end_matches('/'),
            path.trim_start_matches('/')
        )
    }

    /// GET `path` (with its query, if any) under the base URL. Returns the
    /// status and body text; transport failures and timeouts map as they do
    /// for a request.
    pub async fn get(
        &self,
        path: &str,
        timeout: std::time::Duration,
    ) -> crate::error::Result<(reqwest::StatusCode, String)> {
        super::client::get_text(&self.client, &self.url(path), self.headers.clone(), timeout).await
    }

    /// GET `path` and `parse` the reply, shared across the whole process: one
    /// request per URL, header set and parsed type, however many callers ask.
    /// Callers waiting at the same moment share it, and it runs to its end
    /// even if they all give up.
    ///
    /// `parse` sees the status and body and returns the value, and is run once
    /// per request, so whatever it logs is logged once. What is kept, and for
    /// how long:
    /// - a value `parse` returns from a success or a 404 (the thing does not
    ///   exist): an hour;
    /// - a value from any other status, or an error `parse` returns: nothing
    ///   is kept beyond the callers already waiting, so the next caller asks
    ///   again;
    /// - no reply at all (a timeout, a refused connection): a minute, so a
    ///   source that is down costs one wait a minute rather than one per
    ///   caller.
    ///
    /// A failure is logged once, here, and reaches every waiting caller as
    /// [`MiniLLMError::SideRequest`](crate::MiniLLMError::SideRequest). The
    /// client is not part of the key: two clients asking the same URL with the
    /// same headers share the answer.
    pub async fn get_cached<T: Clone + Send + Sync + 'static>(
        &self,
        path: &str,
        timeout: std::time::Duration,
        parse: impl FnOnce(reqwest::StatusCode, &str) -> crate::error::Result<T> + Send + 'static,
    ) -> crate::error::Result<T> {
        type Any = std::sync::Arc<dyn std::any::Any + Send + Sync>;
        static REPLIES: std::sync::LazyLock<super::memo::Memo<Any>> =
            std::sync::LazyLock::new(super::memo::Memo::new);
        const KEPT: std::time::Duration = std::time::Duration::from_secs(3600);

        let url = self.url(path);
        let key = {
            use std::hash::{Hash, Hasher};
            let mut headers = std::collections::hash_map::DefaultHasher::new();
            for (name, value) in &self.headers {
                name.as_str().hash(&mut headers);
                value.as_bytes().hash(&mut headers);
            }
            format!(
                "{url}\n{:x}\n{:?}",
                headers.finish(),
                std::any::TypeId::of::<T>()
            )
        };
        let call = self.clone();
        let value = REPLIES
            .get_or_lookup(&key, move || {
                Box::pin(async move {
                    let reply =
                        super::client::get_text(&call.client, &url, call.headers.clone(), timeout)
                            .await;
                    let (outcome, ttl) = match reply {
                        Err(e) => (Err(e), super::memo::FAILURE_TTL),
                        Ok((status, body)) => {
                            let lasting =
                                status.is_success() || status == reqwest::StatusCode::NOT_FOUND;
                            let ttl = if lasting {
                                KEPT
                            } else {
                                std::time::Duration::ZERO
                            };
                            match parse(status, &body) {
                                Ok(value) => (Ok(std::sync::Arc::new(value) as Any), ttl),
                                Err(e) => (Err(e), std::time::Duration::ZERO),
                            }
                        }
                    };
                    if let Err(e) = &outcome {
                        tracing::warn!("{url}: {e}");
                    }
                    (outcome, ttl)
                })
            })
            .await
            .map_err(crate::error::MiniLLMError::SideRequest)?;
        Ok(value
            .downcast_ref::<T>()
            .expect("the key names the type, so the value is that type")
            .clone())
    }
}

/// Context for an out-of-band usage query (a stream that ended without its
/// usage, a cancelled one): where to ask, and which generation.
pub struct UsageQuery<'a> {
    pub call: SideCall,
    pub generation_id: &'a str,
}

/// Boxed future returned by [`Provider::usage_out_of_band`] (keeps the trait
/// object-safe since async-fn-in-trait is not yet dyn-compatible).
pub type UsageFuture<'a> = Pin<Box<dyn Future<Output = Option<Usage>> + Send + 'a>>;

/// What the library knows about one model: what it charges, and how much it
/// can produce and read. Cost estimation and a required output cap are figured
/// from it, the same way for every provider. Unknown is `None`.
///
/// The caller can give it ([`GeneratorInfo::with_model_info`](crate::GeneratorInfo::with_model_info));
/// whatever the caller leaves unset, the generator's provider looks up
/// ([`Provider::model_info`]).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelInfo {
    /// What the model charges per token.
    pub price: Option<TokenPrice>,
    /// The most tokens the model may produce in one response, reasoning
    /// included.
    pub max_output_tokens: Option<u32>,
    /// The model's whole context window, input plus output.
    pub context_length: Option<u32>,
}

impl ModelInfo {
    /// Every field `self` leaves unset, filled from `other`.
    pub fn or(self, other: ModelInfo) -> ModelInfo {
        ModelInfo {
            price: self.price.or(other.price),
            max_output_tokens: self.max_output_tokens.or(other.max_output_tokens),
            context_length: self.context_length.or(other.context_length),
        }
    }
}

/// Which facts about a model a caller needs looked up: the price, the limits
/// (largest output and context window), or both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Wanted {
    pub price: bool,
    pub limits: bool,
}

impl Wanted {
    /// Everything.
    pub const ALL: Wanted = Wanted {
        price: true,
        limits: true,
    };
}

/// What a provider gets to look a model up with.
pub struct ModelLookup {
    /// Where the generator's requests go: its client, address and headers.
    pub call: SideCall,
    /// The model's id on this wire.
    pub model: String,
    /// The model's id in OpenRouter's catalog, when the caller set one
    /// ([`GeneratorInfo::with_openrouter_name`](crate::GeneratorInfo::with_openrouter_name)).
    pub openrouter_name: Option<String>,
    /// The facts the caller needs and does not have: a source that would only
    /// fill the others need not be asked.
    pub wanted: Wanted,
}

/// Boxed future returned by [`Provider::model_info`]. A lookup never fails: a
/// source that cannot answer is logged and leaves its fields unknown.
pub type InfoFuture = Pin<Box<dyn Future<Output = ModelInfo> + Send>>;

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

    /// HTTP auth headers for this provider's wire, given the generator's
    /// [`Auth`]. Default OpenAI-wire: a key or token becomes
    /// `Authorization: Bearer <secret>`. Anthropic maps `ApiKey` to `x-api-key`.
    fn auth_headers(&self, auth: &Auth) -> crate::error::Result<Vec<(String, String)>> {
        super::openai_wire::openai_auth_headers(auth)
    }

    /// How this provider finds out about a model (see [`ModelInfo`]), or
    /// `None` when it has no way to. Fill what you can find and leave the rest
    /// unset; a source that cannot answer is logged and leaves its fields
    /// unset. Ask through [`SideCall::get_cached`], which shares each request
    /// across the process, so a lookup costs one round trip per model, not one
    /// per call. The default has no way.
    fn model_info(&self, _lookup: ModelLookup) -> Option<InfoFuture> {
        None
    }

    /// Whether this wire's API refuses a request without an output cap. When
    /// it does and the caller set none, the request carries the model's
    /// largest ([`ModelInfo::max_output_tokens`]), or 4096 when that is
    /// unknown. Default: no.
    fn requires_max_tokens(&self) -> bool {
        false
    }

    /// Build the request body from normalized inputs. `include_usage` asks the
    /// provider to opt into usage reporting if its wire requires a flag.
    /// Default = the OpenAI body shape (typed params + `model`/`messages`/
    /// `stream` + the provider's token-limit key + usage opt-in + merged
    /// `extra`).
    fn build_request(
        &self,
        model: &str,
        messages: &[Message],
        params: &CompletionParameters,
        stream: bool,
        include_usage: bool,
    ) -> crate::error::Result<serde_json::Value> {
        super::openai_wire::openai_build_request(
            model,
            messages,
            params,
            stream,
            include_usage,
            self,
        )
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

    /// (OpenAI-default helper) the request key and value a
    /// [`Reasoning`](crate::generator::Reasoning) becomes in this wire's
    /// dialect. Only consulted by the default
    /// [`build_request`](Self::build_request); a provider with its own envelope
    /// translates reasoning inside its own `build_request`.
    ///
    /// The default is OpenAI's dialect: `reasoning_effort`, `"none"` for off,
    /// `"low"` (the lowest effort every OpenAI reasoning model takes) for on
    /// with no effort, and the caller's word as written for on with one.
    fn openai_reasoning_field(
        &self,
        reasoning: &crate::generator::Reasoning,
    ) -> (&'static str, serde_json::Value) {
        super::openai_wire::openai_reasoning_field(reasoning)
    }

    /// How many cache breakpoints this provider's wire accepts per request.
    /// Consulted by the wires that emit `cache_control` markers (Anthropic's
    /// native wire and OpenRouter's passthrough, both capped at Anthropic's 4);
    /// when more messages are marked, the LAST ones win (the most-recent
    /// prefixes are the largest reusable spans). The default is unlimited for
    /// wires without a marker concept.
    fn max_cache_breakpoints(&self) -> usize {
        usize::MAX
    }

    /// (OpenAI-default helper) the wire value for the `messages` array. Only
    /// consulted by the default [`build_request`](Self::build_request); override
    /// it for an OpenAI-envelope wire that carries extra per-message fields
    /// (OpenRouter's Anthropic `cache_control` passthrough).
    fn openai_messages_value(&self, model: &str, messages: &[Message]) -> Vec<serde_json::Value> {
        let _ = model;
        super::openai_wire::messages_to_payload(messages, self.wire_keeps_estimation_metadata())
    }

    /// (OpenAI-default helper) the wire value for the `tools` array. Only
    /// consulted by the default [`build_request`](Self::build_request); override
    /// it for an OpenAI-envelope server whose tool shape deviates.
    fn openai_tools_value(&self, tools: &[crate::tools::ToolDefinition]) -> serde_json::Value {
        serde_json::Value::Array(
            tools
                .iter()
                .map(super::openai_wire::tool_definition_value)
                .collect(),
        )
    }

    /// (OpenAI-default helper) the wire value for `tool_choice`. Only consulted
    /// by the default [`build_request`](Self::build_request); override it for an
    /// OpenAI-envelope server whose tool-choice shape deviates.
    fn openai_tool_choice_value(&self, choice: &crate::tools::ToolChoice) -> serde_json::Value {
        super::openai_wire::tool_choice_value(choice)
    }

    /// Parse a completed (non-streaming) raw response into a normalized
    /// [`CompletionResponse`] (content, usage, tool calls, finish reason). Default
    /// parses the OpenAI `choices[]` envelope.
    fn parse_response(&self, raw: serde_json::Value) -> crate::error::Result<CompletionResponse> {
        super::openai_wire::parse_openai_response(raw, self)
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
        super::openai_wire::parse_openai_chunk(data, self)
    }

    /// Extract a normalized [`Usage`] from a raw object (a non-streaming response
    /// body OR a streaming chunk; both OpenAI-wire put usage under `usage`).
    /// Consulted by the default `parse_response`/`parse_chunk`; a provider with a
    /// different envelope parses usage inside its own overrides instead.
    fn parse_usage(&self, raw: &serde_json::Value) -> Option<Usage> {
        super::openai_wire::parse_openai_usage_field(raw)
    }

    /// Extract the media the model RETURNED from the completed `message`
    /// object, as normalized typed [`Media`](crate::message::Media). The
    /// GENERAL shape is [`CompletionResponse::media`]; HOW a wire carries
    /// returned media is this hook's per-provider concern. The default
    /// parses the OpenAI-wire `message.images` entries (OpenRouter's
    /// normalized image-output field); a provider whose wire returns
    /// media elsewhere (an `audio` object, content blocks) overrides
    /// this without re-implementing the whole response parse.
    fn parse_response_media(
        &self,
        message: &serde_json::Value,
    ) -> crate::error::Result<Vec<crate::message::Media>> {
        super::openai_wire::parse_openai_response_images(message)
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

    /// Whether this provider's wire tolerates the media parts' estimation
    /// metadata (`duration_secs`, `width`, `height`) riding the request
    /// payload. Default FALSE: a strict schema (OpenAI's) rejects unknown
    /// keys, so the payload sheds them. A wire that provably ignores unknown
    /// keys (OpenRouter normalizes requests before forwarding upstream)
    /// keeps them, so anything metering the request in flight (a client-side
    /// estimator, a billing gateway) can price the media exactly from the
    /// bytes.
    fn wire_keeps_estimation_metadata(&self) -> bool {
        false
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

    /// The usage of a generation whose response carried none (a stream that
    /// ended without its usage chunk, a cancelled one), from the provider's own
    /// record of it, or `None` when it has no such record. The cost is then
    /// worked out from it like any other usage ([`cost_of`](Self::cost_of)).
    /// Default: no record.
    fn usage_out_of_band<'a>(&'a self, _ctx: UsageQuery<'a>) -> UsageFuture<'a> {
        Box::pin(async { None })
    }
}

/// Indices of the messages that KEEP their cache breakpoint under the
/// provider's [`Provider::max_cache_breakpoints`] cap: of all marked
/// messages, the LAST `max` (the most-recent prefixes are the largest
/// reusable spans). Warns when marks are dropped. Provider-agnostic:
/// the cap and the marker's wire form are each provider's own, so a custom
/// `build_request` that writes cache markers calls this with its own cap.
pub fn kept_cache_breakpoints(
    messages: &[Message],
    max: usize,
) -> std::collections::HashSet<usize> {
    let marked: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.cache_breakpoint)
        .map(|(i, _)| i)
        .collect();
    if marked.len() > max {
        tracing::warn!(
            "this provider allows at most {} cache breakpoints per request; {} were marked, keeping the last {}",
            max,
            marked.len(),
            max
        );
    }
    marked.iter().rev().take(max).copied().collect()
}

/// Cost for a provider that returns no native USD: derive it from a configured
/// `TokenPrice`, otherwise report `Unpriced` (real tokens, unknown price, never a
/// fake $0). Shared by every token-only provider.
pub(crate) fn price_or_unpriced(usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
    match price {
        Some(p) => CostOutcome::resolved(p.cost_of(&usage), usage),
        None => CostOutcome::unpriced(usage),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fully-uncached usage (all input in the full-price bucket).
    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            uncached_input_tokens: prompt,
            completion_tokens: completion,
            ..Default::default()
        }
    }

    #[test]
    fn token_price_costs_prompt_and_completion_per_mtok() {
        let price = TokenPrice::new(3.0, 15.0); // $3/Mtok in, $15/Mtok out
        let u = usage(1_000_000, 1_000_000);
        assert!((price.cost_of(&u) - 18.0).abs() < 1e-9);
    }

    #[test]
    fn token_price_bills_cache_read_and_write_at_their_own_rates() {
        // read 0.3/Mtok, write 3.75/Mtok (1.25× the 3.0 input).
        let price = TokenPrice::new(3.0, 15.0).with_cache_rates(0.3, 3.75);
        // Disjoint: 200k uncached, 800k cache-read, 100k cache-write, 0 output.
        let u = Usage {
            uncached_input_tokens: 200_000,
            cache_read_tokens: 800_000,
            cache_write_tokens: 100_000,
            ..Default::default()
        };
        // 200k×3.0 ($0.6) + 800k×0.3 ($0.24) + 100k×3.75 ($0.375) = $1.215
        assert!(
            (price.cost_of(&u) - 1.215).abs() < 1e-9,
            "got {}",
            price.cost_of(&u)
        );
    }

    #[test]
    fn cache_rates_fall_back_to_input_rate_when_unset() {
        // No cache rates set → read and write both bill at the input rate.
        let price = TokenPrice::new(2.0, 0.0);
        let u = Usage {
            uncached_input_tokens: 0,
            cache_read_tokens: 1_000_000,
            cache_write_tokens: 1_000_000,
            ..Default::default()
        };
        // 1M×2.0 + 1M×2.0 = $4.0 (both at input rate)
        assert!((price.cost_of(&u) - 4.0).abs() < 1e-9);
    }
}
