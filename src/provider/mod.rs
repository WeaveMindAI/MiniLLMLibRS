//! LLM Provider implementations

mod auth;
#[cfg(feature = "estimate")]
mod bpe;
pub(crate) mod client;
#[cfg(feature = "estimate")]
mod estimate;
pub(crate) mod memo;
pub(crate) mod model_id;
/// The shared OpenAI `/chat/completions` dialect: request building and
/// envelope parsing, reusable by custom OpenAI-compatible providers.
pub mod openai_wire;
mod providers;
pub(crate) mod response;
mod streaming;
pub(crate) mod wire;

pub use auth::Auth;
pub use client::{global_client, LLMClient};
#[cfg(feature = "estimate")]
pub use estimate::{estimate_cost_usd, estimate_prompt_tokens, PromptEstimate, SAFETY_MULTIPLIER};
pub use providers::{catalog_call, catalog_info};
pub use providers::{
    AnthropicProvider, GenericProvider, OpenAiProvider, OpenRouterProvider, OpenRouterRouting,
};
pub use response::{
    CompletionResponse, CostCallback, CostInfo, CostResolution, StreamChunk, Usage,
};
pub use streaming::StreamingCompletion;
pub use wire::{
    kept_cache_breakpoints, AppIdentity, CostOutcome, InfoFuture, ModelInfo, ModelLookup, Provider,
    SideCall, TokenPrice, UsageFuture, UsageQuery, Wanted,
};
