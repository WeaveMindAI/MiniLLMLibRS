//! LLM Provider implementations

mod auth;
#[cfg(feature = "estimate")]
mod bpe;
pub(crate) mod client;
#[cfg(feature = "estimate")]
mod estimate;
/// The shared OpenAI `/chat/completions` dialect: request building and
/// envelope parsing, reusable by custom OpenAI-compatible providers.
pub mod openai_wire;
mod providers;
mod response;
mod streaming;
pub(crate) mod wire;

pub use auth::Auth;
pub use client::{global_client, LLMClient};
#[cfg(feature = "estimate")]
pub use estimate::{estimate_cost_usd, estimate_prompt_tokens, PromptEstimate, SAFETY_MULTIPLIER};
pub use providers::resolve_claude_subscription_auth;
pub use providers::{AnthropicProvider, GenericProvider, OpenAiProvider, OpenRouterProvider};
pub use response::{
    CompletionResponse, CostCallback, CostInfo, CostResolution, StreamChunk, Usage,
};
pub use streaming::StreamingCompletion;
pub use wire::{AppIdentity, CostOutcome, PostStreamCtx, Provider, TokenPrice};
