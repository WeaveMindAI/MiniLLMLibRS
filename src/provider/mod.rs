//! LLM Provider implementations

mod auth;
mod client;
mod providers;
mod response;
mod streaming;
mod wire;

pub use auth::{resolve_claude_subscription_auth, Auth};
pub use client::{global_client, LLMClient};
pub use providers::{AnthropicProvider, GenericProvider, OpenAiProvider, OpenRouterProvider};
pub use response::{
    CompletionResponse, CostCallback, CostInfo, CostResolution, StreamChunk, Usage,
};
pub use streaming::StreamingCompletion;
pub use wire::{AppIdentity, CostOutcome, PostStreamCtx, Provider, TokenPrice};
