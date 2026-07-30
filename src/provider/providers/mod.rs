//! Built-in [`Provider`](super::wire::Provider) implementations, one
//! file per provider: each file owns EVERYTHING specific to its
//! provider (wire quirks, cost model, parsing where the envelope
//! differs) plus that provider's tests. Shared dialect code lives in
//! `super::openai_wire`; nothing provider-specific lives outside these
//! files.

mod anthropic;
mod generic;
mod openai;
mod openrouter;

pub use anthropic::{resolve_claude_subscription_auth, AnthropicProvider};
pub use generic::GenericProvider;
pub use openai::OpenAiProvider;
pub use openrouter::OpenRouterProvider;
