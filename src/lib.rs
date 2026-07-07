//! MiniLLMLib - A minimalist Rust library for LLM interactions
//!
//! This library provides a clean, async-first interface for interacting with
//! Large Language Models via HTTP APIs (OpenRouter, OpenAI, etc.).
//!
//! # Features
//!
//! - **Conversation Trees**: `ChatNode` provides a tree-based conversation structure
//!   supporting branching dialogues and conversation history
//! - **Streaming Support**: First-class support for streaming completions via SSE
//! - **Multimodal**: Support for images and audio in messages
//! - **Multiple wires**: One [`Provider`] trait owns the full dialect (endpoint,
//!   auth, request body, response/stream envelope). Ships OpenAI/OpenRouter,
//!   native Anthropic (`/v1/messages`, `content[]`), and a generic compatible
//!   provider; a custom enterprise API is a small `impl Provider`.
//! - **Subscription auth**: a Claude Pro/Max OAuth token works via
//!   [`Auth::BearerToken`] (see [`GeneratorInfo::claude_subscription`]); cost is a
//!   token-count ESTIMATE through [`TokenPrice`].
//! - **Cost Tracking**: Per-provider usage & cost accounting behind the [`Provider`]
//!   trait; enforced tracking via [`CompletionContext`], with honest
//!   [`CostResolution`] (`Resolved`/`Unpriced`/`Unknown`, never a fake $0)
//! - **JSON Repair**: Robust handling of malformed JSON from LLM outputs
//! - **Async/Parallel**: Built on Tokio for high-performance async operations
//!
//! # Quick Start
//!
//! ```no_run
//! use minillmlib::{ChatNode, GeneratorInfo};
//!
//! #[tokio::main]
//! async fn main() -> minillmlib::error::Result<()> {
//!     // Create a generator for OpenRouter
//!     let generator = GeneratorInfo::openrouter("anthropic/claude-3.5-sonnet");
//!
//!     // Start a conversation
//!     let root = ChatNode::root("You are a helpful assistant.");
//!     let response = root.chat("Hello!", &generator).await?;
//!
//!     println!("Assistant: {}", response.text().unwrap_or_default());
//!     Ok(())
//! }
//! ```

// Core modules
pub mod chat_node;
pub mod error;
pub mod generator;
pub mod json_repair;
pub mod message;
pub mod provider;
pub mod tools;
pub mod tracking;
pub mod utils;

// Re-export main types for convenience
pub use chat_node::{
    format_conversation, pretty_messages, ChatNode, ConversationBuilder, PrettyPrintConfig,
    ThreadData,
};
pub use error::{MiniLLMError, Result};
pub use generator::{
    CompletionParameters, GeneratorInfo, NodeCompletionParameters, ProviderSettings,
    ReasoningConfig,
};
pub use json_repair::{loads, repair_json, JsonValue, RepairOptions};
pub use message::{
    AudioData, AudioInput, ContentPart, ImageData, ImageUrl, Media, MediaData, Message,
    MessageContent, Role, VideoData, VideoUrl,
};
pub use provider::{
    resolve_claude_subscription_auth, AnthropicProvider, AppIdentity, Auth, CompletionResponse,
    CostCallback, CostInfo, CostOutcome, CostResolution, GenericProvider, LLMClient,
    OpenAiProvider, OpenRouterProvider, PostStreamCtx, Provider, StreamChunk, StreamingCompletion,
    TokenPrice, Usage,
};
pub use tools::{ToolCall, ToolCallAccumulator, ToolCallDelta, ToolChoice, ToolDefinition};
pub use tracking::{AsyncCostCallback, CompletionContext, CompletionMeta, TrackedStream};
pub use utils::{
    configure_logging, extract_json, extract_json_value, pretty_json, to_dict, LogLevel,
};

/// Initialize the library with default settings
///
/// This loads environment variables from .env files and configures logging.
pub fn init() {
    // Load .env file if present
    let _ = dotenvy::dotenv();

    // Configure default logging
    utils::configure_logging(utils::LogLevel::Info);
}

/// Initialize with a specific log level
pub fn init_with_logging(level: LogLevel) {
    let _ = dotenvy::dotenv();
    utils::configure_logging(level);
}
