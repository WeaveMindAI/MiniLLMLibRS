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
pub mod utils;

// Re-export main types for convenience
pub use chat_node::{
    format_conversation, pretty_messages, ChatNode, ConversationBuilder, PrettyPrintConfig,
};
pub use error::{MiniLLMError, Result};
pub use generator::{
    CompletionParameters, GeneratorInfo, NodeCompletionParameters, ProviderSettings,
};
pub use json_repair::{loads, repair_json, JsonValue, RepairOptions};
pub use message::{AudioData, ImageData, Message, MessageContent, Role};
pub use provider::{CompletionResponse, LLMClient, StreamingCompletion};
pub use utils::{
    configure_logging, extract_json, extract_json_value, pretty_json, to_dict,
    validate_json_response, LogLevel,
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
