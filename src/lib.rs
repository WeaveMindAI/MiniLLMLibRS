//! MiniLLMLib - A minimalist Rust library for LLM interactions
//!
//! This library provides utilities for working with LLM outputs, including
//! JSON repair capabilities for handling malformed JSON from language models.

pub mod json_repair;

// Re-export the main API for convenience
pub use json_repair::{repair_json, loads, JsonValue, RepairOptions};
