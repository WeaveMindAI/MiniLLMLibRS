//! Normalized tool / function calling types.
//!
//! These are provider-agnostic *intent*, not a wire shape (same principle as
//! [`CompletionParameters`](crate::CompletionParameters)). Each provider's
//! [`Provider::build_request`](crate::Provider::build_request) translates them to
//! its own wire:
//! - OpenAI-wire (OpenAI, OpenRouter, compatibles): `tools: [{"type":"function",
//!   "function":{name, description, parameters, strict}}]`, `tool_choice:
//!   "auto"|"none"|"required"|{"type":"function","function":{"name":...}}`, and a
//!   top-level `parallel_tool_calls` bool.
//! - Anthropic `/v1/messages`: `tools: [{name, description, input_schema,
//!   strict}]`, `tool_choice: {"type":"auto"|"none"|"any"|"tool"}` with
//!   `disable_parallel_tool_use` folded in.
//!
//! The response side is normalized the same way: every provider parses its wire
//! into [`ToolCall`] (complete calls) and [`ToolCallDelta`] (streaming
//! fragments), which [`ToolCallAccumulator`] assembles.

mod args;

pub use args::{ArgumentStream, FieldHandle};

use crate::error::{MiniLLMError, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

// =============================================================================
// Request side: definitions + choice
// =============================================================================

/// A tool the model may call: a name, an optional description, and a JSON
/// Schema for its arguments. Provider-agnostic; each provider projects it
/// into its own wire shape.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// The tool's name (what the model calls it by).
    pub name: String,

    /// What the tool does and when to use it. Strongly recommended: the model
    /// decides when to call based on this text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// JSON Schema for the tool's arguments (an object schema with
    /// `properties`/`required`). OpenAI-wire sends it as `parameters`,
    /// Anthropic as `input_schema`.
    pub parameters: serde_json::Value,

    /// Ask the provider to guarantee schema conformance (OpenAI structured
    /// outputs `strict`, Anthropic strict tool use). Omitted from the wire when
    /// `None` (provider default).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl ToolDefinition {
    /// New tool definition from a name, description, and argument JSON Schema.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: Some(description.into()),
            parameters,
            strict: None,
        }
    }

    /// Ask the provider to enforce exact schema conformance on the arguments.
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = Some(strict);
        self
    }
}

/// How the model must treat the provided tools.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolChoice {
    /// The model decides whether to call a tool (the provider default when
    /// tools are present).
    Auto,
    /// The model must NOT call any tool.
    None,
    /// The model must call at least one tool (OpenAI `"required"`, Anthropic
    /// `{"type":"any"}`).
    Required,
    /// The model must call this specific tool (by name).
    Tool(String),
}

// =============================================================================
// Response side: complete calls + streaming deltas
// =============================================================================

/// A complete tool call made by the model, normalized across providers.
///
/// `arguments` is the RAW JSON TEXT of the call's arguments, exactly as the
/// model produced it (OpenAI-wire sends a JSON string; Anthropic's `input`
/// object is serialized to text on parse). Kept as text so a malformed
/// model output is preserved verbatim rather than silently repaired; use
/// [`arguments_json`](Self::arguments_json) to parse it, failing loudly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The provider's id for this call; echo it back in the tool-result message
    /// ([`Message::tool`](crate::Message::tool)).
    pub id: String,

    /// The name of the tool being called.
    pub name: String,

    /// The call's arguments as raw JSON text.
    pub arguments: String,
}

impl ToolCall {
    /// New tool call from id, name, and raw JSON argument text.
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }

    /// Parse the raw argument text as JSON, failing loudly when the model
    /// produced invalid JSON.
    pub fn arguments_json(&self) -> Result<serde_json::Value> {
        serde_json::from_str(&self.arguments).map_err(|e| {
            MiniLLMError::InvalidParameter(format!(
                "tool call '{}' ({}) carries invalid JSON arguments: {} (raw: {})",
                self.name, self.id, e, self.arguments
            ))
        })
    }

    /// Parse the raw argument text through this crate's JSON REPAIR first
    /// (the non-streaming counterpart of [`ArgumentStream::lenient`]): a
    /// sloppy model's trailing commas, unclosed braces, or single quotes
    /// still yield the arguments. Fails loudly only when nothing parseable
    /// can be recovered.
    pub fn arguments_json_repaired(&self) -> Result<serde_json::Value> {
        crate::utils::extract_json_value(&self.arguments).map_err(|e| {
            MiniLLMError::InvalidParameter(format!(
                "tool call '{}' ({}) carries unrepairable JSON arguments: {} (raw: {})",
                self.name, self.id, e, self.arguments
            ))
        })
    }
}

/// One streaming fragment of a tool call, normalized across providers.
///
/// OpenAI-wire streams `delta.tool_calls` entries (first delta carries
/// id/name, later ones argument fragments); Anthropic streams a
/// `content_block_start` (id/name) then `input_json_delta` fragments. Both
/// key fragments by a block/call `index`; indices may be sparse (Anthropic's
/// index space is shared with text blocks), so the accumulator maps by index
/// rather than assuming contiguity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ToolCallDelta {
    /// The wire's index for this call (de-multiplexes parallel calls).
    pub index: u64,

    /// The call id (usually only on the first fragment).
    pub id: Option<String>,

    /// The tool name (usually only on the first fragment).
    pub name: Option<String>,

    /// A fragment of the raw JSON argument text, to be concatenated in order.
    pub arguments_fragment: Option<String>,
}

/// Assembles [`ToolCallDelta`] fragments into complete [`ToolCall`]s.
///
/// Slots are keyed by the wire index in an ordered map, so sparse or
/// interleaved indices are handled and a hostile index can never size an
/// allocation.
#[derive(Debug, Default)]
pub struct ToolCallAccumulator {
    slots: BTreeMap<u64, PartialToolCall>,
}

#[derive(Debug, Default)]
struct PartialToolCall {
    id: Option<String>,
    name: Option<String>,
    arguments: String,
}

impl ToolCallAccumulator {
    /// Fold a batch of deltas into the accumulator.
    pub fn ingest(&mut self, deltas: &[ToolCallDelta]) {
        for delta in deltas {
            let slot = self.slots.entry(delta.index).or_default();
            if let Some(id) = &delta.id {
                slot.id = Some(id.clone());
            }
            if let Some(name) = &delta.name {
                slot.name = Some(name.clone());
            }
            if let Some(frag) = &delta.arguments_fragment {
                slot.arguments.push_str(frag);
            }
        }
    }

    /// Whether nothing has been accumulated.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Assemble the completed calls, in index order. A slot that never received
    /// an id or name (a stream cancelled mid-call, or a malformed wire) cannot
    /// be a usable call; it is dropped with a loud warning rather than
    /// fabricated.
    pub fn finish(&self) -> Vec<ToolCall> {
        self.slots
            .iter()
            .filter_map(|(index, slot)| match (&slot.id, &slot.name) {
                (Some(id), Some(name)) => {
                    Some(ToolCall::new(id.clone(), name.clone(), slot.arguments.clone()))
                }
                _ => {
                    tracing::warn!(
                        index,
                        has_id = slot.id.is_some(),
                        has_name = slot.name.is_some(),
                        "incomplete tool call fragment dropped (stream cancelled mid-call or malformed wire)"
                    );
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn call_arguments_json_parses_or_fails_loudly() {
        let ok = ToolCall::new("c1", "get_weather", r#"{"city":"Paris"}"#);
        assert_eq!(ok.arguments_json().unwrap()["city"], "Paris");
        let bad = ToolCall::new("c2", "get_weather", "{not json");
        assert!(bad.arguments_json().is_err());
    }

    #[test]
    fn accumulator_assembles_fragments_by_index() {
        let mut acc = ToolCallAccumulator::default();
        acc.ingest(&[ToolCallDelta {
            index: 0,
            id: Some("c0".into()),
            name: Some("search".into()),
            arguments_fragment: Some("{\"q\":".into()),
        }]);
        acc.ingest(&[ToolCallDelta {
            index: 0,
            arguments_fragment: Some("\"rust\"}".into()),
            ..Default::default()
        }]);
        let calls = acc.finish();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].id, "c0");
        assert_eq!(calls[0].name, "search");
        assert_eq!(calls[0].arguments, r#"{"q":"rust"}"#);
    }

    #[test]
    fn accumulator_handles_sparse_and_interleaved_indices() {
        // Anthropic shares the index space with text blocks: a tool call can
        // start at index 1 (or higher) with no slot 0, and parallel calls
        // interleave. Order of output follows index order.
        let mut acc = ToolCallAccumulator::default();
        acc.ingest(&[
            ToolCallDelta {
                index: 3,
                id: Some("c3".into()),
                name: Some("b".into()),
                ..Default::default()
            },
            ToolCallDelta {
                index: 1,
                id: Some("c1".into()),
                name: Some("a".into()),
                ..Default::default()
            },
        ]);
        acc.ingest(&[
            ToolCallDelta {
                index: 1,
                arguments_fragment: Some("{}".into()),
                ..Default::default()
            },
            ToolCallDelta {
                index: 3,
                arguments_fragment: Some("{}".into()),
                ..Default::default()
            },
        ]);
        let calls = acc.finish();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id, "c1", "index order preserved");
        assert_eq!(calls[1].id, "c3");
    }

    #[test]
    fn accumulator_drops_incomplete_slots_and_never_allocates_by_index() {
        let mut acc = ToolCallAccumulator::default();
        // A hostile/huge index is just a map key, never an allocation size.
        acc.ingest(&[ToolCallDelta {
            index: 4_000_000_000,
            arguments_fragment: Some("junk".into()),
            ..Default::default()
        }]);
        // No id/name ever arrived: the slot is unusable and dropped (loudly).
        assert!(acc.finish().is_empty());
    }

    #[test]
    fn tool_call_round_trips_serde() {
        // Message/node persistence serializes ToolCall; the round trip is the
        // wire-shape contract for saved trees.
        let call = ToolCall::new("c1", "get_weather", r#"{"city":"Paris"}"#);
        let json = serde_json::to_string(&call).unwrap();
        let back: ToolCall = serde_json::from_str(&json).unwrap();
        assert_eq!(back, call);
    }
}
