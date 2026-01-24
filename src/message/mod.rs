//! Message types for LLM conversations

mod audio;
mod content;
mod image;
mod media;
mod role;
mod video;

pub use audio::AudioData;
pub use content::{ContentPart, MessageContent, VideoUrl};
pub use image::ImageData;
pub use media::{Media, MediaData};
pub use role::Role;
pub use video::VideoData;

use serde::{Deserialize, Serialize};

/// A single message in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender
    pub role: Role,

    /// The content of the message
    pub content: MessageContent,

    /// Optional name for the message sender
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    /// Tool call ID (for tool responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,

    /// Tool calls made by the assistant
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<serde_json::Value>>,
}

impl Message {
    /// Create a new user message
    pub fn user(content: impl Into<MessageContent>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Create a new assistant message
    pub fn assistant(content: impl Into<MessageContent>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Create a new system message
    pub fn system(content: impl Into<MessageContent>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: None,
        }
    }

    /// Create a tool response message
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<MessageContent>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: None,
        }
    }

    /// Add a name to this message
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Check if this message has multimodal content
    pub fn has_multimodal_content(&self) -> bool {
        self.content.has_multimodal()
    }

    /// Get the text content of this message (if any)
    pub fn text(&self) -> Option<&str> {
        self.content.get_text()
    }
}

/// Convert a list of messages to the API payload format
pub fn messages_to_payload(messages: &[Message]) -> Vec<serde_json::Value> {
    messages
        .iter()
        .map(|msg| {
            let mut obj = serde_json::json!({
                "role": msg.role,
                "content": msg.content.to_api_format(),
            });

            if let Some(name) = &msg.name {
                obj["name"] = serde_json::json!(name);
            }
            if let Some(tool_call_id) = &msg.tool_call_id {
                obj["tool_call_id"] = serde_json::json!(tool_call_id);
            }
            if let Some(tool_calls) = &msg.tool_calls {
                obj["tool_calls"] = serde_json::json!(tool_calls);
            }

            obj
        })
        .collect()
}

/// Merge contiguous messages with the same role
pub fn merge_contiguous_messages(messages: Vec<Message>) -> Vec<Message> {
    if messages.is_empty() {
        return messages;
    }

    let mut result: Vec<Message> = Vec::new();

    for msg in messages {
        if let Some(last) = result.last_mut() {
            if last.role == msg.role && last.tool_call_id.is_none() && msg.tool_call_id.is_none() {
                // Merge content
                last.content = last.content.merge(&msg.content);
                continue;
            }
        }
        result.push(msg);
    }

    result
}
