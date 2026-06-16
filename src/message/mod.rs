//! Message types for LLM conversations

mod audio;
mod content;
mod image;
mod media;
mod role;
mod video;

pub use audio::AudioData;
pub use content::{AudioInput, ContentPart, ImageUrl, MessageContent, VideoUrl};
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

    /// Normalized cache **breakpoint**: when true, the prefix UP TO AND INCLUDING
    /// this message is a candidate for prompt caching. This is provider-agnostic
    /// intent: the provider decides what to do with it
    /// ([`Provider::build_request`](crate::Provider::build_request)): Anthropic
    /// emits a `cache_control` marker (honoring its 4-breakpoint / min-size
    /// limits); OpenAI/OpenRouter ignore it (they auto-cache). The user sets it via
    /// [`ChatNode::cache_breakpoint`](crate::ChatNode::cache_breakpoint).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub cache_breakpoint: bool,
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
            cache_breakpoint: false,
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
            cache_breakpoint: false,
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
            cache_breakpoint: false,
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
            cache_breakpoint: false,
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

/// Merge contiguous messages with the same role.
///
/// Only "plain" same-role messages are merged: a message carrying `tool_calls`,
/// a `tool_call_id`, or a distinct `name` is structurally significant and must
/// be kept intact, merging it would silently drop those fields. Such messages
/// are passed through untouched.
pub fn merge_contiguous_messages(messages: Vec<Message>) -> Vec<Message> {
    let mut result: Vec<Message> = Vec::new();

    for msg in messages {
        let mergeable_with_last = result.last().is_some_and(|last| {
            last.role == msg.role && is_plain(last) && is_plain(&msg) && last.name == msg.name
        });

        if mergeable_with_last {
            let last = result.last_mut().expect("checked non-empty above");
            last.content = last.content.merge(&msg.content);
            // A cache breakpoint on either merged message survives on the result,
            // so marking any node in a run still caches the prefix through it.
            last.cache_breakpoint |= msg.cache_breakpoint;
        } else {
            result.push(msg);
        }
    }

    result
}

/// A message is "plain" (safe to merge by content alone) when it carries no
/// tool-call structure that would be lost by merging.
fn is_plain(msg: &Message) -> bool {
    msg.tool_call_id.is_none() && msg.tool_calls.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merges_plain_contiguous_same_role() {
        let merged =
            merge_contiguous_messages(vec![Message::user("Hello"), Message::user("World")]);
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].text(), Some("Hello\nWorld"));
    }

    #[test]
    fn does_not_merge_across_roles() {
        let merged = merge_contiguous_messages(vec![Message::system("S"), Message::user("U")]);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn does_not_merge_away_tool_calls() {
        // An assistant message carrying tool_calls must not be merged into a
        // following assistant message, which would silently drop the tool_calls.
        let mut with_tools = Message::assistant("calling");
        with_tools.tool_calls = Some(vec![serde_json::json!({"id": "c1"})]);

        let merged = merge_contiguous_messages(vec![with_tools, Message::assistant("after")]);
        assert_eq!(merged.len(), 2, "tool-call message must stay separate");
        assert!(merged[0].tool_calls.is_some());
    }

    #[test]
    fn does_not_merge_different_names() {
        let merged = merge_contiguous_messages(vec![
            Message::user("a").with_name("alice"),
            Message::user("b").with_name("bob"),
        ]);
        assert_eq!(merged.len(), 2);
    }
}
