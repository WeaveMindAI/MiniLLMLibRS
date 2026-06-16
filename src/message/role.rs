//! Message roles

use serde::{Deserialize, Serialize};

/// The role of a message sender in a conversation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message (instructions/context)
    System,
    /// User message
    User,
    /// Assistant (LLM) response
    Assistant,
    /// Tool/function response
    Tool,
}

impl Role {
    /// Check if this is a user role
    pub fn is_user(&self) -> bool {
        matches!(self, Role::User)
    }

    /// Check if this is an assistant role
    pub fn is_assistant(&self) -> bool {
        matches!(self, Role::Assistant)
    }

    /// Check if this is a system role
    pub fn is_system(&self) -> bool {
        matches!(self, Role::System)
    }

    /// Get the role as a string slice
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
