//! ChatNode - Core conversation tree structure
//!
//! ChatNode represents a single node in a conversation tree. Each node contains
//! a message and can have multiple children (branches). This allows for:
//! - Linear conversations
//! - Branching conversations (exploring different paths)
//! - Tree-structured dialogues

use crate::error::{MiniLLMError, Result};
use crate::generator::{GeneratorInfo, NodeCompletionParameters};
use crate::message::{merge_contiguous_messages, Message, MessageContent, Role};
use crate::provider::{global_client, LLMClient, StreamingCompletion};
use std::sync::{Arc, RwLock, Weak};
use std::time::Duration;
use uuid::Uuid;

/// A node in the conversation tree
pub struct ChatNode {
    /// Unique identifier for this node
    pub id: String,

    /// The message at this node
    pub message: Message,

    /// Child nodes (branches from this point)
    children: RwLock<Vec<Arc<ChatNode>>>,

    /// Parent node (weak reference to avoid cycles)
    parent: RwLock<Option<Weak<ChatNode>>>,

    /// Metadata for this node
    pub metadata: RwLock<serde_json::Value>,
}

impl ChatNode {
    /// Create a new root node with a system message
    pub fn root(system_prompt: impl Into<String>) -> Arc<Self> {
        let prompt: String = system_prompt.into();
        Arc::new(Self {
            id: Uuid::new_v4().to_string(),
            message: Message::system(prompt),
            children: RwLock::new(Vec::new()),
            parent: RwLock::new(None),
            metadata: RwLock::new(serde_json::json!({})),
        })
    }

    /// Create a new node with a message
    pub fn new(message: Message) -> Arc<Self> {
        Arc::new(Self {
            id: Uuid::new_v4().to_string(),
            message,
            children: RwLock::new(Vec::new()),
            parent: RwLock::new(None),
            metadata: RwLock::new(serde_json::json!({})),
        })
    }

    /// Create a user message node
    pub fn user(content: impl Into<MessageContent>) -> Arc<Self> {
        Self::new(Message::user(content))
    }

    /// Create an assistant message node
    pub fn assistant(content: impl Into<MessageContent>) -> Arc<Self> {
        Self::new(Message::assistant(content))
    }

    /// Add a child node to this node
    pub fn add_child(self: &Arc<Self>, child: Arc<ChatNode>) -> Arc<ChatNode> {
        // Set parent reference
        {
            let mut parent_lock = child.parent.write().unwrap();
            *parent_lock = Some(Arc::downgrade(self));
        }

        // Add to children
        {
            let mut children_lock = self.children.write().unwrap();
            children_lock.push(child.clone());
        }

        child
    }

    /// Add a user message as a child
    pub fn add_user(self: &Arc<Self>, content: impl Into<MessageContent>) -> Arc<ChatNode> {
        self.add_child(Self::user(content))
    }

    /// Add an assistant message as a child
    pub fn add_assistant(self: &Arc<Self>, content: impl Into<MessageContent>) -> Arc<ChatNode> {
        self.add_child(Self::assistant(content))
    }

    /// Get the parent node
    pub fn parent(&self) -> Option<Arc<ChatNode>> {
        self.parent
            .read()
            .unwrap()
            .as_ref()
            .and_then(|w| w.upgrade())
    }

    /// Get all children
    pub fn children(&self) -> Vec<Arc<ChatNode>> {
        self.children.read().unwrap().clone()
    }

    /// Get the number of children
    pub fn child_count(&self) -> usize {
        self.children.read().unwrap().len()
    }

    /// Check if this is a root node
    pub fn is_root(&self) -> bool {
        self.parent.read().unwrap().is_none()
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.children.read().unwrap().is_empty()
    }

    /// Get the thread (path from root to this node)
    pub fn thread(&self) -> Vec<Message> {
        let mut messages = Vec::new();
        self.collect_thread(&mut messages);
        messages.reverse();
        messages
    }

    /// Helper to collect thread messages
    fn collect_thread(&self, messages: &mut Vec<Message>) {
        messages.push(self.message.clone());
        if let Some(parent) = self.parent() {
            parent.collect_thread(messages);
        }
    }

    /// Get the thread with contiguous messages merged
    pub fn merged_thread(&self) -> Vec<Message> {
        merge_contiguous_messages(self.thread())
    }

    /// Get the depth of this node in the tree
    pub fn depth(&self) -> usize {
        match self.parent() {
            Some(parent) => 1 + parent.depth(),
            None => 0,
        }
    }

    /// Find a node by ID in the subtree rooted at this node
    pub fn find_by_id(self: &Arc<Self>, id: &str) -> Option<Arc<ChatNode>> {
        if self.id == id {
            return Some(self.clone());
        }

        for child in self.children() {
            if let Some(found) = child.find_by_id(id) {
                return Some(found);
            }
        }

        None
    }

    /// Get the last child (most recent branch)
    pub fn last_child(&self) -> Option<Arc<ChatNode>> {
        self.children.read().unwrap().last().cloned()
    }

    /// Get the leaf node following the last child at each level
    pub fn get_leaf(self: &Arc<Self>) -> Arc<ChatNode> {
        match self.last_child() {
            Some(child) => child.get_leaf(),
            None => self.clone(),
        }
    }

    /// Set metadata for this node
    pub fn set_metadata(&self, key: &str, value: serde_json::Value) {
        let mut metadata = self.metadata.write().unwrap();
        metadata[key] = value;
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<serde_json::Value> {
        let metadata = self.metadata.read().unwrap();
        metadata.get(key).cloned()
    }

    // =========================================================================
    // Completion methods
    // =========================================================================

    /// Complete the conversation at this node (non-streaming)
    pub async fn complete(
        self: &Arc<Self>,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<Arc<ChatNode>> {
        let client = global_client();
        self.complete_with_client(client, generator, params).await
    }

    /// Complete with a specific client
    pub async fn complete_with_client(
        self: &Arc<Self>,
        client: &LLMClient,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<Arc<ChatNode>> {
        // Build messages
        let mut messages = self.merged_thread();

        // Add system prompt if specified in params
        if let Some(p) = params {
            if let Some(system) = &p.system_prompt {
                // Prepend system message if not already present
                if messages.first().map(|m| m.role) != Some(Role::System) {
                    messages.insert(0, Message::system(system.clone()));
                }
            }
        }

        // Handle force_prepend - add an assistant message that the model should continue from
        if let Some(p) = params {
            if let Some(prepend) = &p.force_prepend {
                messages.push(Message::assistant(prepend.clone()));
            }
        }

        // Get completion parameters
        let completion_params = params
            .and_then(|p| p.params.as_ref())
            .map(|p| generator.default_params.merge(p))
            .unwrap_or_else(|| generator.default_params.clone());

        // Get retry parameters
        let max_retries = params.map(|p| p.retry).unwrap_or(4);
        let exp_back_off = params.map(|p| p.exp_back_off).unwrap_or(false);
        let back_off_time = params.map(|p| p.back_off_time).unwrap_or(1.0);
        let max_back_off = params.map(|p| p.max_back_off).unwrap_or(15.0);
        let parse_json = params.map(|p| p.parse_json).unwrap_or(false);
        let crash_on_refusal = params.map(|p| p.crash_on_refusal).unwrap_or(false);
        let crash_on_empty = params.map(|p| p.crash_on_empty_response).unwrap_or(false);
        let force_prepend = params.and_then(|p| p.force_prepend.clone());

        let mut last_error: Option<MiniLLMError> = None;
        let mut current_back_off = back_off_time;

        for attempt in 0..=max_retries {
            if attempt > 0 {
                // Apply backoff before retry
                let sleep_time = if exp_back_off {
                    current_back_off.min(max_back_off)
                } else {
                    back_off_time
                };
                tokio::time::sleep(Duration::from_secs_f64(sleep_time)).await;
                if exp_back_off {
                    current_back_off *= 2.0;
                }
                tracing::debug!(attempt = attempt, "Retrying completion request");
            }

            // Make the request
            let response = match client
                .complete(generator, &messages, &completion_params)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };

            let mut content = response.content.clone();

            // Prepend force_prepend if it wasn't already included
            if let Some(ref prepend) = force_prepend {
                if !content.starts_with(prepend) {
                    content = format!("{}{}", prepend, content);
                }
            }

            // Check for empty response
            if crash_on_empty && content.trim().is_empty() {
                last_error = Some(MiniLLMError::Other("Empty response from model".to_string()));
                continue;
            }

            // Handle JSON parsing/repair if needed
            if parse_json {
                match self.process_json_response(&content, crash_on_refusal) {
                    Ok(parsed) => content = parsed,
                    Err(e) => {
                        last_error = Some(e);
                        continue;
                    }
                }
            }

            // Success! Create and add the assistant node
            let assistant_node = Self::new(Message::assistant(content));

            // Store response metadata
            assistant_node.set_metadata("response_id", serde_json::json!(response.id));
            assistant_node.set_metadata("model", serde_json::json!(response.model));
            if let Some(usage) = &response.usage {
                assistant_node.set_metadata("usage", serde_json::json!(usage));
            }
            if let Some(finish_reason) = &response.finish_reason {
                assistant_node.set_metadata("finish_reason", serde_json::json!(finish_reason));
            }

            return Ok(self.add_child(assistant_node));
        }

        // All retries exhausted
        Err(last_error.unwrap_or_else(|| MiniLLMError::Other("Max retries exceeded".to_string())))
    }

    /// Process JSON response with optional crash_on_refusal
    fn process_json_response(&self, content: &str, crash_on_refusal: bool) -> Result<String> {
        use crate::json_repair::{repair_json, RepairOptions};

        // Check if response contains any JSON-like content
        if crash_on_refusal && !content.contains('{') && !content.contains('[') {
            return Err(MiniLLMError::Other(format!(
                "No JSON found in response: {}",
                content
            )));
        }

        let repaired = repair_json(content, &RepairOptions::default())?;

        // Check for empty JSON
        if crash_on_refusal && (repaired == "\"\"" || repaired == "{}" || repaired.is_empty()) {
            return Err(MiniLLMError::Other(format!(
                "Empty JSON in response: {}",
                content
            )));
        }

        Ok(repaired)
    }

    /// Complete with streaming
    pub async fn complete_streaming(
        self: &Arc<Self>,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<StreamingCompletion> {
        let client = global_client();
        self.complete_streaming_with_client(client, generator, params)
            .await
    }

    /// Complete with streaming using a specific client
    pub async fn complete_streaming_with_client(
        self: &Arc<Self>,
        client: &LLMClient,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<StreamingCompletion> {
        // Build messages
        let mut messages = self.merged_thread();

        // Add system prompt if specified
        if let Some(p) = params {
            if let Some(system) = &p.system_prompt {
                if messages.first().map(|m| m.role) != Some(Role::System) {
                    messages.insert(0, Message::system(system.clone()));
                }
            }
        }

        // Get completion parameters
        let completion_params = params
            .and_then(|p| p.params.as_ref())
            .map(|p| generator.default_params.merge(p))
            .unwrap_or_else(|| generator.default_params.clone());

        // Start streaming
        client
            .complete_streaming(generator, &messages, &completion_params)
            .await
    }

    /// Complete streaming and collect into a new node
    pub async fn complete_streaming_collect(
        self: &Arc<Self>,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<Arc<ChatNode>> {
        let stream = self.complete_streaming(generator, params).await?;
        let response = stream.collect().await?;

        let parse_json = params.map(|p| p.parse_json).unwrap_or(false);
        let crash_on_refusal = params.map(|p| p.crash_on_refusal).unwrap_or(false);
        let force_prepend = params.and_then(|p| p.force_prepend.clone());

        let mut content = response.content;

        // Prepend force_prepend if it wasn't already included
        if let Some(ref prepend) = force_prepend {
            if !content.starts_with(prepend) {
                content = format!("{}{}", prepend, content);
            }
        }

        // Handle JSON repair if needed
        if parse_json {
            content = self.process_json_response(&content, crash_on_refusal)?;
        }

        // Create and add the assistant node
        let assistant_node = Self::new(Message::assistant(content));

        // Store metadata
        assistant_node.set_metadata("response_id", serde_json::json!(response.id));
        assistant_node.set_metadata("model", serde_json::json!(response.model));
        if let Some(usage) = &response.usage {
            assistant_node.set_metadata("usage", serde_json::json!(usage));
        }

        Ok(self.add_child(assistant_node))
    }

    // =========================================================================
    // Convenience methods for common patterns
    // =========================================================================

    /// Send a user message and get a completion
    pub async fn chat(
        self: &Arc<Self>,
        user_message: impl Into<MessageContent>,
        generator: &GeneratorInfo,
    ) -> Result<Arc<ChatNode>> {
        let user_node = self.add_user(user_message);
        user_node.complete(generator, None).await
    }

    /// Send a user message and get a streaming completion
    pub async fn chat_streaming(
        self: &Arc<Self>,
        user_message: impl Into<MessageContent>,
        generator: &GeneratorInfo,
    ) -> Result<(Arc<ChatNode>, StreamingCompletion)> {
        let user_node = self.add_user(user_message);
        let stream = user_node.complete_streaming(generator, None).await?;
        Ok((user_node, stream))
    }

    /// Get the text content of this node's message
    pub fn text(&self) -> Option<&str> {
        self.message.text()
    }

    /// Get the role of this node's message
    pub fn role(&self) -> Role {
        self.message.role
    }
}

impl std::fmt::Debug for ChatNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatNode")
            .field("id", &self.id)
            .field("role", &self.message.role)
            .field("children_count", &self.child_count())
            .finish()
    }
}

// =========================================================================
// Pretty printing for conversations
// =========================================================================

/// Configuration for pretty printing messages
#[derive(Debug, Clone)]
pub struct PrettyPrintConfig {
    /// Prefix for system messages
    pub system_prefix: String,
    /// Prefix for user messages
    pub user_prefix: String,
    /// Prefix for assistant messages
    pub assistant_prefix: String,
    /// Separator between messages
    pub separator: String,
}

impl Default for PrettyPrintConfig {
    fn default() -> Self {
        Self {
            system_prefix: "SYSTEM: ".to_string(),
            user_prefix: "\n\nUSER: ".to_string(),
            assistant_prefix: "\n\nASSISTANT: ".to_string(),
            separator: "".to_string(),
        }
    }
}

impl PrettyPrintConfig {
    /// Create a new config with custom prefixes
    pub fn new(system: &str, user: &str, assistant: &str) -> Self {
        Self {
            system_prefix: system.to_string(),
            user_prefix: user.to_string(),
            assistant_prefix: assistant.to_string(),
            separator: "".to_string(),
        }
    }

    /// Set the separator between messages
    pub fn with_separator(mut self, sep: &str) -> Self {
        self.separator = sep.to_string();
        self
    }
}

/// Pretty print a conversation thread
///
/// # Example
/// ```ignore
/// let root = ChatNode::root("You are helpful");
/// let user = root.add_user("Hello");
/// let assistant = user.add_assistant("Hi there!");
///
/// let pretty = pretty_messages(&assistant, None);
/// // Output:
/// // SYSTEM: You are helpful
/// //
/// // USER: Hello
/// //
/// // ASSISTANT: Hi there!
/// ```
pub fn pretty_messages(node: &Arc<ChatNode>, config: Option<&PrettyPrintConfig>) -> String {
    let default_config = PrettyPrintConfig::default();
    let config = config.unwrap_or(&default_config);

    let messages = node.thread();
    let mut result = String::new();

    for (i, msg) in messages.iter().enumerate() {
        if i > 0 && !config.separator.is_empty() {
            result.push_str(&config.separator);
        }

        let prefix = match msg.role {
            Role::System => &config.system_prefix,
            Role::User => &config.user_prefix,
            Role::Assistant => &config.assistant_prefix,
            Role::Tool => "\n\nTOOL: ",
        };

        result.push_str(prefix);
        if let Some(text) = msg.content.get_text() {
            result.push_str(text);
        } else {
            result.push_str("[multimodal content]");
        }
    }

    result
}

/// Pretty print messages as a formatted string (convenience function)
pub fn format_conversation(node: &Arc<ChatNode>) -> String {
    pretty_messages(node, None)
}

// =========================================================================
// Builder pattern for creating conversations
// =========================================================================

/// Builder for creating conversation trees
pub struct ConversationBuilder {
    root: Arc<ChatNode>,
    current: Arc<ChatNode>,
}

impl ConversationBuilder {
    /// Create a new conversation with a system prompt
    pub fn new(system_prompt: impl Into<String>) -> Self {
        let root = ChatNode::root(system_prompt);
        Self {
            current: root.clone(),
            root,
        }
    }

    /// Add a user message
    pub fn user(mut self, content: impl Into<MessageContent>) -> Self {
        self.current = self.current.add_user(content);
        self
    }

    /// Add an assistant message
    pub fn assistant(mut self, content: impl Into<MessageContent>) -> Self {
        self.current = self.current.add_assistant(content);
        self
    }

    /// Get the root node
    pub fn root(&self) -> Arc<ChatNode> {
        self.root.clone()
    }

    /// Get the current (last) node
    pub fn current(&self) -> Arc<ChatNode> {
        self.current.clone()
    }

    /// Build and return the current node
    pub fn build(self) -> Arc<ChatNode> {
        self.current
    }
}
