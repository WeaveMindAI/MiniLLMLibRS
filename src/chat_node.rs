//! ChatNode - Core conversation tree structure
//!
//! ChatNode represents a single node in a conversation tree. Each node contains
//! a message and can have multiple children (branches). This allows for:
//! - Linear conversations
//! - Branching conversations (exploring different paths)
//! - Tree-structured dialogues

use crate::error::{MiniLLMError, Result};
use crate::generator::{GeneratorInfo, NodeCompletionParameters};
use crate::message::{merge_contiguous_messages, ContentPart, Message, MessageContent, Role};
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

    /// Strong reference to the root node, keeping it alive as long as any
    /// descendant exists. Root nodes have this set to None (they ARE the root).
    root_ref: RwLock<Option<Arc<ChatNode>>>,

    /// Metadata for this node
    pub metadata: RwLock<serde_json::Value>,

    /// Format kwargs for template substitution (e.g., {name} -> "Alice")
    /// These are stored on each node and propagated to the root
    format_kwargs: RwLock<std::collections::HashMap<String, String>>,
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
            root_ref: RwLock::new(None),
            metadata: RwLock::new(serde_json::json!({})),
            format_kwargs: RwLock::new(std::collections::HashMap::new()),
        })
    }

    /// Create a new node with a message
    pub fn new(message: Message) -> Arc<Self> {
        Arc::new(Self {
            id: Uuid::new_v4().to_string(),
            message,
            children: RwLock::new(Vec::new()),
            parent: RwLock::new(None),
            root_ref: RwLock::new(None),
            metadata: RwLock::new(serde_json::json!({})),
            format_kwargs: RwLock::new(std::collections::HashMap::new()),
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

        // Propagate root_ref to the child and its entire subtree
        {
            let root = self.get_root();
            child.set_root_ref_recursive(&root);
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

    /// Get the root node of the tree
    pub fn get_root(self: &Arc<Self>) -> Arc<ChatNode> {
        self.root_ref
            .read()
            .unwrap()
            .clone()
            .unwrap_or_else(|| self.clone())
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

    // =========================================================================
    // Root reference management
    // =========================================================================

    /// Recursively set root_ref for this node and all descendants
    fn set_root_ref_recursive(self: &Arc<Self>, root: &Arc<ChatNode>) {
        {
            let mut root_lock = self.root_ref.write().unwrap();
            *root_lock = Some(root.clone());
        }
        for child in self.children() {
            child.set_root_ref_recursive(root);
        }
    }

    // =========================================================================
    // Tree manipulation
    // =========================================================================

    /// Detach this node from its parent
    ///
    /// Removes this node from its parent's children list and clears the parent reference.
    /// Also clears root_ref for this node and all descendants.
    /// Returns self for chaining.
    pub fn detach(self: &Arc<Self>) -> Arc<ChatNode> {
        // Remove from parent's children
        if let Some(parent) = self.parent() {
            let mut children = parent.children.write().unwrap();
            children.retain(|c| c.id != self.id);
        }

        // Clear parent reference
        {
            let mut parent_lock = self.parent.write().unwrap();
            *parent_lock = None;
        }

        // This node becomes the new root of its subtree
        {
            let mut root_lock = self.root_ref.write().unwrap();
            *root_lock = None;
        }
        // Update descendants to point to self as the new root
        for child in self.children() {
            child.set_root_ref_recursive(self);
        }

        self.clone()
    }

    /// Merge another tree into this node
    ///
    /// Attaches the root of the other tree as a child of this node.
    /// Returns the leaf of the merged tree.
    pub fn merge(self: &Arc<Self>, other: &Arc<ChatNode>) -> Arc<ChatNode> {
        let other_root = other.get_root();
        let merged_root = self.add_child(other_root);
        // add_child only sets root_ref on the direct child; propagate to all descendants
        let tree_root = self.get_root();
        for child in merged_root.children() {
            child.set_root_ref_recursive(&tree_root);
        }
        other.get_leaf()
    }

    // =========================================================================
    // Tree iteration
    // =========================================================================

    /// Iterate over all nodes in the subtree rooted at this node (depth-first, pre-order)
    pub fn iter_depth_first(self: &Arc<Self>) -> Vec<Arc<ChatNode>> {
        let mut result = vec![self.clone()];
        for child in self.children() {
            result.extend(child.iter_depth_first());
        }
        result
    }

    /// Iterate over all nodes in the subtree rooted at this node (breadth-first)
    pub fn iter_breadth_first(self: &Arc<Self>) -> Vec<Arc<ChatNode>> {
        let mut result = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(self.clone());

        while let Some(node) = queue.pop_front() {
            result.push(node.clone());
            for child in node.children() {
                queue.push_back(child);
            }
        }

        result
    }

    /// Get all leaf nodes in the subtree rooted at this node
    pub fn iter_leaves(self: &Arc<Self>) -> Vec<Arc<ChatNode>> {
        if self.is_leaf() {
            return vec![self.clone()];
        }

        let mut result = Vec::new();
        for child in self.children() {
            result.extend(child.iter_leaves());
        }
        result
    }

    /// Count total nodes in the subtree rooted at this node
    pub fn node_count(self: &Arc<Self>) -> usize {
        1 + self
            .children()
            .iter()
            .map(|c| c.node_count())
            .sum::<usize>()
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
    // Format kwargs (template substitution)
    // =========================================================================

    /// Set a format kwarg for this node
    ///
    /// Format kwargs are used for template substitution in message content.
    /// For example, if the message content is "Hello {name}", calling
    /// `set_format_kwarg("name", "Alice")` will substitute it.
    pub fn set_format_kwarg(&self, key: &str, value: &str) {
        let mut kwargs = self.format_kwargs.write().unwrap();
        kwargs.insert(key.to_string(), value.to_string());
    }

    /// Set multiple format kwargs at once
    pub fn set_format_kwargs(&self, kwargs: &std::collections::HashMap<String, String>) {
        let mut current = self.format_kwargs.write().unwrap();
        for (k, v) in kwargs {
            current.insert(k.clone(), v.clone());
        }
    }

    /// Get a format kwarg value
    pub fn get_format_kwarg(&self, key: &str) -> Option<String> {
        let kwargs = self.format_kwargs.read().unwrap();
        kwargs.get(key).cloned()
    }

    /// Get all format kwargs for this node
    pub fn get_format_kwargs(&self) -> std::collections::HashMap<String, String> {
        self.format_kwargs.read().unwrap().clone()
    }

    /// Update format kwargs and propagate to parent nodes
    ///
    /// This mimics Python's behavior where format_kwargs are propagated up to the root.
    pub fn update_format_kwargs(
        self: &Arc<Self>,
        kwargs: &std::collections::HashMap<String, String>,
        propagate: bool,
    ) {
        // Update this node's kwargs
        {
            let mut current = self.format_kwargs.write().unwrap();
            for (k, v) in kwargs {
                current.insert(k.clone(), v.clone());
            }
        }

        // Propagate to parent if requested
        if propagate {
            if let Some(parent) = self.parent() {
                parent.update_format_kwargs(kwargs, true);
            }
        }
    }

    /// Get the formatted text content of this node's message
    ///
    /// Applies format_kwargs substitution to the message content.
    pub fn formatted_text(&self) -> Option<String> {
        let text = self.message.content.get_text()?;
        Some(self.format_string(text))
    }

    /// Format a string using this node's format_kwargs
    pub fn format_string(&self, template: &str) -> String {
        let kwargs = self.format_kwargs.read().unwrap();
        let mut result = template.to_string();
        for (key, value) in kwargs.iter() {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Get the thread with format_kwargs applied to all messages
    pub fn formatted_thread(&self) -> Vec<Message> {
        // Collect all format_kwargs from root to this node
        let mut all_kwargs = std::collections::HashMap::new();
        self.collect_format_kwargs(&mut all_kwargs);

        // Apply formatting to each message
        self.thread()
            .into_iter()
            .map(|mut msg| {
                // Only apply formatting to text-only content, preserve multimodal content
                match &msg.content {
                    MessageContent::Text(text) => {
                        let mut formatted = text.clone();
                        for (key, value) in &all_kwargs {
                            let placeholder = format!("{{{}}}", key);
                            formatted = formatted.replace(&placeholder, value);
                        }
                        msg.content = MessageContent::Text(formatted);
                    }
                    MessageContent::Parts(parts) => {
                        // Format text parts while preserving other parts
                        let formatted_parts: Vec<_> = parts
                            .iter()
                            .map(|part| {
                                if let Some(text) = part.as_text() {
                                    let mut formatted = text.to_string();
                                    for (key, value) in &all_kwargs {
                                        let placeholder = format!("{{{}}}", key);
                                        formatted = formatted.replace(&placeholder, value);
                                    }
                                    ContentPart::text(formatted)
                                } else {
                                    part.clone()
                                }
                            })
                            .collect();
                        msg.content = MessageContent::Parts(formatted_parts);
                    }
                }
                msg
            })
            .collect()
    }

    /// Helper to collect format_kwargs from all ancestors
    fn collect_format_kwargs(&self, kwargs: &mut std::collections::HashMap<String, String>) {
        // First collect from parent (so child values override parent)
        if let Some(parent) = self.parent() {
            parent.collect_format_kwargs(kwargs);
        }
        // Then add this node's kwargs
        let my_kwargs = self.format_kwargs.read().unwrap();
        for (k, v) in my_kwargs.iter() {
            kwargs.insert(k.clone(), v.clone());
        }
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
        if let Some(secs) = params.and_then(|p| p.timeout_secs) {
            let client = LLMClient::with_timeout(Duration::from_secs(secs));
            return self.complete_with_client(&client, generator, params).await;
        }
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
        // Build messages with format_kwargs applied, then merge contiguous
        let mut messages = merge_contiguous_messages(self.formatted_thread());

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

        // Cost tracking
        use crate::provider::CostTrackingType;
        let cost_tracking = params
            .map(|p| p.cost_tracking)
            .unwrap_or(CostTrackingType::None);
        let cost_callback = params.and_then(|p| p.cost_callback.clone());
        let include_usage = cost_tracking == CostTrackingType::OpenRouter;

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

            // Make the request (with usage tracking if enabled)
            let response = match client
                .complete_with_usage_tracking(
                    generator,
                    &messages,
                    &completion_params,
                    include_usage,
                )
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

            // Call cost callback if provided
            if let Some(ref callback) = cost_callback {
                if let Some(usage) = &response.usage {
                    use crate::provider::CostInfo;
                    let cost_info = CostInfo {
                        cost: usage.cost.unwrap_or(0.0),
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        total_tokens: usage.total_tokens,
                        cached_tokens: usage.cached_tokens,
                        reasoning_tokens: usage.reasoning_tokens,
                        model: response.model.clone(),
                        response_id: response.id.clone(),
                    };
                    callback(cost_info);
                }
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
        if let Some(secs) = params.and_then(|p| p.timeout_secs) {
            let client = LLMClient::with_timeout(Duration::from_secs(secs));
            return self.complete_streaming_with_client(&client, generator, params).await;
        }
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
        // Build messages with format_kwargs applied, then merge contiguous
        let mut messages = merge_contiguous_messages(self.formatted_thread());

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

        // Check if cost tracking is enabled
        use crate::provider::CostTrackingType;
        let cost_tracking = params
            .map(|p| p.cost_tracking)
            .unwrap_or(CostTrackingType::None);
        let include_usage = cost_tracking == CostTrackingType::OpenRouter;

        // Start streaming (with usage tracking if enabled)
        client
            .complete_streaming_with_usage(generator, &messages, &completion_params, include_usage)
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

        // Call cost callback if provided
        if let Some(p) = params {
            if let Some(ref callback) = p.cost_callback {
                if let Some(usage) = &response.usage {
                    use crate::provider::CostInfo;
                    let cost_info = CostInfo {
                        cost: usage.cost.unwrap_or(0.0),
                        prompt_tokens: usage.prompt_tokens,
                        completion_tokens: usage.completion_tokens,
                        total_tokens: usage.total_tokens,
                        cached_tokens: usage.cached_tokens,
                        reasoning_tokens: usage.reasoning_tokens,
                        model: response.model.clone(),
                        response_id: response.id.clone(),
                    };
                    callback(cost_info);
                }
            }
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
/// ```
/// use minillmlib::{ChatNode, pretty_messages};
///
/// let root = ChatNode::root("You are helpful");
/// let user = root.add_user("Hello");
/// let assistant = user.add_assistant("Hi there!");
///
/// let pretty = pretty_messages(&assistant, None);
/// assert!(pretty.contains("SYSTEM: You are helpful"));
/// assert!(pretty.contains("USER: Hello"));
/// assert!(pretty.contains("ASSISTANT: Hi there!"));
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

    // Use formatted_thread to apply format_kwargs
    let messages = node.formatted_thread();
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

// =========================================================================
// Thread Serialization/Deserialization
// =========================================================================

use serde::{Deserialize, Serialize};

/// Serializable representation of a thread (for saving/loading)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadData {
    /// List of messages in the thread
    pub prompts: Vec<ThreadMessage>,

    /// Format kwargs for template substitution
    /// Values can be null (meaning "placeholder, not yet set") or strings
    #[serde(default)]
    pub required_kwargs: std::collections::HashMap<String, Option<String>>,
}

impl ThreadData {
    /// Get only the non-null kwargs as a HashMap<String, String>
    pub fn get_kwargs(&self) -> std::collections::HashMap<String, String> {
        self.required_kwargs
            .iter()
            .filter_map(|(k, v)| v.as_ref().map(|val| (k.clone(), val.clone())))
            .collect()
    }

    /// Get the list of kwargs that are null (placeholders)
    pub fn get_placeholder_keys(&self) -> Vec<String> {
        self.required_kwargs
            .iter()
            .filter_map(|(k, v)| if v.is_none() { Some(k.clone()) } else { None })
            .collect()
    }
}

/// A single message in a serialized thread
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMessage {
    /// Role of the message sender
    pub role: String,

    /// Text content of the message
    pub content: String,

    /// Optional image data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub image_data: Option<ThreadImageData>,

    /// Optional audio data
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_data: Option<ThreadAudioData>,
}

/// Serializable image data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadImageData {
    /// List of image URLs or base64 data
    pub images: Vec<String>,
}

/// Serializable audio data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadAudioData {
    /// List of audio file paths
    #[serde(default)]
    pub audio_paths: Vec<String>,

    /// Map of audio IDs
    #[serde(default)]
    pub audio_ids: std::collections::HashMap<String, String>,
}

impl ChatNode {
    /// Save the thread (path from root to this node) to a JSON file
    pub fn save_thread(&self, path: &str) -> Result<()> {
        let thread_data = self.to_thread_data();
        let json = serde_json::to_string_pretty(&thread_data)
            .map_err(|e| MiniLLMError::Other(format!("Failed to serialize thread: {}", e)))?;

        std::fs::write(path, json)
            .map_err(|e| MiniLLMError::Other(format!("Failed to write file: {}", e)))?;

        Ok(())
    }

    /// Convert the thread to serializable ThreadData
    pub fn to_thread_data(&self) -> ThreadData {
        let messages = self.thread();

        let prompts: Vec<ThreadMessage> = messages
            .iter()
            .map(|msg| ThreadMessage {
                role: msg.role.as_str().to_string(),
                content: msg.content.get_text().unwrap_or("").to_string(),
                image_data: None, // TODO: Extract image data if present
                audio_data: None, // TODO: Extract audio data if present
            })
            .collect();

        // Collect all format_kwargs from the tree and wrap in Some()
        let mut all_kwargs = std::collections::HashMap::new();
        self.collect_format_kwargs(&mut all_kwargs);
        let required_kwargs = all_kwargs.into_iter().map(|(k, v)| (k, Some(v))).collect();

        ThreadData {
            prompts,
            required_kwargs,
        }
    }

    /// Load a thread from a JSON file
    ///
    /// Returns a tuple of (root_node, leaf_node) so the caller can keep the root alive.
    pub fn from_thread_file(path: &str) -> Result<(Arc<ChatNode>, Arc<ChatNode>)> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| MiniLLMError::Other(format!("Failed to read file: {}", e)))?;

        Self::from_thread_json(&json)
    }

    /// Load a thread from a JSON string
    ///
    /// Returns a tuple of (root_node, leaf_node) so the caller can keep the root alive.
    pub fn from_thread_json(json: &str) -> Result<(Arc<ChatNode>, Arc<ChatNode>)> {
        let thread_data: ThreadData = serde_json::from_str(json)
            .map_err(|e| MiniLLMError::Other(format!("Failed to parse thread JSON: {}", e)))?;

        Self::from_thread_data(&thread_data)
    }

    /// Load a thread from ThreadData
    ///
    /// Returns a tuple of (root_node, leaf_node) so the caller can keep the root alive.
    /// If you only need the leaf, make sure to keep the root in scope.
    pub fn from_thread_data(data: &ThreadData) -> Result<(Arc<ChatNode>, Arc<ChatNode>)> {
        if data.prompts.is_empty() {
            return Err(MiniLLMError::Other("Thread has no messages".to_string()));
        }

        let mut root: Option<Arc<ChatNode>> = None;
        let mut current: Option<Arc<ChatNode>> = None;

        for msg in &data.prompts {
            let role = match msg.role.as_str() {
                "system" => Role::System,
                "user" => Role::User,
                "assistant" => Role::Assistant,
                "tool" => Role::Tool,
                _ => return Err(MiniLLMError::Other(format!("Unknown role: {}", msg.role))),
            };

            let message = Message {
                role,
                content: MessageContent::text(&msg.content),
                name: None,
                tool_call_id: None,
                tool_calls: None,
            };

            let node = ChatNode::new(message);

            current = Some(match current {
                Some(parent) => parent.add_child(node),
                None => {
                    root = Some(node.clone());
                    node
                }
            });
        }

        // Apply format_kwargs to the root node (only non-null values)
        if let Some(ref root_node) = root {
            root_node.set_format_kwargs(&data.get_kwargs());
        }

        Ok((root.unwrap(), current.unwrap()))
    }

    /// Load a thread from a list of messages
    ///
    /// Returns a tuple of (root_node, leaf_node) so the caller can keep the root alive.
    pub fn from_messages(messages: &[Message]) -> Result<(Arc<ChatNode>, Arc<ChatNode>)> {
        if messages.is_empty() {
            return Err(MiniLLMError::Other("No messages provided".to_string()));
        }

        let mut root: Option<Arc<ChatNode>> = None;
        let mut current: Option<Arc<ChatNode>> = None;

        for msg in messages {
            let node = ChatNode::new(msg.clone());

            current = Some(match current {
                Some(parent) => parent.add_child(node),
                None => {
                    root = Some(node.clone());
                    node
                }
            });
        }

        Ok((root.unwrap(), current.unwrap()))
    }
}
