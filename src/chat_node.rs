//! ChatNode - Core conversation tree structure
//!
//! A `ChatNode` is a lightweight, cloneable *handle* into a shared `Tree` arena
//! that owns all node data. Each node has a message and can have multiple
//! children (branches), supporting linear conversations, branching/alternate
//! paths, and tree-structured dialogues.
//!
//! Ownership: holding ANY node handle keeps the entire tree alive (ancestors,
//! descendants, and sibling branches), because every handle holds an `Arc` to the
//! shared arena that owns all node data. When the last handle is dropped the
//! whole arena frees at once. Inter-node links are ids into the arena, not
//! `Arc`s, so there is no reference cycle (nothing leaks) and teardown is a flat
//! map drop (a deep tree can't overflow the stack). See [`ChatNode`] for details.

use crate::error::{MiniLLMError, Result};
use crate::generator::{GeneratorInfo, NodeCompletionParameters};
use crate::json_repair::{loads, repair_json, JsonValue, RepairOptions};
use crate::message::{merge_contiguous_messages, ContentPart, Message, MessageContent, Role};
use crate::provider::{global_client, CompletionResponse, StreamingCompletion};
use std::sync::{Arc, RwLock};
use std::time::Duration;
use uuid::Uuid;

/// Resolved per-request completion settings, extracted from
/// [`NodeCompletionParameters`] once so the completion pipeline reads typed
/// fields instead of repeatedly unwrapping `Option<&params>` at every step.
struct CompletionSettings {
    params: Option<crate::generator::CompletionParameters>,
    system_prompt: Option<String>,
    force_prepend: Option<String>,
    format_kwargs: std::collections::HashMap<String, String>,
    add_child: bool,
    parse_json: bool,
    use_cache: bool,
    crash_on_refusal: bool,
    crash_on_empty: bool,
    retry: u32,
    exp_back_off: bool,
    back_off_time: f64,
    max_back_off: f64,
    timeout: Option<Duration>,
    track_cost: bool,
    token_price: Option<crate::provider::TokenPrice>,
    cost_callback: Option<crate::provider::CostCallback>,
}

impl CompletionSettings {
    fn from_params(params: Option<&NodeCompletionParameters>) -> Self {
        let defaults = NodeCompletionParameters::default();
        let p = params.unwrap_or(&defaults);
        Self {
            params: p.params.clone(),
            system_prompt: p.system_prompt.clone(),
            force_prepend: p.force_prepend.clone(),
            format_kwargs: p.format_kwargs.clone(),
            add_child: p.add_child,
            parse_json: p.parse_json,
            use_cache: p.use_cache,
            crash_on_refusal: p.crash_on_refusal,
            crash_on_empty: p.crash_on_empty_response,
            retry: p.retry,
            exp_back_off: p.exp_back_off,
            back_off_time: p.back_off_time,
            max_back_off: p.max_back_off,
            // 0 means "no timeout", not an instantly-firing 0s deadline.
            timeout: p.timeout_secs.filter(|&s| s > 0).map(Duration::from_secs),
            track_cost: p.track_cost,
            token_price: p.token_price.clone(),
            cost_callback: p.cost_callback.clone(),
        }
    }

    /// Merge per-request params over the generator's defaults.
    fn merged_params(&self, generator: &GeneratorInfo) -> crate::generator::CompletionParameters {
        match &self.params {
            Some(p) => generator.default_params.merge(p),
            None => generator.default_params.clone(),
        }
    }

    /// Effective per-token price: per-request override wins over the generator's.
    fn price<'a>(
        &'a self,
        generator: &'a GeneratorInfo,
    ) -> Option<&'a crate::provider::TokenPrice> {
        self.token_price.as_ref().or(generator.token_price.as_ref())
    }

    /// Fire the user-supplied (non-enforced) cost callback. Costs the usage via
    /// the provider's accounting; best-effort, so it only fires when usage is
    /// present (no out-of-band resolution on this path).
    fn fire_cost_callback(&self, generator: &GeneratorInfo, response: &CompletionResponse) {
        if let (Some(callback), Some(usage)) = (&self.cost_callback, &response.usage) {
            let outcome = generator
                .provider
                .cost_of(usage.clone(), self.price(generator));
            callback(outcome.into_cost_info(response.model.clone(), response.id.clone()));
        }
    }
}

/// How a completion's raw response is obtained. The only axis on which the
/// streaming and non-streaming completion paths differ; everything else (retry,
/// post-processing, node construction, cost) is shared.
#[derive(Debug, Clone, Copy)]
enum ResponseMode {
    NonStreaming,
    Streaming,
}

/// Whether a failed request is worth retrying. Auth/validation errors (4xx other
/// than 408/429) are terminal and must not re-issue paid LLM calls; rate limits,
/// server errors, timeouts, and transport failures are transient.
fn is_retryable(error: &MiniLLMError) -> bool {
    match error {
        MiniLLMError::Api { status, .. } => *status == 408 || *status == 429 || *status >= 500,
        // Timeouts and SSE stream errors are transient.
        MiniLLMError::Timeout | MiniLLMError::Stream(_) => true,
        // A transport error is only worth retrying if it's a connect/request
        // failure (transient); a bad-builder/decode error won't succeed on retry.
        MiniLLMError::Http(e) => e.is_connect() || e.is_request() || e.is_timeout(),
        // Anything else (malformed response, auth/validation) is not retryable.
        _ => false,
    }
}

/// Repair LLM JSON output and, when `crash_on_refusal` is set, reject responses
/// that did not yield a usable JSON value. Refusal is decided on the *parsed*
/// value (empty/null), not on substring scanning of the raw text.
fn repair_and_validate_json(content: &str, crash_on_refusal: bool) -> Result<String> {
    let opts = RepairOptions::default();

    if crash_on_refusal {
        let value = loads(content, &opts)?;
        if json_value_is_empty(&value) {
            return Err(MiniLLMError::NoJsonFound(content.to_string()));
        }
    }

    Ok(repair_json(content, &opts)?)
}

/// A repaired JSON value that carries no usable content: empty string, empty
/// object, empty array, or null. Used to detect a model refusal.
fn json_value_is_empty(value: &JsonValue) -> bool {
    match value {
        JsonValue::Null => true,
        JsonValue::String(s) => s.is_empty(),
        JsonValue::Array(a) => a.is_empty(),
        JsonValue::Object(o) => o.is_empty(),
        _ => false,
    }
}

/// Force usage/cost tracking on and clear any user callback (the tracked path
/// reports cost itself via the CompletionContext's enforced callback).
fn tracked_params(params: Option<&NodeCompletionParameters>) -> NodeCompletionParameters {
    let mut p = params.cloned().unwrap_or_default();
    p.track_cost = true;
    p.cost_callback = None;
    p
}

/// Default max silence between streamed chunks for a tracked stream. A live
/// generation emits tokens far more often; this bounds a wedged connection so it
/// fails loudly (Timeout, retryable) instead of parking the consumer until the
/// connection-pool timeout (and starving the cancellation cost-settle). Applied
/// ONLY on streaming tracked paths: for them `timeout_secs` means idle silence;
/// on the non-streaming path it would be a total-response deadline (wrong for a
/// slow reasoning model the user gave no deadline).
///
/// A tracked stream ALWAYS gets an idle floor: both `None` (unset) and `Some(0)`
/// (the "no timeout" escape hatch) are overridden, because the tracked path's
/// reason for being is the cost-settle backstop, which a dead connection with no
/// idle bound would starve.
fn tracked_streaming_params(params: Option<&NodeCompletionParameters>) -> NodeCompletionParameters {
    const DEFAULT_TRACKED_IDLE_TIMEOUT_SECS: u64 = 120;
    let mut p = tracked_params(params);
    if p.timeout_secs.is_none_or(|s| s == 0) {
        p.timeout_secs = Some(DEFAULT_TRACKED_IDLE_TIMEOUT_SECS);
    }
    p
}

/// Substitute every `{key}` placeholder in `template` with its kwarg value.
///
/// Single left-to-right pass: a matched `{key}`'s value is emitted verbatim and
/// never re-scanned, so the result is independent of map iteration order and a
/// value that itself contains `{another_key}` is not re-expanded. An unmatched
/// `{...}` is left as-is. The one place template substitution happens, so node
/// text, thread text, and `format_string` all behave identically.
fn apply_kwargs(template: &str, kwargs: &std::collections::HashMap<String, String>) -> String {
    let mut result = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        result.push_str(&rest[..open]);
        let after_open = &rest[open + 1..];
        match after_open.find('}') {
            Some(close) => {
                let key = &after_open[..close];
                match kwargs.get(key) {
                    Some(value) => result.push_str(value),
                    // Unknown placeholder: leave it literally intact.
                    None => {
                        result.push('{');
                        result.push_str(key);
                        result.push('}');
                    }
                }
                rest = &after_open[close + 1..];
            }
            // Unbalanced '{' with no closing '}': emit the rest verbatim.
            None => {
                result.push_str(&rest[open..]);
                rest = "";
            }
        }
    }
    result.push_str(rest);
    result
}

/// The per-node mutable/structural data, owned by the `Tree` arena (keyed by
/// node id). Inter-node links are ids into the same arena, never `Arc`s, so the
/// tree has no reference cycle.
struct NodeData {
    /// The message at this node (immutable once created; also cached on the
    /// handle for cheap field access).
    message: Message,
    /// Per-node metadata (mutable).
    metadata: serde_json::Value,
    /// Node-scoped format kwargs (mutable).
    format_kwargs: std::collections::HashMap<String, String>,
    /// Whether this node is a cache breakpoint (mutable). Stamped onto the
    /// outgoing message at thread-build time, like `format_kwargs`, so it is read
    /// fresh from the arena (a mark set after a handle was cloned still applies).
    cache_breakpoint: bool,
    /// Parent node id (`None` for a root).
    parent: Option<String>,
    /// Ordered child node ids (REGISTERED children only; phantoms are excluded by
    /// design so traversal never sees them).
    children: Vec<String>,
    /// Number of PHANTOM children that name this node as their `parent` without
    /// appearing in `children` (see [`ChatNode::insert_phantom_child`]). A phantom
    /// walks UP through this node, so this node must stay alive while any phantom
    /// descends from it, exactly as a registered child keeps it alive. Bumped when
    /// a phantom is inserted, decremented when a phantom is reclaimed. Without this,
    /// reclaiming a node whose only descendant is a held phantom would dangle the
    /// phantom's parent id and abort the process on the next upward walk.
    phantom_child_count: usize,
    /// Number of live [`ChatNode`] handles pointing at THIS node. A node is kept
    /// alive while `refcount > 0` OR it has any child (registered or phantom). When
    /// all reach zero the node is reclaimed and its parent is re-checked (cascade).
    /// See [`Tree::release`].
    refcount: usize,
}

/// The arena that owns every node of one conversation tree.
///
/// All node data lives here in `nodes` (id → [`NodeData`]); nodes reference each
/// other only by id, never by `Arc`, so there is no reference cycle.
///
/// # Lifetime of a node
///
/// Each node carries a `refcount` of the live handles pointing at it. A node is
/// retained while it has a live handle (`refcount > 0`) OR any descendant that
/// must walk up through it: a registered child (in `children`) OR a phantom
/// (counted in `phantom_child_count`). When all three reach zero the node is
/// removed and its parent is re-checked, so an unheld leaf branch, a dropped
/// phantom, or a detached-then-dropped subtree is reclaimed bottom-up, while the
/// ancestors of any held node (registered or phantom descendant) stay alive. The
/// arena `HashMap` therefore does not grow without bound. Teardown is a flat map
/// drop (no recursive destructor), so a deep tree can't overflow the stack.
struct Tree {
    nodes: RwLock<std::collections::HashMap<String, NodeData>>,
}

impl Tree {
    /// A fresh tree seeded with one node (refcount 1, for the handle the caller
    /// gets). Returns the tree and the new node's id.
    fn with_root(message: Message) -> (Arc<Tree>, String) {
        let id = Uuid::new_v4().to_string();
        let mut nodes = std::collections::HashMap::new();
        nodes.insert(id.clone(), NodeData::new(message, None));
        (
            Arc::new(Tree {
                nodes: RwLock::new(nodes),
            }),
            id,
        )
    }

    /// Drop one handle's claim on `id`: decrement its refcount, then reclaim it
    /// (and cascade upward) if it is now unheld and childless.
    fn release(&self, id: &str) {
        let mut nodes = self.nodes.write().unwrap();
        if let Some(node) = nodes.get_mut(id) {
            node.refcount = node.refcount.saturating_sub(1);
        }
        Self::reclaim_dead(&mut nodes, id.to_string());
    }

    /// Reclaim `start` and cascade to its ancestors while each is unheld
    /// (`refcount == 0`) and childless. The single place node removal happens:
    /// removing a node drops it from its parent's child list, which may make the
    /// parent reclaimable in turn. Caller holds the write lock.
    fn reclaim_dead(nodes: &mut NodeMap, start: String) {
        let mut current = Some(start);
        while let Some(cur) = current {
            let Some(node) = nodes.get(&cur) else {
                break;
            };
            // Retained while held, or while ANY child (registered or phantom) still
            // descends from it: a phantom walks up through this node just like a
            // registered child, so it must keep this node alive too.
            if node.refcount > 0 || !node.children.is_empty() || node.phantom_child_count > 0 {
                break;
            }
            let parent = node.parent.clone();
            nodes.remove(&cur);
            if let Some(pid) = &parent {
                if let Some(p) = nodes.get_mut(pid) {
                    // Drop this node's incoming edge from its parent. A registered
                    // child is in the parent's `children`; a phantom is not, and
                    // instead counted in `phantom_child_count`. `retain` reports
                    // whether it removed a registered entry, which disambiguates.
                    let before = p.children.len();
                    p.children.retain(|c| c != &cur);
                    if p.children.len() == before {
                        // Was not a registered child, so it was a phantom. Inserts
                        // and reclaims of phantoms are 1:1 under the write lock, so
                        // an underflow here means the registered-vs-phantom invariant
                        // broke (a non-phantom reclaimed as a phantom, or a phantom
                        // reclaimed twice). Fail loudly at the fault site rather than
                        // clamp to a wrong count that would later dangle a parent id.
                        p.phantom_child_count = p.phantom_child_count.checked_sub(1).expect(
                            "phantom_child_count underflow: registered/phantom invariant broken",
                        );
                    }
                }
            }
            current = parent; // re-check the parent (it just lost a child)
        }
    }

    /// Cut `id`'s edge to its current parent (drop it from the parent's child
    /// list and clear its `parent`), then reclaim the former parent if that left
    /// it unheld and childless. The single place a parent edge is severed, shared
    /// by `detach` (which keeps `id` as a new root) and `add_child`'s same-tree
    /// re-link (which immediately re-parents `id`). Caller holds the write lock.
    /// `id` itself is never reclaimed here (the caller holds a handle to it).
    fn unlink_from_parent(nodes: &mut NodeMap, id: &str) {
        let parent_id = nodes.get(id).and_then(|n| n.parent.clone());
        if let Some(pid) = &parent_id {
            if let Some(p) = nodes.get_mut(pid) {
                // Drop `id`'s incoming edge. A registered child is in `children`; a
                // phantom is not, and is instead counted in `phantom_child_count`.
                // The same disambiguation `reclaim_dead` uses (retain removed
                // nothing → it was a phantom edge), so detaching/re-parenting a
                // phantom transfers its count instead of leaking the old parent.
                let before = p.children.len();
                p.children.retain(|c| c != id);
                if p.children.len() == before {
                    p.phantom_child_count = p.phantom_child_count.checked_sub(1).expect(
                        "phantom_child_count underflow: registered/phantom invariant broken",
                    );
                }
            }
        }
        if let Some(n) = nodes.get_mut(id) {
            n.parent = None;
        }
        if let Some(pid) = parent_id {
            Self::reclaim_dead(nodes, pid);
        }
    }

    /// Whether `target` is `start` or one of its ancestors, walking the parent-id
    /// chain over an already-locked map (no lock taken). The cycle test for
    /// `add_child`, evaluated UNDER the same write guard as the re-link so the
    /// check and the mutation are atomic (a concurrent re-parent can't slip a
    /// cycle in between a separate check and mutation).
    fn is_self_or_ancestor_locked(nodes: &NodeMap, start: &str, target: &str) -> bool {
        let mut cur = Some(start.to_string());
        while let Some(c) = cur {
            if c == target {
                return true;
            }
            cur = nodes.get(&c).and_then(|n| n.parent.clone());
        }
        false
    }
}

/// The locked node maps of two distinct trees, returned in the caller's
/// (`first`, `second`) logical order but ACQUIRED in a global order (by `Arc`
/// pointer address). This gives every pair of trees a single, consistent lock
/// order regardless of which is `first`/`second`, so two threads co-locking the
/// same pair in opposite logical directions can't deadlock. Both are write
/// guards (a write guard also permits reads); the caller may use one read-only.
type NodeMap = std::collections::HashMap<String, NodeData>;
fn lock_two_write<'a>(
    first: &'a Arc<Tree>,
    second: &'a Arc<Tree>,
) -> (
    std::sync::RwLockWriteGuard<'a, NodeMap>,
    std::sync::RwLockWriteGuard<'a, NodeMap>,
) {
    assert!(
        !Arc::ptr_eq(first, second),
        "lock_two_write requires two distinct trees"
    );
    // Acquire in address order, then return in logical order.
    if Arc::as_ptr(first) < Arc::as_ptr(second) {
        let f = first.nodes.write().unwrap();
        let s = second.nodes.write().unwrap();
        (f, s)
    } else {
        let s = second.nodes.write().unwrap();
        let f = first.nodes.write().unwrap();
        (f, s)
    }
}

impl NodeData {
    /// A fresh node with one handle claim (`refcount: 1`).
    fn new(message: Message, parent: Option<String>) -> Self {
        NodeData {
            message,
            metadata: serde_json::json!({}),
            format_kwargs: std::collections::HashMap::new(),
            cache_breakpoint: false,
            parent,
            children: Vec::new(),
            phantom_child_count: 0,
            refcount: 1,
        }
    }
}

/// A handle to a node in a conversation tree.
///
/// # Ownership
///
/// A handle is `{ id, message, tree }`: a cheap, cloneable reference into a
/// shared `Tree` arena that owns all node data. The handle caches the node's id
/// and (immutable) message for ergonomic, lock-free field access, and holds an
/// `Arc<Tree>` plus a per-node refcount claim.
///
/// **Holding a node keeps that node and its full ancestor chain alive** (its
/// ancestors retain it as a child, and it retains them by walking up). A branch
/// you stop holding (an unheld leaf, a dropped phantom, a detached-then-dropped
/// subtree) is reclaimed; the arena does not grow without bound.
/// Inter-node links are ids, not `Arc`s, so there is no reference cycle, and the
/// last handle dropping frees its node bottom-up with no recursive destructor.
pub struct ChatNode {
    /// Unique identifier for this node.
    pub id: String,

    /// The message at this node (immutable; cached from the arena).
    pub message: Message,

    /// The arena this node belongs to.
    tree: Arc<Tree>,
}

impl Clone for ChatNode {
    /// Cloning a handle adds a refcount claim on the node (so the node lives as
    /// long as any clone does).
    fn clone(&self) -> Self {
        self.tree
            .nodes
            .write()
            .unwrap()
            .get_mut(&self.id)
            .expect("node id present in its own tree")
            .refcount += 1;
        Self {
            id: self.id.clone(),
            message: self.message.clone(),
            tree: self.tree.clone(),
        }
    }
}

impl Drop for ChatNode {
    /// Dropping a handle releases its refcount claim, reclaiming the node (and
    /// cascading to ancestors) if nothing else holds or descends from it.
    fn drop(&mut self) {
        self.tree.release(&self.id);
    }
}

impl ChatNode {
    // ---- arena access helpers --------------------------------------------

    /// Read a node's data via `f`. The node is always present for a live handle.
    fn with_node<R>(&self, id: &str, f: impl FnOnce(&NodeData) -> R) -> R {
        let nodes = self.tree.nodes.read().unwrap();
        f(nodes.get(id).expect("node id present in its own tree"))
    }

    /// Mutate a node's data via `f`.
    fn with_node_mut<R>(&self, id: &str, f: impl FnOnce(&mut NodeData) -> R) -> R {
        let mut nodes = self.tree.nodes.write().unwrap();
        f(nodes.get_mut(id).expect("node id present in its own tree"))
    }

    /// Build an ADDITIONAL handle for an existing `id` in this node's tree,
    /// adding a refcount claim (so the node lives as long as this handle does).
    fn handle(&self, id: String) -> ChatNode {
        let message = {
            let mut nodes = self.tree.nodes.write().unwrap();
            let n = nodes.get_mut(&id).expect("node id present in its own tree");
            n.refcount += 1;
            n.message.clone()
        };
        ChatNode {
            id,
            message,
            tree: self.tree.clone(),
        }
    }

    /// Wrap an id whose refcount claim has ALREADY been accounted for (e.g. a
    /// freshly-inserted node born with `refcount: 1`) into a handle, without
    /// adding another claim.
    fn handle_owned(&self, id: String, message: Message) -> ChatNode {
        ChatNode {
            id,
            message,
            tree: self.tree.clone(),
        }
    }

    /// Number of nodes currently allocated in this node's arena (test-only; the
    /// proof that dead branches/phantoms are reclaimed rather than accumulating).
    #[cfg(test)]
    fn arena_len(&self) -> usize {
        self.tree.nodes.read().unwrap().len()
    }

    /// Insert a fresh node (carrying `message`) into this node's tree as a child
    /// of `parent_id`, returning a handle to it (the node is born `refcount: 1`
    /// for that handle).
    fn insert_child(&self, parent_id: &str, message: Message) -> ChatNode {
        let id = Uuid::new_v4().to_string();
        {
            let mut nodes = self.tree.nodes.write().unwrap();
            nodes.insert(
                id.clone(),
                NodeData::new(message.clone(), Some(parent_id.to_string())),
            );
            nodes
                .get_mut(parent_id)
                .expect("parent id present")
                .children
                .push(id.clone());
        }
        self.handle_owned(id, message)
    }

    /// Copy the subtree rooted at `src` (a node in a DIFFERENT tree) into this
    /// tree under `parent_id`, with fresh ids, preserving structure/metadata/
    /// kwargs. Returns the id of the copied subtree's new root. Iterative (no
    /// recursion). Copied nodes are born `refcount: 0` (kept alive by their
    /// children, and the root by the parent's child list); the caller turns the
    /// returned id into a handle via [`handle`](Self::handle), which adds its claim.
    ///
    /// `src` must be in a different arena than `self`: this locks both arenas at
    /// once, so a same-arena call would deadlock. The only caller (`add_child`'s
    /// cross-tree branch) guarantees this.
    fn copy_subtree_under(&self, parent_id: &str, src: &ChatNode) -> String {
        debug_assert!(
            !Arc::ptr_eq(&self.tree, &src.tree),
            "copy_subtree_under must be cross-tree (same-tree would deadlock)"
        );
        let new_root_id = Uuid::new_v4().to_string();
        let mut stack = vec![(
            src.id.clone(),
            Some(parent_id.to_string()),
            new_root_id.clone(),
        )];
        // Lock both arenas in a global order (by Arc pointer address) so a
        // concurrent reverse-direction cross-tree copy can't deadlock.
        let (src_nodes, mut nodes) = lock_two_write(&src.tree, &self.tree);
        while let Some((src_id, new_parent, new_id)) = stack.pop() {
            let src_data = src_nodes.get(&src_id).expect("src node present");
            // Pre-mint child ids so we can record them and recurse.
            let child_pairs: Vec<(String, String)> = src_data
                .children
                .iter()
                .map(|c| (c.clone(), Uuid::new_v4().to_string()))
                .collect();
            nodes.insert(
                new_id.clone(),
                NodeData {
                    message: src_data.message.clone(),
                    metadata: src_data.metadata.clone(),
                    format_kwargs: src_data.format_kwargs.clone(),
                    cache_breakpoint: src_data.cache_breakpoint,
                    parent: new_parent.clone(),
                    children: child_pairs.iter().map(|(_, n)| n.clone()).collect(),
                    // Only registered children are copied (the walk follows
                    // `children`), so the copy descends from no phantoms.
                    phantom_child_count: 0,
                    refcount: 0,
                },
            );
            if let Some(p) = &new_parent {
                if p == parent_id {
                    nodes
                        .get_mut(parent_id)
                        .expect("parent id present")
                        .children
                        .push(new_id.clone());
                }
            }
            for (src_child, new_child) in child_pairs {
                stack.push((src_child, Some(new_id.clone()), new_child));
            }
        }
        new_root_id
    }

    /// Create a new root node with a system message
    pub fn root(system_prompt: impl Into<String>) -> Self {
        Self::new(Message::system(system_prompt.into()))
    }

    /// Create a new node with a message (in its own fresh single-node tree).
    pub fn new(message: Message) -> Self {
        let (tree, id) = Tree::with_root(message.clone());
        Self { id, message, tree }
    }

    /// Create a user message node
    pub fn user(content: impl Into<MessageContent>) -> Self {
        Self::new(Message::user(content))
    }

    /// Create an assistant message node
    pub fn assistant(content: impl Into<MessageContent>) -> Self {
        Self::new(Message::assistant(content))
    }

    /// Add an (arbitrary) node's subtree as a child of this node.
    ///
    /// `child` may belong to a different tree (or the same one). Its subtree is
    /// **copied** into this node's tree with fresh ids; the returned handle is the
    /// copy's root, bound to this tree. The original `child` and its tree are
    /// untouched. (Copying, not moving, keeps each tree's handles valid and the
    /// arena cycle-free.)
    ///
    /// Fails if `child` is `self` or one of its ancestors *in this tree*: that
    /// would graft a node under its own descendant, a structural cycle. For normal
    /// conversation building use [`add_user`](Self::add_user) /
    /// [`add_assistant`](Self::add_assistant); to join two trees use
    /// [`merge`](Self::merge).
    pub fn add_child(&self, child: ChatNode) -> Result<ChatNode> {
        if Arc::ptr_eq(&self.tree, &child.tree) {
            // Same tree: re-link in place (a move), so existing handles to `child`
            // stay valid and observe the new parent. The cycle check AND the
            // mutation happen under one write guard so they are atomic: a
            // concurrent re-parent of `self` can't slip a cycle in between a
            // separate check and a separate mutation (which would make the parent
            // chain loop, hanging every later walk).
            let mut nodes = self.tree.nodes.write().unwrap();
            if Tree::is_self_or_ancestor_locked(&nodes, &self.id, &child.id) {
                return Err(MiniLLMError::InvalidParameter(
                    "add_child would create a cycle: the child is this node or one of its ancestors"
                        .to_string(),
                ));
            }
            // Cut `child`'s old parent edge (reclaiming that parent if it is now an
            // unheld, childless orphan, symmetric with `detach`, so a re-parent
            // can't leak the old parent), then link `child` under self.
            Tree::unlink_from_parent(&mut nodes, &child.id);
            nodes.get_mut(&child.id).expect("child present").parent = Some(self.id.clone());
            nodes
                .get_mut(&self.id)
                .expect("self present")
                .children
                .push(child.id.clone());
            drop(nodes);
            Ok(child)
        } else {
            // Different tree: copy the subtree in (both trees' handles stay valid).
            let new_id = self.copy_subtree_under(&self.id, &child);
            Ok(self.handle(new_id))
        }
    }

    /// Add a user message as a child (a fresh node in this tree).
    pub fn add_user(&self, content: impl Into<MessageContent>) -> ChatNode {
        self.insert_child(&self.id, Message::user(content))
    }

    /// Add an assistant message as a child (a fresh node in this tree).
    pub fn add_assistant(&self, content: impl Into<MessageContent>) -> ChatNode {
        self.insert_child(&self.id, Message::assistant(content))
    }

    /// Add a tool-result message as a child, answering one of this node's tool
    /// calls (`call_id` is [`ToolCall::id`](crate::tools::ToolCall::id)). For
    /// parallel calls, chain one `add_tool_result` per call; the provider groups
    /// them on its wire (Anthropic packs consecutive results into one user turn).
    pub fn add_tool_result(
        &self,
        call_id: impl Into<String>,
        content: impl Into<MessageContent>,
    ) -> ChatNode {
        self.insert_child(&self.id, Message::tool(call_id, content))
    }

    /// The tool calls carried by this node's message, if the assistant made any
    /// (set by the completion when the model called tools).
    pub fn tool_calls(&self) -> Option<Vec<crate::tools::ToolCall>> {
        self.message.tool_calls.clone()
    }

    /// Get the parent node (`None` for a root).
    pub fn parent(&self) -> Option<ChatNode> {
        let parent_id = self.with_node(&self.id, |n| n.parent.clone())?;
        Some(self.handle(parent_id))
    }

    /// Get this node's children (in insertion order).
    pub fn children(&self) -> Vec<ChatNode> {
        let ids = self.with_node(&self.id, |n| n.children.clone());
        ids.into_iter().map(|id| self.handle(id)).collect()
    }

    /// Get the number of children.
    pub fn child_count(&self) -> usize {
        self.with_node(&self.id, |n| n.children.len())
    }

    /// Check if this is a root node.
    pub fn is_root(&self) -> bool {
        self.with_node(&self.id, |n| n.parent.is_none())
    }

    /// Get the root node of the tree by walking the parent id chain.
    pub fn get_root(&self) -> ChatNode {
        let mut id = self.id.clone();
        while let Some(parent) = self.with_node(&id, |n| n.parent.clone()) {
            id = parent;
        }
        self.handle(id)
    }

    /// Check if this is a leaf node (no children).
    pub fn is_leaf(&self) -> bool {
        self.with_node(&self.id, |n| n.children.is_empty())
    }

    /// Get the thread (path from root to this node)
    pub fn thread(&self) -> Vec<Message> {
        self.node_path().iter().map(|n| n.message.clone()).collect()
    }

    /// Get the thread with contiguous messages merged
    pub fn merged_thread(&self) -> Vec<Message> {
        merge_contiguous_messages(self.thread())
    }

    /// Get the depth of this node in the tree (0 for a root).
    pub fn depth(&self) -> usize {
        let mut depth = 0;
        let mut node = self.parent();
        while let Some(n) = node {
            depth += 1;
            node = n.parent();
        }
        depth
    }

    /// Find a node by ID in the subtree rooted at this node
    pub fn find_by_id(&self, id: &str) -> Option<ChatNode> {
        self.iter_depth_first().into_iter().find(|n| n.id == id)
    }

    /// Get the last live child (most recent branch)
    pub fn last_child(&self) -> Option<ChatNode> {
        self.children().pop()
    }

    /// Get the leaf node following the last child at each level
    pub fn get_leaf(&self) -> ChatNode {
        let mut node = self.clone();
        while let Some(child) = node.last_child() {
            node = child;
        }
        node
    }

    // =========================================================================
    // Tree manipulation
    // =========================================================================

    /// Detach this node from its parent, making it a root of its subtree.
    ///
    /// The node and its descendants stay in the same arena (so existing handles
    /// remain valid); only the parent edge is cut: this node's `parent` is cleared
    /// and it is removed from its parent's child list. After this, `self` is a root
    /// (`is_root()` true) and its former parent no longer reaches it. Returns self.
    pub fn detach(&self) -> ChatNode {
        // Cut the parent edge: drop `self` from its parent's child list, clear its
        // `parent`, and reclaim the former parent if that left it an unheld,
        // childless orphan. `self` is unaffected (the caller holds it, refcount ≥ 1).
        let mut nodes = self.tree.nodes.write().unwrap();
        Tree::unlink_from_parent(&mut nodes, &self.id);
        drop(nodes);
        self.clone()
    }

    /// Merge another tree into this node: copies the root of `other`'s tree (and
    /// its whole subtree) in as a child of this node, returning a handle to the
    /// copied leaf (in this tree).
    ///
    /// `other` and its tree are left untouched (copy, not move). Fails if `other`
    /// is in *this* tree and is an ancestor of `self` (would graft under a
    /// descendant). Typically used to join two *separate* trees.
    pub fn merge(&self, other: &ChatNode) -> Result<ChatNode> {
        let other_root = other.get_root();
        let copied_root = self.add_child(other_root)?;
        Ok(copied_root.get_leaf())
    }

    /// Deep-clone the tree this node belongs to into a fresh independent tree,
    /// returning the handle to this node's copy.
    ///
    /// Every node is duplicated with a fresh id, copying message, metadata and
    /// format kwargs; the clone shares nothing with the original. Holding the
    /// returned handle keeps the entire cloned tree alive (any handle keeps the
    /// whole tree). Use this before per-node edits when a node may be shared.
    pub fn clone_tree(&self) -> ChatNode {
        let src_nodes = self.tree.nodes.read().unwrap();

        // Map every original id to a fresh id (covers all nodes in the arena,
        // including any phantom that isn't reachable from the root via children).
        let id_map: std::collections::HashMap<String, String> = src_nodes
            .keys()
            .map(|id| (id.clone(), Uuid::new_v4().to_string()))
            .collect();

        let new_id = id_map[&self.id].clone();

        // Copy each node, translating ids through the map. Cloned nodes are born
        // with `refcount: 0` (no handles into the clone yet); the one node we hand
        // back gets `refcount: 1` for the returned handle, and is then kept alive
        // exactly like any node (by that handle and/or its children).
        let new_nodes: std::collections::HashMap<String, NodeData> = src_nodes
            .iter()
            .map(|(id, data)| {
                let new_id_for_node = id_map[id].clone();
                let refcount = usize::from(new_id_for_node == new_id);
                (
                    new_id_for_node,
                    NodeData {
                        message: data.message.clone(),
                        metadata: data.metadata.clone(),
                        format_kwargs: data.format_kwargs.clone(),
                        cache_breakpoint: data.cache_breakpoint,
                        parent: data.parent.as_ref().map(|p| id_map[p].clone()),
                        children: data.children.iter().map(|c| id_map[c].clone()).collect(),
                        // clone_tree copies EVERY arena node (phantoms included, with
                        // their parent ids remapped), so the parent's phantom count
                        // stays accurate, so preserve it verbatim.
                        phantom_child_count: data.phantom_child_count,
                        refcount,
                    },
                )
            })
            .collect();

        let new_tree = Arc::new(Tree {
            nodes: RwLock::new(new_nodes),
        });
        ChatNode {
            id: new_id,
            message: self.message.clone(),
            tree: new_tree,
        }
    }

    // =========================================================================
    // Tree iteration
    // =========================================================================

    /// Iterate over all nodes in the subtree rooted at this node (depth-first,
    /// pre-order). Iterative work-stack walk (no recursion → no overflow on a
    /// deep subtree).
    pub fn iter_depth_first(&self) -> Vec<ChatNode> {
        let mut result = Vec::new();
        let mut stack = vec![self.clone()];
        while let Some(node) = stack.pop() {
            // Push children in reverse so they pop in original (pre-)order.
            let children = node.children();
            result.push(node);
            stack.extend(children.into_iter().rev());
        }
        result
    }

    /// Iterate over all nodes in the subtree rooted at this node (breadth-first)
    pub fn iter_breadth_first(&self) -> Vec<ChatNode> {
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

    /// Get all leaf nodes in the subtree rooted at this node (iterative).
    pub fn iter_leaves(&self) -> Vec<ChatNode> {
        self.iter_depth_first()
            .into_iter()
            .filter(|n| n.is_leaf())
            .collect()
    }

    /// Count total nodes in the subtree rooted at this node (iterative).
    pub fn node_count(&self) -> usize {
        self.iter_depth_first().len()
    }

    /// Set metadata for this node
    pub fn set_metadata(&self, key: &str, value: serde_json::Value) {
        self.with_node_mut(&self.id, |n| n.metadata[key] = value);
    }

    /// Get metadata value
    pub fn get_metadata(&self, key: &str) -> Option<serde_json::Value> {
        self.with_node(&self.id, |n| n.metadata.get(key).cloned())
    }

    // =========================================================================
    // Prompt caching (normalized intent: the provider decides the wire)
    // =========================================================================

    /// Mark this node as a **cache breakpoint**: the conversation prefix up to and
    /// including this node becomes a candidate for prompt caching. Provider-agnostic
    /// intent: [`AnthropicProvider`](crate::AnthropicProvider) emits a `cache_control`
    /// marker here (honoring its 4-breakpoint / minimum-size limits); OpenAI/OpenRouter
    /// ignore it (they auto-cache). Returns self for chaining.
    ///
    /// To cache just the system prompt, mark the root; to cache the whole stable
    /// prefix of a conversation, mark the last node before the volatile turn. Set
    /// several marks for multiple breakpoints (e.g. tools vs system vs context).
    pub fn cache_breakpoint(&self) -> ChatNode {
        self.with_node_mut(&self.id, |n| n.cache_breakpoint = true);
        self.clone()
    }

    /// Whether this node is currently marked as a cache breakpoint.
    pub fn is_cache_breakpoint(&self) -> bool {
        self.with_node(&self.id, |n| n.cache_breakpoint)
    }

    /// Clear the cache breakpoint mark on THIS node.
    pub fn clear_cache_breakpoint(&self) {
        self.with_node_mut(&self.id, |n| n.cache_breakpoint = false);
    }

    /// Clear every cache breakpoint mark in the WHOLE tree this node belongs to.
    /// Useful to reset caching intent before re-marking, or to stop caching a
    /// conversation entirely.
    pub fn clear_all_cache_breakpoints(&self) {
        let mut nodes = self.tree.nodes.write().unwrap();
        for node in nodes.values_mut() {
            node.cache_breakpoint = false;
        }
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
        self.with_node_mut(&self.id, |n| {
            n.format_kwargs.insert(key.to_string(), value.to_string());
        });
    }

    /// Set multiple format kwargs at once
    pub fn set_format_kwargs(&self, kwargs: &std::collections::HashMap<String, String>) {
        self.with_node_mut(&self.id, |n| {
            for (k, v) in kwargs {
                n.format_kwargs.insert(k.clone(), v.clone());
            }
        });
    }

    /// Get a format kwarg value
    pub fn get_format_kwarg(&self, key: &str) -> Option<String> {
        self.with_node(&self.id, |n| n.format_kwargs.get(key).cloned())
    }

    /// Get all format kwargs for this node
    pub fn get_format_kwargs(&self) -> std::collections::HashMap<String, String> {
        self.with_node(&self.id, |n| n.format_kwargs.clone())
    }

    /// Get the formatted text content of this node's message.
    ///
    /// Applies only this node's own format_kwargs (node-level kwargs are scoped
    /// to the node they are set on; they do not bleed to ancestors or descendants).
    pub fn formatted_text(&self) -> Option<String> {
        let text = self.message.content.get_text()?;
        Some(apply_kwargs(text, &self.get_format_kwargs()))
    }

    /// Substitute `{placeholder}`s in a template using only this node's own kwargs.
    pub fn format_string(&self, template: &str) -> String {
        apply_kwargs(template, &self.get_format_kwargs())
    }

    /// Get the thread with format_kwargs applied to each message's text.
    pub fn formatted_thread(&self) -> Vec<Message> {
        self.formatted_thread_with_base(&std::collections::HashMap::new())
    }

    /// Like [`formatted_thread`](Self::formatted_thread) but with `base`
    /// (completion-level kwargs) applied thread-wide to every message.
    ///
    /// Resolution is per-message: each node's message is formatted with `base`
    /// overlaid by that node's *own* node-level kwargs (which win on collision).
    /// A node-level kwarg therefore only affects its own message, never the
    /// ancestors' or descendants' text.
    pub fn formatted_thread_with_base(
        &self,
        base: &std::collections::HashMap<String, String>,
    ) -> Vec<Message> {
        self.node_path()
            .into_iter()
            .map(|node| {
                // base, then this node's own kwargs on top.
                let mut kwargs = base.clone();
                for (k, v) in node.get_format_kwargs() {
                    kwargs.insert(k, v);
                }

                let mut msg = node.message.clone();
                // Stamp the cache breakpoint from live node state (read fresh from
                // the arena, like format_kwargs), so a mark set after the handle
                // was cloned still applies.
                msg.cache_breakpoint = node.with_node(&node.id, |n| n.cache_breakpoint);
                msg.content = match msg.content {
                    // Substitute text content; preserve multimodal parts untouched.
                    MessageContent::Text(text) => {
                        MessageContent::Text(apply_kwargs(&text, &kwargs))
                    }
                    MessageContent::Parts(parts) => MessageContent::Parts(
                        parts
                            .into_iter()
                            .map(|part| match part.as_text() {
                                Some(text) => ContentPart::text(apply_kwargs(text, &kwargs)),
                                None => part,
                            })
                            .collect(),
                    ),
                };
                msg
            })
            .collect()
    }

    /// The path of nodes from the root down to this node (inclusive).
    /// Iterative (walks the parent chain) so a deep thread can't overflow the stack.
    fn node_path(&self) -> Vec<ChatNode> {
        let mut path = vec![self.clone()];
        let mut node = self.parent();
        while let Some(n) = node {
            node = n.parent();
            path.push(n);
        }
        path.reverse();
        path
    }

    // =========================================================================
    // Completion methods
    // =========================================================================

    /// Complete the conversation at this node (non-streaming).
    ///
    /// Returns the new assistant node. For the typed response (used internally by
    /// the tracked variants and by streaming collect), see [`Self::complete_collect`].
    pub async fn complete(
        &self,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<ChatNode> {
        Ok(self.complete_collect(generator, params).await?.0)
    }

    /// Complete (non-streaming) and return both the new node and the typed
    /// response, so cost can flow as typed `Usage` rather than through a lossy
    /// JSON metadata round-trip. This is the single non-streaming request path.
    pub async fn complete_collect(
        &self,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<(ChatNode, CompletionResponse)> {
        let settings = CompletionSettings::from_params(params);
        self.run_with_retry(generator, &settings, ResponseMode::NonStreaming)
            .await
    }

    /// Warm the prompt cache for this node's prefix WITHOUT generating a real
    /// completion: fire a `max_tokens: 0` request with the prefix marked for
    /// caching. The provider decides what this means (Anthropic writes/refreshes
    /// the cache entry and returns immediately with no output; OpenAI auto-caches,
    /// so this just primes its cache). Returns the [`crate::CostInfo`] of the warm request
    /// so the caller can see what priming cost.
    ///
    /// Semantics that make this cheap to call repeatedly (e.g. before every agent
    /// run): if the cache is COLD it pays the one-time `cache_write` premium (which
    /// you would pay on the next real call anyway); if it is already WARM, the
    /// request is a cheap `cache_read` that also refreshes the cache TTL for free.
    /// Either way the next real completion is a guaranteed cheap cache hit. There
    /// is no way to make the first write free on any provider; the point is you
    /// never pay it twice.
    ///
    /// `params` is honored for cache-relevant settings (system prompt, format
    /// kwargs, token price, cost tracking); `max_tokens`/`use_cache` are forced.
    pub async fn ensure_cached(
        &self,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<crate::provider::CostInfo> {
        // Force the cache mark on the prefix and a zero-output warm request.
        let mut warm = params.cloned().unwrap_or_default();
        warm.use_cache = true;
        let mut completion_params = warm.params.unwrap_or_default();
        completion_params.max_tokens = Some(0);
        warm.params = Some(completion_params);

        let settings = CompletionSettings::from_params(Some(&warm));
        let messages = self.prepare_messages(&settings);
        let completion_params = settings.merged_params(generator);

        // Track usage so we get real cache_write/cache_read counts back.
        let response = global_client()
            .complete_with_usage_tracking(
                generator,
                &messages,
                &completion_params,
                true,
                settings.timeout,
            )
            .await?;

        // Price the warm request through the shared cost decision, and fire the
        // user callback if one was set (so warming cost is accounted like any other).
        let price = settings.price(generator);
        let outcome = match &response.usage {
            Some(usage) => generator.provider.cost_of(usage.clone(), price),
            None => crate::provider::CostOutcome::unknown(),
        };
        let info = outcome.into_cost_info(response.model.clone(), response.id.clone());
        if let Some(cb) = &settings.cost_callback {
            cb(info.clone());
        }
        Ok(info)
    }

    /// Fetch one raw completion response, either non-streaming or by draining a
    /// stream. The single point where the two transports diverge; everything
    /// after (post-processing, node construction, cost, retry policy) is shared.
    async fn fetch_response(
        &self,
        generator: &GeneratorInfo,
        settings: &CompletionSettings,
        mode: ResponseMode,
    ) -> Result<CompletionResponse> {
        match mode {
            ResponseMode::NonStreaming => {
                let messages = self.prepare_messages(settings);
                let completion_params = settings.merged_params(generator);
                global_client()
                    .complete_with_usage_tracking(
                        generator,
                        &messages,
                        &completion_params,
                        settings.track_cost,
                        settings.timeout,
                    )
                    .await
            }
            ResponseMode::Streaming => {
                self.start_streaming(generator, settings)
                    .await?
                    .collect()
                    .await
            }
        }
    }

    /// The single completion retry loop, shared by streaming and non-streaming.
    ///
    /// Applies backoff between attempts and retries on transient transport errors
    /// (per [`is_retryable`]) and on content rejection (`crash_on_empty` /
    /// `crash_on_refusal`, the user-opted-in "retry until the model complies"
    /// behavior). Non-retryable provider errors (4xx auth/validation) fail
    /// immediately so paid calls are never re-issued pointlessly.
    async fn run_with_retry(
        &self,
        generator: &GeneratorInfo,
        settings: &CompletionSettings,
        mode: ResponseMode,
    ) -> Result<(ChatNode, CompletionResponse)> {
        let mut last_error: Option<MiniLLMError> = None;
        let mut current_back_off = settings.back_off_time;

        for attempt in 0..=settings.retry {
            if attempt > 0 {
                let sleep_time = if settings.exp_back_off {
                    current_back_off.min(settings.max_back_off)
                } else {
                    settings.back_off_time
                };
                tokio::time::sleep(Duration::from_secs_f64(sleep_time)).await;
                if settings.exp_back_off {
                    current_back_off *= 2.0;
                }
                tracing::debug!(attempt = attempt, "Retrying completion request");
            }

            let response = match self.fetch_response(generator, settings, mode).await {
                Ok(r) => r,
                Err(e) => {
                    // Non-retryable provider errors (4xx auth/validation) will never
                    // succeed on retry, so fail immediately rather than burn paid calls.
                    if !is_retryable(&e) {
                        return Err(e);
                    }
                    last_error = Some(e);
                    continue;
                }
            };

            let content = match self.postprocess_content(&response.content, settings) {
                Ok(c) => c,
                Err(e) => {
                    last_error = Some(e);
                    continue;
                }
            };

            let node = self.build_assistant_node(content, &response, settings.add_child);
            settings.fire_cost_callback(generator, &response);
            return Ok((node, response));
        }

        Err(MiniLLMError::MaxRetriesExceeded(Box::new(
            last_error.unwrap_or(MiniLLMError::EmptyResponse),
        )))
    }

    /// Build the outgoing message list: formatted thread, contiguous merge,
    /// optional system-prompt prepend, optional force_prepend continuation.
    /// Shared by every completion path so message construction lives in one place.
    fn prepare_messages(&self, settings: &CompletionSettings) -> Vec<Message> {
        let mut messages =
            merge_contiguous_messages(self.formatted_thread_with_base(&settings.format_kwargs));

        if let Some(system) = &settings.system_prompt {
            if messages.first().map(|m| m.role) != Some(Role::System) {
                messages.insert(0, Message::system(system.clone()));
            }
        }

        // force_prepend: prime the assistant turn so the model continues from it.
        if let Some(prepend) = &settings.force_prepend {
            messages.push(Message::assistant(prepend.clone()));
        }

        // use_cache: auto-mark the whole prompt prefix as a cache breakpoint (the
        // last message), in addition to any explicit per-node marks. The provider
        // decides the wire (Anthropic caches it; OpenAI auto-caches regardless).
        if settings.use_cache {
            if let Some(last) = messages.last_mut() {
                last.cache_breakpoint = true;
            }
        }

        messages
    }

    /// Post-process raw response content: re-attach force_prepend, enforce
    /// crash_on_empty, and repair/validate JSON. The single place these rules
    /// live, so streaming and non-streaming behave identically.
    fn postprocess_content(&self, raw: &str, settings: &CompletionSettings) -> Result<String> {
        let mut content = raw.to_string();

        // OpenAI-wire providers return only the continuation, not the prefill we
        // sent, so re-attach it. The starts_with guard handles the rare provider
        // that does echo the prefill, keeping this idempotent.
        if let Some(prepend) = &settings.force_prepend {
            if !content.starts_with(prepend) {
                content = format!("{}{}", prepend, content);
            }
        }

        if settings.crash_on_empty && content.trim().is_empty() {
            return Err(MiniLLMError::EmptyResponse);
        }

        if settings.parse_json {
            content = repair_and_validate_json(&content, settings.crash_on_refusal)?;
        }

        Ok(content)
    }

    /// Build the assistant node from finished content plus the typed response,
    /// threading tool_calls and recording response metadata. The single node
    /// constructor for all completion paths, so tool_calls/finish_reason can
    /// never be dropped by one variant and kept by another.
    fn build_assistant_node(
        &self,
        content: String,
        response: &CompletionResponse,
        add_child: bool,
    ) -> ChatNode {
        let mut message = Message::assistant(content);
        message.tool_calls = response.tool_calls.clone();

        // A real child is registered in this node's child list; a phantom lives in
        // the same tree with this node as its parent but is NOT listed (so it reads
        // its `thread()` yet leaves the shared tree's branches untouched).
        let node = if add_child {
            self.insert_child(&self.id, message)
        } else {
            self.insert_phantom_child(&self.id, message)
        };

        node.set_metadata("response_id", serde_json::json!(response.id));
        node.set_metadata("model", serde_json::json!(response.model));
        if let Some(usage) = &response.usage {
            node.set_metadata("usage", serde_json::json!(usage));
        }
        if let Some(finish_reason) = &response.finish_reason {
            node.set_metadata("finish_reason", serde_json::json!(finish_reason));
        }
        node
    }

    /// Insert a node into this tree with `parent_id` as its parent but WITHOUT
    /// registering it in the parent's child list (a "phantom"). It reads its
    /// `thread()` (the parent chain is intact) but no traversal from the parent
    /// finds it, so the shared tree's branch structure is untouched.
    fn insert_phantom_child(&self, parent_id: &str, message: Message) -> ChatNode {
        let id = Uuid::new_v4().to_string();
        // Born refcount 1 for the returned handle, parent set but NOT registered in
        // the parent's child list. Instead we bump the parent's
        // `phantom_child_count` so the parent (and its whole ancestor chain) stays
        // alive while the phantom descends from it, exactly as a registered child
        // would keep it alive. Without this the parent could be reclaimed out from
        // under the held phantom, dangling its parent id. When the phantom's handle
        // drops, `reclaim_dead` decrements that count (detecting the phantom because
        // it is not in the parent's `children`), so the speculative node frees
        // itself and the tree is untouched.
        {
            let mut nodes = self.tree.nodes.write().unwrap();
            nodes.insert(
                id.clone(),
                NodeData::new(message.clone(), Some(parent_id.to_string())),
            );
            nodes
                .get_mut(parent_id)
                .expect("parent id present")
                .phantom_child_count += 1;
        }
        self.handle_owned(id, message)
    }

    /// Complete with streaming. The returned stream applies none of the
    /// post-processing params (force_prepend, parse_json, crash_on_*); those
    /// require the full response and are applied by [`Self::complete_streaming_collect`].
    /// Use this only when consuming chunks directly.
    pub async fn complete_streaming(
        &self,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<StreamingCompletion> {
        let settings = CompletionSettings::from_params(params);
        self.start_streaming(generator, &settings).await
    }

    /// Start a stream from already-derived settings. Lets the collect wrappers
    /// build `CompletionSettings` once and reuse it for both the request and the
    /// post-processing, instead of re-deriving (and re-cloning) it per hop.
    async fn start_streaming(
        &self,
        generator: &GeneratorInfo,
        settings: &CompletionSettings,
    ) -> Result<StreamingCompletion> {
        let messages = self.prepare_messages(settings);
        let completion_params = settings.merged_params(generator);

        global_client()
            .complete_streaming_with_usage(
                generator,
                &messages,
                &completion_params,
                settings.track_cost,
                settings.timeout,
            )
            .await
    }

    /// Append the assistant node for a response you consumed manually, e.g. a
    /// streaming loop over [`StreamingCompletion::next_chunk`] followed by
    /// [`StreamingCompletion::collect`]. Threads content, tool_calls, and
    /// response metadata exactly like the built-in completion paths, so a
    /// hand-driven stream ends in the same tree shape as `complete`.
    pub fn append_response(&self, response: &CompletionResponse) -> ChatNode {
        self.build_assistant_node(response.content.clone(), response, true)
    }

    /// Complete with streaming and collect into a new node.
    ///
    /// Shares the retry/post-processing/cost pipeline with the non-streaming
    /// path, so `crash_on_empty`/`crash_on_refusal` + `retry` behave identically
    /// in both (a rejected streamed completion is retried the same way).
    pub async fn complete_streaming_collect(
        &self,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<ChatNode> {
        let settings = CompletionSettings::from_params(params);
        let (node, _response) = self
            .run_with_retry(generator, &settings, ResponseMode::Streaming)
            .await?;
        Ok(node)
    }

    // =========================================================================
    // Tracked completion methods (enforced cost tracking via CompletionContext)
    // =========================================================================

    /// Complete with enforced cost tracking via a [`crate::CompletionContext`].
    ///
    /// One of two delivery shapes for the SAME accounting: this one pushes the
    /// cost into the sink the context registered, for callers that route every
    /// completion's cost to one place regardless of call site. When the caller
    /// itself acts on the bill, use [`complete_costed`](Self::complete_costed),
    /// which returns it with the result instead.
    ///
    /// Always enables usage tracking and reports cost on every completion through
    /// the context's callback. How cost is determined is the generator's provider
    /// accounting: from the response's usage when present, otherwise via the
    /// provider's out-of-band resolution (e.g. OpenRouter's `/generation` query;
    /// providers without one report `Unknown`). An undeterminable cost is
    /// reported `Unknown`/`Unpriced`, never silently counted as a free request.
    pub async fn complete_tracked(
        &self,
        ctx: &crate::tracking::CompletionContext,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<ChatNode> {
        let tracked_params = tracked_params(params);
        let (node, response) = self
            .complete_collect(&ctx.generator, Some(&tracked_params))
            .await?;

        // Cost flows from the typed response via the one shared decision: the
        // provider's `cost_of` aggregates native cost (or prices tokens), and the
        // backoff out-of-band query covers absent usage, reporting Unknown rather
        // than a fake $0 if unresolvable.
        ctx.report_cost(ctx.cost_for_response(&response).await)
            .await;
        Ok(node)
    }

    /// Complete and hand back what it cost, no callback ceremony.
    ///
    /// One of two delivery shapes for the SAME accounting (usage from the
    /// response, the provider's out-of-band resolution as the backstop, never a
    /// fake $0): this one returns the [`CostInfo`](crate::CostInfo) WITH the
    /// result, for the caller that acts on the bill itself. When many call
    /// sites should feed one central sink instead, use
    /// [`complete_tracked`](Self::complete_tracked); streaming always goes
    /// through the tracked shape, since a stream's cost resolves only after it
    /// ends. An errored completion carries no cost info: the request failed
    /// before a billable response existed.
    pub async fn complete_costed(
        &self,
        generator: &GeneratorInfo,
        params: Option<&NodeCompletionParameters>,
    ) -> (Result<ChatNode>, Option<crate::CostInfo>) {
        let tracked_params = tracked_params(params);
        match self.complete_collect(generator, Some(&tracked_params)).await {
            Ok((node, response)) => {
                let info = crate::tracking::cost_for_response(generator, &response).await;
                (Ok(node), Some(info))
            }
            Err(e) => (Err(e), None),
        }
    }

    /// Complete streaming with enforced cost tracking.
    ///
    /// Returns a TrackedStream that will automatically report costs when
    /// the stream finishes or is dropped (cancelled).
    pub async fn complete_streaming_tracked(
        &self,
        ctx: &crate::tracking::CompletionContext,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<crate::tracking::TrackedStream> {
        let tracked_params = tracked_streaming_params(params);
        let stream = self
            .complete_streaming(&ctx.generator, Some(&tracked_params))
            .await?;

        Ok(crate::tracking::TrackedStream::new(stream, ctx))
    }

    /// Complete streaming, collect all chunks, and report cost. Tracked variant.
    pub async fn complete_streaming_collect_tracked(
        &self,
        ctx: &crate::tracking::CompletionContext,
        params: Option<&NodeCompletionParameters>,
    ) -> Result<ChatNode> {
        // Mirror complete_tracked exactly, only the transport differs: the shared
        // retry loop drains the stream, post-processes, and builds the node;
        // run_with_retry returns Ok only for an accepted completion, so cost is
        // never booked for a rejected one. Then report cost from the typed
        // response (same enforced path as the non-streaming tracked variant).
        let tracked_params = tracked_streaming_params(params);
        let settings = CompletionSettings::from_params(Some(&tracked_params));
        let (node, response) = self
            .run_with_retry(&ctx.generator, &settings, ResponseMode::Streaming)
            .await?;

        ctx.report_cost(ctx.cost_for_response(&response).await)
            .await;
        Ok(node)
    }

    // =========================================================================
    // Convenience methods for common patterns
    // =========================================================================

    /// Send a user message and get a completion
    pub async fn chat(
        &self,
        user_message: impl Into<MessageContent>,
        generator: &GeneratorInfo,
    ) -> Result<ChatNode> {
        let user_node = self.add_user(user_message);
        user_node.complete(generator, None).await
    }

    /// Send a user message and get a streaming completion
    pub async fn chat_streaming(
        &self,
        user_message: impl Into<MessageContent>,
        generator: &GeneratorInfo,
    ) -> Result<(ChatNode, StreamingCompletion)> {
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
pub fn pretty_messages(node: &ChatNode, config: Option<&PrettyPrintConfig>) -> String {
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
        // Use all_text so a multimodal message's later text parts aren't dropped
        // from the human-facing view.
        let text = msg.content.all_text();
        if text.is_empty() && msg.content.has_multimodal() {
            result.push_str("[multimodal content]");
        } else {
            result.push_str(&text);
        }
    }

    result
}

/// Pretty print messages as a formatted string (convenience function)
pub fn format_conversation(node: &ChatNode) -> String {
    pretty_messages(node, None)
}

// =========================================================================
// Builder pattern for creating conversations
// =========================================================================

/// Builder for creating conversation trees
pub struct ConversationBuilder {
    root: ChatNode,
    current: ChatNode,
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
    pub fn root(&self) -> ChatNode {
        self.root.clone()
    }

    /// Get the current (last) node
    pub fn current(&self) -> ChatNode {
        self.current.clone()
    }

    /// Build and return the current node
    pub fn build(self) -> ChatNode {
        self.current
    }
}

// =========================================================================
// Thread Serialization/Deserialization
// =========================================================================

use serde::{Deserialize, Serialize};

/// One node of a serialized thread: its message plus the node-level format
/// kwargs that belong to that node. `Message` serializes every modality (text,
/// images, audio, video, tool calls) via its own serde derives, and the kwargs
/// ride alongside, so save/load is a faithful per-node round-trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadNode {
    /// The message at this node.
    pub message: Message,

    /// This node's own format kwargs (scoped to this node's text).
    #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
    pub format_kwargs: std::collections::HashMap<String, String>,
}

/// Serializable representation of a thread (for saving/loading), root to leaf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadData {
    /// The nodes in the thread, root to leaf.
    pub prompts: Vec<ThreadNode>,
}

impl ChatNode {
    /// Save the thread (path from root to this node) to a JSON file
    pub fn save_thread(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(&self.to_thread_data())?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Convert the thread to serializable ThreadData, capturing each node's own
    /// message and node-level kwargs.
    pub fn to_thread_data(&self) -> ThreadData {
        ThreadData {
            prompts: self
                .node_path()
                .into_iter()
                .map(|node| ThreadNode {
                    message: node.message.clone(),
                    format_kwargs: node.get_format_kwargs(),
                })
                .collect(),
        }
    }

    /// Load a thread from a JSON file.
    ///
    /// Returns (root_node, leaf_node); either handle keeps the whole thread alive
    /// (any handle into a tree keeps the entire tree alive).
    pub fn from_thread_file(path: &str) -> Result<(ChatNode, ChatNode)> {
        Self::from_thread_json(&std::fs::read_to_string(path)?)
    }

    /// Load a thread from a JSON string.
    ///
    /// Returns (root_node, leaf_node); either handle keeps the whole thread alive.
    pub fn from_thread_json(json: &str) -> Result<(ChatNode, ChatNode)> {
        Self::from_thread_data(&serde_json::from_str(json)?)
    }

    /// Load a thread from ThreadData, restoring each node's own kwargs.
    ///
    /// Returns (root_node, leaf_node); either handle keeps the whole thread alive
    /// (any handle into a tree keeps the entire tree alive).
    pub fn from_thread_data(data: &ThreadData) -> Result<(ChatNode, ChatNode)> {
        Self::build_chain(
            data.prompts
                .iter()
                .map(|e| (e.message.clone(), e.format_kwargs.clone())),
        )
    }

    /// Load a thread from a list of messages, building a linear chain.
    ///
    /// Returns (root_node, leaf_node); either handle keeps the whole thread alive.
    pub fn from_messages(messages: &[Message]) -> Result<(ChatNode, ChatNode)> {
        Self::build_chain(
            messages
                .iter()
                .map(|m| (m.clone(), std::collections::HashMap::new())),
        )
    }

    /// Build a linear root→leaf chain from an iterator of (message, node kwargs).
    /// The single chain-construction path shared by every thread loader.
    fn build_chain(
        nodes: impl IntoIterator<Item = (Message, std::collections::HashMap<String, String>)>,
    ) -> Result<(ChatNode, ChatNode)> {
        let mut iter = nodes.into_iter();
        let (first_msg, first_kwargs) = iter.next().ok_or(MiniLLMError::EmptyThread)?;

        let root = ChatNode::new(first_msg);
        root.set_format_kwargs(&first_kwargs);
        let mut current = root.clone();
        for (msg, kwargs) in iter {
            // Fresh node inserted into the same tree as a child of `current`.
            current = current.insert_child(&current.id, msg);
            current.set_format_kwargs(&kwargs);
        }
        Ok((root, current))
    }
}

#[cfg(test)]
mod completion_pipeline_tests {
    use super::*;
    use crate::provider::CompletionResponse;

    fn response_with(content: &str) -> CompletionResponse {
        CompletionResponse::new("gen-1", "test-model", content)
    }

    // ---- is_retryable ----------------------------------------------------

    #[test]
    fn retryable_classification() {
        assert!(is_retryable(&MiniLLMError::Api {
            status: 429,
            message: "rate".into()
        }));
        assert!(is_retryable(&MiniLLMError::Api {
            status: 503,
            message: "down".into()
        }));
        assert!(is_retryable(&MiniLLMError::Timeout));
        assert!(is_retryable(&MiniLLMError::Stream("boom".into())));
        // 4xx auth/validation must NOT burn paid retries.
        assert!(!is_retryable(&MiniLLMError::Api {
            status: 401,
            message: "bad key".into()
        }));
        assert!(!is_retryable(&MiniLLMError::Api {
            status: 400,
            message: "bad req".into()
        }));
        assert!(!is_retryable(&MiniLLMError::EmptyResponse));
    }

    // ---- repair_and_validate_json / json_value_is_empty ------------------

    #[test]
    fn json_value_emptiness() {
        use crate::json_repair::{loads, RepairOptions};
        let opts = RepairOptions::default();
        assert!(json_value_is_empty(&loads("null", &opts).unwrap()));
        assert!(json_value_is_empty(&loads("\"\"", &opts).unwrap()));
        assert!(json_value_is_empty(&loads("{}", &opts).unwrap()));
        assert!(json_value_is_empty(&loads("[]", &opts).unwrap()));
        assert!(!json_value_is_empty(&loads("{\"a\":1}", &opts).unwrap()));
        assert!(!json_value_is_empty(&loads("[1]", &opts).unwrap()));
    }

    #[test]
    fn repair_validate_crash_on_refusal_rejects_empty_value() {
        // A refusal that contains a brace but yields an empty object must be
        // rejected on the PARSED value, not slip through a substring check.
        let err = repair_and_validate_json("I can't help. {}", true).unwrap_err();
        assert!(matches!(err, MiniLLMError::NoJsonFound(_)));
    }

    #[test]
    fn repair_validate_accepts_real_json_and_repairs() {
        let out = repair_and_validate_json("{'a': 1,}", true).unwrap();
        assert_eq!(out, r#"{"a": 1}"#);
    }

    #[test]
    fn repair_validate_no_crash_when_flag_off() {
        // Without crash_on_refusal, even an empty repair is returned, not errored.
        assert!(repair_and_validate_json("no json here", false).is_ok());
    }

    // ---- postprocess_content --------------------------------------------

    #[test]
    fn postprocess_reattaches_force_prepend_once() {
        let node = ChatNode::root("sys");
        let params = NodeCompletionParameters::new().with_force_prepend("Score: ");
        let settings = CompletionSettings::from_params(Some(&params));
        // Provider returned only the continuation.
        assert_eq!(
            node.postprocess_content("8/10", &settings).unwrap(),
            "Score: 8/10"
        );
        // Provider echoed the prefill, not doubled.
        assert_eq!(
            node.postprocess_content("Score: 8/10", &settings).unwrap(),
            "Score: 8/10"
        );
    }

    #[test]
    fn postprocess_crash_on_empty() {
        let node = ChatNode::root("sys");
        let params = NodeCompletionParameters::new().with_crash_on_empty(true);
        let settings = CompletionSettings::from_params(Some(&params));
        assert!(matches!(
            node.postprocess_content("   \n ", &settings).unwrap_err(),
            MiniLLMError::EmptyResponse
        ));
        // Non-empty passes.
        assert_eq!(node.postprocess_content("hi", &settings).unwrap(), "hi");
    }

    #[test]
    fn postprocess_parse_json_repairs() {
        let node = ChatNode::root("sys");
        let params = NodeCompletionParameters::new().with_parse_json(true);
        let settings = CompletionSettings::from_params(Some(&params));
        assert_eq!(
            node.postprocess_content("{'a': 1,}", &settings).unwrap(),
            r#"{"a": 1}"#
        );
    }

    #[test]
    fn postprocess_default_is_passthrough() {
        let node = ChatNode::root("sys");
        let settings = CompletionSettings::from_params(None);
        assert_eq!(
            node.postprocess_content("plain text", &settings).unwrap(),
            "plain text"
        );
    }

    // ---- build_assistant_node: real child vs phantom --------------------

    #[test]
    fn build_node_as_real_child_threads_tool_calls_and_metadata() {
        let root = ChatNode::root("sys");
        let user = root.add_user("hi");

        let mut response = response_with("answer");
        response.finish_reason = Some("tool_calls".into());
        response.tool_calls = Some(vec![crate::tools::ToolCall::new("c1", "t", "{}")]);
        let node = user.build_assistant_node("answer".into(), &response, true);

        // Real child: parent lists it.
        assert_eq!(user.child_count(), 1);
        assert_eq!(node.parent().unwrap().id, user.id);
        // tool_calls + finish_reason threaded.
        assert!(node.message.tool_calls.is_some());
        assert_eq!(
            node.get_metadata("finish_reason"),
            Some(serde_json::json!("tool_calls"))
        );
        assert_eq!(
            node.get_metadata("model"),
            Some(serde_json::json!("test-model"))
        );

        // The typed accessor exposes the calls, and a tool result answers one:
        // the result node carries role=tool + the call id in its message.
        let calls = node.tool_calls().expect("accessor exposes tool calls");
        assert_eq!(calls[0].id, "c1");
        let result = node.add_tool_result(&calls[0].id, "42");
        assert_eq!(result.role(), Role::Tool);
        assert_eq!(result.message.tool_call_id.as_deref(), Some("c1"));
        assert_eq!(result.text(), Some("42"));
    }

    #[test]
    fn append_response_builds_the_same_node_as_the_completion_paths() {
        // The public escape hatch for hand-driven streams must produce the same
        // tree shape as `complete`: a real child carrying content, tool_calls,
        // and response metadata.
        let root = ChatNode::root("sys");
        let user = root.add_user("hi");
        let mut response = response_with("answer");
        response.tool_calls = Some(vec![crate::tools::ToolCall::new("c1", "t", "{}")]);

        let node = user.append_response(&response);
        assert_eq!(user.child_count(), 1);
        assert_eq!(node.text(), Some("answer"));
        assert_eq!(node.tool_calls().unwrap()[0].id, "c1");
        assert_eq!(
            node.get_metadata("model"),
            Some(serde_json::json!("test-model"))
        );
    }

    #[test]
    fn build_node_as_phantom_does_not_register_in_parent() {
        let root = ChatNode::root("sys");
        let user = root.add_user("hi");

        let response = response_with("answer");
        let phantom = user.build_assistant_node("answer".into(), &response, false);

        // Phantom: parent does NOT list it...
        assert_eq!(user.child_count(), 0);
        // ...but the phantom knows its parent and can read the full thread.
        assert_eq!(phantom.parent().unwrap().id, user.id);
        let thread = phantom.thread();
        assert_eq!(thread.len(), 3); // sys, hi, answer
        assert_eq!(thread[2].text(), Some("answer"));
        // And the phantom resolves the real root.
        assert_eq!(phantom.get_root().id, root.id);
    }

    // ---- clone_tree ------------------------------------------------------

    #[test]
    fn clone_tree_copies_the_whole_tree_isolated() {
        let root = ChatNode::root("sys");
        let a = root.add_user("a");
        a.set_format_kwarg("k", "v");
        let _b = root.add_user("b"); // a sibling branch (handle dropped)

        let cloned_a = a.clone_tree();
        // Fresh id, same content + kwargs.
        assert_ne!(cloned_a.id, a.id);
        assert_eq!(cloned_a.text(), Some("a"));
        assert_eq!(cloned_a.get_format_kwarg("k"), Some("v".to_string()));

        // The WHOLE tree is cloned: ancestor spine AND the sibling branch.
        let cloned_root = cloned_a.get_root();
        assert_ne!(cloned_root.id, root.id);
        assert_eq!(cloned_root.text(), Some("sys"));
        assert_eq!(
            cloned_root.node_count(),
            3,
            "clone has root + both branches"
        );
        assert_eq!(cloned_a.thread().len(), 2);

        // Isolation: mutating/extending the clone never touches the original.
        cloned_a.set_format_kwarg("k", "changed");
        assert_eq!(a.get_format_kwarg("k"), Some("v".to_string()));
        cloned_a.add_assistant("reply");
        assert!(
            a.is_leaf(),
            "extending the clone must not touch the original"
        );
    }

    #[test]
    fn clone_tree_works_on_a_phantom_node() {
        // A phantom (parent set, not in parent's children) must clone without
        // panicking, preserving its history.
        let root = ChatNode::root("sys");
        let user = root.add_user("u");
        let phantom = user.build_assistant_node(
            "answer".into(),
            &crate::provider::CompletionResponse::new("g", "m", "answer"),
            false, // add_child = false -> phantom
        );
        assert_eq!(
            user.child_count(),
            0,
            "precondition: phantom not registered"
        );

        let cloned = phantom.clone_tree();
        assert_eq!(cloned.text(), Some("answer"));
        assert_eq!(cloned.thread().len(), 3); // sys, u, answer
        assert_ne!(cloned.id, phantom.id);
    }

    // ---- ownership invariants (arena: any handle keeps the whole tree) ----

    #[test]
    fn holding_a_node_keeps_its_full_ancestor_history_alive() {
        // The core contract: hold any node, never the root, and its whole history
        // stays alive. Drop every ancestor handle; the leaf must still resolve its
        // full thread and its true root.
        let leaf = {
            let root = ChatNode::root("sys");
            let user = root.add_user("hi");
            user.add_assistant("there")
            // root and user handles dropped here.
        };
        assert_eq!(leaf.thread().len(), 3);
        assert_eq!(leaf.get_root().text(), Some("sys"));
        assert_eq!(leaf.thread()[0].text(), Some("sys"));
    }

    #[test]
    fn held_node_keeps_its_ancestor_chain_but_unheld_branches_are_reclaimed() {
        // Refcounting contract (Decision A): holding a node keeps that node and its
        // ANCESTOR chain alive (ancestors retain it as a child). A sibling/alternate
        // branch whose handle is dropped is RECLAIMED, so the arena does not grow
        // with dead branches.
        let root = ChatNode::root("sys");
        let kept = root.add_user("kept");
        root.add_user("dropped"); // returned handle dropped immediately → reclaimed
        root.add_user("alsodropped"); // another, dropped → reclaimed

        // root still held here: its arena currently has sys + kept (dropped two
        // siblings already reclaimed on handle drop).
        assert_eq!(
            root.arena_len(),
            2,
            "unheld sibling branches were reclaimed"
        );
        assert_eq!(root.child_count(), 1);

        // Drop the root handle; only `kept` (a leaf) remains held. Its ancestor
        // chain (root → kept) survives because `kept` retains the root.
        drop(root);
        let root = kept.get_root();
        assert_eq!(root.text(), Some("sys"));
        assert_eq!(root.node_count(), 2); // sys + kept
        assert_eq!(kept.arena_len(), 2);
    }

    #[test]
    fn dropping_every_handle_frees_the_tree() {
        // When the LAST handle is dropped, the arena frees. The Arc<Tree> strong
        // count proves it: a held handle keeps the tree alive; dropping all handles
        // brings strong_count back to our explicit clone alone (no lingering cycle).
        let leaf = {
            let root = ChatNode::root("sys");
            let u = root.add_user("u");
            u.add_assistant("a")
        };
        assert_eq!(leaf.get_root().node_count(), 3);
        let tree = leaf.tree.clone();
        assert_eq!(Arc::strong_count(&tree), 2); // `leaf` + our `tree` clone
        drop(leaf);
        assert_eq!(Arc::strong_count(&tree), 1);
    }

    #[test]
    fn phantom_node_is_reclaimed_when_its_handle_drops() {
        // Every speculative completion (add_child=false) mints a phantom node. Its
        // handle dropping must reclaim it, not accumulate in the arena.
        let root = ChatNode::root("sys");
        let user = root.add_user("hi");
        let resp = response_with("speculative");
        {
            let phantom = user.build_assistant_node("speculative".into(), &resp, false);
            assert_eq!(
                phantom.thread().len(),
                3,
                "phantom reads its ancestor chain"
            );
            assert_eq!(root.arena_len(), 3, "phantom present while held"); // sys, hi, phantom
        }
        // phantom handle dropped → reclaimed; the real tree (sys, hi) is untouched.
        assert_eq!(root.arena_len(), 2, "phantom reclaimed on drop");
        assert!(user.is_leaf(), "phantom never registered as a child");
    }

    #[test]
    fn held_phantom_keeps_its_parent_chain_alive_when_ancestors_are_dropped() {
        // A phantom walks UP through its parent chain, so holding the phantom must
        // keep that chain alive even after every ancestor HANDLE is dropped, exactly
        // like holding a registered leaf. Otherwise reclaim removes the parent out
        // from under the phantom and the next upward walk aborts the process.
        let phantom = {
            let root = ChatNode::root("sys");
            let user = root.add_user("hi");
            let resp = response_with("speculative");
            user.build_assistant_node("speculative".into(), &resp, false)
            // root and user handles dropped here; only the phantom is held.
        };
        // The phantom's whole ancestor chain survives (kept alive by phantom_child_count).
        assert_eq!(phantom.thread().len(), 3, "sys, hi, phantom");
        assert_eq!(phantom.get_root().text(), Some("sys"));
        assert_eq!(phantom.parent().unwrap().text(), Some("hi"));
        // And clone_tree of the phantom still works (it must not hit a dangling id).
        let cloned = phantom.clone_tree();
        assert_eq!(cloned.thread().len(), 3);
    }

    #[test]
    fn reparenting_a_sibling_out_from_under_a_phantoms_ancestor_keeps_the_phantom_alive() {
        // The exact path this round's `add_child` reclaim widened: a node A whose
        // only REGISTERED child is moved away, but which still has a held phantom
        // descendant, must NOT be reclaimed. Build root -> a -> reg (registered) plus
        // a phantom under `a`; drop `a`'s handle; move `reg` under root. `a` is now
        // registered-childless but still has the phantom → must survive.
        let root = ChatNode::root("sys");
        let a = root.add_user("a");
        let reg = a.add_assistant("reg"); // registered child of a
        let resp = response_with("ph");
        let phantom = a.build_assistant_node("ph".into(), &resp, false); // phantom under a
        drop(a); // a kept alive only by reg (registered) + phantom (phantom_child_count)

        // Move reg from a to root. a loses its only registered child but keeps the phantom.
        root.add_child(reg.clone()).unwrap();

        // a must still be present (the phantom descends from it); the phantom's
        // thread and root must still resolve, no abort.
        assert_eq!(phantom.thread().len(), 3, "sys, a, phantom");
        assert_eq!(phantom.get_root().text(), Some("sys"));
        assert_eq!(phantom.parent().unwrap().text(), Some("a"));

        // Dropping the phantom now reclaims it AND, since a is then fully unheld and
        // childless, cascades a away too (reg moved to root, so a has nothing left).
        drop(phantom);
        // a reclaimed: arena is sys + reg (now under root). reg is still held.
        assert_eq!(reg.get_root().text(), Some("sys"));
        assert!(
            reg.thread().iter().all(|m| m.text() != Some("a")),
            "a gone from reg's thread"
        );
    }

    #[test]
    fn detaching_a_phantom_decrements_its_old_parents_phantom_count() {
        // Cutting a PHANTOM's own parent edge (detach / re-parent) must decrement the
        // old parent's phantom_child_count, or the old parent leaks (never reclaimed)
        // because a stale count pins it forever.
        let root = ChatNode::root("sys");
        let user = root.add_user("hi"); // the phantom's parent
        let resp = response_with("ph");
        let phantom = user.build_assistant_node("ph".into(), &resp, false);
        assert_eq!(root.arena_len(), 3, "sys, hi, phantom");

        // Detach the phantom: it becomes its own root, and `user` must lose its
        // phantom claim so it can be reclaimed once unheld.
        phantom.detach();
        assert!(phantom.is_root(), "detached phantom is a root");

        // Drop `user`'s handle: `user` is now unheld, has no registered children, and
        // (crucially) phantom_child_count back to 0 → it must be reclaimed, leaving
        // only sys (held by root) and the detached phantom (held).
        drop(user);
        assert_eq!(
            root.arena_len(),
            2,
            "old parent reclaimed (no leak): sys + detached phantom"
        );
        assert_eq!(phantom.text(), Some("ph"));
    }

    #[test]
    fn re_parenting_a_phantom_transfers_it_without_leaking_the_old_parent() {
        // Moving a phantom via add_child registers it under the new parent AND clears
        // its claim on the old parent (no leak), symmetric with detach.
        let root = ChatNode::root("sys");
        let a = root.add_user("a");
        let b = root.add_user("b");
        let resp = response_with("ph");
        let phantom = a.build_assistant_node("ph".into(), &resp, false); // phantom under a
        assert_eq!(root.arena_len(), 4, "sys, a, b, phantom");

        // Move the phantom under b: it becomes a REGISTERED child of b, and a loses
        // its phantom claim.
        b.add_child(phantom.clone()).unwrap();
        assert_eq!(phantom.parent().unwrap().id, b.id);
        assert_eq!(b.child_count(), 1, "phantom is now a registered child of b");

        // Drop a's handle: a has no registered children and no phantom claim now → reclaimed.
        drop(a);
        assert_eq!(
            root.arena_len(),
            3,
            "old parent `a` reclaimed (no leak): sys, b, phantom-under-b"
        );
    }

    #[test]
    fn detached_then_dropped_subtree_is_reclaimed() {
        // Detaching a branch and dropping every handle into it reclaims it from the
        // arena (the former parent keeps living because root is held).
        let root = ChatNode::root("sys");
        let a = root.add_user("a");
        let _a_child = a.add_assistant("a-reply");
        root.add_user("b"); // sibling, dropped immediately → reclaimed
        assert_eq!(root.arena_len(), 3); // sys, a, a-reply (b already gone)

        a.detach(); // a becomes a root of its own subtree (still held by `a`/`_a_child`)
        assert_eq!(root.child_count(), 0, "root lost its only child");
        assert_eq!(root.arena_len(), 3, "a's subtree still held, still present");

        drop(_a_child);
        drop(a); // last handles into the detached subtree gone → reclaimed
        assert_eq!(
            root.arena_len(),
            1,
            "detached subtree reclaimed; only sys remains"
        );
        assert_eq!(root.node_count(), 1);
    }

    #[test]
    fn concurrent_reverse_cross_tree_merges_do_not_deadlock() {
        // Two threads merging in opposite directions (A←B and B←A) must not
        // deadlock: `copy_subtree_under` acquires both arena locks in a global
        // pointer-address order, so there is no cyclic wait. We run many rounds to
        // make a lock-ordering bug overwhelmingly likely to surface (and the test
        // would hang, not pass, if it regressed).
        use std::sync::Barrier;
        for _ in 0..200 {
            let a = ChatNode::root("A");
            let b = ChatNode::root("B");
            let a2 = a.clone();
            let b2 = b.clone();
            let barrier = std::sync::Arc::new(Barrier::new(2));
            let (ba, bb) = (barrier.clone(), barrier.clone());
            let t1 = std::thread::spawn(move || {
                ba.wait();
                let _ = a.merge(&b); // A ← B
            });
            let t2 = std::thread::spawn(move || {
                bb.wait();
                let _ = b2.merge(&a2); // B ← A (opposite order)
            });
            t1.join().unwrap();
            t2.join().unwrap();
        }
    }

    #[test]
    fn deep_tree_builds_traverses_and_drops_without_overflow() {
        // A long linear thread must build, resolve its full thread, and tear down
        // without a stack overflow (arena teardown is a flat map drop; the release
        // cascade is iterative).
        let leaf = {
            let root = ChatNode::root("sys");
            let mut cur = root.clone();
            for i in 0..50_000 {
                cur = cur.add_user(format!("m{i}"));
            }
            cur
        };
        assert_eq!(leaf.thread().len(), 50_001);
        drop(leaf); // iterative release cascade up 50k ancestors, no recursion
    }

    #[test]
    fn merge_rejects_same_tree_cycle() {
        // Merging a node onto its own ancestor would form a structural cycle in the
        // arena; it must fail loudly, not corrupt the tree.
        let root = ChatNode::root("sys");
        let user = root.add_user("u");
        // root is an ancestor of user → cycle if attached under user.
        assert!(user.merge(&root).is_err());
        // Merging two SEPARATE trees is fine.
        let other = ChatNode::root("other");
        assert!(user.merge(&other).is_ok());
    }

    #[test]
    fn add_child_rejects_ancestor_and_self_cycle() {
        // The primitive itself guards the cycle (merge relies on it). Attaching an
        // ancestor (or self) as a child must fail loudly, never form a cycle.
        let root = ChatNode::root("sys");
        let user = root.add_user("u");
        assert!(user.add_child(root.clone()).is_err(), "ancestor → cycle");
        assert!(user.add_child(user.clone()).is_err(), "self → cycle");
        // A detached separate subtree is fine.
        let other = ChatNode::root("other");
        assert!(root.add_child(other).is_ok());
    }

    #[test]
    fn same_tree_reparent_reclaims_the_orphaned_old_parent() {
        // Moving a node to a new parent must reclaim its OLD parent if that left it
        // unheld and childless, symmetric with detach. Otherwise the old parent
        // leaks in the arena forever (it was kept alive only by the moved child).
        let root = ChatNode::root("sys");
        let a = root.add_user("a"); // a child of root
        let b = a.add_assistant("b"); // b child of a
        drop(a); // a now refcount 0, kept alive ONLY by its child b
        assert_eq!(root.arena_len(), 3); // sys, a, b

        // Move b from a to root. a is now unheld AND childless → must be reclaimed.
        root.add_child(b.clone()).unwrap();
        assert_eq!(
            root.arena_len(),
            2,
            "orphaned old parent `a` must be reclaimed, not leaked"
        );
        assert_eq!(b.parent().unwrap().id, root.id);
        assert_eq!(root.child_count(), 1);
    }

    #[test]
    fn concurrent_same_tree_reparents_cannot_form_a_cycle() {
        // Two threads re-parenting in opposite directions (X under Y and Y under X)
        // must not commit a parent-chain cycle: the cycle check and the re-link
        // happen under one write lock, so the second mover sees the first's edge and
        // is rejected. A regressed (check-then-mutate non-atomic) version would
        // occasionally commit a 2-cycle X↔Y. That cycle is orphaned from `root`, so
        // we must probe it via handles held INTO X and Y and walk UP the parent
        // chain (an up-walk inside a cycle never terminates); a bounded walk that
        // panics catches the regression loudly instead of hanging on the harness.
        use std::sync::Barrier;
        for _ in 0..200 {
            let root = ChatNode::root("root");
            let x = root.add_user("x");
            let y = root.add_user("y"); // x and y are siblings under root
                                        // Keep handles INTO the cycle candidates in this thread so we can probe
                                        // them after the race (the threads get their own clones).
            let (xa, ya) = (x.clone(), y.clone());
            let (x_mover, y_target) = (x.clone(), y.clone());
            let (y_mover, x_target) = (y.clone(), x.clone());
            drop((x, y));
            let barrier = std::sync::Arc::new(Barrier::new(2));
            let (b1, b2) = (barrier.clone(), barrier.clone());
            let t1 = std::thread::spawn(move || {
                b1.wait();
                let _ = x_mover.add_child(y_target); // Y under X
            });
            let t2 = std::thread::spawn(move || {
                b2.wait();
                let _ = y_mover.add_child(x_target); // X under Y (opposite)
            });
            t1.join().unwrap();
            t2.join().unwrap();

            // Walk up from each node with a hard cap: a parent chain in a sound
            // arena is at most a handful of nodes deep here. If it ever exceeds the
            // cap, a cycle was committed: fail loudly rather than loop forever.
            for node in [&xa, &ya] {
                let mut steps = 0;
                let mut cur = Some(node.clone());
                while let Some(n) = cur {
                    steps += 1;
                    assert!(
                        steps < 100,
                        "parent chain did not terminate: a cycle was committed"
                    );
                    cur = n.parent();
                }
            }
        }
    }

    // ---- node_path / prepare_messages -----------------------------------

    #[test]
    fn node_path_is_root_to_self() {
        let root = ChatNode::root("sys");
        let u = root.add_user("u");
        let a = u.add_assistant("a");
        let path = a.node_path();
        let ids: Vec<_> = path.iter().map(|n| n.id.clone()).collect();
        assert_eq!(ids, vec![root.id.clone(), u.id.clone(), a.id.clone()]);
    }

    #[test]
    fn prepare_messages_applies_system_prompt_and_force_prepend() {
        let root = ChatNode::root("base sys");
        let user = root.add_user("hello");
        // No system prompt override: keep existing system message.
        let p1 = NodeCompletionParameters::new();
        let s1 = CompletionSettings::from_params(Some(&p1));
        let m1 = user.prepare_messages(&s1);
        assert_eq!(m1.first().unwrap().role, Role::System);
        assert_eq!(m1.first().unwrap().text(), Some("base sys"));

        // force_prepend pushes a trailing assistant primer.
        let p2 = NodeCompletionParameters::new().with_force_prepend("Answer: ");
        let s2 = CompletionSettings::from_params(Some(&p2));
        let m2 = user.prepare_messages(&s2);
        let last = m2.last().unwrap();
        assert_eq!(last.role, Role::Assistant);
        assert_eq!(last.text(), Some("Answer: "));
    }

    #[test]
    fn prepare_messages_applies_completion_kwargs_base() {
        let root = ChatNode::root("I am {bot}");
        let user = root.add_user("hi");
        let params = NodeCompletionParameters::new().with_format_kwarg("bot", "Claude");
        let settings = CompletionSettings::from_params(Some(&params));
        let msgs = user.prepare_messages(&settings);
        assert_eq!(msgs.first().unwrap().text(), Some("I am Claude"));
    }

    // ---- cache breakpoints -----------------------------------------------

    #[test]
    fn marking_a_node_propagates_breakpoint_into_prepared_messages() {
        let root = ChatNode::root("sys");
        let user = root.add_user("hi");
        // Mark the system node (cache just the system prompt).
        root.cache_breakpoint();
        assert!(root.is_cache_breakpoint());

        let settings = CompletionSettings::from_params(None);
        let msgs = user.prepare_messages(&settings);
        // The system message carries the breakpoint; the user message does not.
        assert!(msgs[0].cache_breakpoint, "system marked");
        assert!(!msgs[1].cache_breakpoint, "user not marked");
    }

    #[test]
    fn use_cache_flag_marks_the_whole_prefix() {
        let root = ChatNode::root("sys");
        let user = root.add_user("hi");
        let params = NodeCompletionParameters::new().with_cache(true);
        let settings = CompletionSettings::from_params(Some(&params));
        let msgs = user.prepare_messages(&settings);
        // The last message of the prefix is auto-marked (caches everything before).
        assert!(msgs.last().unwrap().cache_breakpoint);
    }

    #[test]
    fn clear_cache_breakpoint_and_clear_all() {
        let root = ChatNode::root("sys");
        let a = root.add_user("a");
        let b = a.add_assistant("b");
        root.cache_breakpoint();
        b.cache_breakpoint();
        assert!(root.is_cache_breakpoint() && b.is_cache_breakpoint());

        // Clear one node.
        b.clear_cache_breakpoint();
        assert!(!b.is_cache_breakpoint());
        assert!(root.is_cache_breakpoint());

        // Clear the whole tree from any node.
        root.cache_breakpoint();
        b.cache_breakpoint();
        a.clear_all_cache_breakpoints();
        assert!(!root.is_cache_breakpoint());
        assert!(!b.is_cache_breakpoint());
    }

    #[test]
    fn cache_breakpoint_survives_clone_tree_and_thread_serialization() {
        let root = ChatNode::root("sys");
        let user = root.add_user("hi");
        root.cache_breakpoint();

        // clone_tree preserves the mark.
        let cloned = user.clone_tree();
        assert!(cloned.get_root().is_cache_breakpoint());

        // The mark rides on the serialized message (so saved threads keep it).
        let msgs = user.prepare_messages(&CompletionSettings::from_params(None));
        let json = serde_json::to_value(&msgs[0]).unwrap();
        assert_eq!(json["cache_breakpoint"], true);
    }
}
