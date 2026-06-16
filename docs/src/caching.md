# Prompt Caching

Caching intent is marked on the conversation tree; the provider decides the wire.
Anthropic emits `cache_control` markers (honoring its 4-breakpoint cap); OpenAI
and OpenRouter auto-cache and ignore the marks. Switch the provider and the same
code works.

## Mark what to cache

```rust,no_run
# use minillmlib::ChatNode;
let root = ChatNode::root("a large, stable system prompt ...");
root.cache_breakpoint();          // cache just the system prompt

// ...or cache the whole stable prefix of a conversation:
# let some_node = root.clone();
some_node.cache_breakpoint();
```

Or, per request, auto-mark the entire prompt prefix without touching individual
nodes:

```rust,no_run
# use minillmlib::NodeCompletionParameters;
let params = NodeCompletionParameters::new().with_cache(true);
```

Explicit per-node marks are always honored in addition.

## Clearing marks

```rust,no_run
# use minillmlib::ChatNode;
# let node = ChatNode::root("x");
node.clear_cache_breakpoint();        // this node
node.clear_all_cache_breakpoints();   // the whole tree
```

## Warming the cache

`ensure_cached` fires a zero-output request that writes/refreshes the cache for a
node's prefix, returning the `CostInfo` of the warm call. Cheap to call before an
agent run: cold pays the one-time write (which you'd pay on the next real call
anyway); warm is a cheap read that refreshes the TTL.

```rust,no_run
# use minillmlib::{ChatNode, GeneratorInfo};
# async fn run(some_node: ChatNode, generator: GeneratorInfo) -> minillmlib::Result<()> {
let warm_cost = some_node.ensure_cached(&generator, None).await?;
# let _ = warm_cost;
# Ok(()) }
```

## Pricing cached tokens

Cache reads and writes have their own rates (read is a discount, write a premium):

```rust,no_run
use minillmlib::TokenPrice;

let price = TokenPrice::new(1.0, 5.0)      // $/Mtok input, output
    .with_cache_rates(0.1, 1.25);          // $/Mtok cache-read, cache-write
```

The three input buckets (uncached / cache-read / cache-write) are billed at their
own rates; see [Cost Tracking](./cost-tracking.md).
