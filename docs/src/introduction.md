# MiniLLMLib (Rust)

A minimalist, async-first Rust library for talking to Large Language Models over
HTTP, with one consistent API across every provider.

The headline idea: **`ChatNode::root(...).chat(...)` is identical no matter what
is behind it.** A `Provider` owns the entire wire dialect (endpoint, auth,
request body, response and stream envelope, cost accounting). Your code only ever
deals in normalized types, so switching from OpenRouter to OpenAI, to a native
Anthropic key, to a Claude subscription, or to your own self-hosted server is a
one-line change.

## What's here

- **Conversation trees.** A conversation is a tree of [`ChatNode`] handles. Linear
  chats, branching, and prebuilt history all use the same structure.
- **Multiple providers.** OpenRouter, OpenAI, native Anthropic (`/v1/messages`),
  a generic OpenAI-compatible provider for self-hosted servers, and your own
  hand-written `impl Provider` for any other wire.
- **Streaming** over SSE, with an idle-timeout that won't kill a long live
  generation but fails loudly on a dead connection.
- **Honest cost tracking.** Per-provider usage and cost, with disjoint
  cached/uncached/cache-write token buckets and a `CostResolution`
  (`Resolved` / `Unpriced` / `Unknown`) that never reports a fake `$0`.
- **Prompt caching**, marked on the tree and enforced per-provider.
- **Claude subscription auth**: use your Pro/Max plan instead of an API key.
- **JSON repair** for malformed model output.

## Two layers of documentation

| Layer | What | Where |
|---|---|---|
| **This guide** | Tutorials, patterns, worked examples | the pages on the left |
| **API reference** | Every public type, method, and signature | [docs.rs/minillmlib](https://docs.rs/minillmlib) |

The guide teaches you how to use the library; the API reference (auto-generated
from the source by docs.rs) is the exhaustive signature lookup. Start with
[Quickstart](./quickstart.md).

[`ChatNode`]: https://docs.rs/minillmlib/latest/minillmlib/struct.ChatNode.html
