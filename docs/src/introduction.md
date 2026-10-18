# MiniLLMLib (Rust)

A minimalist, async-first Rust library for talking to Large Language Models over
HTTP.

`ChatNode::root(...).chat(...)` is the same call whatever provider is behind
it. A `Provider` owns everything that differs between APIs (endpoint, auth,
request body, response and stream shapes, cost accounting), and your code deals
only in the library's own types. So moving from OpenRouter to OpenAI, to
Anthropic's own API, or to your own server means changing the line that makes
the generator. What each API can carry still differs: Anthropic's takes no
audio or video, and has no `seed`, penalties or JSON mode.

## What's here

- **Conversation trees.** A conversation is a tree of [`ChatNode`] handles; a
  linear chat is just a tree with one branch.
- **Multiple providers.** OpenRouter, OpenAI, native Anthropic (`/v1/messages`),
  a generic OpenAI-compatible provider for self-hosted servers, and your own
  hand-written `impl Provider` for any other wire.
- **Streaming** over SSE, with an idle-timeout that won't kill a long live
  generation but fails loudly on a dead connection.
- **Cost tracking.** What each request cost, with cached and uncached tokens
  priced at their own rates, and a `CostResolution` that tells you when the cost
  is unknown, so a `0.0` is never mistaken for a free request.
- **Prompt caching**: you mark what to cache on the tree, and each provider writes the markers its API takes.
- **JSON repair** for malformed model output.

Start with [Quickstart](./quickstart.md). For every public type and signature,
go and read the [API reference](https://docs.rs/minillmlib).

[`ChatNode`]: https://docs.rs/minillmlib/latest/minillmlib/chat_node/struct.ChatNode.html
