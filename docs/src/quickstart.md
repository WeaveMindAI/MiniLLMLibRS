# Quickstart

## Install

```toml
# Cargo.toml
[dependencies]
minillmlib = "0.6"
tokio = { version = "1", features = ["full"] }
```

## One call

```rust,no_run
use minillmlib::{ChatNode, GeneratorInfo};

#[tokio::main]
async fn main() -> minillmlib::Result<()> {
    // Pick a provider. OpenRouter reads OPENROUTER_API_KEY from the environment
    // (if it is in a .env file, call `minillmlib::init()` first, which loads it).
    let generator = GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");

    // A conversation is a tree; `root` is the system prompt.
    let root = ChatNode::root("You are a helpful assistant. Be brief.");

    // `chat` = add a user message + get the assistant reply (returns the new node).
    let answer = root.chat("Say hello in five words.", &generator).await?;

    println!("{}", answer.message.text().unwrap_or(""));
    Ok(())
}
```

If you want another provider, change that one line:

```rust,no_run
# use minillmlib::GeneratorInfo;
GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");      // OPENROUTER_API_KEY
GeneratorInfo::openai("gpt-4o-mini");                           // OPENAI_API_KEY
GeneratorInfo::anthropic("claude-haiku-4-5");                   // ANTHROPIC_API_KEY, native /v1/messages
GeneratorInfo::custom("my", "http://localhost:8000/v1", "m");   // your own OpenAI-compatible server
```

For what each preset does, go and read [Providers](./providers.md), and to
connect your own server, [Custom Providers](./custom-providers.md).

## A multi-turn conversation

Each `chat` returns the assistant node; chain from it to continue the thread.

```rust,no_run
use minillmlib::{ChatNode, GeneratorInfo};

#[tokio::main]
async fn main() -> minillmlib::Result<()> {
    let gen = GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");
    let root = ChatNode::root("You are a terse assistant.");

    let a1 = root.chat("What's the capital of France?", &gen).await?;
    let a2 = a1.chat("And its population, roughly?", &gen).await?;

    println!("{}", a2.message.text().unwrap_or(""));
    // a2 knows its whole history: a2.thread() is the full root-to-leaf message list.
    Ok(())
}
```

## Errors

Every fallible call returns [`minillmlib::Result<T>`], whose error is a
`MiniLLMError` you can match on: `Api { status, message }` when the provider
refuses, `Timeout`, and `MaxRetriesExceeded` once retries you turned on run
out. An empty reply counts as a success unless you pass
`NodeCompletionParameters::new().with_crash_on_empty(true)` to `complete` (see
[Completion Parameters](./parameters.md)).

[`minillmlib::Result<T>`]: https://docs.rs/minillmlib/latest/minillmlib/error/type.Result.html
