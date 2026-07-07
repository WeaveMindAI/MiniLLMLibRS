# Tool Calling

Tools are normalized *intent*, like every other parameter: you declare
`ToolDefinition`s and a `ToolChoice` once, and each provider emits its own wire
shape. The same code drives OpenRouter, OpenAI, any OpenAI-compatible server,
and native Anthropic.

| Normalized | OpenAI-wire (OpenRouter, OpenAI, compatibles) | Anthropic `/v1/messages` |
|---|---|---|
| `ToolDefinition { name, description, parameters, strict }` | `{"type":"function","function":{name, description, parameters, strict}}` | `{name, description, input_schema, strict}` |
| `ToolChoice::Auto` / `None` / `Required` / `Tool(name)` | `"auto"` / `"none"` / `"required"` / `{"type":"function","function":{"name"}}` | `{"type":"auto"}` / `{"type":"none"}` / `{"type":"any"}` / `{"type":"tool","name"}` |
| `parallel_tool_calls: false` | top-level `"parallel_tool_calls": false` | `tool_choice.disable_parallel_tool_use: true` |
| assistant `ToolCall`s | `message.tool_calls[]` (arguments as a JSON string) | `tool_use` content blocks (input as an object) |
| `Message::tool(call_id, content)` | `{"role":"tool","tool_call_id","content"}` | a `user` turn with `tool_result` blocks |

## The loop

```rust,no_run
use minillmlib::{
    ChatNode, CompletionParameters, GeneratorInfo, NodeCompletionParameters,
    ToolChoice, ToolDefinition,
};

# async fn run() -> minillmlib::Result<()> {
let gen = GeneratorInfo::openrouter("anthropic/claude-sonnet-4.5");

let params = NodeCompletionParameters::new().with_params(
    CompletionParameters::new()
        .with_tool(ToolDefinition::new(
            "get_weather",
            "Get the current weather for a city",
            serde_json::json!({
                "type": "object",
                "properties": { "city": { "type": "string" } },
                "required": ["city"],
            }),
        ))
        .with_tool_choice(ToolChoice::Auto),
);

let node = ChatNode::root("You are helpful.")
    .add_user("What's the weather in Paris?")
    .complete(&gen, Some(&params))
    .await?;

// The model called a tool: run it and answer each call, then complete again.
if let Some(calls) = node.tool_calls() {
    let mut current = node.clone();
    for call in &calls {
        let args = call.arguments_json()?;           // parsed arguments, fails loudly
        let result = format!("15 degrees in {}", args["city"]);
        current = current.add_tool_result(&call.id, result);
    }
    let answer = current.complete(&gen, Some(&params)).await?;
    println!("{}", answer.text().unwrap_or(""));
}
# Ok(()) }
```

Notes:

- **Keep the same `tools` in the follow-up request.** Providers require the tool
  definitions to still be present when you send back the results.
- **Parallel calls**: the model may return several `ToolCall`s in one turn; add
  one `add_tool_result` per call (in any order). The Anthropic provider packs
  consecutive results into the single `user` turn its wire requires. Forbid
  parallelism with `.with_parallel_tool_calls(false)`.
- **Arguments are raw JSON text** (`ToolCall::arguments`), exactly as the model
  produced them; `arguments_json()` parses them and fails loudly on invalid
  JSON instead of silently repairing.
- **Forcing a call**: `ToolChoice::Required` (any tool) or
  `ToolChoice::Tool("get_weather".into())` (that one).
- **Strict schemas**: `ToolDefinition::with_strict(true)` asks the provider to
  guarantee the arguments match your schema (OpenAI structured outputs,
  Anthropic strict tool use).
- **Streaming works too**: tool-call fragments are accumulated across chunks and
  the final `CompletionResponse::tool_calls` (and the node) carry the assembled
  calls.

## Streaming a tool call as it is generated

You don't have to wait for the model to finish a call before acting on it. The
streaming chunks expose typed `ToolCallDelta` fragments with the same timing on
both wires: the first fragment carries the call's `name` (and `id`), then each
later fragment carries a piece of the raw JSON argument text, in order. That
lets you start the tool the moment the model names it and pipe the argument
bytes in while the model is still generating them.

```rust,no_run
use minillmlib::{
    ChatNode, CompletionParameters, GeneratorInfo, NodeCompletionParameters,
    ToolChoice, ToolDefinition,
};

# async fn run() -> minillmlib::Result<()> {
let gen = GeneratorInfo::openrouter("anthropic/claude-sonnet-4.5");

let params = NodeCompletionParameters::new().with_params(
    CompletionParameters::new()
        .with_tool(
            ToolDefinition::new(
                "run_python",
                "Execute Python code",
                serde_json::json!({
                    "type": "object",
                    "properties": { "code": { "type": "string" } },
                    "required": ["code"],
                }),
            )
            .with_strict(true),
        )
        .with_tool_choice(ToolChoice::Tool("run_python".into())),
);

let root = ChatNode::root("You are helpful.");
let user = root.add_user("Compute the 100th Fibonacci number.");
let mut stream = user.complete_streaming(&gen, Some(&params)).await?;

let mut tool_started = false;
while let Some(chunk) = stream.next_chunk().await {
    let chunk = chunk?;
    if let Some(deltas) = &chunk.tool_calls {
        for delta in deltas {
            // First fragment carries the name: start the tool NOW
            // (e.g. spawn the interpreter process here).
            if let Some(name) = &delta.name {
                println!(">> model is calling {name}, starting process");
                tool_started = true;
            }
            // Later fragments: raw JSON argument text, in order.
            if let Some(frag) = &delta.arguments_fragment {
                if tool_started {
                    // CAVEAT: this is escaped JSON source, e.g.
                    // {"code": "print(\"hi\")... — see the note below.
                    print!("{frag}");
                }
            }
        }
    }
}

// The stream assembled the complete calls in parallel: append the assistant
// node and finish the normal loop (add_tool_result + complete again).
let response = stream.collect().await?;
let node = user.append_response(&response);
if let Some(calls) = node.tool_calls() {
    let result = node.add_tool_result(&calls[0].id, "354224848179261915075");
    let answer = result.complete(&gen, Some(&params)).await?;
    println!("{}", answer.text().unwrap_or(""));
}
# Ok(()) }
```

Notes:

- **The fragments are JSON source text, not your payload.** For a tool whose
  input is one string field (like `code` above), the bytes arrive escaped and
  wrapped in the object syntax (`{"code": "print(\"hi` ...). To feed a tool the
  clean payload as it streams, put an incremental extractor between the
  fragment and the tool's stdin: lock onto the field, un-escape the string
  content as it grows.
- **Parallel calls**: `delta.index` disambiguates concurrent calls; key your
  spawned tools by it. Forcing a single call with `ToolChoice::Tool(..)` (and
  `.with_parallel_tool_calls(false)`) sidesteps this.
- **Key order**: models may emit argument keys in any order, so with several
  fields your payload field can arrive last. `strict: true` plus a one-property
  schema keeps the stream predictable.

For the complete pattern (a multi-turn agent loop mixing a streaming tool and a
buffered tool, forwarding all prose live), see
[`examples/agent_loop.rs`](https://github.com/WeaveMindAI/MiniLLMLibRS/blob/main/examples/agent_loop.rs):
`cargo run --example agent_loop`. The key mental model: **a tool call always
ends the model's turn**; "the model continues after the tool" is always a new
API request that your loop makes after `add_tool_result`, and the consumer of
your stream never sees the seams.

## Custom wire shapes

An OpenAI-envelope server whose *tool* shape deviates only needs to override the
two tool hooks on `Provider` (`openai_tools_value`, `openai_tool_choice_value`);
a fully different wire translates `params.tools` / `params.tool_choice` /
`message.tool_calls` itself in its `build_request`. See
[Custom Providers](./custom-providers.md).
