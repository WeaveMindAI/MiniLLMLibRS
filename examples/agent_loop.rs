//! A full agent loop with tool calling: the model talks, calls tools, gets
//! results injected back, and continues, until it answers without a tool call.
//!
//! Demonstrates the two execution styles at once:
//! - a STREAMING tool (`draft_notes`): started the moment the model names it,
//!   fed the raw argument fragments while the model is still generating them,
//! - a BUFFERED tool (`get_weather`): executed after the turn ends, from the
//!   fully assembled `ToolCall`.
//!
//! Everything the model says is forwarded as it arrives (stdout here, an SSE
//! pipe to a browser in a real app); the seams between the loop's API requests
//! are invisible to the consumer.
//!
//! Run with: `OPENROUTER_API_KEY=... cargo run --example agent_loop`
//! (makes real, billed API calls).

use minillmlib::{
    ArgumentStream, ChatNode, CompletionParameters, GeneratorInfo, NodeCompletionParameters,
    ToolDefinition,
};
use std::collections::HashMap;
use std::io::Write;

const MODEL: &str = "google/gemini-2.5-flash-lite";

/// Hard cap on model turns so a tool-happy model can't loop forever.
const MAX_TURNS: usize = 8;

/// The streaming tool: starts on the first fragment, consumes the payload as
/// the model generates it. An `ArgumentStream` decodes the raw JSON fragments
/// and the tool takes a `FieldHandle` on `content`: a spawned consumer task
/// drinks its decoded deltas live (escapes undone, exactly the text the model
/// meant) while the driver keeps feeding. Lenient mode tolerates a model that
/// is sloppy with escaping or forgets to close the JSON.
struct DraftNotesSession {
    /// The call id from the first fragment, used to match this session back to
    /// the assembled `ToolCall` after the turn ends.
    call_id: String,
    args: ArgumentStream,
    consumer: tokio::task::JoinHandle<usize>,
}

impl DraftNotesSession {
    fn start(call_id: &str) -> Self {
        println!("\n[draft_notes {call_id}] started, receiving decoded payload live:");
        let mut args = ArgumentStream::lenient();
        let mut content = args.field("content");
        // The tool runs concurrently with the wire: a real one would pipe
        // into a process's stdin here.
        let consumer = tokio::spawn(async move {
            let mut bytes = 0;
            while let Some(text) = content.delta().await {
                print!("{text}");
                std::io::stdout().flush().ok();
                bytes += text.len();
            }
            bytes
        });
        Self {
            call_id: call_id.to_string(),
            args,
            consumer,
        }
    }

    fn feed(&mut self, fragment: &str) -> minillmlib::Result<()> {
        self.args.feed(fragment)
    }

    async fn finish(mut self) -> String {
        if let Err(e) = self.args.finish() {
            eprintln!("\n[draft_notes {}] malformed call: {e}", self.call_id);
        }
        let bytes = self.consumer.await.unwrap_or(0);
        println!(
            "\n[draft_notes {}] done ({} decoded bytes streamed)",
            self.call_id, bytes
        );
        "notes saved".to_string()
    }
}

/// The buffered tool: runs after the turn ends, from complete arguments.
fn get_weather(args: &serde_json::Value) -> String {
    format!(
        "15 degrees and sunny in {}",
        args["city"].as_str().unwrap_or("unknown")
    )
}

fn tool_params() -> NodeCompletionParameters {
    NodeCompletionParameters::new().with_params(
        CompletionParameters::new()
            .with_max_tokens(600)
            .with_tool(ToolDefinition::new(
                "get_weather",
                "Get the current weather for a city",
                serde_json::json!({
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"],
                }),
            ))
            .with_tool(ToolDefinition::new(
                "draft_notes",
                "Save working notes. Use it to record your findings.",
                serde_json::json!({
                    "type": "object",
                    "properties": { "content": { "type": "string" } },
                    "required": ["content"],
                }),
            )),
    )
}

#[tokio::main]
async fn main() -> minillmlib::Result<()> {
    minillmlib::init();
    let gen = GeneratorInfo::openrouter(MODEL);
    let params = tool_params();

    let root = ChatNode::root(
        "You are a helpful assistant. When asked about weather, use get_weather. \
         Record what you learned with draft_notes before giving your final answer.",
    );
    let mut node = root.add_user("What's the weather in Paris and in Lyon? Then note it down.");

    for turn in 1..=MAX_TURNS {
        println!("\n=== model turn {turn} ===");
        let mut stream = node.complete_streaming(&gen, Some(&params)).await?;

        // Streaming-tool sessions in flight this turn, keyed by the wire's call
        // index (the model may call tools in parallel; the index de-multiplexes
        // the fragments, the call id ties a session to its assembled call).
        let mut streaming: HashMap<u64, DraftNotesSession> = HashMap::new();

        while let Some(chunk) = stream.next_chunk().await {
            let chunk = chunk?;

            // 1. Prose → forward as it arrives (browser SSE in a real app).
            if !chunk.delta.is_empty() {
                print!("{}", chunk.delta);
                std::io::stdout().flush().ok();
            }

            // 2. Tool fragments: start streaming tools on their first fragment
            //    (which carries name + id), feed them live; buffered tools are
            //    ignored here (the library assembles them for after the turn).
            for delta in chunk.tool_calls.as_deref().unwrap_or_default() {
                if delta.name.as_deref() == Some("draft_notes") {
                    let id = delta.id.as_deref().unwrap_or_default();
                    streaming.insert(delta.index, DraftNotesSession::start(id));
                }
                if let Some(frag) = &delta.arguments_fragment {
                    if let Some(session) = streaming.get_mut(&delta.index) {
                        session.feed(frag)?;
                    }
                }
            }
        }

        // Turn over: append the assistant node (content + assembled calls).
        let response = stream.collect().await?;
        let assistant = node.append_response(&response);

        // No tool calls → that was the final answer.
        let Some(calls) = assistant.tool_calls() else {
            println!("\n\n=== final answer delivered in {turn} turn(s) ===");
            return Ok(());
        };

        // Answer EVERY call, then loop: the next request carries the results.
        let mut sessions: HashMap<String, DraftNotesSession> = streaming
            .into_values()
            .map(|s| (s.call_id.clone(), s))
            .collect();
        let mut current = assistant;
        for call in &calls {
            let output = match call.name.as_str() {
                "draft_notes" => match sessions.remove(&call.id) {
                    Some(session) => session.finish().await,
                    // Defensive only for a call the stream never fragmented.
                    None => "notes saved".to_string(),
                },
                "get_weather" => get_weather(&call.arguments_json()?),
                other => format!("Error: unknown tool '{other}'"),
            };
            println!("\n[{}] -> {}", call.name, output);
            current = current.add_tool_result(&call.id, output);
        }
        node = current;
    }

    Err(minillmlib::MiniLLMError::InvalidParameter(format!(
        "agent did not finish within {MAX_TURNS} turns"
    )))
}
