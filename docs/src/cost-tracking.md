# Cost Tracking

The library tracks usage and cost per request, and is honest about when a cost is
actually known.

## Token buckets

Input tokens are split into three **disjoint, additive** buckets so caching is
priced correctly across every provider's differing wire conventions:

- `uncached_input_tokens`: full-price prompt tokens,
- `cache_read_tokens`: served from a warm cache (cheap),
- `cache_write_tokens`: written to the cache this request (a premium).

Total input is the sum of the three; cost is a clean weighted sum, no subtraction.

## Resolution: never a fake $0

Every reported `CostInfo` carries a `CostResolution`:

| Resolution | Meaning |
|---|---|
| `Resolved` | The USD cost is authoritative (native, or tokens × a configured `TokenPrice`) |
| `Unpriced` | Tokens are real, but no native cost and no `TokenPrice` was set. `cost` is `0.0` but must NOT be treated as a free request. Set a `TokenPrice` to resolve it. |
| `Unknown` | Cost could not be determined at all (no usage, and any out-of-band query failed) |

Check `resolution` before trusting `cost`.

## The reply and the bill together: `complete_costed`

The simplest tracked shape, and the right default when the caller itself acts
on the cost (settles a budget hold, gates the next call, logs it right there):
the `CostInfo` comes back WITH the result, no callback to register.

```rust,no_run
use minillmlib::{ChatNode, GeneratorInfo};

# async fn run() {
let generator = GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");
let root = ChatNode::root("You are helpful.");

let (result, cost) = root.add_user("Hi").complete_costed(&generator, None).await;
let reply = result.expect("completion");
if let Some(cost) = cost {
    println!("that cost ${:.6} ({:?})", cost.cost, cost.resolution);
}
# }
```

Same accounting as every other tracked path (usage from the response, the
provider's out-of-band resolution as backstop, never a fake $0). An errored
completion carries no cost info: the request failed before a billable response
existed.

When many call sites should feed one central sink instead, use the callback
shapes below; streaming always goes through them, since a stream's cost
resolves only after it ends.

## A callback per completion

```rust,no_run
use minillmlib::{ChatNode, GeneratorInfo, NodeCompletionParameters, CompletionParameters, CostInfo};
use std::sync::{Arc, Mutex};

# async fn run() -> minillmlib::Result<()> {
let gen = GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");
let total = Arc::new(Mutex::new(0.0));
let sink = total.clone();

let params = NodeCompletionParameters::new()
    .with_params(CompletionParameters::new().with_max_tokens(200))
    .with_cost_tracking(true)
    .with_cost_callback(move |info: CostInfo| {
        // info.cost, .prompt_tokens, .completion_tokens,
        // .cache_read_tokens, .cache_write_tokens, .resolution
        *sink.lock().unwrap() += info.cost;
    });

let root = ChatNode::root("You are helpful.");
root.add_user("Hi").complete(&gen, Some(&params)).await?;
println!("total spent: {}", *total.lock().unwrap());
# Ok(()) }
```

## Enforced tracking via CompletionContext

When you want cost reporting to be structurally guaranteed (not opt-in per call),
wrap the generator in a `CompletionContext` and use `complete_tracked`. It always
reports cost through the context's async callback, and on a cancelled or
usage-less stream it resolves out-of-band (e.g. OpenRouter's `/generation` query)
or reports `Unknown`, rather than silently booking `$0`.

```rust,no_run
use minillmlib::{CompletionContext, CostInfo, AsyncCostCallback, CompletionMeta, GeneratorInfo, ChatNode};
use std::sync::Arc;

# async fn run() -> minillmlib::Result<()> {
# let generator = GeneratorInfo::openrouter("m");
let callback: AsyncCostCallback = Arc::new(|cost: CostInfo, _meta: CompletionMeta| {
    Box::pin(async move {
        // persist `cost` to your DB / metering here
        let _ = cost;
    })
});
let ctx = CompletionContext::new(generator, serde_json::json!({}), callback, "https://app", "App");

let root = ChatNode::root("You are helpful.");
let _answer = root.add_user("Hi").complete_tracked(&ctx, None).await?;
# Ok(()) }
```

For streaming, `complete_streaming_tracked` returns a `TrackedStream` that settles
cost when it finishes or is cancelled (use `cancel().await` for a reliable
settle; a plain drop is best-effort).
