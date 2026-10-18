# Cost Tracking

The library reports what each request cost, and says so when it does not know.

## Token buckets

Input tokens are split into three buckets that never overlap, so each token is
priced once, at its own rate, whatever the provider:

- `uncached_input_tokens`: full-price prompt tokens,
- `cache_read_tokens`: served from a warm cache (cheap),
- `cache_write_tokens`: written to the cache this request (a premium).

These are the fields of `Usage`, on `CompletionResponse::usage`. A `CostInfo`
carries `cache_read_tokens`, `cache_write_tokens` and `prompt_tokens`, which is
the sum of all three, so the uncached count is `prompt_tokens` minus the other
two. The cost is each bucket times its rate, plus the output tokens times the
output rate.

## Resolution: whether the cost is known

Every reported `CostInfo` carries a `CostResolution`:

| Resolution | Meaning |
|---|---|
| `Resolved` | The USD cost is real: the provider's own figure, or the tokens at the model's price (yours, or the one looked up) |
| `Unpriced` | Tokens are real, but the provider returned no dollar amount and the model's price is unknown (none given, none found in the catalog). `cost` is `0.0` but must NOT be treated as a free request. Give the price (`with_token_price`), or, on OpenAI or Anthropic, the catalog id (`with_openrouter_name`) so it can be looked up. |
| `Unknown` | Cost could not be worked out at all: no usage came back, and asking the provider afterwards (only OpenRouter can be asked) found none |

Check `resolution` before trusting `cost`.

## The reply and the bill together: `complete_costed`

If you want the cost handed back with the reply, call `complete_costed`:

```rust,no_run
use minillmlib::{ChatNode, GeneratorInfo};

# async fn run() {
let generator = GeneratorInfo::openrouter("google/gemini-2.5-flash-lite");
let root = ChatNode::root("You are helpful.");

let (result, costs) = root.add_user("Hi").complete_costed(&generator, None).await;
for cost in &costs {
    println!("a request cost ${:.6} ({:?})", cost.cost, cost.resolution);
}
let reply = result.expect("completion");
# }
```

There is one `CostInfo` per response received: a response rejected by
`crash_on_empty_response` or `crash_on_refusal` was billed too, so it is listed,
also when the completion fails in the end. A request that fails without a
response lists nothing, though a provider may still bill it (one that timed out
after the model had finished, for example).

If you would rather have the cost pushed to you, set a callback, as below. If you stream, there is no costed version: use
`complete_streaming_collect` with a callback, or `complete_streaming_tracked`.

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

Setting a callback already asks the provider for usage, so you do not need
`with_cost_tracking(true)`. The `_tracked` methods refuse a callback, because
they report to their `CompletionContext`.

## Enforced tracking via CompletionContext

If every completion in your app must report its cost to one place, wrap the
generator in a `CompletionContext` and call `complete_tracked`. It reports every
billed response through the context's async callback.

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

The last two arguments are your app's URL and name; on OpenRouter they become
the `HTTP-Referer` and `X-Title` headers it credits usage to.

If you stream, `complete_streaming_tracked` returns a `TrackedStream`. If you
only want the finished node, call `complete_streaming_collect_tracked` instead:
it collects the stream, reports the cost and adds the reply to the tree for
you. If you keep the `TrackedStream` and want the whole reply,
call `collect().await`, which reads whatever is left of the stream, then
`report_cost(&response).await`, and `append_response(&response)` on the node you
started the stream from, if you want it in the tree. If you want a kill switch, `collect_or_cancel(interrupt)`
collects and reports in one call, or cancels and reports if `interrupt` fires
first. It returns a `CollectOutcome`: `Finished(response)`, which is not yet in
the tree, `Interrupted`, or `Failed(error)` when the stream broke off (a dropped
connection, an idle timeout, or an error the provider sent mid-stream), in
which case nothing is booked. If you stop
early, call `cancel().await`, which waits until the cost is reported. Just
dropping the stream reports it from a background task, which is lost if your
runtime shuts down first.
