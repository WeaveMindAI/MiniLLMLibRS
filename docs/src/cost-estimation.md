# Cost Estimation

Cost tracking tells you what a call *did* cost. Estimation tells you what it
*will* cost, before you send it, so you can decide whether to allow it. Turn on
the `estimate` feature:

```toml
minillmlib = { version = "0.6", features = ["estimate"] }
```

## Ask the generator

The `GeneratorInfo` you send completions with gives the estimate:

```rust,no_run
use minillmlib::{ChatNode, CompletionParameters, GeneratorInfo};

# async fn run() {

let generator = GeneratorInfo::openrouter("anthropic/claude-haiku-4.5");
let params = CompletionParameters::new().with_max_tokens(1024);
let prompt = ChatNode::root("You are terse.").add_user("Name three primary colours.");

match generator.estimate_cost_usd(&prompt.thread(), &params).await {
    Some(usd) => println!("reserve ${usd:.6} for this call"),
    None => println!("price or output limit unknown"),
}
# }
```

Pass the messages you will send. If you fill `{placeholders}`, pass
`prompt.formatted_thread()`, or, if you give the values with
`with_format_kwargs`, pass `prompt.formatted_thread_with_base(&kwargs)` with
those same values. A system prompt or forced prefix set on
`NodeCompletionParameters` is added only when the request is sent, so it is not
counted: add it to the messages you pass if you want it in the figure.

You get `None` when the library cannot work it out: the model's price is
unknown, or nothing limits how much it can write (you set no `max_tokens`, and
neither the model's largest output nor its context window is known).

## What the library knows about a model

An estimate needs three facts about the model, which the library keeps in a
`ModelInfo`: its price per token, the most it can write in one reply, and its
context window. The price is also what the library reports a finished call as
costing when the provider returns token counts but no dollar amount (OpenAI,
Anthropic).

If you already know them, give them yourself. Anything you set wins over a
lookup:

```rust,no_run
use minillmlib::{GeneratorInfo, ModelInfo, TokenPrice};

let generator = GeneratorInfo::custom("local", "http://localhost:8000/v1", "my-model")
    .with_token_price(TokenPrice::new(0.2, 0.6)) // $/Mtok in, out
    .with_model_info(ModelInfo {
        max_output_tokens: Some(8192),
        context_length: Some(32768),
        ..Default::default()
    });
```

For whatever you leave unset, the generator asks its provider:

| Provider | Where it looks |
|---|---|
| OpenRouter | OpenRouter's catalog, at the highest rate of any provider serving the model, since routing may land on any of them (your `OpenRouterRouting` is not read) |
| OpenAI | OpenRouter's catalog, at OpenAI's own rate, under `openai/<model>` |
| Anthropic | Anthropic's Models API for the largest output and the context window, and OpenRouter's catalog, under `anthropic/<model>`, for the price at Anthropic's own rate |
| Generic / custom | nowhere: give the facts yourself |

If you use Anthropic directly, the catalog id `anthropic/<model>` usually
misses, because the catalog writes versions with dots
(`anthropic/claude-haiku-4.5`) where Anthropic uses dashes and a date
(`claude-haiku-4-5-20251001`). When it misses, the price stays unknown and a
warning says to name the catalog id:

```rust,no_run
# use minillmlib::GeneratorInfo;
let generator = GeneratorInfo::anthropic("claude-haiku-4-5-20251001")
    .with_openrouter_name("anthropic/claude-haiku-4.5");
```

Each lookup is shared by your whole program, and its answer, including
"the catalog does not list this model", is kept for an hour. If you fix a miss
with `with_openrouter_name`, the new id is looked up straight away. A source
that does not answer is asked again after a minute, and one that answers with
an error is asked again on the next call. Either way its facts stay unknown and
a warning says why. A request that needs Anthropic's cap asks only the Models
API, and billing asks only the catalog. If you want to see what the library
knows, call `generator.model_info().await`.

## What the figure assumes

The figure leans high so you can reserve against it, except in the three cases
below that say it can come in low. Replace it with the tracked cost once the
call returns. It assumes:

- everything priced as uncached input. A cache read costs less, but a request
  that writes the cache (`with_cache(true)`, or a node marked with
  `cache_breakpoint`) pays more than the input rate on the cached part (1.25x
  on Anthropic), so on that request the estimate can come in low;
- the longest reply the request allows: your `max_tokens`, capped at the
  model's largest output, or the largest output itself if you set none. Where
  the largest output is unknown, the context window stands in for it;
- tool definitions, and the tool calls already in the conversation, counted
  as input text, like the messages;
- text counted with GPT's tokenizer and multiplied by 1.6, which covers the
  worst case measured against Claude's tokenizer;
- 1,600 tokens for an image of unknown size. **Set its size** with
  `ImageData::with_dimensions` and it is priced at the higher of OpenAI's
  tile price and Anthropic's pixel price instead, never above 1,600;
- a minute of audio or video for a clip whose length you did not state. **Set
  the real length** with `AudioData::with_duration` / `VideoData::with_duration`.
  The one-minute guess counts an 8-second clip at 7.5 times its length and
  undercounts anything longer than a minute. That matters most for audio,
  which can bill up to a thousand times the text rate;
- 20 pages for a PDF whose page count you did not state, each counted as
  4,000 tokens (2,400 of text after the 1.6 multiplier, plus 1,600 for the page
  as an image). **Set the real count** with `DocumentData::with_page_count`;
  a longer PDF without one is undercounted.

If the prompt comes out larger than the model's context window, and the window
is known, the estimate prices one full window instead, filled with the most
expensive kinds of token first.
