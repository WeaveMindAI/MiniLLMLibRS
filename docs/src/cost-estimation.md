# Cost Estimation

Cost tracking tells you what a call *did* cost. Estimation tells you what it
*will* cost, before you send it, so you can decide whether to allow it (reserve
budget, gate a request, refuse an over-priced job). Enable the `estimate`
feature:

```toml
minillmlib = { version = "0.5", features = ["estimate"] }
```

## One call on the generator

The `GeneratorInfo` you already send completions with answers directly:

```rust
use minillmlib::{ChatNode, CompletionParameters, GeneratorInfo};

let generator = GeneratorInfo::openrouter("anthropic/claude-haiku-4.5");
let params = CompletionParameters::new().with_max_tokens(1024);

let root = ChatNode::root("You are terse.");
let prompt = root.add_user("Name three primary colours.");

let usd = generator.estimate_cost_usd(&prompt.thread(), &params).await?;
println!("this call will cost at most ${usd:.6}");
```

If you want the raw rates instead (to price several prompts, or to combine with
[cost tracking](./cost-tracking.md)'s token buckets), use
`generator.model_rates()`, which returns the per-million-token prices and the
model's limits.

## Where the prices come from

OpenRouter's catalog is the one public price sheet, and it lists the first-party
vendors (OpenAI, Anthropic) at their own standard rates. So the model is looked
up by its OpenRouter id:

- An **OpenRouter generator's** model id already is a catalog id. Nothing to set.
- A **direct-vendor generator** whose id differs sets the catalog id explicitly;
  that is what unlocks estimation:

```rust
let generator = GeneratorInfo::anthropic("claude-haiku-4-5-20251001")
    .with_openrouter_name("anthropic/claude-haiku-4.5");
```

The only failure is a model the catalog does not know at all. Everything else
produces a number.

## Whose rates: the provider question

One model is served by many providers at different prices (some models span a
5x range). The rates used are:

1. the serving provider's own, when the generator's provider knows its catalog
   slug (the built-in Anthropic and OpenAI providers do: a direct call is billed
   by the vendor itself);
2. otherwise the **dearest** rate any provider of the model charges, taken
   bucket by bucket. That is the only figure that is a ceiling wherever
   OpenRouter's routing lands, and routing really does land on expensive
   endpoints: a real request was once billed at nearly 4x the advertised rate.

If a request pins one provider through routing settings (a single-entry `order`
with fallbacks off), `ProviderSettings::billing_provider()` yields its slug and
`generator.model_rates_served_by(Some(&slug))` prices at exactly that provider.

## What the figure means

The estimate is **deliberately high**, never a best guess: only the low side
lets you overspend. It is still an estimate (tokenizers differ across model
families), so treat it as a strong ceiling to reserve against, not a guarantee,
and replace it with the tracked real cost once the call returns. It assumes:

- no prompt caching (caching only ever lowers the real cost);
- the largest completion the request permits, including any reasoning budget,
  which providers bill *on top of* `max_tokens`;
- a minute of media for a clip whose length you did not state. **Set the real
  length** with `AudioData::with_duration` / `VideoData::with_duration` whenever
  your prompt carries audio or video: the one-minute assumption overshoots short
  clips by a lot (measured live: 5-7x on an 8-second clip) and *undershoots*
  anything longer than a minute, which is the one way the estimate can come in
  below the real cost. With the length declared the media estimate is tight
  (the video token model matched Gemini's real billing within 1%). Audio bills
  by the second, at up to 1000x the text rate on some models, so this is where
  the money is.

There is no error case beyond an uncatalogued model: a prompt counted larger
than the model accepts is priced as the largest input the model *does* accept,
so you always get a number.

## Keep your generators alive

Each `GeneratorInfo` caches the prices it fetches for an hour, and clones share
the cache. Reuse the same generator for both completions and estimates: only the
first estimate in an hour touches the network, and concurrent estimates for the
same model share a single fetch. Recreating the generator per call throws the
cache away and refetches the price sheet every time. (A generator whose `model`
you change notices and refetches; it never serves another model's prices.)

The library deliberately holds no registry of generators; pooling is yours to
do, and only worth doing if you estimate costs. For code that multiplexes many
models, `generator.pricing_key()` gives you the map key: the catalog model id
plus the provider slug, exactly the pair that determines the price. Build a
generator on a miss, reuse it on a hit:

```rust
use std::collections::HashMap;

struct Generators(HashMap<String, GeneratorInfo>);

impl Generators {
    fn for_model(&mut self, model: &str) -> &GeneratorInfo {
        let generator = GeneratorInfo::openrouter(model);
        self.0.entry(generator.pricing_key()).or_insert(generator)
    }
}
```

If you never estimate costs, ignore all of this: the cache sits unused and
costs nothing.
