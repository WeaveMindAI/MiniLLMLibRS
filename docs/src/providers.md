# Providers

A `GeneratorInfo` bundles a model, a base URL, a credential, and a `Provider`
that speaks one API's format. The crate ships these presets:

| Preset | Wire | Auth (env var) | Cost |
|---|---|---|---|
| `GeneratorInfo::openrouter(model)` | OpenAI `/chat/completions` | `OPENROUTER_API_KEY` | OpenRouter's own dollar figure; if a stream ends without it, the library asks OpenRouter's `/generation` endpoint |
| `GeneratorInfo::openai(model)` | OpenAI `/chat/completions` | `OPENAI_API_KEY` | token counts, priced from OpenRouter's catalog |
| `GeneratorInfo::anthropic(model)` | Anthropic's own `/v1/messages` (images and PDFs go through; audio and video are refused, since the API takes neither) | `ANTHROPIC_API_KEY` (`x-api-key`) | token counts, priced from OpenRouter's catalog, which usually needs its catalog id (`with_openrouter_name`) |
| `GeneratorInfo::custom(name, base_url, model)` | OpenAI-compatible (default) | none unless you add one | token counts, priced only if you give a price |

## Auth

The credential lives on the generator, and each provider turns it into its own
header:

```rust,no_run
# use minillmlib::GeneratorInfo;
# let g = GeneratorInfo::openai("gpt-4o-mini");
g.clone().with_api_key("sk-...");                 // provider picks the header (Bearer / x-api-key)
g.clone().with_api_key_from_env("MY_KEY");        // no-op if the var is unset
g.clone().with_bearer_token("token");             // always Authorization: Bearer
g.clone().with_header("X-Tenant", "acme");        // any extra header
```

## Your own HTTP client

Every generator shares one HTTP client, which keeps connections open between
calls. If you want your own (a proxy, custom TLS, middleware), pass it with
`with_http_client`. It takes a `reqwest::Client`, or a
`reqwest_middleware::ClientWithMiddleware` if you want middleware, and every
request for that generator goes through it, cost lookups included. Your
`reqwest` has to be 0.12, the version this crate uses, or the client will not
convert.

```rust,no_run
# use minillmlib::GeneratorInfo;
let gen = GeneratorInfo::openai("gpt-4o-mini").with_http_client(reqwest::Client::new());
```

A client you pass in replaces the shared one completely, including its
600-second limit on a silent connection. If you want that limit, build your
client with `reqwest::Client::builder().read_timeout(Duration::from_secs(600))`.

If your program starts more than one tokio runtime (one per thread, or a
throwaway `Runtime::new().block_on(...)` for a single call), give each
runtime's generators their own client. A connection belongs to the runtime that
opened it, so once that runtime shuts down, the shared client can hand that
dead connection to a call running on a different runtime. The call then fails
with `HTTP error: error sending request for url (...)`, and the error's full
cause chain ends in "dispatch task is gone: runtime dropped the dispatch task".

## Cost for token-only providers

OpenAI and Anthropic send back token counts but no dollar amount, so the
library multiplies the tokens by the model's price. If you know the price, give
it with `with_token_price` (USD per million tokens). If you don't, the library
looks it up, and when it finds none, tracking reports `Unpriced`:

```rust,no_run
use minillmlib::{GeneratorInfo, TokenPrice};

let gen = GeneratorInfo::anthropic("claude-haiku-4-5")
    .with_token_price(TokenPrice::new(1.0, 5.0)); // $1/Mtok in, $5/Mtok out
```

For where the price is looked up, and the catalog id an Anthropic model needs,
go and read [Cost Estimation](./cost-estimation.md#what-the-library-knows-about-a-model).

## OpenRouter routing

If you use OpenRouter, its routing settings (provider order, sort, ignore list,
data collection, fallbacks) go in the request's `provider` key:

```rust,no_run
use minillmlib::{CompletionParameters, OpenRouterRouting};

let routing = OpenRouterRouting::new()
    .sort_by_throughput()
    .deny_data_collection();

let params = CompletionParameters::new()
    .with_openrouter_routing(routing);
```

Set routing only on an OpenRouter generator: OpenAI and Anthropic refuse a
request that carries a `provider` key.
