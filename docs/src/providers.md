# Providers

A `GeneratorInfo` bundles a model, a base URL, an auth strategy, and a
`Provider` (the wire dialect). The provider owns everything that differs between
APIs; your calling code never changes. The crate ships these presets:

| Preset | Wire | Auth (env var) | Cost |
|---|---|---|---|
| `GeneratorInfo::openrouter(model)` | OpenAI `/chat/completions` | `OPENROUTER_API_KEY` | native USD, with a `/generation` fallback |
| `GeneratorInfo::openai(model)` | OpenAI `/chat/completions` | `OPENAI_API_KEY` | token-only (set a `TokenPrice`) |
| `GeneratorInfo::anthropic(model)` | native `/v1/messages`, `content[]` | `ANTHROPIC_API_KEY` (`x-api-key`) | token-only (set a `TokenPrice`) |
| `GeneratorInfo::claude_subscription(model)` | native `/v1/messages` | Pro/Max OAuth token | token-only ESTIMATE |
| `GeneratorInfo::custom(name, base_url, model)` | OpenAI-compatible (default) | none unless you add one | token-only |

## Auth

Auth is a strategy on the generator, mapped to concrete headers by the provider
(so the same Anthropic provider serves both an API key and a subscription token):

```rust,no_run
# use minillmlib::GeneratorInfo;
# let g = GeneratorInfo::openai("gpt-4o-mini");
g.clone().with_api_key("sk-...");                 // provider picks the header (Bearer / x-api-key)
g.clone().with_api_key_from_env("MY_KEY");        // no-op if the var is unset
g.clone().with_bearer_token("token");             // always Authorization: Bearer
g.clone().with_header("X-Tenant", "acme");        // any extra header
```

## Cost for token-only providers

OpenAI and Anthropic return token counts but no dollar amount. Attach a
`TokenPrice` (USD per **million** tokens, the unit every price sheet quotes) to
get a resolved cost; otherwise tracking reports `Unpriced` (never a fake `$0`):

```rust,no_run
use minillmlib::{GeneratorInfo, TokenPrice};

let gen = GeneratorInfo::anthropic("claude-haiku-4-5")
    .with_token_price(TokenPrice::new(1.0, 5.0)); // $1/Mtok in, $5/Mtok out
```

See [Cost Tracking](./cost-tracking.md) for the full picture.

## OpenRouter routing

OpenRouter-specific routing (provider order, sort, data-collection) is attached
honestly through the `extra` escape hatch rather than masquerading as a universal
parameter:

```rust,no_run
use minillmlib::{CompletionParameters, ProviderSettings};

let routing = ProviderSettings::new()
    .sort_by_throughput()
    .deny_data_collection();

let params = CompletionParameters::new()
    .with_openrouter_routing(routing);
```

Non-OpenRouter providers simply ignore it.
