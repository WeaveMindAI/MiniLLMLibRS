# Claude Subscription

Use your Claude **Pro/Max subscription** instead of a pay-as-you-go API key. A
subscription OAuth token authenticates against the same native Anthropic API as
an API key, but draws on your subscription's rolling quota (the 5-hour / 7-day
window) rather than API billing.

```rust,no_run
use minillmlib::{ChatNode, GeneratorInfo, TokenPrice};

# async fn run() -> minillmlib::Result<()> {
// Anthropic returns token counts but no dollar cost, so set a price for a
// resolved cost ESTIMATE (otherwise tracking reports `Unpriced`).
let generator = GeneratorInfo::claude_subscription("claude-haiku-4-5")
    .with_token_price(TokenPrice::new(1.0, 5.0)); // $/Mtok in, $/Mtok out

let root = ChatNode::root("You are helpful.");
let response = root.chat("Hello!", &generator).await?;
# let _ = response;
# Ok(()) }
```

## How the token is resolved

`claude_subscription` resolves the bearer token in this order:

1. the `ANTHROPIC_AUTH_TOKEN` env var, if set (explicit override; you keep it
   fresh, e.g. from `claude setup-token`);
2. otherwise the live Claude Code credential at `~/.claude/.credentials.json`
   (`claudeAiOauth.accessToken`), which Claude Code keeps refreshed, so if you're
   logged into Claude Code with your subscription, it just works.

If neither source yields a token, the request fails loudly as unauthenticated
rather than silently using the wrong account.

## Subscription vs Console

> A subscription token (from Claude Code) bills your **Pro/Max plan**. A
> Console/API OAuth token bills your **API account**, not the subscription. For
> Console use an API key via `GeneratorInfo::anthropic(model)`, and this preset
> only for the actual Pro/Max subscription token.

Cost is always an ESTIMATE here: Anthropic returns only token counts, so the
`TokenPrice` you set (reflecting the model's published price) produces a
`Resolved` USD estimate; without it, tracking reports `Unpriced`.
