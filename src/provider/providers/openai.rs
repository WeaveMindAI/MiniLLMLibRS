//! OpenAI: the native platform speaking its own dialect unmodified.
//! Everything rides the trait defaults; the only OpenAI-specific facts
//! are the streaming usage opt-in and token-only pricing.

use super::super::response::Usage;
use super::super::wire::{
    price_or_unpriced, CostOutcome, InfoFuture, ModelLookup, Provider, TokenPrice,
};

/// OpenAI: OpenAI-wire, returns token counts but no dollar cost (price them via a
/// configured `TokenPrice`). Streaming usage requires the
/// `stream_options:{include_usage:true}` opt-in.
#[derive(Debug, Clone, Default)]
pub struct OpenAiProvider;

impl Provider for OpenAiProvider {
    /// The model from OpenRouter's catalog, at OpenAI's own rate (OpenAI
    /// publishes no prices in its API). An OpenAI id is found there as
    /// `openai/<id>`, unless the generator names another catalog id.
    fn model_info(&self, lookup: ModelLookup) -> Option<InfoFuture> {
        let model = lookup
            .openrouter_name
            .unwrap_or_else(|| format!("openai/{}", lookup.model));
        let call = super::catalog_call(lookup.call.client);
        Some(Box::pin(async move {
            super::catalog_info(&call, &model, Some("openai")).await
        }))
    }

    fn openai_request_usage(&self, body: &mut serde_json::Value, stream: bool) {
        // OpenAI only emits a usage chunk on streaming when explicitly asked.
        if stream {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }
    }

    // parse_usage uses the default (`parse_openai_usage_field`): OpenAI reports
    // no native cost fields, so the base OpenAI-wire parse is exactly right.

    /// OpenAI reports no native cost: the tokens at the model's price, or
    /// `Unpriced` when it is unknown.
    fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
        price_or_unpriced(usage, price)
    }

    // No out-of-band endpoint: a cancelled stream that never delivered usage
    // has no usage to find, so the default `usage_out_of_band` (none, so the
    // cost is Unknown) is right. A stream that DID deliver usage prices it via
    // `cost_of`.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::CostResolution;

    /// A fully-uncached usage (all input in the full-price bucket).
    fn usage(prompt: u32, completion: u32) -> Usage {
        Usage {
            uncached_input_tokens: prompt,
            completion_tokens: completion,
            ..Default::default()
        }
    }

    /// OpenAI's own dialect: `reasoning_effort`, `"none"` for off, `"low"`
    /// for on with no effort, and the caller's word as written otherwise.
    #[test]
    fn openai_reasoning_is_reasoning_effort() {
        use crate::generator::{CompletionParameters, Reasoning};
        use crate::message::Message;
        let p = OpenAiProvider;
        let messages = vec![Message::user("Hi")];
        for (reasoning, wire) in [
            (Reasoning::Off, "none"),
            (Reasoning::On(None), "low"),
            (Reasoning::On(Some("xhigh".into())), "xhigh"),
        ] {
            let params = CompletionParameters::new().with_reasoning(reasoning);
            let body = p
                .build_request("gpt-5", &messages, &params, false, true)
                .unwrap();
            assert_eq!(body["reasoning_effort"], wire);
            assert!(body.get("reasoning").is_none(), "{body}");
        }
    }

    #[test]
    fn openai_is_unpriced_without_a_price_and_resolved_with_one() {
        let acct = OpenAiProvider;
        let unpriced = acct.cost_of(usage(100, 50), None);
        assert_eq!(unpriced.resolution, CostResolution::Unpriced);
        assert_eq!(unpriced.usd, 0.0);
        // tokens survive so the consumer can price them later
        assert_eq!(unpriced.usage.prompt_tokens(), 100);

        let price = TokenPrice::new(1.0, 1.0); // $1/Mtok both ways
        let resolved = acct.cost_of(usage(1_000_000, 0), Some(&price));
        assert_eq!(resolved.resolution, CostResolution::Resolved);
        assert!((resolved.usd - 1.0).abs() < 1e-9);
    }
}
