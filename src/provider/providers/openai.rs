//! OpenAI: the native platform speaking its own dialect unmodified.
//! Everything rides the trait defaults; the only OpenAI-specific facts
//! are the streaming usage opt-in and token-only pricing.

use super::super::response::Usage;
use super::super::wire::{price_or_unpriced, CostOutcome, Provider, TokenPrice};

/// OpenAI: OpenAI-wire, returns token counts but no dollar cost (price them via a
/// configured `TokenPrice`). Streaming usage requires the
/// `stream_options:{include_usage:true}` opt-in.
#[derive(Debug, Clone, Default)]
pub struct OpenAiProvider;

impl Provider for OpenAiProvider {
    fn openrouter_slug(&self) -> Option<&'static str> {
        Some("openai")
    }

    fn openai_request_usage(&self, body: &mut serde_json::Value, stream: bool) {
        // OpenAI only emits a usage chunk on streaming when explicitly asked.
        if stream {
            body["stream_options"] = serde_json::json!({ "include_usage": true });
        }
    }

    // parse_usage uses the default (`parse_openai_usage_field`): OpenAI reports
    // no native cost fields, so the base OpenAI-wire parse is exactly right.

    /// OpenAI reports no native cost; price tokens via `TokenPrice` or report
    /// `Unpriced`.
    fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
        price_or_unpriced(usage, price)
    }

    // No out-of-band endpoint: a cancelled stream that never delivered usage is
    // genuinely unresolvable, so the default `resolve_post_stream` (Unknown) is
    // correct. A stream that DID deliver usage prices it via `cost_of`.
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
