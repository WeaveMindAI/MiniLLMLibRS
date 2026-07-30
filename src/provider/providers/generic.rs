//! Generic OpenAI-compatible server: the trait defaults with the two
//! honest unknowns of a bare server (legacy token-limit key on older
//! ones, and no usage chunk to wait for).

use super::super::response::Usage;
use super::super::wire::{price_or_unpriced, CostOutcome, Provider, TokenPrice};

/// A minimal OpenAI-compatible provider: token counts only, no native cost, no
/// usage opt-in flag assumed, no out-of-band endpoint. The default for
/// [`GeneratorInfo::custom`](crate::GeneratorInfo::custom).
#[derive(Debug, Clone, Default)]
pub struct GenericProvider {
    /// Some older OpenAI-compatible servers only accept the legacy `max_tokens`
    /// request key. Set true for those.
    pub legacy_token_limit: bool,
}

impl Provider for GenericProvider {
    fn openai_token_limit_field(&self) -> &'static str {
        if self.legacy_token_limit {
            "max_tokens"
        } else {
            "max_completion_tokens"
        }
    }

    /// A bare OpenAI-compatible server has no usage opt-in (the default
    /// `openai_request_usage` is a no-op) and may never emit a usage chunk, so the
    /// streaming reader must NOT wait for one (it would wedge the stream until the
    /// idle timeout). Cost is still parsed opportunistically if one arrives.
    fn emits_stream_usage(&self, _requested: bool) -> bool {
        false
    }

    // parse_usage uses the default (`parse_openai_usage_field`).

    fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
        price_or_unpriced(usage, price)
    }
}
