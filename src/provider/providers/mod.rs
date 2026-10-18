//! Built-in [`Provider`](super::wire::Provider) implementations, one
//! file per provider: each file owns EVERYTHING specific to its
//! provider (wire quirks, cost model, parsing where the envelope
//! differs) plus that provider's tests. `catalog` reads OpenRouter's
//! model catalog, which every built-in provider prices from. Shared
//! dialect code lives in `super::openai_wire`; nothing provider-specific
//! lives outside these files.

mod anthropic;
mod catalog;
mod generic;
mod openai;
mod openrouter;

pub use anthropic::AnthropicProvider;
pub use catalog::{catalog_call, catalog_info};
pub use generic::GenericProvider;
pub use openai::OpenAiProvider;
pub use openrouter::{OpenRouterProvider, OpenRouterRouting};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::CompletionParameters;
    use crate::message::Message;
    use crate::provider::Provider;

    /// Every built-in provider, each with the request-body key its wire uses
    /// for the output cap, and whether its API requires that key.
    fn providers() -> Vec<(Box<dyn Provider>, &'static str, bool)> {
        vec![
            (Box::new(OpenAiProvider), "max_completion_tokens", false),
            (Box::new(OpenRouterProvider), "max_completion_tokens", false),
            (
                Box::new(GenericProvider::default()),
                "max_completion_tokens",
                false,
            ),
            (
                Box::new(GenericProvider {
                    legacy_token_limit: true,
                }),
                "max_tokens",
                false,
            ),
            (Box::new(AnthropicProvider), "max_tokens", true),
        ]
    }

    /// The request settings `CompletionParameters` can put on a wire. None of
    /// them may appear in a minimal call's body unless the API requires it.
    const SETTING_KEYS: &[&str] = &[
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "stop",
        "stop_sequences",
        "seed",
        "response_format",
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "reasoning",
        "reasoning_effort",
        "thinking",
        "output_config",
    ];

    /// A minimal call carries no setting nobody asked for, on every provider.
    /// A wire whose API requires a cap is built from params that already carry
    /// one ([`GeneratorInfo::request_params`](crate::GeneratorInfo::request_params)
    /// fills it), so it is checked with that cap and sends only that.
    #[test]
    fn a_minimal_call_sends_no_setting() {
        let messages = vec![Message::user("Hi")];
        for (provider, cap_key, requires_cap) in providers() {
            assert_eq!(provider.requires_max_tokens(), requires_cap, "{provider:?}");
            let params = CompletionParameters {
                max_tokens: requires_cap.then_some(64_000),
                ..Default::default()
            };
            for stream in [false, true] {
                let body = provider
                    .build_request("m", &messages, &params, stream, true)
                    .unwrap();
                for key in SETTING_KEYS {
                    match body.get(*key) {
                        Some(v) if requires_cap && *key == cap_key => {
                            assert_eq!(v, 64_000, "{provider:?}: {body}")
                        }
                        Some(_) => panic!("{provider:?} sent `{key}` unasked: {body}"),
                        None => {}
                    }
                }
            }
        }
    }

    /// What the caller sets goes through unchanged, under the wire's own key.
    #[test]
    fn explicit_temperature_and_max_tokens_go_through_unchanged() {
        let messages = vec![Message::user("Hi")];
        let params = CompletionParameters::new()
            .with_temperature(0.3)
            .with_max_tokens(77);
        for (provider, cap_key, _) in providers() {
            let body = provider
                .build_request("m", &messages, &params, false, true)
                .unwrap();
            assert_eq!(body["temperature"], 0.3, "{provider:?}: {body}");
            assert_eq!(body[cap_key], 77, "{provider:?}: {body}");
        }
    }
}
