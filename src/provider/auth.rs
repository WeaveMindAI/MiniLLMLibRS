//! Authentication strategy for a generator.
//!
//! Auth is deliberately NOT a property of the provider *dialect*: the same
//! provider (e.g. [`AnthropicProvider`](super::AnthropicProvider)) must serve both
//! a pay-as-you-go API key and a Claude **subscription** OAuth bearer token. So
//! the strategy lives on [`GeneratorInfo`](crate::GeneratorInfo), and each
//! provider maps it to ITS concrete wire headers via
//! [`Provider::auth_headers`](super::Provider::auth_headers):
//! - an OpenAI-wire provider turns `ApiKey`/`BearerToken` into
//!   `Authorization: Bearer <secret>`,
//! - Anthropic turns `ApiKey` into `x-api-key: <key>` and `BearerToken` into
//!   `Authorization: Bearer <token>` (the subscription path).

use secrecy::SecretString;

/// How a request authenticates. The provider decides which header(s) express it.
#[derive(Clone, Default)]
pub enum Auth {
    /// A provider-issued API key. The provider chooses the header: OpenAI-wire
    /// uses `Authorization: Bearer <key>`, Anthropic uses `x-api-key: <key>`.
    ApiKey(SecretString),

    /// An OAuth/bearer token (e.g. a Claude Pro/Max subscription token from
    /// `ant auth print-credentials`). Always carried as `Authorization: Bearer
    /// <token>`. Draws on the subscription's quota rather than API billing.
    BearerToken(SecretString),

    /// No authentication (a local or pre-authenticated gateway).
    #[default]
    None,
}

impl Auth {
    /// The underlying secret, if any (key or token). Used by out-of-band cost
    /// queries that must re-authenticate (e.g. OpenRouter's `/generation`).
    pub fn secret(&self) -> Option<&SecretString> {
        match self {
            Auth::ApiKey(s) | Auth::BearerToken(s) => Some(s),
            Auth::None => None,
        }
    }

    /// Whether any credential is present.
    pub fn is_some(&self) -> bool {
        !matches!(self, Auth::None)
    }
}

impl std::fmt::Debug for Auth {
    /// Never print the secret material.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Auth::ApiKey(_) => f.write_str("Auth::ApiKey(***)"),
            Auth::BearerToken(_) => f.write_str("Auth::BearerToken(***)"),
            Auth::None => f.write_str("Auth::None"),
        }
    }
}
