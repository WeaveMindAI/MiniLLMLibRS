//! Authentication for a generator.
//!
//! Auth is deliberately NOT a property of the provider *dialect*: the same
//! credential means different headers on different wires. So the credential
//! lives on [`GeneratorInfo`](crate::GeneratorInfo), and each provider maps it
//! to ITS concrete wire headers via
//! [`Provider::auth_headers`](super::Provider::auth_headers):
//! - an OpenAI-wire provider turns `ApiKey`/`BearerToken` into
//!   `Authorization: Bearer <secret>`,
//! - Anthropic turns `ApiKey` into `x-api-key: <key>` and `BearerToken` into
//!   `Authorization: Bearer <token>`.

use secrecy::SecretString;

/// What a generator's requests authenticate with. The provider decides which
/// header(s) express it.
#[derive(Clone, Default)]
pub enum Auth {
    /// A provider-issued API key. The provider chooses the header: OpenAI-wire
    /// uses `Authorization: Bearer <key>`, Anthropic uses `x-api-key: <key>`.
    ApiKey(SecretString),

    /// An OAuth/bearer token. Always carried as `Authorization: Bearer <token>`.
    BearerToken(SecretString),

    /// No authentication (a local or pre-authenticated gateway).
    #[default]
    None,
}

impl Auth {
    /// The secret, if any (key or token), for a provider that sends it in a
    /// header of its own (see the custom-provider guide).
    pub fn secret(&self) -> Option<&SecretString> {
        match self {
            Auth::ApiKey(s) | Auth::BearerToken(s) => Some(s),
            Auth::None => None,
        }
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
