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

/// Resolve a Claude **subscription** bearer token, env superseding the on-disk
/// Claude Code credential.
///
/// 1. `ANTHROPIC_AUTH_TOKEN` if set and non-empty (explicit override; the caller
///    is responsible for keeping it fresh, e.g. from `claude setup-token`);
/// 2. else the Claude Code credential at `~/.claude/.credentials.json`
///    (`claudeAiOauth.accessToken`), the live Pro/Max subscription token, which
///    Claude Code refreshes on disk, so reading it each call always gets a current
///    token without the library having to manage OAuth refresh.
///
/// Returns [`Auth::None`] when neither source yields a token (the request then
/// fails loudly as unauthenticated, rather than silently using the wrong account).
pub fn resolve_claude_subscription_auth() -> Auth {
    if let Ok(token) = std::env::var("ANTHROPIC_AUTH_TOKEN") {
        if !token.trim().is_empty() {
            return Auth::BearerToken(SecretString::from(token));
        }
    }
    match dirs_home().map(|h| h.join(".claude/.credentials.json")) {
        Some(path) => match read_claude_code_token(&path) {
            Some(token) => Auth::BearerToken(SecretString::from(token)),
            None => Auth::None,
        },
        None => Auth::None,
    }
}

/// The user's home directory (`$HOME`), or `None` if unset.
fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

/// Extract `claudeAiOauth.accessToken` from a Claude Code credentials file.
/// Pure over the file contents (the path is read here, parsing is in
/// [`parse_claude_code_token`]) so the parse is unit-testable without a real file.
fn read_claude_code_token(path: &std::path::Path) -> Option<String> {
    let contents = std::fs::read_to_string(path).ok()?;
    parse_claude_code_token(&contents)
}

/// Parse the subscription access token out of a Claude Code credentials JSON body.
fn parse_claude_code_token(contents: &str) -> Option<String> {
    let json: serde_json::Value = serde_json::from_str(contents).ok()?;
    json["claudeAiOauth"]["accessToken"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(String::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_claude_code_subscription_token() {
        let body = r#"{"claudeAiOauth":{"accessToken":"sk-ant-oat01-abc",
            "refreshToken":"sk-ant-ort01-x","subscriptionType":"max"}}"#;
        assert_eq!(
            parse_claude_code_token(body).as_deref(),
            Some("sk-ant-oat01-abc")
        );
    }

    #[test]
    fn missing_or_empty_token_is_none() {
        assert!(parse_claude_code_token(r#"{"claudeAiOauth":{}}"#).is_none());
        assert!(parse_claude_code_token(r#"{"claudeAiOauth":{"accessToken":""}}"#).is_none());
        assert!(parse_claude_code_token("not json").is_none());
        assert!(parse_claude_code_token(r#"{"other":"shape"}"#).is_none());
    }
}
