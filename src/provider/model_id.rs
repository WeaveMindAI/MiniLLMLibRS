//! Model ids that go into a URL path: a price read, a model lookup.

use crate::error::{MiniLLMError, Result};

/// Every character a real model id is made of. Stated positively, so a new
/// separator a vendor invents fails loudly here rather than silently reshaping a
/// URL.
fn is_model_id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':' | '/')
}

/// Refuse anything that is not a model id.
///
/// A model id is caller-supplied and legitimately contains `/`
/// (`anthropic/claude-sonnet-4.6`). Everything else is refused rather than
/// escaped: `evil?free=1` would push the rest of the path into a query string
/// and hit a different route, and `../../../models` would climb out of the API
/// version prefix. Both would return a plausible response for the wrong thing.
/// Escaping does not save either: a URL parser decodes `%2E%2E` and traverses
/// anyway, and the `url` crate's segment writer silently drops dot segments,
/// which would fetch answers for a DIFFERENT model. A wrong answer is worse
/// than an error.
pub(crate) fn validate_model_id(model: &str) -> Result<()> {
    let refuse = |why: &str| {
        Err(MiniLLMError::InvalidParameter(format!(
            "model id {model:?} {why}, so it cannot go in a lookup URL"
        )))
    };

    if model.is_empty() {
        return refuse("is empty");
    }
    if let Some(bad) = model.chars().find(|c| !is_model_id_char(*c)) {
        return refuse(&format!("contains {bad:?}, which no model id contains"));
    }
    // Made of legal characters, yet `.` and `..` mean "here" and "up one" to every
    // URL parser. A model is never named either.
    if model
        .split('/')
        .any(|segment| segment.is_empty() || segment == "." || segment == "..")
    {
        return refuse("has an empty or dotted path segment");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An id that is not a model id is refused, never escaped.
    ///
    /// Escaping does not work: a URL parser decodes `%2E%2E` and traverses anyway,
    /// and the `url` crate's segment writer silently drops the dot segments, which
    /// would fetch answers for a DIFFERENT model. Both were verified against
    /// `reqwest::Url`. A wrong answer is worse than an error.
    #[test]
    fn an_id_that_is_not_a_model_id_is_refused() {
        let hostile = [
            ("evil?free=1", "a query would swallow the /endpoints suffix"),
            ("x#frag", "a fragment would truncate the path"),
            (
                "../../../models",
                "traversal would climb out of the api version",
            ),
            ("a/../b", "traversal, buried mid-path"),
            ("a/./b", "a dot segment resolves away"),
            ("a//b", "an empty segment collapses"),
            ("a b", "a space is not a model id character"),
            ("a\nb", "nor is a newline"),
            ("a%2fb", "nor is a percent escape"),
            ("", "an empty id names nothing"),
        ];
        for (bad, why) in hostile {
            let Err(err) = validate_model_id(bad) else {
                panic!("{bad:?} was accepted, but {why}");
            };
            assert!(
                matches!(err, MiniLLMError::InvalidParameter(_)),
                "{bad:?}: {err:?}"
            );
        }
    }

    /// A real model id is accepted, separator and all.
    #[test]
    fn a_real_model_id_is_accepted() {
        for real in [
            "anthropic/claude-sonnet-4.6",
            "openai/gpt-5.5",
            "o3",
            "z-ai/glm-5.2",
        ] {
            assert!(validate_model_id(real).is_ok(), "{real} is a real id");
        }
    }
}
