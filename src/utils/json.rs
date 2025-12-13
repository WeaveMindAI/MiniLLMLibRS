//! JSON utility functions

use crate::error::{MiniLLMError, Result};
use crate::json_repair::{repair_json, RepairOptions};

/// Extract and repair JSON from an LLM completion string
///
/// This function handles common issues with LLM JSON output:
/// - Markdown code fences (```json ... ```)
/// - Missing quotes
/// - Trailing commas
/// - Single quotes instead of double quotes
pub fn extract_json(completion: &str) -> Result<String> {
    let repaired = repair_json(completion, &RepairOptions::default())?;
    Ok(repaired)
}

/// Extract JSON and parse into a serde_json::Value
pub fn extract_json_value(completion: &str) -> Result<serde_json::Value> {
    let repaired = extract_json(completion)?;
    let value: serde_json::Value = serde_json::from_str(&repaired)?;
    Ok(value)
}

/// Convert any serializable value to a JSON dictionary representation
///
/// This is similar to Python's recursive to_dict function that converts
/// objects with __dict__ to dictionaries.
pub fn to_dict<T: serde::Serialize>(value: &T) -> Result<serde_json::Value> {
    let json = serde_json::to_value(value)?;
    Ok(json)
}

/// Pretty print a JSON value
pub fn pretty_json<T: serde::Serialize>(value: &T) -> Result<String> {
    let json = serde_json::to_string_pretty(value)?;
    Ok(json)
}

/// Validate and extract content from an API JSON response
///
/// This validates that the response has the expected structure:
/// - `choices` array exists
/// - First choice has `message.content`
///
/// Returns the content string or an error with details.
pub fn validate_json_response(response: &serde_json::Value) -> Result<String> {
    let choices = response.get("choices").ok_or_else(|| {
        MiniLLMError::Other(format!("Missing 'choices' key in response: {}", response))
    })?;

    let choices_arr = choices
        .as_array()
        .ok_or_else(|| MiniLLMError::Other(format!("'choices' must be an array: {}", response)))?;

    if choices_arr.is_empty() {
        return Err(MiniLLMError::Other(format!(
            "'choices' array is empty: {}",
            response
        )));
    }

    let first_choice = &choices_arr[0];

    let message = first_choice.get("message").ok_or_else(|| {
        MiniLLMError::Other(format!("Missing 'message' in first choice: {}", response))
    })?;

    let content = message.get("content").ok_or_else(|| {
        MiniLLMError::Other(format!("Missing 'content' in message: {}", response))
    })?;

    content
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| MiniLLMError::Other(format!("'content' is not a string: {}", content)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_json_simple() {
        let input = r#"{"key": "value"}"#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_with_markdown() {
        let input = r#"```json
{"key": "value"}
```"#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_with_single_quotes() {
        let input = "{'key': 'value'}";
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }

    #[test]
    fn test_extract_json_with_trailing_comma() {
        let input = r#"{"key": "value",}"#;
        let result = extract_json(input).unwrap();
        assert_eq!(result, r#"{"key": "value"}"#);
    }
}
