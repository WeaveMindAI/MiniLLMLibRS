//! JSON Repair module - fixes malformed JSON from LLM outputs
//!
//! This is a Rust port of the Python `json_repair` library by Stefano Baccianella.
//! It handles common JSON errors produced by language models:
//! - Missing or mismatched quotes
//! - Trailing commas
//! - Single quotes instead of double quotes
//! - Unquoted keys/values
//! - Missing closing brackets
//! - Comments (// and /* */)
//! - Code fences (```json ... ```)

mod context;
mod parser;
mod value;

// Re-export the public API
pub use parser::RepairOptions;
pub use value::JsonValue;

use parser::JsonParser;

/// Repair a malformed JSON string and return the fixed JSON as a string.
///
/// # Arguments
/// * `json_str` - The potentially malformed JSON string
///
/// # Returns
/// The repaired JSON string, or an error if repair is impossible
///
/// # Example
/// ```
/// use minillmlib::json_repair::repair_json;
///
/// let broken = r#"{'name': 'John', age: 30,}"#;
/// let fixed = repair_json(broken, &Default::default()).unwrap();
/// assert_eq!(fixed, r#"{"name": "John", "age": 30}"#);
/// ```
pub fn repair_json(json_str: &str, options: &RepairOptions) -> Result<String, JsonRepairError> {
    // Both the standard-parse and repair paths converge through `loads`, so the
    // empty-string special case below applies uniformly regardless of which path
    // produced the value.
    let value = loads(json_str, options)?;

    // Handle empty string specially - return empty string, not quoted empty.
    if let JsonValue::String(s) = &value {
        if s.is_empty() {
            return Ok(String::new());
        }
    }

    Ok(value.to_json_string_with_options(options.ensure_ascii))
}

/// Parse a malformed JSON string and return the repaired value as a JsonValue.
///
/// This is similar to Python's `json.loads()` but with repair capabilities.
///
/// # Arguments
/// * `json_str` - The potentially malformed JSON string
///
/// # Returns
/// The parsed and repaired JSON value
pub fn loads(json_str: &str, options: &RepairOptions) -> Result<JsonValue, JsonRepairError> {
    // First, try standard JSON parsing
    if !options.skip_json_loads {
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
            return Ok(JsonValue::from(value));
        }
    }

    // Parse with our repair parser
    let mut parser = JsonParser::new(json_str, options);
    parser.parse()
}

/// Errors that can occur during JSON repair
#[derive(Debug, thiserror::Error)]
pub enum JsonRepairError {
    #[error("JSON serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Strict mode violation: {0}")]
    StrictModeError(String),

    #[error("Parse error: {0}")]
    ParseError(String),
}

#[cfg(test)]
mod tests;
