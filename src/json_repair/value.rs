//! JSON Value type for representing parsed JSON data
//!
//! This is our own JSON value type that we use during parsing.
//! It's similar to serde_json::Value but tailored for our repair needs.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON null
    Null,
    /// JSON boolean (true/false)
    Bool(bool),
    /// JSON integer number
    Integer(i64),
    /// JSON floating-point number
    Float(f64),
    /// JSON string
    String(String),
    /// JSON array (ordered list of values)
    Array(Vec<JsonValue>),
    /// JSON object (key-value map)
    /// We use a Vec of tuples to preserve insertion order (important for LLM outputs)
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Convert this value to a JSON string representation
    /// If ensure_ascii is true, non-ASCII characters are escaped to \uXXXX
    pub fn to_json_string(&self) -> String {
        self.to_json_string_with_options(true)
    }
    
    /// Convert this value to a JSON string with configurable ASCII escaping
    pub fn to_json_string_with_options(&self, ensure_ascii: bool) -> String {
        match self {
            JsonValue::Null => "null".to_string(),
            JsonValue::Bool(b) => if *b { "true" } else { "false" }.to_string(),
            JsonValue::Integer(n) => n.to_string(),
            JsonValue::Float(f) => {
                // Handle special float formatting
                if f.fract() == 0.0 && f.abs() < 1e15 {
                    format!("{:.1}", f)
                } else {
                    f.to_string()
                }
            }
            JsonValue::String(s) => format!("\"{}\"", escape_json_string(s, ensure_ascii)),
            JsonValue::Array(arr) => {
                let items: Vec<String> = arr.iter().map(|v| v.to_json_string_with_options(ensure_ascii)).collect();
                format!("[{}]", items.join(", "))
            }
            JsonValue::Object(obj) => {
                let items: Vec<String> = obj
                    .iter()
                    .map(|(k, v)| format!("\"{}\": {}", escape_json_string(k, ensure_ascii), v.to_json_string_with_options(ensure_ascii)))
                    .collect();
                format!("{{{}}}", items.join(", "))
            }
        }
    }

    /// Check if this value is empty (empty string, array, or object)
    pub fn is_empty(&self) -> bool {
        match self {
            JsonValue::String(s) => s.is_empty(),
            JsonValue::Array(arr) => arr.is_empty(),
            JsonValue::Object(obj) => obj.is_empty(),
            _ => false,
        }
    }

    /// Check if this value is strictly empty (empty container, not null/false/0)
    pub fn is_strictly_empty(&self) -> bool {
        match self {
            JsonValue::String(s) => s.is_empty(),
            JsonValue::Array(arr) => arr.is_empty(),
            JsonValue::Object(obj) => obj.is_empty(),
            _ => false,
        }
    }
}

/// Escape special characters in a JSON string
/// If ensure_ascii is true, non-ASCII characters are escaped to \uXXXX
/// If ensure_ascii is false, non-ASCII characters are preserved as-is
fn escape_json_string(s: &str, ensure_ascii: bool) -> String {
    let mut result = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => result.push_str("\\\""),
            '\\' => result.push_str("\\\\"),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            '\x08' => result.push_str("\\b"),
            '\x0c' => result.push_str("\\f"),
            c if c.is_control() => {
                // Unicode escape for control characters
                result.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if !c.is_ascii() && ensure_ascii => {
                // Escape non-ASCII characters to \uXXXX (matching Python's ensure_ascii=True)
                let code = c as u32;
                if code <= 0xFFFF {
                    result.push_str(&format!("\\u{:04x}", code));
                } else {
                    // For characters outside BMP, use surrogate pairs
                    let code = code - 0x10000;
                    let high = 0xD800 + (code >> 10);
                    let low = 0xDC00 + (code & 0x3FF);
                    result.push_str(&format!("\\u{:04x}\\u{:04x}", high, low));
                }
            }
            c => result.push(c),
        }
    }
    result
}

impl From<serde_json::Value> for JsonValue {
    fn from(value: serde_json::Value) -> Self {
        match value {
            serde_json::Value::Null => JsonValue::Null,
            serde_json::Value::Bool(b) => JsonValue::Bool(b),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    JsonValue::Integer(i)
                } else if let Some(f) = n.as_f64() {
                    JsonValue::Float(f)
                } else {
                    // Fallback for very large numbers
                    JsonValue::String(n.to_string())
                }
            }
            serde_json::Value::String(s) => JsonValue::String(s),
            serde_json::Value::Array(arr) => {
                JsonValue::Array(arr.into_iter().map(JsonValue::from).collect())
            }
            serde_json::Value::Object(obj) => {
                // Clean up keys: strip trailing whitespace/newlines (common LLM issue)
                JsonValue::Object(obj.into_iter().map(|(k, v)| {
                    let clean_key = k.trim_end_matches(|c: char| c.is_whitespace()).to_string();
                    (clean_key, JsonValue::from(v))
                }).collect())
            }
        }
    }
}

/// Display implementation for debugging
impl fmt::Display for JsonValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_json_string())
    }
}

/// Default value is Null
impl Default for JsonValue {
    fn default() -> Self {
        JsonValue::Null
    }
}
