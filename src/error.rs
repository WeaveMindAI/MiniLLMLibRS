//! Error types for MiniLLMLib

use thiserror::Error;

/// Main error type for MiniLLMLib operations
#[derive(Debug, Error)]
pub enum MiniLLMError {
    /// HTTP request failed
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// JSON repair failed
    #[error("JSON repair error: {0}")]
    JsonRepair(#[from] crate::json_repair::JsonRepairError),

    /// API returned an error response
    #[error("API error ({status}): {message}")]
    Api { status: u16, message: String },

    /// Invalid parameter
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),

    /// Stream error during SSE
    #[error("Stream error: {0}")]
    Stream(String),

    /// IO error (file operations)
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Base64 encoding/decoding error
    #[error("Base64 error: {0}")]
    Base64(#[from] base64::DecodeError),

    /// Timeout error
    #[error("Request timed out")]
    Timeout,

    /// The model returned an empty response (when crash_on_empty_response is set)
    #[error("Model returned an empty response")]
    EmptyResponse,

    /// The model returned no usable JSON (when crash_on_refusal is set).
    /// Carries the raw response content for diagnosis.
    #[error("Model returned no usable JSON: {0}")]
    NoJsonFound(String),

    /// The API returned a 2xx body that is not a well-formed completion
    /// (no choices/message and no error object). Carries a diagnostic preview.
    #[error("Malformed completion response: {0}")]
    MalformedResponse(String),

    /// All retry attempts were exhausted without a successful completion.
    /// Carries the last underlying error.
    #[error("Max retries exceeded: {0}")]
    MaxRetriesExceeded(Box<MiniLLMError>),

    /// A thread had no messages to load.
    #[error("Thread has no messages")]
    EmptyThread,
}

/// Result type alias for MiniLLMLib operations
pub type Result<T> = std::result::Result<T, MiniLLMError>;
