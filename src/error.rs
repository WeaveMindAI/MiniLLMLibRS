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

    /// Missing required configuration
    #[error("Missing configuration: {0}")]
    MissingConfig(String),

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

    /// URL parsing error
    #[error("URL error: {0}")]
    Url(#[from] url::ParseError),

    /// Timeout error
    #[error("Request timed out")]
    Timeout,

    /// Node not found in conversation tree
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// Generic error with message
    #[error("{0}")]
    Other(String),
}

/// Result type alias for MiniLLMLib operations
pub type Result<T> = std::result::Result<T, MiniLLMError>;
