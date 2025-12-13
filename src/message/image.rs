//! Image data handling

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Image data for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// Base64-encoded image data
    pub base64_data: String,

    /// MIME type (e.g., "image/png", "image/jpeg")
    pub mime_type: String,

    /// Optional detail level for vision models ("low", "high", "auto")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ImageData {
    /// Create ImageData from base64 string
    pub fn from_base64(base64_data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            base64_data: base64_data.into(),
            mime_type: mime_type.into(),
            detail: None,
        }
    }

    /// Create ImageData from raw bytes
    pub fn from_bytes(bytes: &[u8], mime_type: impl Into<String>) -> Self {
        Self {
            base64_data: BASE64.encode(bytes),
            mime_type: mime_type.into(),
            detail: None,
        }
    }

    /// Load ImageData from a file path
    pub fn from_file(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let mime_type = Self::guess_mime_type(path);
        Ok(Self::from_bytes(&bytes, mime_type))
    }

    /// Load ImageData from a file path (async)
    pub async fn from_file_async(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        let path = path.as_ref();
        let bytes = tokio::fs::read(path).await?;
        let mime_type = Self::guess_mime_type(path);
        Ok(Self::from_bytes(&bytes, mime_type))
    }

    /// Create ImageData from a URL (the URL will be used directly)
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            base64_data: url.into(),
            mime_type: "url".to_string(),
            detail: None,
        }
    }

    /// Set the detail level
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }

    /// Convert to data URL format for API
    pub fn to_data_url(&self) -> String {
        if self.mime_type == "url" {
            // Already a URL, return as-is
            self.base64_data.clone()
        } else {
            format!("data:{};base64,{}", self.mime_type, self.base64_data)
        }
    }

    /// Decode the base64 data to bytes
    pub fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        Ok(BASE64.decode(&self.base64_data)?)
    }

    /// Guess MIME type from file extension
    fn guess_mime_type(path: &Path) -> String {
        match path.extension().and_then(|e| e.to_str()) {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("bmp") => "image/bmp",
            Some("svg") => "image/svg+xml",
            _ => "application/octet-stream",
        }
        .to_string()
    }
}
