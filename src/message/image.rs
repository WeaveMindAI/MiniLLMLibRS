//! Image data handling

use super::media::MediaData;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Image data for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageData {
    /// Base64-encoded image data, OR the URL verbatim when [`is_url`](Self::is_url).
    pub base64_data: String,

    /// MIME type (e.g., "image/png", "image/jpeg"). Empty for a URL reference.
    pub mime_type: String,

    /// Whether `base64_data` holds a remote URL rather than inline base64. Explicit
    /// flag, NOT a magic `mime_type == "url"` value, so no caller-supplied mime
    /// string can turn inline bytes into a counterfeit URL.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_url: bool,

    /// Optional detail level for vision models ("low", "high", "auto")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl MediaData for ImageData {
    fn base64_data(&self) -> &str {
        &self.base64_data
    }

    fn mime_type(&self) -> String {
        self.mime_type.clone()
    }

    fn is_url(&self) -> bool {
        self.is_url
    }

    fn from_base64(base64_data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Self {
            base64_data: base64_data.into(),
            mime_type: mime_type.into(),
            is_url: false,
            detail: None,
        }
    }

    fn guess_format(path: &Path) -> Option<String> {
        // Images have an honest unknown default (`application/octet-stream` is a
        // valid binary content type), so this never returns `None`.
        let mime = match path.extension().and_then(|e| e.to_str()) {
            Some("png") => "image/png",
            Some("jpg") | Some("jpeg") => "image/jpeg",
            Some("gif") => "image/gif",
            Some("webp") => "image/webp",
            Some("bmp") => "image/bmp",
            Some("svg") => "image/svg+xml",
            _ => "application/octet-stream",
        };
        Some(mime.to_string())
    }
}

// Shared inherent forwarders (from_base64/from_bytes/from_file/from_file_async/
// to_bytes/mime_type/to_data_url) generated once for every media type.
crate::impl_media_forwarders!(ImageData, mime_type);

impl ImageData {
    /// Create ImageData from a URL (the URL will be used directly)
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            base64_data: url.into(),
            mime_type: String::new(),
            is_url: true,
            detail: None,
        }
    }

    /// Set the detail level
    pub fn with_detail(mut self, detail: impl Into<String>) -> Self {
        self.detail = Some(detail.into());
        self
    }
}
