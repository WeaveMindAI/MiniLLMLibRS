//! Video data handling

use super::media::MediaData;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Video data for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoData {
    /// Base64-encoded video data
    pub base64_data: String,

    /// Video format (e.g., "mp4", "webm", "mov")
    pub format: String,

    /// Duration in seconds (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,

    /// Width in pixels (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,

    /// Height in pixels (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,

    /// Frame rate (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frame_rate: Option<f32>,
}

impl MediaData for VideoData {
    fn base64_data(&self) -> &str {
        &self.base64_data
    }

    fn format_id(&self) -> &str {
        &self.format
    }

    fn mime_type(&self) -> String {
        match self.format.as_str() {
            "mp4" => "video/mp4",
            "webm" => "video/webm",
            "mov" => "video/quicktime",
            "avi" => "video/x-msvideo",
            "mkv" => "video/x-matroska",
            "flv" => "video/x-flv",
            "wmv" => "video/x-ms-wmv",
            "m4v" => "video/x-m4v",
            "3gp" => "video/3gpp",
            "ogv" => "video/ogg",
            _ => "video/mp4",
        }
        .to_string()
    }

    fn from_base64(base64_data: impl Into<String>, format: impl Into<String>) -> Self {
        Self {
            base64_data: base64_data.into(),
            format: format.into(),
            duration_secs: None,
            width: None,
            height: None,
            frame_rate: None,
        }
    }

    fn guess_format(path: &Path) -> String {
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("mp4")
            .to_lowercase()
    }
}

impl VideoData {
    /// Create VideoData from base64 string
    pub fn from_base64(base64_data: impl Into<String>, format: impl Into<String>) -> Self {
        <Self as MediaData>::from_base64(base64_data, format)
    }

    /// Create VideoData from raw bytes
    pub fn from_bytes(bytes: &[u8], format: impl Into<String>) -> Self {
        <Self as MediaData>::from_bytes(bytes, format)
    }

    /// Load VideoData from a file path
    pub fn from_file(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        <Self as MediaData>::from_file(path)
    }

    /// Load VideoData from a file path (async)
    pub async fn from_file_async(path: impl AsRef<Path> + Send) -> crate::error::Result<Self> {
        <Self as MediaData>::from_file_async(path).await
    }

    /// Create VideoData from a URL (the URL will be used directly)
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            base64_data: url.into(),
            format: "url".to_string(),
            duration_secs: None,
            width: None,
            height: None,
            frame_rate: None,
        }
    }

    /// Set duration in seconds
    pub fn with_duration(mut self, duration_secs: f64) -> Self {
        self.duration_secs = Some(duration_secs);
        self
    }

    /// Set dimensions
    pub fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = Some(width);
        self.height = Some(height);
        self
    }

    /// Set frame rate
    pub fn with_frame_rate(mut self, frame_rate: f32) -> Self {
        self.frame_rate = Some(frame_rate);
        self
    }

    /// Decode the base64 data to bytes
    pub fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        Ok(BASE64.decode(&self.base64_data)?)
    }

    /// Get MIME type for this video format
    pub fn mime_type(&self) -> String {
        <Self as MediaData>::mime_type(self)
    }

    /// Convert to data URL format for API
    pub fn to_data_url(&self) -> String {
        if self.format == "url" {
            self.base64_data.clone()
        } else {
            format!("data:{};base64,{}", self.mime_type(), self.base64_data)
        }
    }
}
