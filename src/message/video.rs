//! Video data handling

use super::media::MediaData;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Video data for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoData {
    /// Base64-encoded video data, OR the URL verbatim when [`is_url`](Self::is_url).
    pub base64_data: String,

    /// Video format/codec (e.g., "mp4", "webm", "mov"). Empty for a URL reference.
    pub format: String,

    /// Whether `base64_data` holds a remote URL rather than inline base64. Explicit
    /// flag, NOT a magic `format == "url"` value, so no caller-supplied format
    /// string can turn inline bytes into a counterfeit URL.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_url: bool,

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

    fn mime_type(&self) -> String {
        match self.format.as_str() {
            "mp4" => "video/mp4".to_string(),
            "webm" => "video/webm".to_string(),
            "mov" => "video/quicktime".to_string(),
            "avi" => "video/x-msvideo".to_string(),
            "mkv" => "video/x-matroska".to_string(),
            "flv" => "video/x-flv".to_string(),
            "wmv" => "video/x-ms-wmv".to_string(),
            "m4v" => "video/x-m4v".to_string(),
            "3gp" => "video/3gpp".to_string(),
            "ogv" => "video/ogg".to_string(),
            // Unknown: derive from the format rather than mislabeling it as mp4.
            other => format!("video/{}", other),
        }
    }

    fn is_url(&self) -> bool {
        self.is_url
    }

    fn from_base64(base64_data: impl Into<String>, format: impl Into<String>) -> Self {
        Self {
            base64_data: base64_data.into(),
            format: format.into(),
            is_url: false,
            duration_secs: None,
            width: None,
            height: None,
            frame_rate: None,
        }
    }

    fn guess_format(path: &Path) -> Option<String> {
        // Real extension only; no extension → `None` so `from_file` fails loudly
        // rather than fabricating a codec (e.g. "mp4") for arbitrary bytes.
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
    }
}

// Shared inherent forwarders generated once for every media type.
crate::impl_media_forwarders!(VideoData, format);

impl VideoData {
    /// Create VideoData from a URL (the URL will be used directly)
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            base64_data: url.into(),
            format: String::new(),
            is_url: true,
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
}
