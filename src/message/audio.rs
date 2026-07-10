//! Audio data handling

use super::media::MediaData;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Audio data for multimodal messages.
///
/// Grows as more metadata proves billing-relevant, so it is built through
/// `from_*` plus the `with_*` setters, never a struct literal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AudioData {
    /// Base64-encoded audio data, OR the URL verbatim when [`is_url`](Self::is_url).
    pub base64_data: String,

    /// Audio format/codec (e.g., "wav", "mp3", "ogg"). Empty for a URL reference.
    pub format: String,

    /// Whether `base64_data` holds a remote URL rather than inline base64. This is
    /// an explicit flag, NOT a magic `format == "url"` value, so no caller-supplied
    /// format string can ever turn inline bytes into a counterfeit URL.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_url: bool,

    /// Duration in seconds (if known).
    ///
    /// Providers bill audio by the second (Gemini at 32 tokens per second), and
    /// the duration lives inside the container's header, which this type does not
    /// parse. A caller that already opened the file knows it and should say so:
    /// without it a request's cost cannot be bounded before it is sent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,

    /// Sample rate in Hz (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,

    /// Number of channels (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<u8>,
}

impl MediaData for AudioData {
    fn base64_data(&self) -> &str {
        &self.base64_data
    }

    fn mime_type(&self) -> String {
        match self.format.as_str() {
            "wav" => "audio/wav".to_string(),
            "mp3" => "audio/mpeg".to_string(),
            "ogg" => "audio/ogg".to_string(),
            "flac" => "audio/flac".to_string(),
            "webm" => "audio/webm".to_string(),
            "m4a" | "aac" => "audio/aac".to_string(),
            // Unknown format: derive the MIME from it rather than mislabeling it
            // as wav (which would ship wrong bytes under a lying content type).
            other => format!("audio/{}", other),
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
            sample_rate: None,
            channels: None,
        }
    }

    fn guess_format(path: &Path) -> Option<String> {
        // Use the real extension (even an uncommon one: `mime_type` derives
        // `audio/<ext>` from it). No extension → `None`: there is no safe codec to
        // assume, so `from_file` fails loudly rather than shipping a guess.
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
    }
}

// Shared inherent forwarders generated once for every media type.
crate::impl_media_forwarders!(AudioData, format);

impl AudioData {
    /// Create AudioData from a URL (the URL will be passed directly to the API)
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            base64_data: url.into(),
            format: String::new(),
            is_url: true,
            duration_secs: None,
            sample_rate: None,
            channels: None,
        }
    }

    /// Set the clip's duration in seconds. Cost estimation assumes a default
    /// length for a clip without one, which overshoots short clips and
    /// undershoots long ones; providers bill audio by the second, so pass the
    /// real length for an accurate figure.
    pub fn with_duration(mut self, duration_secs: f64) -> Self {
        self.duration_secs = Some(duration_secs);
        self
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, rate: u32) -> Self {
        self.sample_rate = Some(rate);
        self
    }

    /// Set number of channels
    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = Some(channels);
        self
    }
}
