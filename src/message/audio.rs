//! Audio data handling

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Audio data for multimodal messages
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioData {
    /// Base64-encoded audio data
    pub base64_data: String,

    /// Audio format (e.g., "wav", "mp3", "ogg")
    pub format: String,

    /// Sample rate in Hz (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_rate: Option<u32>,

    /// Number of channels (if known)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub channels: Option<u8>,
}

impl AudioData {
    /// Create AudioData from base64 string
    pub fn from_base64(base64_data: impl Into<String>, format: impl Into<String>) -> Self {
        Self {
            base64_data: base64_data.into(),
            format: format.into(),
            sample_rate: None,
            channels: None,
        }
    }

    /// Create AudioData from raw bytes
    pub fn from_bytes(bytes: &[u8], format: impl Into<String>) -> Self {
        Self {
            base64_data: BASE64.encode(bytes),
            format: format.into(),
            sample_rate: None,
            channels: None,
        }
    }

    /// Load AudioData from a file path
    pub fn from_file(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let format = Self::guess_format(path);
        Ok(Self::from_bytes(&bytes, format))
    }

    /// Load AudioData from a file path (async)
    pub async fn from_file_async(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        let path = path.as_ref();
        let bytes = tokio::fs::read(path).await?;
        let format = Self::guess_format(path);
        Ok(Self::from_bytes(&bytes, format))
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

    /// Decode the base64 data to bytes
    pub fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        Ok(BASE64.decode(&self.base64_data)?)
    }

    /// Get MIME type for this audio format
    pub fn mime_type(&self) -> String {
        match self.format.as_str() {
            "wav" => "audio/wav",
            "mp3" => "audio/mpeg",
            "ogg" => "audio/ogg",
            "flac" => "audio/flac",
            "webm" => "audio/webm",
            "m4a" | "aac" => "audio/aac",
            _ => "audio/wav",
        }
        .to_string()
    }

    /// Guess format from file extension
    fn guess_format(path: &Path) -> String {
        path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("wav")
            .to_lowercase()
    }
}
