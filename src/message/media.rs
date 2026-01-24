//! Unified media handling for multimodal messages
//!
//! This module provides a common abstraction for different media types (Image, Audio, Video).

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use std::path::Path;

/// Common trait for all media types
///
/// This trait defines the shared behavior across Image, Audio, and Video data.
pub trait MediaData: Sized {
    /// Get the base64-encoded data
    fn base64_data(&self) -> &str;

    /// Get the format/mime type identifier
    fn format_id(&self) -> &str;

    /// Get the MIME type for this media
    fn mime_type(&self) -> String;

    /// Create from base64 string and format
    fn from_base64(base64_data: impl Into<String>, format: impl Into<String>) -> Self;

    /// Create from raw bytes
    fn from_bytes(bytes: &[u8], format: impl Into<String>) -> Self {
        Self::from_base64(BASE64.encode(bytes), format)
    }

    /// Load from a file path
    fn from_file(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let format = Self::guess_format(path);
        Ok(Self::from_bytes(&bytes, format))
    }

    /// Load from a file path (async)
    fn from_file_async(
        path: impl AsRef<Path> + Send,
    ) -> impl std::future::Future<Output = crate::error::Result<Self>> + Send
    where
        Self: Send,
    {
        let path = path.as_ref().to_path_buf();
        async move {
            let bytes = tokio::fs::read(&path).await?;
            let format = Self::guess_format(&path);
            Ok(Self::from_bytes(&bytes, format))
        }
    }

    /// Decode the base64 data to bytes
    fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        Ok(BASE64.decode(self.base64_data())?)
    }

    /// Guess format from file extension
    fn guess_format(path: &Path) -> String;
}

/// Unified media enum that can hold any media type
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "media_type")]
pub enum Media {
    /// Image media
    #[serde(rename = "image")]
    Image(super::ImageData),

    /// Audio media
    #[serde(rename = "audio")]
    Audio(super::AudioData),

    /// Video media
    #[serde(rename = "video")]
    Video(super::VideoData),
}

impl Media {
    /// Create an image media from ImageData
    pub fn image(data: super::ImageData) -> Self {
        Self::Image(data)
    }

    /// Create an audio media from AudioData
    pub fn audio(data: super::AudioData) -> Self {
        Self::Audio(data)
    }

    /// Create a video media from VideoData
    pub fn video(data: super::VideoData) -> Self {
        Self::Video(data)
    }

    /// Get the MIME type for this media
    pub fn mime_type(&self) -> String {
        match self {
            Self::Image(img) => {
                if img.mime_type == "url" {
                    "url".to_string()
                } else {
                    img.mime_type.clone()
                }
            }
            Self::Audio(audio) => audio.mime_type(),
            Self::Video(video) => video.mime_type(),
        }
    }

    /// Check if this is an image
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image(_))
    }

    /// Check if this is audio
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Audio(_))
    }

    /// Check if this is video
    pub fn is_video(&self) -> bool {
        matches!(self, Self::Video(_))
    }

    /// Get as ImageData if this is an image
    pub fn as_image(&self) -> Option<&super::ImageData> {
        match self {
            Self::Image(img) => Some(img),
            _ => None,
        }
    }

    /// Get as AudioData if this is audio
    pub fn as_audio(&self) -> Option<&super::AudioData> {
        match self {
            Self::Audio(audio) => Some(audio),
            _ => None,
        }
    }

    /// Get as VideoData if this is video
    pub fn as_video(&self) -> Option<&super::VideoData> {
        match self {
            Self::Video(video) => Some(video),
            _ => None,
        }
    }
}

impl From<super::ImageData> for Media {
    fn from(data: super::ImageData) -> Self {
        Self::Image(data)
    }
}

impl From<super::AudioData> for Media {
    fn from(data: super::AudioData) -> Self {
        Self::Audio(data)
    }
}

impl From<super::VideoData> for Media {
    fn from(data: super::VideoData) -> Self {
        Self::Video(data)
    }
}
