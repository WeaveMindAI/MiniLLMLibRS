//! Message content types

use super::{AudioData, ImageData, Media, VideoData};
use serde::{Deserialize, Serialize};

/// A single part of message content
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    /// Text content
    #[serde(rename = "text")]
    Text { text: String },

    /// Image content
    #[serde(rename = "image_url")]
    Image { image_url: ImageUrl },

    /// Audio content (for models that support it)
    #[serde(rename = "input_audio")]
    Audio { input_audio: AudioInput },

    /// Video content (for models that support it)
    #[serde(rename = "video_url")]
    Video { video_url: VideoUrl },
}

/// Image URL structure for API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Audio input structure for API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInput {
    pub data: String,
    pub format: String,
}

/// Video URL structure for API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
}

impl ContentPart {
    /// Create a text content part
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an image content part from ImageData
    pub fn image(image: &ImageData) -> Self {
        Self::Image {
            image_url: ImageUrl {
                url: image.to_data_url(),
                detail: image.detail.clone(),
            },
        }
    }

    /// Create an audio content part from AudioData
    pub fn audio(audio: &AudioData) -> Self {
        Self::Audio {
            input_audio: AudioInput {
                data: audio.base64_data.clone(),
                format: audio.format.clone(),
            },
        }
    }

    /// Create a video content part from VideoData
    pub fn video(video: &VideoData) -> Self {
        Self::Video {
            video_url: VideoUrl {
                url: video.to_data_url(),
                duration_secs: video.duration_secs,
            },
        }
    }

    /// Create a content part from any Media type
    pub fn from_media(media: &Media) -> Self {
        match media {
            Media::Image(img) => Self::image(img),
            Media::Audio(audio) => Self::audio(audio),
            Media::Video(video) => Self::video(video),
        }
    }

    /// Check if this is text content
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text { .. })
    }

    /// Get text if this is text content
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text { text } => Some(text),
            _ => None,
        }
    }
}

/// Message content - can be simple text or multimodal
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    /// Simple text content
    Text(String),

    /// Multimodal content (text + images + audio)
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    /// Create text content
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text(text.into())
    }

    /// Create multimodal content with parts
    pub fn parts(parts: Vec<ContentPart>) -> Self {
        Self::Parts(parts)
    }

    /// Create content with text and images
    pub fn with_images(text: impl Into<String>, images: &[ImageData]) -> Self {
        let mut parts = vec![ContentPart::text(text)];
        parts.extend(images.iter().map(ContentPart::image));
        Self::Parts(parts)
    }

    /// Create content with text and audio
    pub fn with_audio(text: impl Into<String>, audio: &[AudioData]) -> Self {
        let mut parts = vec![ContentPart::text(text)];
        parts.extend(audio.iter().map(ContentPart::audio));
        Self::Parts(parts)
    }

    /// Create content with text and video
    pub fn with_video(text: impl Into<String>, video: &[VideoData]) -> Self {
        let mut parts = vec![ContentPart::text(text)];
        parts.extend(video.iter().map(ContentPart::video));
        Self::Parts(parts)
    }

    /// Create content with text and any media types
    pub fn with_media(text: impl Into<String>, media: &[Media]) -> Self {
        let mut parts = vec![ContentPart::text(text)];
        parts.extend(media.iter().map(ContentPart::from_media));
        Self::Parts(parts)
    }

    /// Check if this content has multimodal elements
    pub fn has_multimodal(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts.iter().any(|p| !p.is_text()),
        }
    }

    /// Get the text content (first text part if multimodal)
    pub fn get_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Parts(parts) => {
                // Return first text part
                parts.iter().find_map(|p| p.as_text())
            }
        }
    }

    /// Get all text content concatenated
    pub fn all_text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| p.as_text())
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    /// Convert to API format
    pub fn to_api_format(&self) -> serde_json::Value {
        match self {
            Self::Text(text) => serde_json::json!(text),
            Self::Parts(parts) => serde_json::json!(parts),
        }
    }

    /// Merge two contents together
    pub fn merge(&self, other: &MessageContent) -> MessageContent {
        match (self, other) {
            (Self::Text(a), Self::Text(b)) => Self::Text(format!("{}\n{}", a, b)),
            (Self::Text(a), Self::Parts(b)) => {
                let mut parts = vec![ContentPart::text(a)];
                parts.extend(b.clone());
                Self::Parts(parts)
            }
            (Self::Parts(a), Self::Text(b)) => {
                let mut parts = a.clone();
                parts.push(ContentPart::text(b));
                Self::Parts(parts)
            }
            (Self::Parts(a), Self::Parts(b)) => {
                let mut parts = a.clone();
                parts.extend(b.clone());
                Self::Parts(parts)
            }
        }
    }
}

impl From<String> for MessageContent {
    fn from(text: String) -> Self {
        Self::Text(text)
    }
}

impl From<&str> for MessageContent {
    fn from(text: &str) -> Self {
        Self::Text(text.to_string())
    }
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}
