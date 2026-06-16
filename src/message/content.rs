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

/// Audio input structure for API.
///
/// `data` carries either base64-encoded audio or, for URL-backed audio, the URL
/// verbatim. `format` is omitted for URL-backed audio (no `"url"` sentinel leaks
/// to the wire); the provider infers it from the URL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInput {
    pub data: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
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

    /// Create an audio content part from AudioData.
    ///
    /// For URL-backed audio the URL is sent verbatim in `data` with `format`
    /// omitted (the `"url"` sentinel never reaches the wire); for inline audio
    /// the base64 data and real format are sent.
    pub fn audio(audio: &AudioData) -> Self {
        let format = if audio.is_url() {
            None
        } else {
            Some(audio.format.clone())
        };
        Self::Audio {
            input_audio: AudioInput {
                data: audio.base64_data.clone(),
                format,
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

    /// Get the FIRST text part (borrowed). For a single-text message this is the
    /// whole text; for a multimodal message with several text parts it returns
    /// only the first, so use [`all_text`](Self::all_text) when you need every
    /// text part (e.g. for display). Named `get_text` for the common single-text
    /// case; it does not promise "all" the text.
    pub fn get_text(&self) -> Option<&str> {
        match self {
            Self::Text(text) => Some(text),
            Self::Parts(parts) => parts.iter().find_map(|p| p.as_text()),
        }
    }

    /// Get all text content concatenated (every text part, newline-joined).
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::AudioData;

    #[test]
    fn audio_content_part_emits_base64_with_format() {
        let audio = AudioData::from_bytes(&[0u8; 4], "mp3");
        let part = ContentPart::audio(&audio);
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "input_audio");
        assert_eq!(json["input_audio"]["format"], "mp3");
        assert!(json["input_audio"]["data"].as_str().is_some());
    }

    #[test]
    fn audio_content_part_url_does_not_leak_sentinel() {
        // Regression: URL-backed audio must NOT emit format:"url"; the URL goes
        // in `data` and `format` is omitted for the provider to infer.
        let audio = AudioData::from_url("https://example.com/clip.mp3");
        let part = ContentPart::audio(&audio);
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["input_audio"]["data"], "https://example.com/clip.mp3");
        assert!(
            json["input_audio"].get("format").is_none(),
            "format must be omitted for URL audio, got {:?}",
            json["input_audio"].get("format")
        );
    }
}
