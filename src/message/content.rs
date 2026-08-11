//! Message content types

use super::{AudioData, DocumentData, ImageData, Media, VideoData};
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

    /// Document content (PDFs, for models that support them)
    #[serde(rename = "file")]
    File { file: FileData },
}

/// Image URL structure for API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    /// Pixel dimensions, when the caller knows them. Estimation metadata,
    /// like [`AudioInput::duration_secs`]: kept by serde, shed from the
    /// wire unless the provider's wire tolerates it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
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
    /// Clip length in seconds, when the caller knows it. Estimation metadata,
    /// not a wire field: it sharpens the pre-send cost estimate, survives serde
    /// round trips (saved conversation trees keep it), and is stripped from the
    /// request payload so a provider's schema never sees an unknown key.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
}

/// Video URL structure for API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoUrl {
    pub url: String,
    /// Clip length in seconds. Estimation metadata, exactly like
    /// [`AudioInput::duration_secs`]: kept by serde, shed from the wire
    /// unless the provider's wire tolerates it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f64>,
    /// Pixel dimensions, when the caller knows them. Same estimation-
    /// metadata rules as the duration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
}

/// File (document) structure for API.
///
/// The OpenAI-compatible `file` part: `file_data` carries a
/// `data:<mime>;base64,<payload>` URL for inline documents, or the URL verbatim
/// for URL-backed ones; `filename` is the display name the model sees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileData {
    pub filename: String,
    pub file_data: String,
    /// Page count, when the caller knows it. Estimation metadata, like
    /// [`AudioInput::duration_secs`]: kept by serde, shed from the wire
    /// unless the provider's wire tolerates it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_count: Option<u32>,
}

/// The display filename a URL-backed document derives from its URL: the last
/// PATH segment. The query (`?`) and fragment (`#`) are cut FIRST, so nothing
/// after them (even a stray "://") influences the result. `None` when the URL
/// yields no honest name, so the caller falls back to the generic one:
/// a `data:` URL in any case (its "segment" would be base64 noise; schemes are
/// case-insensitive), a URL with a host but no path in either spelling
/// ("https://example.com" or the protocol-relative "//example.com": the
/// hostname is not a filename), a segment containing a backslash (a Windows
/// path is not a URL and cannot be split reliably) or a colon ("mailto:a@b" is
/// an address, not a file; colons are illegal in filenames anyway), or an
/// empty segment (trailing slash).
fn derived_url_filename(url: &str) -> Option<String> {
    if url
        .get(..5)
        .is_some_and(|s| s.eq_ignore_ascii_case("data:"))
    {
        return None;
    }
    let path = &url[..url.find(['?', '#']).unwrap_or(url.len())];
    // Skip the authority when one is present, in EITHER spelling:
    // "<scheme>://host" or the protocol-relative "//host". The path starts at
    // the first '/' after the host; a URL with a host but no path has no
    // filename. The detection is STRUCTURAL on purpose, no scheme validation:
    // validating the scheme was tried and it made rejected-scheme URLs
    // ("my_app://host") fall into the bare-path branch and leak the hostname
    // as a filename, the exact outcome this block exists to prevent. The
    // protocol-relative check comes first so a "://" inside such a URL's path
    // is never mistaken for the authority marker.
    let host_start = if path.starts_with("//") {
        Some(2)
    } else {
        path.find("://").map(|pos| pos + 3)
    };
    let path = match host_start {
        Some(host_start) => {
            let authority_and_path = &path[host_start..];
            let slash = authority_and_path.find('/')?;
            &authority_and_path[slash + 1..]
        }
        None => path,
    };
    let segment = path.rsplit('/').next().unwrap_or("");
    (!segment.is_empty() && !segment.contains(['\\', ':'])).then(|| segment.to_string())
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
                width: image.width,
                height: image.height,
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
                duration_secs: audio.duration_secs,
            },
        }
    }

    /// Create a video content part from VideoData
    pub fn video(video: &VideoData) -> Self {
        Self::Video {
            video_url: VideoUrl {
                url: video.to_data_url(),
                duration_secs: video.duration_secs,
                width: video.width,
                height: video.height,
            },
        }
    }

    /// Create a document content part from DocumentData.
    ///
    /// Some wires require a filename, so a missing one is derived: for a
    /// URL-backed document, from the URL's last path segment (the name the
    /// model would reasonably be told, and distinct across several
    /// attachments); for inline bytes, the fixed "document.pdf", since PDF is
    /// the only format the wires accept ([`MediaData::guess_format`](super::MediaData::guess_format) is the
    /// one place that decides which formats exist). No name is ever derived
    /// from the MIME string: string surgery on a caller-supplied MIME produced
    /// garbage names for anything unexpected.
    pub fn document(document: &DocumentData) -> Self {
        // A blank explicit filename (empty or whitespace-only) is a caller bug
        // (an unfilled variable), not a choice: it falls through to derivation
        // instead of shipping a blank name to a wire that requires one. A real
        // name is shipped trimmed.
        let named = document
            .filename
            .as_deref()
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .map(str::to_string);
        let filename = named
            .or_else(|| {
                document
                    .is_url
                    .then(|| derived_url_filename(&document.base64_data))
                    .flatten()
            })
            .unwrap_or_else(|| "document.pdf".to_string());
        Self::File {
            file: FileData {
                filename,
                file_data: document.to_data_url(),
                page_count: document.page_count,
            },
        }
    }

    /// Create a content part from any Media type
    pub fn from_media(media: &Media) -> Self {
        match media {
            Media::Image(img) => Self::image(img),
            Media::Audio(audio) => Self::audio(audio),
            Media::Video(video) => Self::video(video),
            Media::Document(document) => Self::document(document),
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

    /// Multimodal content (text plus any media parts)
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

    /// Create content with text and documents
    pub fn with_documents(text: impl Into<String>, documents: &[DocumentData]) -> Self {
        let mut parts = vec![ContentPart::text(text)];
        parts.extend(documents.iter().map(ContentPart::document));
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

    /// A clip's length decides its cost, so it must survive a serde round trip
    /// (a saved conversation tree keeps it). It is omitted entirely when
    /// unknown: an absent key round-trips as `None`, a null might not.
    #[test]
    fn a_clips_duration_survives_a_round_trip_and_is_omitted_when_unknown() {
        let timed = ContentPart::audio(&AudioData::from_bytes(&[0u8; 4], "mp3").with_duration(3.5));
        let json = serde_json::to_value(&timed).unwrap();
        assert_eq!(json["input_audio"]["duration_secs"], 3.5);

        let ContentPart::Audio { input_audio } = serde_json::from_value(json).unwrap() else {
            panic!("an audio part must deserialize as one");
        };
        assert_eq!(input_audio.duration_secs, Some(3.5));

        // Unknown length: the key is absent, not null.
        let untimed = ContentPart::audio(&AudioData::from_bytes(&[0u8; 4], "mp3"));
        let json = serde_json::to_value(&untimed).unwrap();
        assert!(json["input_audio"].get("duration_secs").is_none(), "{json}");
    }

    /// Video carries the same field, through its own wire shape.
    #[test]
    fn a_videos_duration_survives_a_round_trip_and_is_omitted_when_unknown() {
        use crate::message::VideoData;

        let timed = ContentPart::video(&VideoData::from_url("https://x/y.mp4").with_duration(12.0));
        let json = serde_json::to_value(&timed).unwrap();
        assert_eq!(json["video_url"]["duration_secs"], 12.0);

        let ContentPart::Video { video_url } = serde_json::from_value(json).unwrap() else {
            panic!("a video part must deserialize as one");
        };
        assert_eq!(video_url.duration_secs, Some(12.0));

        let untimed = ContentPart::video(&VideoData::from_url("https://x/y.mp4"));
        let json = serde_json::to_value(&untimed).unwrap();
        assert!(json["video_url"].get("duration_secs").is_none(), "{json}");
    }

    #[test]
    fn document_content_part_emits_openai_file_shape() {
        use crate::message::DocumentData;
        let doc = DocumentData::from_bytes(b"%PDF-1.7", "application/pdf")
            .with_filename("report.pdf")
            .with_page_count(3);
        let part = ContentPart::document(&doc);
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "file");
        assert_eq!(json["file"]["filename"], "report.pdf");
        assert!(json["file"]["file_data"]
            .as_str()
            .unwrap()
            .starts_with("data:application/pdf;base64,"));
        assert_eq!(json["file"]["page_count"], 3);

        // Round trip: the page count (estimation metadata) survives serde.
        let ContentPart::File { file } = serde_json::from_value(json).unwrap() else {
            panic!("a file part must deserialize as one");
        };
        assert_eq!(file.page_count, Some(3));
    }

    #[test]
    fn document_without_filename_gets_a_generic_one_and_url_ships_verbatim() {
        use crate::message::DocumentData;
        let inline = ContentPart::document(&DocumentData::from_bytes(b"x", "application/pdf"));
        let json = serde_json::to_value(&inline).unwrap();
        assert_eq!(json["file"]["filename"], "document.pdf");
        assert!(json["file"].get("page_count").is_none(), "{json}");

        // A weird MIME never leaks into the derived name (the name is NOT
        // derived from the MIME string; a caller-supplied subtype once
        // produced "document.vnd.openxmlformats-...").
        let weird = ContentPart::document(&DocumentData::from_bytes(b"x", "application/weird"));
        let json = serde_json::to_value(&weird).unwrap();
        assert_eq!(json["file"]["filename"], "document.pdf");
    }

    #[test]
    fn a_url_backed_document_is_named_after_its_url() {
        use crate::message::DocumentData;
        let url = ContentPart::document(&DocumentData::from_url(
            "https://x/reports/2024-annual.pdf?dl=1#page=2",
        ));
        let json = serde_json::to_value(&url).unwrap();
        // The URL rides verbatim; the model is told the file's REAL name
        // (query and fragment cut), not a generic placeholder that would
        // collide across several attachments.
        assert_eq!(
            json["file"]["file_data"],
            "https://x/reports/2024-annual.pdf?dl=1#page=2"
        );
        assert_eq!(json["file"]["filename"], "2024-annual.pdf");

        // A URL with no usable last segment falls back to the generic name.
        let bare = ContentPart::document(&DocumentData::from_url("https://example.com/"));
        let json = serde_json::to_value(&bare).unwrap();
        assert_eq!(json["file"]["filename"], "document.pdf");

        // An explicit filename always wins over derivation.
        let named = ContentPart::document(
            &DocumentData::from_url("https://x/paper.pdf").with_filename("the-good-one.pdf"),
        );
        let json = serde_json::to_value(&named).unwrap();
        assert_eq!(json["file"]["filename"], "the-good-one.pdf");

        // A BLANK explicit filename (empty or whitespace-only) is a caller bug
        // (an unfilled variable), never shipped: it falls through to
        // derivation like a missing one. A real name ships trimmed.
        for blank in ["", "   "] {
            let part = ContentPart::document(
                &DocumentData::from_url("https://x/paper.pdf").with_filename(blank),
            );
            let json = serde_json::to_value(&part).unwrap();
            assert_eq!(json["file"]["filename"], "paper.pdf", "blank {blank:?}");
        }
        let padded = ContentPart::document(
            &DocumentData::from_url("https://x/paper.pdf").with_filename("  report.pdf "),
        );
        let json = serde_json::to_value(&padded).unwrap();
        assert_eq!(json["file"]["filename"], "report.pdf");
    }

    /// Each cut and each no-honest-name case is pinned SEPARATELY: a single
    /// query-and-fragment URL would pass with either cut broken (the `?`
    /// comes first, so the `#` never decides).
    #[test]
    fn a_urls_query_and_fragment_are_each_cut_on_their_own() {
        use crate::message::DocumentData;
        let name_of = |url: &str| {
            let part = ContentPart::document(&DocumentData::from_url(url));
            serde_json::to_value(&part).unwrap()["file"]["filename"].clone()
        };
        // Query only, fragment only.
        assert_eq!(name_of("https://x/paper.pdf?dl=1"), "paper.pdf");
        assert_eq!(name_of("https://x/paper.pdf#page=2"), "paper.pdf");

        // A pathless URL never hands the HOSTNAME out as a filename, in ANY
        // spelling: with a scheme, protocol-relative, with or without the
        // trailing slash. All forms agree on the generic fallback, and the
        // protocol-relative spelling WITH a path still derives normally.
        assert_eq!(name_of("https://example.com"), "document.pdf");
        assert_eq!(name_of("https://example.com/"), "document.pdf");
        assert_eq!(name_of("//example.com"), "document.pdf");
        assert_eq!(name_of("//example.com/"), "document.pdf");
        assert_eq!(name_of("//example.com/paper.pdf"), "paper.pdf");
        // A "://" sitting INSIDE a protocol-relative URL's path must not be
        // mistaken for the authority marker: the "//" branch wins.
        assert_eq!(name_of("//example.com/x://y/c.pdf"), "c.pdf");
        // A NON-STANDARD scheme still counts as an authority (detection is
        // structural, no scheme validation): the hostname must not leak as a
        // filename just because the scheme has an unusual character.
        assert_eq!(name_of("my_app://example.com"), "document.pdf");
        assert_eq!(name_of("my_app://example.com/paper.pdf"), "paper.pdf");

        // A non-path URL (no slashes at all) is an address, not a file.
        assert_eq!(name_of("mailto:someone@example.com"), "document.pdf");

        // No honest name → the generic fallback, never a misleading one:
        // a data: URL's "segment" is base64 noise, a backslashed Windows
        // path is not a URL, a trailing slash names a directory-ish thing.
        assert_eq!(
            name_of("data:application/pdf;base64,JVBERi0="),
            "document.pdf"
        );
        // Schemes are case-insensitive, so the data: guard must be too.
        assert_eq!(
            name_of("DATA:application/pdf;base64,JVBERi0="),
            "document.pdf"
        );
        assert_eq!(name_of(r"C:\Users\me\report.pdf"), "document.pdf");
        assert_eq!(name_of("https://x/report.pdf/"), "document.pdf");
    }

    /// The from_media dispatch must route a document to a File part; pinned
    /// at unit level so the fast suite catches it, not only the integration
    /// binary.
    #[test]
    fn from_media_routes_a_document_to_a_file_part() {
        use crate::message::DocumentData;
        let media = Media::Document(
            DocumentData::from_bytes(b"%PDF", "application/pdf").with_page_count(2),
        );
        let ContentPart::File { file } = ContentPart::from_media(&media) else {
            panic!("a Media::Document must become a File part");
        };
        assert_eq!(file.page_count, Some(2));
    }

    /// The untagged `MessageContent` enum must resolve a message carrying a
    /// file part back to `Parts` with the `File` variant intact; a wrong serde
    /// tag would silently misparse the whole multimodal message.
    #[test]
    fn a_message_with_a_document_survives_a_serde_round_trip() {
        use crate::message::DocumentData;
        let content = MessageContent::with_documents(
            "read this",
            &[DocumentData::from_bytes(b"%PDF", "application/pdf").with_page_count(9)],
        );
        let json = serde_json::to_value(&content).unwrap();
        let back: MessageContent = serde_json::from_value(json).unwrap();
        let MessageContent::Parts(parts) = back else {
            panic!("a parts message must deserialize as Parts, not collapse to Text");
        };
        assert_eq!(parts[0].as_text(), Some("read this"));
        let ContentPart::File { file } = &parts[1] else {
            panic!("the file part must come back as File, got {:?}", parts[1]);
        };
        assert_eq!(file.page_count, Some(9));
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
