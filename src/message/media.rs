//! Unified media handling for multimodal messages
//!
//! This module provides a common abstraction for different media types (Image, Audio, Video).

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use std::path::Path;

/// The error for a file whose media format can't be determined from its
/// extension, used by the fail-loud `from_file` paths.
fn undetermined_format_err(path: &Path) -> crate::error::MiniLLMError {
    crate::error::MiniLLMError::InvalidParameter(format!(
        "cannot determine media format for {:?}: no recognized extension. Pass the format explicitly via from_bytes/from_base64.",
        path
    ))
}

/// Common trait for all media types.
///
/// Defines the shared behavior across Image, Audio, and Video data. The
/// data-URL conversion and byte decoding are provided here once as default
/// methods, so the concrete types never re-implement them.
pub trait MediaData: Sized {
    /// Get the base64-encoded data (or the URL, when [`is_url`](Self::is_url)).
    fn base64_data(&self) -> &str;

    /// Get the MIME type for this media
    fn mime_type(&self) -> String;

    /// Whether this media is a remote URL reference rather than inline base64.
    /// When true, `base64_data` holds the URL and is passed to the API verbatim.
    fn is_url(&self) -> bool;

    /// Create from base64 string and format
    fn from_base64(base64_data: impl Into<String>, format: impl Into<String>) -> Self;

    /// Determine the media format from a file's extension. `None` when it cannot
    /// be determined and no honest default exists (e.g. an extensionless audio
    /// file: there is no safe codec to assume). A type with an honest unknown
    /// default (an image → `application/octet-stream`) returns `Some`.
    fn guess_format(path: &Path) -> Option<String>;

    /// Create from raw bytes
    fn from_bytes(bytes: &[u8], format: impl Into<String>) -> Self {
        Self::from_base64(BASE64.encode(bytes), format)
    }

    /// Load from a file path. Fails loudly if the format can't be determined from
    /// the extension (rather than shipping an empty/guessed format to the wire).
    fn from_file(path: impl AsRef<Path>) -> crate::error::Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let format = Self::guess_format(path).ok_or_else(|| undetermined_format_err(path))?;
        Ok(Self::from_bytes(&bytes, format))
    }

    /// Load from a file path (async). Same fail-loud format rule as [`Self::from_file`].
    fn from_file_async(
        path: impl AsRef<Path> + Send,
    ) -> impl std::future::Future<Output = crate::error::Result<Self>> + Send
    where
        Self: Send,
    {
        let path = path.as_ref().to_path_buf();
        async move {
            let bytes = tokio::fs::read(&path).await?;
            let format = Self::guess_format(&path).ok_or_else(|| undetermined_format_err(&path))?;
            Ok(Self::from_bytes(&bytes, format))
        }
    }

    /// Convert to the value the API expects: the URL verbatim when this is a
    /// URL reference, otherwise a `data:<mime>;base64,<data>` URL.
    fn to_data_url(&self) -> String {
        if self.is_url() {
            self.base64_data().to_string()
        } else {
            format!("data:{};base64,{}", self.mime_type(), self.base64_data())
        }
    }

    /// Decode the base64 data to bytes. Errors if this is a URL reference (there
    /// is no inline data to decode).
    fn to_bytes(&self) -> crate::error::Result<Vec<u8>> {
        if self.is_url() {
            return Err(crate::error::MiniLLMError::InvalidParameter(
                "cannot decode bytes from a URL-backed media reference".to_string(),
            ));
        }
        Ok(BASE64.decode(self.base64_data())?)
    }
}

/// Generate the inherent forwarders shared by every concrete media type, so a
/// caller can write `ImageData::from_bytes(..)` / `audio.to_data_url()` without
/// importing [`MediaData`]. These are pure pass-throughs to the trait; the macro
/// is the single source of truth so the three types can't drift (one gaining a
/// forwarder another lacks). Type-specific constructors (`from_url`, `with_*`)
/// are written by hand per type. `$fmt_param` names the format/mime argument so
/// each type's signature reads naturally (`mime_type` for images, `format` for
/// audio/video).
#[macro_export]
#[doc(hidden)]
macro_rules! impl_media_forwarders {
    ($ty:ty, $fmt_param:ident) => {
        impl $ty {
            /// Create from a base64 string and format/mime.
            pub fn from_base64(
                base64_data: impl Into<String>,
                $fmt_param: impl Into<String>,
            ) -> Self {
                <Self as $crate::MediaData>::from_base64(base64_data, $fmt_param)
            }

            /// Create from raw bytes and format/mime.
            pub fn from_bytes(bytes: &[u8], $fmt_param: impl Into<String>) -> Self {
                <Self as $crate::MediaData>::from_bytes(bytes, $fmt_param)
            }

            /// Load from a file path (format inferred from the extension; fails
            /// loudly if it can't be determined).
            pub fn from_file(path: impl AsRef<std::path::Path>) -> $crate::error::Result<Self> {
                <Self as $crate::MediaData>::from_file(path)
            }

            /// Load from a file path, async.
            pub async fn from_file_async(
                path: impl AsRef<std::path::Path> + Send,
            ) -> $crate::error::Result<Self> {
                <Self as $crate::MediaData>::from_file_async(path).await
            }

            /// Decode the base64 data to bytes (errors for a URL reference).
            pub fn to_bytes(&self) -> $crate::error::Result<Vec<u8>> {
                <Self as $crate::MediaData>::to_bytes(self)
            }

            /// The MIME type for this media.
            pub fn mime_type(&self) -> String {
                <Self as $crate::MediaData>::mime_type(self)
            }

            /// Whether this media is a remote URL reference rather than inline bytes.
            pub fn is_url(&self) -> bool {
                <Self as $crate::MediaData>::is_url(self)
            }

            /// Convert to the value the API expects (verbatim URL or data URL).
            pub fn to_data_url(&self) -> String {
                <Self as $crate::MediaData>::to_data_url(self)
            }
        }
    };
}

/// Unified media enum that can hold any media type.
///
/// An in-memory adapter only: it is converted into a `ContentPart` before
/// anything reaches the wire, and is never itself serialized/persisted (the
/// wire-bearing types are `ContentPart`/`Message`), so it carries no serde
/// derive that would imply a persistence contract it doesn't honor.
#[derive(Debug, Clone)]
pub enum Media {
    /// Image media
    Image(super::ImageData),
    /// Audio media
    Audio(super::AudioData),
    /// Video media
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
            Self::Image(img) => MediaData::mime_type(img),
            Self::Audio(audio) => MediaData::mime_type(audio),
            Self::Video(video) => MediaData::mime_type(video),
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

#[cfg(test)]
mod tests {
    use crate::message::{AudioData, ImageData, VideoData};

    /// Write `bytes` to a temp file with the given name (incl. extension) and
    /// return its path.
    fn temp_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!(
            "minillmlib_media_test_{}_{}",
            std::process::id(),
            name
        ));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn audio_from_file_without_extension_fails_loudly() {
        // The fail-loud rule: an extensionless audio file has no safe codec to
        // assume, so from_file errors instead of shipping an empty format.
        let p = temp_file("noext", b"\x00\x01\x02");
        let err = AudioData::from_file(&p).unwrap_err();
        assert!(
            matches!(err, crate::error::MiniLLMError::InvalidParameter(_)),
            "expected InvalidParameter, got {err:?}"
        );
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn video_from_file_without_extension_fails_loudly() {
        let p = temp_file("noext2", b"\x00\x01\x02");
        assert!(VideoData::from_file(&p).is_err());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn audio_from_file_with_extension_uses_it() {
        let p = temp_file("clip.mp3", b"\x00\x01\x02");
        let audio = AudioData::from_file(&p).unwrap();
        assert_eq!(audio.format, "mp3");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn image_from_file_without_extension_uses_octet_stream() {
        // Images DO have an honest unknown default, so they don't fail loudly.
        let p = temp_file("img_noext", b"\x00\x01\x02");
        let img = ImageData::from_file(&p).unwrap();
        assert_eq!(img.mime_type(), "application/octet-stream");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn format_string_url_cannot_mint_a_counterfeit_url_reference() {
        // The is-URL flag is now an explicit bool, NOT `format/mime_type == "url"`.
        // So passing "url" as the format/mime to a normal constructor yields INLINE
        // bytes (is_url() == false), never a fake URL reference that would ship the
        // bytes verbatim as a remote URL. This closes the magic-string overload at
        // the root (every constructor), not just the file-path door.
        let a = AudioData::from_bytes(b"\x00\x01", "url");
        assert!(
            !a.is_url(),
            "format 'url' must NOT flag inline audio as a URL"
        );
        let v = VideoData::from_bytes(b"\x00\x01", "url");
        assert!(
            !v.is_url(),
            "format 'url' must NOT flag inline video as a URL"
        );
        let i = ImageData::from_base64("ZGF0YQ==", "url");
        assert!(
            !i.is_url(),
            "mime 'url' must NOT flag inline image as a URL"
        );

        // The real URL constructors still flag correctly and round-trip the URL.
        let au = AudioData::from_url("https://example.com/a.mp3");
        assert!(au.is_url());
        assert_eq!(au.to_data_url(), "https://example.com/a.mp3");

        // A file literally named `clip.url` now loads as ordinary inline bytes
        // (format "url", is_url false), harmless, since "url" is no longer magic.
        let p = temp_file("clip.url", b"\x00\x01\x02");
        let loaded = AudioData::from_file(&p).unwrap();
        assert_eq!(loaded.format, "url");
        assert!(!loaded.is_url(), "a .url FILE is inline bytes, not a URL");
        std::fs::remove_file(&p).ok();
    }
}
