//! Document data handling (PDFs)

use super::media::MediaData;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Document data for multimodal messages (today: PDF).
///
/// Ships on the OpenAI-compatible wire as a `type: "file"` content part
/// (`file.filename` + `file.file_data`), which OpenRouter translates to each
/// downstream provider's native document shape (e.g. Anthropic `document`
/// blocks). The model sees the document natively: layout, tables, figures,
/// scanned pages.
///
/// Grows as more metadata proves billing-relevant, so it is built through
/// `from_*` plus the `with_*` setters, never a struct literal.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct DocumentData {
    /// Base64-encoded document data, OR the URL verbatim when [`is_url`](Self::is_url).
    pub base64_data: String,

    /// MIME type (e.g., "application/pdf"). Empty for a URL reference.
    pub mime_type: String,

    /// Whether `base64_data` holds a remote URL rather than inline base64. Explicit
    /// flag, NOT a magic `mime_type == "url"` value, so no caller-supplied mime
    /// string can turn inline bytes into a counterfeit URL.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub is_url: bool,

    /// Display name the provider shows the model (e.g. "report.pdf"). Some
    /// wires require one, so the wire projection ships a real name trimmed of
    /// surrounding whitespace, and derives one when this is unset or blank
    /// (empty or whitespace-only, the unfilled-variable bug): a URL-backed
    /// document is named after its URL's last path segment, inline bytes get
    /// the generic "document.pdf".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,

    /// Page count, when the caller knows it. Estimation metadata (sharpens a
    /// pre-send cost estimate, like an audio clip's `duration_secs`); never
    /// required and shed from the wire.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_count: Option<u32>,
}

impl MediaData for DocumentData {
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
            filename: None,
            page_count: None,
        }
    }

    fn guess_format(path: &Path) -> Option<String> {
        // PDF is the only document format the provider wires accept today, so
        // any other extension → `None` and `from_file` fails loudly rather than
        // shipping bytes under a MIME type no provider will take.
        match path.extension().and_then(|e| e.to_str()) {
            Some(ext) if ext.eq_ignore_ascii_case("pdf") => Some("application/pdf".to_string()),
            _ => None,
        }
    }
}

// Shared inherent forwarders generated once for every media type.
crate::impl_media_forwarders!(DocumentData, mime_type);

impl DocumentData {
    /// Create DocumentData from a URL (the URL will be used directly)
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            base64_data: url.into(),
            mime_type: String::new(),
            is_url: true,
            filename: None,
            page_count: None,
        }
    }

    /// Set the display filename the model sees.
    pub fn with_filename(mut self, filename: impl Into<String>) -> Self {
        self.filename = Some(filename.into());
        self
    }

    /// Declare the document's page count (estimation metadata).
    pub fn with_page_count(mut self, page_count: u32) -> Self {
        self.page_count = Some(page_count);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pdf_extension_maps_to_the_pdf_mime_and_others_fail_loudly() {
        assert_eq!(
            DocumentData::guess_format(Path::new("report.pdf")).as_deref(),
            Some("application/pdf")
        );
        assert_eq!(
            DocumentData::guess_format(Path::new("REPORT.PDF")).as_deref(),
            Some("application/pdf")
        );
        // No honest default exists for a non-PDF document: None → from_file errors.
        assert!(DocumentData::guess_format(Path::new("notes.docx")).is_none());
        assert!(DocumentData::guess_format(Path::new("noext")).is_none());
    }

    #[test]
    fn inline_bytes_round_trip_and_url_reference_stays_verbatim() {
        let doc = DocumentData::from_bytes(b"%PDF-1.7", "application/pdf");
        assert!(!doc.is_url());
        assert_eq!(doc.to_bytes().unwrap(), b"%PDF-1.7");
        assert!(doc
            .to_data_url()
            .starts_with("data:application/pdf;base64,"));

        let url = DocumentData::from_url("https://example.com/paper.pdf");
        assert!(url.is_url());
        assert_eq!(url.to_data_url(), "https://example.com/paper.pdf");
    }
}
