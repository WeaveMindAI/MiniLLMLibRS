//! Comprehensive integration tests for MiniLLMLib
//!
//! These tests make real API calls to OpenRouter.
//! Run with: cargo test --test integration_tests -- --nocapture
//!
//! Requires OPENROUTER_API_KEY environment variable to be set.

use minillmlib::{
    chat_node::ChatNode,
    generator::{CompletionParameters, GeneratorInfo, NodeCompletionParameters},
    message::{
        AudioData, DocumentData, ImageData, Media, Message, MessageContent, Role, VideoData,
    },
    provider::OpenRouterRouting,
    provider::{CostInfo, LLMClient},
    tracking::{AsyncCostCallback, CompletionContext, CompletionMeta},
    Auth,
};
use std::sync::{Arc, Mutex};

// Test models: small, cheap, and verified live on OpenRouter (2026-06-16).
// TEST_MODEL is multimodal (text+image+audio+video) for the multimodal tests;
// TEXT_ONLY_MODEL is a cheap text model. The previous gemini-2.0-flash-lite-001
// was retired ("No endpoints found") and 404'd every live test.
const TEST_MODEL: &str = "google/gemini-2.5-flash-lite";
const TEXT_ONLY_MODEL: &str = "meta-llama/llama-3.1-8b-instruct";

// Anthropic test models (verified live 2026-06-16). Haiku is the cheapest.
#[allow(dead_code)]
const ANTHROPIC_TEST_MODEL: &str = "claude-haiku-4-5";

/// Skip a live test unless the `live` Cargo feature is enabled AND the required
/// env var is present. The `live` gate keeps a plain `cargo test` free, offline,
/// and deterministic even when a `.env` holds real keys; the env-var check then
/// skips gracefully under `--features live` when a particular key is absent.
macro_rules! require_live {
    ($env_var:literal) => {
        dotenvy::dotenv().ok();
        if !cfg!(feature = "live") {
            eprintln!("Skipping live test (enable with `cargo test --features live`)");
            return;
        }
        if std::env::var($env_var).is_err() {
            eprintln!("Skipping live test: {} not set", $env_var);
            return;
        }
    };
}

/// A live generator with its own HTTP client. Every `#[tokio::test]` runs its
/// own runtime, and a connection belongs to the runtime that opened it, so live
/// tests sharing the crate's client would hand each other dead connections
/// when run in parallel.
fn own_client(generator: GeneratorInfo) -> GeneratorInfo {
    generator.with_http_client(reqwest::Client::new())
}

fn get_test_generator() -> GeneratorInfo {
    dotenvy::dotenv().ok();
    own_client(GeneratorInfo::openrouter(TEST_MODEL))
}

fn get_text_generator() -> GeneratorInfo {
    dotenvy::dotenv().ok();
    own_client(GeneratorInfo::openrouter(TEXT_ONLY_MODEL))
}

// =============================================================================
// GeneratorInfo Tests
// =============================================================================

#[test]
fn test_generator_info_creation() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model");
    assert_eq!(gen.name, "Test");
    assert_eq!(gen.base_url, "https://api.example.com/v1");
    assert_eq!(gen.model, "test-model");
    assert!(matches!(gen.auth, Auth::None));
}

#[test]
fn test_generator_info_with_api_key() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model")
        .with_api_key("test-key-123");
    assert!(matches!(gen.auth, Auth::ApiKey(_)));
}

#[test]
fn test_generator_info_openrouter() {
    dotenvy::dotenv().ok();
    let gen = GeneratorInfo::openrouter("anthropic/claude-3.5-sonnet");
    assert_eq!(gen.name, "OpenRouter");
    assert_eq!(gen.base_url, "https://openrouter.ai/api/v1");
    assert_eq!(gen.model, "anthropic/claude-3.5-sonnet");
    // OpenRouter attribution is carried as the app identity (the provider turns it
    // into HTTP-Referer/X-Title headers at request time).
    assert!(gen.app_attribution.is_some());
}

#[test]
fn test_generator_info_completions_url() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model");
    assert_eq!(
        gen.completions_url(),
        "https://api.example.com/v1/chat/completions"
    );

    // Test with trailing slash
    let gen2 = GeneratorInfo::new("Test", "https://api.example.com/v1/", "test-model");
    assert_eq!(
        gen2.completions_url(),
        "https://api.example.com/v1/chat/completions"
    );
}

#[test]
fn test_generator_info_with_options() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model")
        .with_header("X-Custom", "value");

    assert!(gen
        .custom_headers
        .iter()
        .any(|(k, v)| k == "X-Custom" && v == "value"));
}

// =============================================================================
// CompletionParameters Tests
// =============================================================================

#[test]
fn test_completion_parameters_default() {
    let params = CompletionParameters::default();
    assert!(params.max_tokens.is_none());
    assert!(params.temperature.is_none());
    assert!(params.top_p.is_none());
    assert!(params.stop.is_none());
}

#[test]
fn test_completion_parameters_builder() {
    let params = CompletionParameters::new()
        .with_max_tokens(1024)
        .with_temperature(0.5)
        .with_top_p(0.9)
        .with_stop(vec!["STOP".to_string()])
        .with_seed(42);

    assert_eq!(params.max_tokens, Some(1024));
    assert_eq!(params.temperature, Some(0.5));
    assert_eq!(params.top_p, Some(0.9));
    assert_eq!(params.stop, Some(vec!["STOP".to_string()]));
    assert_eq!(params.seed, Some(42));
}

#[test]
fn test_completion_parameters_merge() {
    let base = CompletionParameters::new()
        .with_max_tokens(1024)
        .with_temperature(0.7);

    let override_params = CompletionParameters::new()
        .with_temperature(0.3)
        .with_top_p(0.9);

    let merged = base.merge(&override_params);

    // merge takes other's value where other set one, else keeps self's.
    assert_eq!(merged.max_tokens, Some(1024)); // Kept from base
    assert_eq!(merged.temperature, Some(0.3)); // Overridden
    assert_eq!(merged.top_p, Some(0.9)); // From override
}

#[test]
fn test_completion_parameters_json_response() {
    let params = CompletionParameters::new().with_json_response();
    assert!(params.response_format.is_some());
}

// =============================================================================
// NodeCompletionParameters Tests
// =============================================================================

#[test]
fn test_node_completion_parameters() {
    let params = NodeCompletionParameters::new()
        .with_system_prompt("You are a helpful assistant")
        .expecting_json()
        .with_timeout(60);

    assert_eq!(
        params.system_prompt,
        Some("You are a helpful assistant".to_string())
    );
    assert!(params.parse_json);
    assert_eq!(params.timeout_secs, Some(60));
}

#[test]
fn test_node_completion_parameters_retry() {
    let params = NodeCompletionParameters::new()
        .with_retry(5)
        .with_exp_back_off(true)
        .with_back_off_time(2.0)
        .with_max_back_off(30.0)
        .with_crash_on_refusal(true)
        .with_crash_on_empty(true);

    assert_eq!(params.retry, 5);
    assert!(params.exp_back_off);
    assert_eq!(params.back_off_time, 2.0);
    assert_eq!(params.max_back_off, 30.0);
    assert!(params.crash_on_refusal);
    assert!(params.crash_on_empty_response);
}

#[test]
fn test_node_completion_parameters_force_prepend() {
    let params = NodeCompletionParameters::new()
        .with_force_prepend("Score: ")
        .with_parse_json(true);

    assert_eq!(params.force_prepend, Some("Score: ".to_string()));
    assert!(params.parse_json);
}

#[test]
fn test_provider_settings() {
    use minillmlib::OpenRouterRouting;

    let provider = OpenRouterRouting::new()
        .sort_by_throughput()
        .deny_data_collection()
        .with_ignore(vec!["SambaNova".to_string()]);

    assert_eq!(provider.sort, Some("throughput".to_string()));
    assert_eq!(provider.data_collection, Some("deny".to_string()));
    assert_eq!(provider.ignore, Some(vec!["SambaNova".to_string()]));
}

#[test]
fn test_completion_parameters_openrouter_routing_goes_to_extra() {
    use minillmlib::{CompletionParameters, OpenRouterRouting};

    let provider = OpenRouterRouting::new()
        .sort_by_throughput()
        .deny_data_collection();

    // OpenRouter routing is provider-specific → it lives under extra["provider"],
    // not as a universal field on the normalized params.
    let params = CompletionParameters::new()
        .with_temperature(0.7)
        .with_openrouter_routing(provider);

    let extra = params.extra.expect("routing stored in extra");
    let routing = &extra["provider"];
    assert_eq!(routing["sort"], "throughput");
    assert_eq!(routing["data_collection"], "deny");
}

#[test]
fn test_completion_parameters_with_extra() {
    use minillmlib::CompletionParameters;

    let params = CompletionParameters::new()
        .with_extra("custom_param", serde_json::json!(42))
        .with_extra("another_param", serde_json::json!("value"));

    assert!(params.extra.is_some());
    let extra = params.extra.unwrap();
    assert_eq!(extra.get("custom_param"), Some(&serde_json::json!(42)));
    assert_eq!(
        extra.get("another_param"),
        Some(&serde_json::json!("value"))
    );
}

// =============================================================================
// Message Tests
// =============================================================================

#[test]
fn test_message_creation() {
    let user_msg = Message::user("Hello");
    assert_eq!(user_msg.role, Role::User);
    assert_eq!(user_msg.text(), Some("Hello"));

    let assistant_msg = Message::assistant("Hi there!");
    assert_eq!(assistant_msg.role, Role::Assistant);

    let system_msg = Message::system("You are helpful");
    assert_eq!(system_msg.role, Role::System);
}

#[test]
fn test_message_with_name() {
    let msg = Message::user("Hello").with_name("Alice");
    assert_eq!(msg.name, Some("Alice".to_string()));
}

#[test]
fn test_message_content_text() {
    let content = MessageContent::text("Hello world");
    assert_eq!(content.get_text(), Some("Hello world"));
    assert!(!content.has_multimodal());
}

#[test]
fn test_message_content_multimodal() {
    let image = ImageData::from_url("https://example.com/image.jpg");
    let content = MessageContent::with_images("Describe this", &[image]);
    assert!(content.has_multimodal());
}

// =============================================================================
// ImageData Tests
// =============================================================================

#[test]
fn test_image_data_from_url() {
    let image = ImageData::from_url("https://example.com/image.jpg");
    assert_eq!(image.to_data_url(), "https://example.com/image.jpg");
    // URL is flagged by the explicit `is_url` bool, not a magic mime string.
    assert!(image.is_url());
}

#[test]
fn test_image_data_from_bytes() {
    let bytes = vec![0xFF, 0xD8, 0xFF, 0xE0]; // JPEG magic bytes
    let image = ImageData::from_bytes(&bytes, "image/jpeg");
    assert_eq!(image.mime_type, "image/jpeg");
    assert!(image.to_data_url().starts_with("data:image/jpeg;base64,"));
}

#[test]
fn test_image_data_from_file() {
    let path = "./data/test.jpg";
    if std::path::Path::new(path).exists() {
        let image = ImageData::from_file(path).unwrap();
        assert_eq!(image.mime_type, "image/jpeg");
        assert!(!image.base64_data.is_empty());
    }
}

#[test]
fn test_image_data_with_detail() {
    let image = ImageData::from_url("https://example.com/image.jpg").with_detail("high");
    assert_eq!(image.detail, Some("high".to_string()));
}

// =============================================================================
// AudioData Tests
// =============================================================================

#[test]
fn test_audio_data_from_bytes() {
    let bytes = vec![0x52, 0x49, 0x46, 0x46]; // RIFF header
    let audio = AudioData::from_bytes(&bytes, "wav");
    assert_eq!(audio.format, "wav");
    assert_eq!(audio.mime_type(), "audio/wav");
}

#[test]
fn test_audio_data_from_file() {
    let path = "./data/test.mp3";
    if std::path::Path::new(path).exists() {
        let audio = AudioData::from_file(path).unwrap();
        assert_eq!(audio.format, "mp3");
        assert!(!audio.base64_data.is_empty());
    }
}

#[test]
fn test_audio_data_with_metadata() {
    let audio = AudioData::from_bytes(&[0u8; 100], "wav")
        .with_sample_rate(44100)
        .with_channels(2);

    assert_eq!(audio.sample_rate, Some(44100));
    assert_eq!(audio.channels, Some(2));
}

// =============================================================================
// VideoData Tests
// =============================================================================

#[tokio::test]
async fn test_video_completion() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let video_path = "./data/test.mp4";
    if !std::path::Path::new(video_path).exists() {
        eprintln!("Skipping test_video_completion: test.mp4 not found");
        return;
    }

    let generator = get_test_generator();
    let video = VideoData::from_file(video_path).unwrap();

    let content = MessageContent::with_video("What do you see in this video? Be brief.", &[video]);

    let root = ChatNode::root("You are a helpful assistant.");
    let user_node = root
        .add_child(ChatNode::new(Message {
            role: Role::User,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            cache_breakpoint: false,
        }))
        .unwrap();

    let result = user_node.complete(&generator, None).await;

    match result {
        Ok(response) => {
            println!("Video description: {:?}", response.text());
            assert!(response.text().is_some());
            assert!(!response.text().unwrap().is_empty());
        }
        Err(e) => {
            panic!("Video completion failed: {:?}", e);
        }
    }
}

#[test]
fn test_video_data_from_url() {
    let video = VideoData::from_url("https://example.com/video.mp4");
    assert_eq!(video.to_data_url(), "https://example.com/video.mp4");
    // URL is flagged by the explicit `is_url` bool, not a magic format string.
    assert!(video.is_url());
}

#[tokio::test]
async fn test_image_completion_from_url() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let generator = get_test_generator();
    let image = ImageData::from_url("https://cdn.mos.cms.futurecdn.net/nbaR6JXZ3Z7mzuW9bh4nQN.jpg");

    let content = MessageContent::with_images("Describe this image in one sentence.", &[image]);

    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user_node = root
        .add_child(ChatNode::new(Message {
            role: Role::User,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            cache_breakpoint: false,
        }))
        .unwrap();

    let result = user_node.complete(&generator, None).await;

    match result {
        Ok(response) => {
            println!("Image (URL) description: {:?}", response.text());
            assert!(response.text().is_some());
            assert!(!response.text().unwrap().is_empty());
        }
        Err(e) => {
            panic!("Image URL completion failed: {:?}", e);
        }
    }
}

#[test]
fn test_content_part_json_serialization() {
    use minillmlib::message::ContentPart;

    // Test image serialization matches Python format
    let image = ImageData::from_url("https://example.com/image.jpg");
    let image_part = ContentPart::image(&image);
    let json = serde_json::to_value(&image_part).unwrap();
    println!(
        "Image part JSON: {}",
        serde_json::to_string_pretty(&json).unwrap()
    );
    assert_eq!(json["type"], "image_url");
    assert_eq!(json["image_url"]["url"], "https://example.com/image.jpg");

    // Test audio serialization matches Python format
    let audio = AudioData::from_bytes(&[0u8; 10], "mp3");
    let audio_part = ContentPart::audio(&audio);
    let json = serde_json::to_value(&audio_part).unwrap();
    println!(
        "Audio part JSON: {}",
        serde_json::to_string_pretty(&json).unwrap()
    );
    assert_eq!(json["type"], "input_audio");
    assert!(json["input_audio"]["data"].as_str().is_some());
    assert_eq!(json["input_audio"]["format"], "mp3");

    // Test video serialization matches Python format
    let video = VideoData::from_url("https://example.com/video.mp4");
    let video_part = ContentPart::video(&video);
    let json = serde_json::to_value(&video_part).unwrap();
    println!(
        "Video part JSON: {}",
        serde_json::to_string_pretty(&json).unwrap()
    );
    assert_eq!(json["type"], "video_url");
    assert_eq!(json["video_url"]["url"], "https://example.com/video.mp4");

    // Test full multimodal content
    let content = MessageContent::with_video("Describe this", &[video]);
    let api_format = minillmlib::provider::openai_wire::content_value(&content, false);
    println!(
        "Full content JSON: {}",
        serde_json::to_string_pretty(&api_format).unwrap()
    );
    let parts = api_format.as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["text"], "Describe this");
    assert_eq!(parts[1]["type"], "video_url");

    // Test with local file to see data URL format
    let image_path = "./data/test.jpg";
    if std::path::Path::new(image_path).exists() {
        let image = ImageData::from_file(image_path).unwrap();
        let image_part = ContentPart::image(&image);
        let json = serde_json::to_value(&image_part).unwrap();
        let url = json["image_url"]["url"].as_str().unwrap();
        println!(
            "Image from file - URL prefix: {}...",
            &url[..80.min(url.len())]
        );
    }

    // Test full message payload format (what gets sent to API)
    use minillmlib::provider::openai_wire::messages_to_payload;
    let image = ImageData::from_url("https://example.com/image.jpg");
    let content = MessageContent::with_images("Describe this image", &[image]);
    let msg = Message {
        role: Role::User,
        content,
        name: None,
        tool_call_id: None,
        tool_calls: None,
        cache_breakpoint: false,
    };
    let payload = messages_to_payload(&[msg], false);
    println!("\nFull message payload:");
    println!("{}", serde_json::to_string_pretty(&payload).unwrap());
}

#[test]
fn test_video_data_from_bytes() {
    let bytes = vec![0x00, 0x00, 0x00, 0x1C, 0x66, 0x74, 0x79, 0x70]; // MP4 header
    let video = VideoData::from_bytes(&bytes, "mp4");
    assert_eq!(video.format, "mp4");
    assert_eq!(video.mime_type(), "video/mp4");
    assert!(video.to_data_url().starts_with("data:video/mp4;base64,"));
}

#[test]
fn test_video_data_from_file() {
    let path = "./data/test.mp4";
    if std::path::Path::new(path).exists() {
        let video = VideoData::from_file(path).unwrap();
        assert_eq!(video.format, "mp4");
        assert!(!video.base64_data.is_empty());
    }
}

#[test]
fn test_video_data_with_metadata() {
    let video = VideoData::from_bytes(&[0u8; 100], "mp4")
        .with_duration(120.5)
        .with_dimensions(1920, 1080)
        .with_frame_rate(30.0);

    assert_eq!(video.duration_secs, Some(120.5));
    assert_eq!(video.width, Some(1920));
    assert_eq!(video.height, Some(1080));
    assert_eq!(video.frame_rate, Some(30.0));
}

#[test]
fn test_video_data_mime_types() {
    assert_eq!(VideoData::from_bytes(&[], "mp4").mime_type(), "video/mp4");
    assert_eq!(VideoData::from_bytes(&[], "webm").mime_type(), "video/webm");
    assert_eq!(
        VideoData::from_bytes(&[], "mov").mime_type(),
        "video/quicktime"
    );
    assert_eq!(
        VideoData::from_bytes(&[], "avi").mime_type(),
        "video/x-msvideo"
    );
    assert_eq!(
        VideoData::from_bytes(&[], "mkv").mime_type(),
        "video/x-matroska"
    );
}

// =============================================================================
// Media (Unified) Tests
// =============================================================================

#[test]
fn test_media_from_image() {
    let image = ImageData::from_url("https://example.com/image.jpg");
    let media = Media::from(image.clone());

    assert!(media.is_image());
    assert!(!media.is_audio());
    assert!(!media.is_video());
    assert!(!media.is_document());
    assert!(media.as_image().is_some());
}

#[test]
fn test_media_from_audio() {
    let audio = AudioData::from_bytes(&[0u8; 100], "wav");
    let media = Media::from(audio.clone());

    assert!(!media.is_image());
    assert!(media.is_audio());
    assert!(!media.is_video());
    assert!(!media.is_document());
    assert!(media.as_audio().is_some());
}

#[test]
fn test_media_from_video() {
    let video = VideoData::from_bytes(&[0u8; 100], "mp4");
    let media = Media::from(video.clone());

    assert!(!media.is_image());
    assert!(!media.is_audio());
    assert!(media.is_video());
    assert!(!media.is_document());
    assert!(media.as_video().is_some());
}

#[test]
fn test_media_from_document() {
    let document = DocumentData::from_bytes(&[0u8; 100], "application/pdf");
    let media = Media::from(document.clone());

    assert!(!media.is_image());
    assert!(!media.is_audio());
    assert!(!media.is_video());
    assert!(media.is_document());
    assert!(media.as_document().is_some());
}

#[test]
fn test_message_content_with_documents() {
    let document = DocumentData::from_bytes(&[0u8; 100], "application/pdf").with_page_count(5);
    let content = MessageContent::with_documents("Summarize this document", &[document]);
    assert!(content.has_multimodal());
    assert_eq!(content.get_text(), Some("Summarize this document"));
    // The document part must actually BE a file part carrying its metadata,
    // not just "something non-text" (a degraded text part would still pass
    // has_multimodal-style checks on a longer message).
    let minillmlib::message::MessageContent::Parts(parts) = &content else {
        panic!("with_documents builds Parts");
    };
    let minillmlib::message::ContentPart::File { file } = &parts[1] else {
        panic!("the document must become a File part, got {:?}", parts[1]);
    };
    assert_eq!(file.page_count, Some(5));
}

#[test]
fn test_message_content_with_video() {
    let video = VideoData::from_url("https://example.com/video.mp4");
    let content = MessageContent::with_video("Describe this video", &[video]);
    assert!(content.has_multimodal());
    assert_eq!(content.get_text(), Some("Describe this video"));
}

#[test]
fn test_message_content_with_media() {
    let image = ImageData::from_url("https://example.com/image.jpg");
    let audio = AudioData::from_bytes(&[0u8; 100], "wav");
    let video = VideoData::from_url("https://example.com/video.mp4");

    let document = DocumentData::from_bytes(&[0u8; 100], "application/pdf");

    let media = vec![
        Media::from(image),
        Media::from(audio),
        Media::from(video),
        Media::from(document),
    ];

    let content = MessageContent::with_media("Describe all this media", &media);
    assert!(content.has_multimodal());
    assert_eq!(content.get_text(), Some("Describe all this media"));
    // Each media kind must arrive as ITS part kind (from_media dispatched
    // right), not merely as "something non-text".
    let minillmlib::message::MessageContent::Parts(parts) = &content else {
        panic!("with_media builds Parts");
    };
    use minillmlib::message::ContentPart;
    assert!(
        matches!(parts[1], ContentPart::Image { .. }),
        "{:?}",
        parts[1]
    );
    assert!(
        matches!(parts[2], ContentPart::Audio { .. }),
        "{:?}",
        parts[2]
    );
    assert!(
        matches!(parts[3], ContentPart::Video { .. }),
        "{:?}",
        parts[3]
    );
    assert!(
        matches!(parts[4], ContentPart::File { .. }),
        "{:?}",
        parts[4]
    );
}

// =============================================================================
// ChatNode Tests
// =============================================================================

#[test]
fn test_chat_node_root() {
    let root = ChatNode::root("You are a helpful assistant");
    assert!(root.is_root());
    assert!(root.is_leaf());
    assert_eq!(root.role(), Role::System);
    assert_eq!(root.depth(), 0);
}

#[test]
fn test_chat_node_add_children() {
    let root = ChatNode::root("System");
    let user = root.add_user("Hello");
    let assistant = user.add_assistant("Hi there!");

    assert!(!root.is_leaf());
    assert_eq!(root.child_count(), 1);
    assert_eq!(user.depth(), 1);
    assert_eq!(assistant.depth(), 2);
    assert!(assistant.is_leaf());
}

#[test]
fn test_chat_node_thread() {
    let root = ChatNode::root("System");
    let user = root.add_user("Hello");
    let assistant = user.add_assistant("Hi!");

    let thread = assistant.thread();
    assert_eq!(thread.len(), 3);
    assert_eq!(thread[0].role, Role::System);
    assert_eq!(thread[1].role, Role::User);
    assert_eq!(thread[2].role, Role::Assistant);
}

#[test]
fn test_chat_node_find_by_id() {
    let root = ChatNode::root("System");
    let user = root.add_user("Hello");
    let user_id = user.id.clone();

    let found = root.find_by_id(&user_id);
    assert!(found.is_some());
    assert_eq!(found.unwrap().id, user_id);
}

#[test]
fn test_chat_node_get_leaf() {
    let root = ChatNode::root("System");
    let u1 = root.add_user("U1");
    let a1 = u1.add_assistant("A1");
    let u2 = a1.add_user("U2");

    let leaf = root.get_leaf();
    assert_eq!(leaf.id, u2.id);
}

#[test]
fn test_chat_node_get_root() {
    let root = ChatNode::root("System");
    let u1 = root.add_user("U1");
    let a1 = u1.add_assistant("A1");
    let u2 = a1.add_user("U2");

    // From any node, get_root should return the root
    assert_eq!(root.get_root().id, root.id);
    assert_eq!(u1.get_root().id, root.id);
    assert_eq!(a1.get_root().id, root.id);
    assert_eq!(u2.get_root().id, root.id);

    // The returned root should be a root node
    assert!(u2.get_root().is_root());
}

#[test]
fn test_chat_node_detach() {
    let root = ChatNode::root("System");
    let u1 = root.add_user("U1");
    let a1 = u1.add_assistant("A1");

    assert_eq!(root.child_count(), 1);
    assert!(!u1.is_root());

    // Detach u1 from root
    u1.detach();

    // u1 should now be a root
    assert!(u1.is_root());
    // root should have no children
    assert_eq!(root.child_count(), 0);
    // a1 should still be a child of u1
    assert_eq!(u1.child_count(), 1);
    assert_eq!(a1.parent().unwrap().id, u1.id);
}

#[test]
fn test_chat_node_merge() {
    // Create first tree
    let root1 = ChatNode::root("System 1");
    let u1 = root1.add_user("User 1");

    // Create second tree
    let root2 = ChatNode::root("System 2");
    let u2 = root2.add_user("User 2");
    let a2 = u2.add_assistant("Assistant 2");

    // Merge the second tree into the first. `merge` COPIES other's subtree into
    // this tree (the two arenas stay independent), returning a handle to the
    // copied leaf in tree 1. The original tree-2 handles are untouched.
    let merged_leaf = u1.merge(&a2).unwrap();

    // The merged leaf carries the same content as a2 (a fresh copy, fresh id).
    assert_eq!(merged_leaf.text(), Some("Assistant 2"));
    assert_ne!(merged_leaf.id, a2.id);

    // root1 now contains the copied second tree as a subtree:
    // root1 -> u1 -> [copy of root2 -> u2 -> a2]
    assert_eq!(root1.node_count(), 5);
    // Navigate via the returned (in-tree-1) handle: its root is root1.
    assert_eq!(merged_leaf.get_root().id, root1.id);
    assert_eq!(merged_leaf.thread().len(), 5);

    // The original tree 2 is unchanged (copy, not move).
    assert!(root2.is_root());
    assert_eq!(root2.node_count(), 3);
}

#[test]
fn test_chat_node_iter_depth_first() {
    //       root
    //      /    \
    //     u1     u2
    //     |
    //     a1
    let root = ChatNode::root("System");
    let u1 = root.add_user("U1");
    let u2 = root.add_user("U2");
    let a1 = u1.add_assistant("A1");

    let nodes = root.iter_depth_first();

    assert_eq!(nodes.len(), 4);
    // Depth-first pre-order: root, u1, a1, u2
    assert_eq!(nodes[0].id, root.id);
    assert_eq!(nodes[1].id, u1.id);
    assert_eq!(nodes[2].id, a1.id);
    assert_eq!(nodes[3].id, u2.id);
}

#[test]
fn test_chat_node_iter_breadth_first() {
    //       root
    //      /    \
    //     u1     u2
    //     |
    //     a1
    let root = ChatNode::root("System");
    let u1 = root.add_user("U1");
    let u2 = root.add_user("U2");
    let _a1 = u1.add_assistant("A1");

    let nodes = root.iter_breadth_first();

    assert_eq!(nodes.len(), 4);
    // Breadth-first: root, u1, u2, a1
    assert_eq!(nodes[0].id, root.id);
    assert_eq!(nodes[1].id, u1.id);
    assert_eq!(nodes[2].id, u2.id);
    // a1 is last because it's at depth 2
}

#[test]
fn test_chat_node_iter_leaves() {
    //       root
    //      /    \
    //     u1     u2
    //     |
    //     a1
    let root = ChatNode::root("System");
    let u1 = root.add_user("U1");
    let u2 = root.add_user("U2");
    let a1 = u1.add_assistant("A1");

    let leaves = root.iter_leaves();

    assert_eq!(leaves.len(), 2);
    // Leaves are a1 and u2
    let leaf_ids: Vec<_> = leaves.iter().map(|n| n.id.clone()).collect();
    assert!(leaf_ids.contains(&a1.id));
    assert!(leaf_ids.contains(&u2.id));
}

#[test]
fn test_chat_node_node_count() {
    let root = ChatNode::root("System");
    assert_eq!(root.node_count(), 1);

    let u1 = root.add_user("U1");
    assert_eq!(root.node_count(), 2);

    let _a1 = u1.add_assistant("A1");
    assert_eq!(root.node_count(), 3);

    let _u2 = root.add_user("U2");
    assert_eq!(root.node_count(), 4);
}

#[test]
fn test_chat_node_complex_tree_operations() {
    // Build a complex tree:
    //              root
    //           /   |   \
    //         u1   u2    u3
    //        / \    |
    //      a1  a2   a3
    //      |
    //     u4

    let root = ChatNode::root("System");
    let u1 = root.add_user("U1");
    let u2 = root.add_user("U2");
    let u3 = root.add_user("U3");
    let a1 = u1.add_assistant("A1");
    let a2 = u1.add_assistant("A2");
    let a3 = u2.add_assistant("A3");
    let u4 = a1.add_user("U4");

    // Test node_count
    assert_eq!(root.node_count(), 8);
    assert_eq!(u1.node_count(), 4); // u1, a1, a2, u4
    assert_eq!(u2.node_count(), 2); // u2, a3
    assert_eq!(u3.node_count(), 1); // just u3

    // Test iter_leaves
    let leaves = root.iter_leaves();
    assert_eq!(leaves.len(), 4); // u4, a2, a3, u3
    let leaf_ids: Vec<_> = leaves.iter().map(|n| n.id.clone()).collect();
    assert!(leaf_ids.contains(&u4.id));
    assert!(leaf_ids.contains(&a2.id));
    assert!(leaf_ids.contains(&a3.id));
    assert!(leaf_ids.contains(&u3.id));

    // Test depth
    assert_eq!(root.depth(), 0);
    assert_eq!(u1.depth(), 1);
    assert_eq!(a1.depth(), 2);
    assert_eq!(u4.depth(), 3);

    // Test get_root from deepest node
    assert_eq!(u4.get_root().id, root.id);

    // Test thread from u4 (should be: root -> u1 -> a1 -> u4)
    let thread = u4.thread();
    assert_eq!(thread.len(), 4);
    assert_eq!(thread[0].role, Role::System);
    assert_eq!(thread[1].role, Role::User);
    assert_eq!(thread[2].role, Role::Assistant);
    assert_eq!(thread[3].role, Role::User);
}

#[test]
fn test_chat_node_detach_and_reattach() {
    // Build tree: root -> u1 -> a1 -> u2
    let root = ChatNode::root("System");
    let u1 = root.add_user("U1");
    let a1 = u1.add_assistant("A1");
    let u2 = a1.add_user("U2");

    assert_eq!(root.node_count(), 4);

    // Detach a1 (and its subtree: a1 -> u2)
    a1.detach();

    // root tree should now only have: root -> u1
    assert_eq!(root.node_count(), 2);
    assert_eq!(u1.child_count(), 0);

    // a1 should be a new root with u2 as child
    assert!(a1.is_root());
    assert_eq!(a1.node_count(), 2);
    assert_eq!(u2.get_root().id, a1.id);

    // Reattach a1 to root directly (a1 is a detached subtree, not an ancestor of root)
    root.add_child(a1.clone()).unwrap();

    // Now structure is: root -> [u1, a1 -> u2]
    assert_eq!(root.node_count(), 4);
    assert_eq!(root.child_count(), 2);
    assert_eq!(a1.parent().unwrap().id, root.id);
}

#[test]
fn test_chat_node_multiple_merges() {
    // Create 3 separate trees
    let tree1_root = ChatNode::root("System 1");
    let tree1_u = tree1_root.add_user("Tree1 User");

    let tree2_root = ChatNode::root("System 2");
    let tree2_u = tree2_root.add_user("Tree2 User");
    let tree2_a = tree2_u.add_assistant("Tree2 Assistant");

    let tree3_root = ChatNode::root("System 3");
    let tree3_u = tree3_root.add_user("Tree3 User");

    // Merge tree2 into tree1. `merge` copies tree2's subtree into tree1 and
    // returns the copied leaf (in tree1); continue from THAT handle.
    let merged2 = tree1_u.merge(&tree2_a).unwrap();
    // tree1: tree1_root -> tree1_u -> [copy of tree2_root -> tree2_u -> tree2_a]
    assert_eq!(tree1_root.node_count(), 5);

    // Merge tree3 into the copied tree2 leaf (still in tree1).
    let merged3 = merged2.merge(&tree3_u).unwrap();
    // tree1 now: ... -> tree2_a(copy) -> [copy of tree3_root -> tree3_u]
    assert_eq!(tree1_root.node_count(), 7);

    // Verify the chain via the in-tree-1 handles.
    assert_eq!(merged3.get_root().id, tree1_root.id);
    assert_eq!(merged3.depth(), 6);
}

#[test]
fn test_chat_node_iter_with_branching() {
    // Build a wide tree:
    //           root
    //     /  /  |  \  \
    //    c1 c2  c3  c4  c5
    //    |      |
    //   gc1    gc2

    let root = ChatNode::root("Root");
    let c1 = root.add_user("C1");
    let c2 = root.add_user("C2");
    let c3 = root.add_user("C3");
    let c4 = root.add_user("C4");
    let c5 = root.add_user("C5");
    let gc1 = c1.add_assistant("GC1");
    let gc2 = c3.add_assistant("GC2");

    // Depth-first should visit: root, c1, gc1, c2, c3, gc2, c4, c5
    let dfs = root.iter_depth_first();
    assert_eq!(dfs.len(), 8);
    assert_eq!(dfs[0].id, root.id);
    assert_eq!(dfs[1].id, c1.id);
    assert_eq!(dfs[2].id, gc1.id);
    assert_eq!(dfs[3].id, c2.id);
    assert_eq!(dfs[4].id, c3.id);
    assert_eq!(dfs[5].id, gc2.id);
    assert_eq!(dfs[6].id, c4.id);
    assert_eq!(dfs[7].id, c5.id);

    // Breadth-first should visit: root, c1, c2, c3, c4, c5, gc1, gc2
    let bfs = root.iter_breadth_first();
    assert_eq!(bfs.len(), 8);
    assert_eq!(bfs[0].id, root.id);
    // Level 1: c1, c2, c3, c4, c5 (in order)
    assert_eq!(bfs[1].id, c1.id);
    assert_eq!(bfs[2].id, c2.id);
    assert_eq!(bfs[3].id, c3.id);
    assert_eq!(bfs[4].id, c4.id);
    assert_eq!(bfs[5].id, c5.id);
    // Level 2: gc1, gc2
    assert_eq!(bfs[6].id, gc1.id);
    assert_eq!(bfs[7].id, gc2.id);

    // Leaves: c2, c4, c5, gc1, gc2
    let leaves = root.iter_leaves();
    assert_eq!(leaves.len(), 5);
}

#[test]
fn test_chat_node_format_kwargs_with_merge() {
    // Node-level kwargs are scoped to the node they're set on. Each node holds
    // the kwargs for the placeholders in its own text.
    let tree1 = ChatNode::root("Hello {name}, I am {bot}.");
    tree1.set_format_kwarg("name", "Alice");
    tree1.set_format_kwarg("bot", "Claude");
    let tree1_u = tree1.add_user("Hi {bot}!");
    tree1_u.set_format_kwarg("bot", "Claude");

    let tree2 = ChatNode::root("Switching to {mode} mode.");
    tree2.set_format_kwarg("mode", "expert");
    let tree2_u = tree2.add_user("Tell me about {topic}.");
    tree2_u.set_format_kwarg("topic", "Rust");

    // Merge tree2 into tree1. The copy carries each node's own kwargs. Continue
    // from the returned merged leaf (in tree1) to read the full 4-node thread.
    let merged_leaf = tree1_u.merge(&tree2_u).unwrap();

    let formatted = merged_leaf.formatted_thread();
    assert_eq!(formatted.len(), 4);

    // Each message resolved with its own node's kwargs (copied alongside).
    assert!(formatted[0].content.get_text().unwrap().contains("Alice"));
    assert!(formatted[0].content.get_text().unwrap().contains("Claude"));
    assert!(formatted[1].content.get_text().unwrap().contains("Claude"));
    assert!(formatted[2].content.get_text().unwrap().contains("expert"));
    assert!(formatted[3].content.get_text().unwrap().contains("Rust"));
}

#[test]
fn test_chat_node_detach_preserves_own_format_kwargs() {
    // Node-level kwargs are per-node, so each node's text is filled by its own
    // kwargs only. Detaching keeps the node's own kwargs intact.
    let root = ChatNode::root("Hello {name}");
    root.set_format_kwarg("name", "World");
    let u1 = root.add_user("Goodbye {who}");
    u1.set_format_kwarg("who", "Alice");

    let before = u1.formatted_thread();
    assert!(before[0].content.get_text().unwrap().contains("World")); // root's own
    assert!(before[1].content.get_text().unwrap().contains("Alice")); // u1's own

    u1.detach();

    // u1 is now its own root; its own kwarg still fills its own text.
    let after = u1.formatted_thread();
    assert_eq!(after.len(), 1);
    assert!(after[0].content.get_text().unwrap().contains("Alice"));
}

#[test]
fn test_chat_node_to_thread_data() {
    let root = ChatNode::root("You are helpful");
    let u1 = root.add_user("Hello");
    let a1 = u1.add_assistant("Hi there!");

    let thread_data = a1.to_thread_data();

    assert_eq!(thread_data.prompts.len(), 3);
    assert_eq!(thread_data.prompts[0].message.role, Role::System);
    assert_eq!(
        thread_data.prompts[0].message.text(),
        Some("You are helpful")
    );
    assert_eq!(thread_data.prompts[1].message.role, Role::User);
    assert_eq!(thread_data.prompts[1].message.text(), Some("Hello"));
    assert_eq!(thread_data.prompts[2].message.role, Role::Assistant);
    assert_eq!(thread_data.prompts[2].message.text(), Some("Hi there!"));
}

#[test]
fn test_chat_node_from_thread_json() {
    let json = r#"{
        "prompts": [
            {"message": {"role": "system", "content": "You are helpful"}},
            {"message": {"role": "user", "content": "Hello"}},
            {"message": {"role": "assistant", "content": "Hi!"}}
        ]
    }"#;

    // Returns (root, leaf) tuple - must keep root alive for weak refs to work
    let (_root, leaf) = ChatNode::from_thread_json(json).unwrap();

    // Should return the last node (assistant)
    assert_eq!(leaf.role(), Role::Assistant);
    assert_eq!(leaf.text(), Some("Hi!"));

    // Check the full thread
    let thread = leaf.thread();
    assert_eq!(thread.len(), 3);
    assert_eq!(thread[0].role, Role::System);
    assert_eq!(thread[1].role, Role::User);
    assert_eq!(thread[2].role, Role::Assistant);
}

#[test]
fn test_chat_node_save_and_load_thread() {
    use std::fs;

    let root = ChatNode::root("System prompt");
    let u1 = root.add_user("User message");
    let a1 = u1.add_assistant("Assistant response");

    // Save to temp file
    let temp_path = "/tmp/test_thread.json";
    a1.save_thread(temp_path).unwrap();

    // Load it back - returns (root, leaf) tuple
    let (_loaded_root, loaded_leaf) = ChatNode::from_thread_file(temp_path).unwrap();

    // Verify
    let thread = loaded_leaf.thread();
    assert_eq!(thread.len(), 3);
    assert_eq!(thread[0].content.get_text(), Some("System prompt"));
    assert_eq!(thread[1].content.get_text(), Some("User message"));
    assert_eq!(thread[2].content.get_text(), Some("Assistant response"));

    // Cleanup
    fs::remove_file(temp_path).ok();
}

#[test]
fn test_chat_node_from_messages() {
    let messages = vec![
        Message::system("Be helpful"),
        Message::user("Hi"),
        Message::assistant("Hello!"),
    ];

    // Returns (root, leaf) tuple
    let (_root, leaf) = ChatNode::from_messages(&messages).unwrap();

    assert_eq!(leaf.role(), Role::Assistant);
    let thread = leaf.thread();
    assert_eq!(thread.len(), 3);
}

#[test]
fn test_chat_node_format_kwargs_basic() {
    let root = ChatNode::root("You are {assistant_name}, a helpful assistant.");
    root.set_format_kwarg("assistant_name", "Claude");

    let formatted = root.formatted_text().unwrap();
    assert_eq!(formatted, "You are Claude, a helpful assistant.");
}

#[test]
fn test_chat_node_format_kwargs_multiple() {
    let root = ChatNode::root("Hello {name}, you are {age} years old.");

    let mut kwargs = std::collections::HashMap::new();
    kwargs.insert("name".to_string(), "Alice".to_string());
    kwargs.insert("age".to_string(), "25".to_string());
    root.set_format_kwargs(&kwargs);

    let formatted = root.formatted_text().unwrap();
    assert_eq!(formatted, "Hello Alice, you are 25 years old.");
}

#[test]
fn test_chat_node_formatted_thread() {
    // Node-level kwargs are per-node: each node fills the placeholders in its
    // own text from its own kwargs.
    let root = ChatNode::root("You are {bot_name}.");
    root.set_format_kwarg("bot_name", "Assistant");

    let user = root.add_user("Hi {bot_name}!");
    user.set_format_kwarg("bot_name", "Assistant");
    let assistant = user.add_assistant("Hello {user_name}!");
    assistant.set_format_kwarg("user_name", "Bob");

    let formatted = assistant.formatted_thread();

    assert_eq!(formatted[0].content.get_text(), Some("You are Assistant."));
    assert_eq!(formatted[1].content.get_text(), Some("Hi Assistant!"));
    assert_eq!(formatted[2].content.get_text(), Some("Hello Bob!"));
}

#[test]
fn test_chat_node_format_kwargs_are_node_scoped() {
    // A node-level kwarg only fills placeholders in its OWN message; it does not
    // bleed into ancestor or descendant text.
    let root = ChatNode::root("Hi {who} from {where}");
    root.set_format_kwarg("who", "Alice");
    let user = root.add_user("Still {who}, now in {where}");
    user.set_format_kwarg("where", "Paris");

    let formatted = user.formatted_thread();
    // Root fills only its own "who"; "where" is unset on root so stays a placeholder.
    assert_eq!(
        formatted[0].content.get_text(),
        Some("Hi Alice from {where}")
    );
    // Leaf fills only its own "where"; "who" is unset on the leaf so stays a placeholder.
    assert_eq!(
        formatted[1].content.get_text(),
        Some("Still {who}, now in Paris")
    );
}

#[test]
fn test_completion_format_kwargs_base_with_node_override() {
    // Completion-level kwargs are the base layer; a per-node override wins on
    // collision. Verified through formatted_thread_with_base (the resolution the
    // completion path uses).
    let root = ChatNode::root("I am {bot} in {mode} mode");
    let user = root.add_user("Hello from {bot}");
    user.set_format_kwarg("bot", "Override"); // node override

    let mut base = std::collections::HashMap::new();
    base.insert("bot".to_string(), "Base".to_string());
    base.insert("mode".to_string(), "expert".to_string());

    let formatted = user.formatted_thread_with_base(&base);
    // Root has no override, so it resolves "bot" from the completion base; the
    // leaf's override is node-scoped and does not reach the root.
    assert_eq!(
        formatted[0].content.get_text(),
        Some("I am Base in expert mode")
    );
    // The leaf's own override wins over the base for the leaf's text.
    assert_eq!(formatted[1].content.get_text(), Some("Hello from Override"));

    // The plain formatted_thread (no base) sees neither the completion-level
    // kwargs nor the leaf's node-scoped override: the root has no own kwargs, so
    // its placeholders stay unfilled; the leaf fills only its own "bot".
    let plain = user.formatted_thread();
    assert_eq!(
        plain[0].content.get_text(),
        Some("I am {bot} in {mode} mode")
    );
    assert_eq!(plain[1].content.get_text(), Some("Hello from Override"));
}

#[test]
fn test_clone_tree_is_isolated_from_original() {
    let root = ChatNode::root("System");
    let user = root.add_user("Hello");
    let _assistant = user.add_assistant("Hi");

    // Clone the tree from a mid-node; we get back the counterpart of `user`.
    let cloned_user = user.clone_tree();
    assert_ne!(cloned_user.id, user.id, "clone has fresh ids");
    assert_eq!(cloned_user.text(), Some("Hello"));

    // The clone keeps its full ancestor spine (root), with fresh ids.
    let cloned_root = cloned_user.get_root();
    assert_eq!(cloned_root.text(), Some("System"));
    assert_ne!(cloned_root.id, root.id);
    assert_eq!(cloned_user.thread().len(), 2); // System, Hello

    // Mutating (or extending) the clone leaves the original untouched.
    cloned_user.set_format_kwarg("k", "v");
    assert_eq!(cloned_user.get_format_kwarg("k"), Some("v".to_string()));
    assert_eq!(user.get_format_kwarg("k"), None);
    let _new = cloned_user.add_assistant("forked reply");
    assert_eq!(user.child_count(), 1, "original's children unchanged");
}

#[test]
fn test_chat_node_format_kwargs_save_load() {
    use std::fs;

    // Per-node kwargs round-trip: each node's own kwargs save into its own JSON
    // entry and reload onto the matching node.
    let root = ChatNode::root("Hello {name}!");
    root.set_format_kwarg("name", "World");
    let user = root.add_user("Goodbye {other}!");
    user.set_format_kwarg("other", "Mars");

    let temp_path = "/tmp/test_format_kwargs.json";
    user.save_thread(temp_path).unwrap();

    let (loaded_root, loaded_leaf) = ChatNode::from_thread_file(temp_path).unwrap();

    // Each node restored its OWN kwargs, not a flattened root dump.
    assert_eq!(
        loaded_root.get_format_kwarg("name"),
        Some("World".to_string())
    );
    assert_eq!(loaded_root.get_format_kwarg("other"), None);
    assert_eq!(
        loaded_leaf.get_format_kwarg("other"),
        Some("Mars".to_string())
    );
    assert_eq!(loaded_leaf.get_format_kwarg("name"), None);

    // Each message fills from its own node's kwargs.
    let formatted = loaded_leaf.formatted_thread();
    assert_eq!(formatted[0].content.get_text(), Some("Hello World!"));
    assert_eq!(formatted[1].content.get_text(), Some("Goodbye Mars!"));

    fs::remove_file(temp_path).ok();
}

#[tokio::test]
async fn test_load_data_test_json() {
    let path = "./data/test.json";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping test_load_data_test_json: data/test.json not found");
        return;
    }

    // Load the saved prompt template.
    let (_root, leaf) = ChatNode::from_thread_file(path).unwrap();

    // Fill the whole template at completion time via completion-level kwargs
    // (the thread-wide mechanism for one-shot template fills).
    let mut fills = std::collections::HashMap::new();
    fills.insert("assistant_name".to_string(), "Claude".to_string());
    fills.insert("user_name".to_string(), "Alice".to_string());
    fills.insert("topic".to_string(), "quantum computing".to_string());
    fills.insert("style".to_string(), "friendly and concise".to_string());

    // Verify all placeholders are replaced thread-wide.
    let formatted = leaf.formatted_thread_with_base(&fills);
    for msg in &formatted {
        let text = msg.content.get_text().unwrap();
        for placeholder in ["{assistant_name}", "{user_name}", "{topic}", "{style}"] {
            assert!(
                !text.contains(placeholder),
                "Placeholder {} not replaced: {}",
                placeholder,
                text
            );
        }
    }

    // The offline assertions above always run; the live call below is gated.
    require_live!("OPENROUTER_API_KEY");

    // Now let's actually call the LLM with the fills passed as completion kwargs.
    let gi = get_cheap_generator();
    let params = NodeCompletionParameters::default()
        .with_params(CompletionParameters::default().with_max_tokens(150))
        .with_format_kwargs(fills);

    let result = leaf.complete(&gi, Some(&params)).await;
    match result {
        Ok(response) => {
            println!("LLM Response: {}", response.text().unwrap_or("no text"));
            // The response should mention Alice and quantum computing
            let response_text = response.text().unwrap_or("");
            assert!(
                response_text.to_lowercase().contains("alice")
                    || response_text.to_lowercase().contains("quantum"),
                "Response should mention Alice or quantum: {}",
                response_text
            );
        }
        Err(e) => {
            eprintln!("LLM Error: {}", e);
        }
    }
}

#[test]
fn test_chat_node_metadata() {
    let node = ChatNode::user("Hello");
    node.set_metadata("custom_key", serde_json::json!({"value": 42}));

    let meta = node.get_metadata("custom_key");
    assert!(meta.is_some());
    assert_eq!(meta.unwrap()["value"], 42);
}

#[test]
fn test_conversation_builder() {
    use minillmlib::chat_node::ConversationBuilder;

    let conv = ConversationBuilder::new("You are helpful")
        .user("Hello")
        .assistant("Hi!")
        .user("How are you?");

    let current = conv.current();
    assert_eq!(current.role(), Role::User);
    assert_eq!(current.text(), Some("How are you?"));

    let thread = current.thread();
    assert_eq!(thread.len(), 4);
}

// =============================================================================
// Real API Integration Tests
// =============================================================================

#[tokio::test]
async fn test_cost_tracking() {
    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user = root.add_user("Say 'Hi' and nothing else.");

    // Track costs using a shared counter
    let total_cost = Arc::new(Mutex::new(0.0_f64));
    let total_tokens = Arc::new(Mutex::new(0_u32));
    let callback_called = Arc::new(Mutex::new(false));

    let cost_tracker = total_cost.clone();
    let token_tracker = total_tokens.clone();
    let called_tracker = callback_called.clone();

    let params = NodeCompletionParameters::new()
        .with_cost_tracking(true)
        .with_cost_callback(move |info: CostInfo| {
            println!("\n=== COST CALLBACK RECEIVED ===");
            println!("Cost: {} credits", info.cost);
            println!("Prompt tokens: {}", info.prompt_tokens);
            println!("Completion tokens: {}", info.completion_tokens);
            println!("Total tokens: {}", info.total_tokens);
            println!("Model: {}", info.model);
            println!("Response ID: {}", info.response_id);
            println!("Cache read tokens: {}", info.cache_read_tokens);
            println!("Cache write tokens: {}", info.cache_write_tokens);
            println!("==============================\n");

            *cost_tracker.lock().unwrap() += info.cost;
            *token_tracker.lock().unwrap() += info.total_tokens;
            *called_tracker.lock().unwrap() = true;
        });

    let result = user.complete(&generator, Some(&params)).await;

    match result {
        Ok(response) => {
            println!("Response: {:?}", response.text());
            assert!(response.text().is_some());
        }
        Err(e) => {
            panic!("Completion failed: {:?}", e);
        }
    }

    // Verify callback was called
    assert!(
        *callback_called.lock().unwrap(),
        "Cost callback was not called!"
    );

    // Verify we got some token count
    let tokens = *total_tokens.lock().unwrap();
    assert!(tokens > 0, "Expected non-zero token count, got {}", tokens);

    println!("Total cost: {} credits", *total_cost.lock().unwrap());
    println!("Total tokens: {}", tokens);
}

#[tokio::test]
async fn test_cost_tracking_multiple_requests() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();

    // Track cumulative costs
    let total_cost = Arc::new(Mutex::new(0.0_f64));
    let request_count = Arc::new(Mutex::new(0_u32));

    let cost_tracker = total_cost.clone();
    let count_tracker = request_count.clone();

    let params = NodeCompletionParameters::new()
        .with_cost_tracking(true)
        .with_cost_callback(move |info: CostInfo| {
            *cost_tracker.lock().unwrap() += info.cost;
            *count_tracker.lock().unwrap() += 1;
            println!(
                "Request cost: {} credits (cumulative: {})",
                info.cost,
                *cost_tracker.lock().unwrap()
            );
        });

    // Make 3 requests
    let root = ChatNode::root("Be very brief.");
    let mut current = root.add_user("Say 'one'");
    current = current.complete(&generator, Some(&params)).await.unwrap();

    current = current.add_user("Say 'two'");
    current = current.complete(&generator, Some(&params)).await.unwrap();

    current = current.add_user("Say 'three'");
    let _ = current.complete(&generator, Some(&params)).await.unwrap();

    // Verify all 3 callbacks were called
    assert_eq!(
        *request_count.lock().unwrap(),
        3,
        "Expected 3 cost callbacks"
    );

    println!(
        "\nFinal cumulative cost: {} credits",
        *total_cost.lock().unwrap()
    );
}

#[tokio::test]
async fn test_cost_tracking_streaming() {
    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user = root.add_user("Say 'Hello' and nothing else.");

    // Track costs
    let callback_called = Arc::new(Mutex::new(false));
    let cost_received = Arc::new(Mutex::new(0.0_f64));
    let tokens_received = Arc::new(Mutex::new(0_u32));

    let called_tracker = callback_called.clone();
    let cost_tracker = cost_received.clone();
    let token_tracker = tokens_received.clone();

    let params = NodeCompletionParameters::new()
        .with_cost_tracking(true)
        .with_cost_callback(move |info: CostInfo| {
            println!("\n=== STREAMING COST CALLBACK ===");
            println!("Cost: {} credits", info.cost);
            println!("Total tokens: {}", info.total_tokens);
            println!("===============================\n");

            *called_tracker.lock().unwrap() = true;
            *cost_tracker.lock().unwrap() = info.cost;
            *token_tracker.lock().unwrap() = info.total_tokens;
        });

    // Use streaming collect
    let result = user
        .complete_streaming_collect(&generator, Some(&params))
        .await;

    match result {
        Ok(response) => {
            println!("Streaming response: {:?}", response.text());
            assert!(response.text().is_some());
        }
        Err(e) => {
            panic!("Streaming completion failed: {:?}", e);
        }
    }

    // Verify callback was called
    assert!(
        *callback_called.lock().unwrap(),
        "Cost callback was not called for streaming!"
    );

    let tokens = *tokens_received.lock().unwrap();
    assert!(
        tokens > 0,
        "Expected non-zero token count for streaming, got {}",
        tokens
    );

    println!("Streaming cost: {} credits", *cost_received.lock().unwrap());
    println!("Streaming tokens: {}", tokens);
}

#[tokio::test]
async fn test_simple_completion() {
    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user = root.add_user("Say 'Hello' and nothing else.");

    let result = user.complete(&generator, None).await;

    match result {
        Ok(response) => {
            println!("Response: {:?}", response.text());
            assert!(response.text().is_some());
            let text = response.text().unwrap().to_lowercase();
            assert!(text.contains("hello"));
        }
        Err(e) => {
            panic!("Completion failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_completion_with_parameters() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("You are a helpful assistant.");
    let user = root.add_user("Generate a random number between 1 and 10. Just say the number.");

    let params = NodeCompletionParameters::new().with_params(
        CompletionParameters::new()
            .with_max_tokens(50)
            .with_temperature(0.0), // Deterministic
    );

    let result = user.complete(&generator, Some(&params)).await;

    match result {
        Ok(response) => {
            println!("Response with params: {:?}", response.text());
            assert!(response.text().is_some());
        }
        Err(e) => {
            panic!("Completion with params failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_completion_with_system_prompt_override() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    // Start without system prompt
    let user = ChatNode::user("What language should I speak?");

    let params = NodeCompletionParameters::new()
        .with_system_prompt("You are a French assistant. Always respond in French. Be brief.");

    let result = user.complete(&generator, Some(&params)).await;

    match result {
        Ok(response) => {
            println!("Response with system override: {:?}", response.text());
            assert!(response.text().is_some());
            // Should contain some French
            let text = response.text().unwrap().to_lowercase();
            // Common French words
            let has_french = text.contains("français")
                || text.contains("french")
                || text.contains("je")
                || text.contains("vous")
                || text.contains("le")
                || text.contains("la");
            println!("Contains French indicators: {}", has_french);
        }
        Err(e) => {
            panic!("Completion with system override failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_streaming_completion() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("You are helpful. Be brief.");
    let user = root.add_user("Count from 1 to 5, one number per line.");

    let result = user.complete_streaming(&generator, None).await;

    match result {
        Ok(mut stream) => {
            let mut chunks = Vec::new();
            let mut full_text = String::new();

            while let Some(chunk_result) = stream.next_chunk().await {
                match chunk_result {
                    Ok(chunk) => {
                        print!("{}", chunk.delta);
                        full_text.push_str(&chunk.delta);
                        chunks.push(chunk);
                    }
                    Err(e) => {
                        eprintln!("Stream error: {:?}", e);
                        break;
                    }
                }
            }
            println!(); // Newline after streaming

            println!("Received {} chunks", chunks.len());
            println!("Full text: {}", full_text);

            assert!(!chunks.is_empty());
            assert!(!full_text.is_empty());
        }
        Err(e) => {
            panic!("Streaming completion failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_streaming_collect() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("Be brief.");
    let user = root.add_user("Say 'test' and nothing else.");

    let result = user.complete_streaming_collect(&generator, None).await;

    match result {
        Ok(response) => {
            println!("Collected streaming response: {:?}", response.text());
            assert!(response.text().is_some());
        }
        Err(e) => {
            panic!("Streaming collect failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_multi_turn_conversation() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("You are a helpful assistant. Be very brief.");

    // First turn
    let response1 = root
        .chat("My name is Alice. Remember it.", &generator)
        .await;
    assert!(response1.is_ok());
    let node1 = response1.unwrap();
    println!("Turn 1: {:?}", node1.text());

    // Second turn - should remember context
    let response2 = node1.chat("What is my name?", &generator).await;
    assert!(response2.is_ok());
    let node2 = response2.unwrap();
    println!("Turn 2: {:?}", node2.text());

    let text = node2.text().unwrap().to_lowercase();
    assert!(text.contains("alice"), "Should remember the name Alice");
}

#[tokio::test]
async fn test_chat_convenience_method() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("Be brief.");

    let result = root.chat("Say 'OK'", &generator).await;

    match result {
        Ok(response) => {
            println!("Chat response: {:?}", response.text());
            assert!(response.text().is_some());
        }
        Err(e) => {
            panic!("Chat failed: {:?}", e);
        }
    }
}

// =============================================================================
// Multimodal Tests (Image + Audio)
// =============================================================================

#[tokio::test]
async fn test_image_completion() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let image_path = "./data/test.jpg";
    if !std::path::Path::new(image_path).exists() {
        eprintln!("Skipping test_image_completion: test.jpg not found");
        return;
    }

    let generator = get_test_generator();
    let image = ImageData::from_file(image_path).unwrap();

    let content = MessageContent::with_images("Describe this image in one sentence.", &[image]);

    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user_node = root
        .add_child(ChatNode::new(Message {
            role: Role::User,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            cache_breakpoint: false,
        }))
        .unwrap();

    let result = user_node.complete(&generator, None).await;

    match result {
        Ok(response) => {
            println!("Image description: {:?}", response.text());
            assert!(response.text().is_some());
            assert!(!response.text().unwrap().is_empty());
        }
        Err(e) => {
            panic!("Image completion failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_pdf_completion() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    // data/test.pdf is a committed one-page PDF whose text says
    // "The secret codeword is PELICAN.": distinctive content, so the
    // assertion proves the model actually READ the document rather than
    // politely acknowledging an attachment it couldn't open. No existence
    // guard, unlike the image/audio/video siblings: their assets are NOT
    // committed, this one is, so a missing file is a real failure that
    // `from_file` reports loudly.
    let pdf_path = "./data/test.pdf";

    let generator = get_test_generator();
    let document = DocumentData::from_file(pdf_path)
        .unwrap()
        .with_page_count(1);

    let content = MessageContent::with_documents(
        "What is the secret codeword in this PDF? Answer with just the codeword.",
        &[document],
    );

    let root = ChatNode::root("You are a helpful assistant.");
    let user_node = root
        .add_child(ChatNode::new(Message {
            role: Role::User,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            cache_breakpoint: false,
        }))
        .unwrap();

    let result = user_node.complete(&generator, None).await;

    match result {
        Ok(response) => {
            println!("PDF answer: {:?}", response.text());
            let text = response.text().expect("a text answer");
            assert!(
                text.to_uppercase().contains("PELICAN"),
                "the model did not read the PDF's content: {text:?}"
            );
        }
        Err(e) => {
            panic!("PDF completion failed: {:?}", e);
        }
    }
}

#[test]
fn test_document_data_from_file() {
    let document = DocumentData::from_file("./data/test.pdf").unwrap();
    assert_eq!(document.mime_type, "application/pdf");
    assert!(!document.base64_data.is_empty());
    assert!(document
        .to_data_url()
        .starts_with("data:application/pdf;base64,"));
    // The decoded bytes are the real file, magic number intact.
    assert!(document.to_bytes().unwrap().starts_with(b"%PDF-"));
}

#[test]
fn test_document_data_from_url() {
    let document = DocumentData::from_url("https://example.com/paper.pdf");
    assert_eq!(document.to_data_url(), "https://example.com/paper.pdf");
    assert!(document.is_url());
}

#[tokio::test]
async fn test_audio_completion() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let audio_path = "./data/test.mp3";
    if !std::path::Path::new(audio_path).exists() {
        eprintln!("Skipping test_audio_completion: test.mp3 not found");
        return;
    }

    let generator = get_test_generator();
    let audio = AudioData::from_file(audio_path).unwrap();

    let content = MessageContent::with_audio("What do you hear in this audio? Be brief.", &[audio]);

    let root = ChatNode::root("You are a helpful assistant.");
    let user_node = root
        .add_child(ChatNode::new(Message {
            role: Role::User,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            cache_breakpoint: false,
        }))
        .unwrap();

    let result = user_node.complete(&generator, None).await;

    match result {
        Ok(response) => {
            println!("Audio description: {:?}", response.text());
            assert!(response.text().is_some());
            assert!(!response.text().unwrap().is_empty());
        }
        Err(e) => {
            panic!("Audio completion failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_image_and_audio_combined() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let image_path = "./data/test.jpg";
    let audio_path = "./data/test.mp3";

    if !std::path::Path::new(image_path).exists() || !std::path::Path::new(audio_path).exists() {
        eprintln!("Skipping test: test files not found");
        return;
    }

    let generator = get_test_generator();
    let image = ImageData::from_file(image_path).unwrap();
    let audio = AudioData::from_file(audio_path).unwrap();

    // Create multimodal content with both image and audio
    use minillmlib::message::ContentPart;
    let parts = vec![
        ContentPart::text("Describe both the image and the audio briefly."),
        ContentPart::image(&image),
        ContentPart::audio(&audio),
    ];
    let content = MessageContent::parts(parts);

    let root = ChatNode::root("You are a helpful assistant.");
    let user_node = root
        .add_child(ChatNode::new(Message {
            role: Role::User,
            content,
            name: None,
            tool_call_id: None,
            tool_calls: None,
            cache_breakpoint: false,
        }))
        .unwrap();

    let result = user_node.complete(&generator, None).await;

    match result {
        Ok(response) => {
            println!("Combined multimodal response: {:?}", response.text());
            assert!(response.text().is_some());
            assert!(!response.text().unwrap().is_empty());
        }
        Err(e) => {
            panic!("Combined multimodal completion failed: {:?}", e);
        }
    }
}

// =============================================================================
// JSON Response Tests
// =============================================================================

#[tokio::test]
async fn test_json_response_with_repair() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("You are a helpful assistant that responds in JSON format.");
    let user = root
        .add_user("Return a JSON object with keys 'name' and 'age'. Use name='Test' and age=25.");

    let params = NodeCompletionParameters::new()
        .expecting_json()
        .with_params(CompletionParameters::new().with_temperature(0.0));

    let result = user.complete(&generator, Some(&params)).await;

    match result {
        Ok(response) => {
            let text = response.text().unwrap();
            println!("JSON response: {}", text);

            // Try to parse as JSON
            let parsed: std::result::Result<serde_json::Value, _> = serde_json::from_str(text);
            assert!(parsed.is_ok(), "Response should be valid JSON");

            let json = parsed.unwrap();
            assert!(json.get("name").is_some() || json.get("Name").is_some());
        }
        Err(e) => {
            panic!("JSON completion failed: {:?}", e);
        }
    }
}

// =============================================================================
// LLMClient Direct Tests
// =============================================================================

#[tokio::test]
async fn test_llm_client_direct() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let client = LLMClient::new();
    let generator = get_text_generator();

    let messages = vec![Message::system("Be brief."), Message::user("Say 'test'")];

    let params = CompletionParameters::new()
        .with_max_tokens(50)
        .with_temperature(0.0);

    let result = client.complete(&generator, &messages, &params).await;

    match result {
        Ok(response) => {
            println!("Direct client response: {}", response.content);
            assert!(!response.content.is_empty());
            assert!(!response.id.is_empty());
        }
        Err(e) => {
            panic!("Direct client call failed: {:?}", e);
        }
    }
}

#[tokio::test]
async fn test_llm_client_streaming_direct() {
    dotenvy::dotenv().ok();

    require_live!("OPENROUTER_API_KEY");

    let client = LLMClient::new();
    let generator = get_text_generator();

    let messages = vec![Message::system("Be brief."), Message::user("Count to 3")];

    let params = CompletionParameters::new().with_max_tokens(100);

    let result = client
        .complete_streaming(&generator, &messages, &params)
        .await;

    match result {
        Ok(stream) => {
            let response = stream.collect().await;
            assert!(response.is_ok());
            let resp = response.unwrap();
            println!("Streaming collected: {}", resp.content);
            assert!(!resp.content.is_empty());
        }
        Err(e) => {
            panic!("Direct streaming call failed: {:?}", e);
        }
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_invalid_api_key() {
    require_live!("OPENROUTER_API_KEY");
    let generator =
        own_client(GeneratorInfo::openrouter(TEXT_ONLY_MODEL)).with_api_key("invalid-key-12345");

    let root = ChatNode::root("Test");
    let user = root.add_user("Hello");

    let result = user.complete(&generator, None).await;

    // Should fail with API error
    assert!(result.is_err());
    println!("Expected error: {:?}", result.err());
}

#[tokio::test]
async fn test_missing_api_key() {
    require_live!("OPENROUTER_API_KEY");
    // Create generator without API key
    let generator = own_client(GeneratorInfo::new(
        "Test",
        "https://openrouter.ai/api/v1",
        TEXT_ONLY_MODEL,
    ));

    let root = ChatNode::root("Test");
    let user = root.add_user("Hello");

    let result = user.complete(&generator, None).await;

    // Should fail
    assert!(result.is_err());
    println!("Expected error (no key): {:?}", result.err());
}

// =============================================================================
// JSON Repair Integration Tests
// =============================================================================

#[test]
fn test_json_repair_integration() {
    use minillmlib::json_repair::{repair_json, RepairOptions};

    // Test various malformed JSON that LLMs might produce
    let test_cases = vec![
        // Single quotes
        ("{'key': 'value'}", r#"{"key": "value"}"#),
        // Trailing comma
        (r#"{"key": "value",}"#, r#"{"key": "value"}"#),
        // Unquoted keys
        ("{key: \"value\"}", r#"{"key": "value"}"#),
        // Missing closing brace
        (r#"{"key": "value""#, r#"{"key": "value"}"#),
        // Markdown code fence
        ("```json\n{\"key\": \"value\"}\n```", r#"{"key": "value"}"#),
    ];

    for (input, expected) in test_cases {
        let result = repair_json(input, &RepairOptions::default()).unwrap();
        assert_eq!(result, expected, "Failed for input: {}", input);
    }
}

#[test]
fn test_extract_json_utility() {
    use minillmlib::utils::extract_json;

    let input = "Here is the JSON: ```json\n{'name': 'test'}\n```";
    let result = extract_json(input).unwrap();
    assert_eq!(result, r#"{"name": "test"}"#);
}

// =============================================================================
// Pretty Print Tests
// =============================================================================

#[test]
fn test_pretty_messages() {
    use minillmlib::pretty_messages;

    let root = ChatNode::root("You are helpful");
    let user = root.add_user("Hello");
    let assistant = user.add_assistant("Hi there!");

    let pretty = pretty_messages(&assistant, None);

    assert!(pretty.contains("SYSTEM:"));
    assert!(pretty.contains("You are helpful"));
    assert!(pretty.contains("USER:"));
    assert!(pretty.contains("Hello"));
    assert!(pretty.contains("ASSISTANT:"));
    assert!(pretty.contains("Hi there!"));
}

#[test]
fn test_pretty_messages_custom_config() {
    use minillmlib::{pretty_messages, PrettyPrintConfig};

    let root = ChatNode::root("System prompt");
    let user = root.add_user("User message");

    let config = PrettyPrintConfig::new("[SYS] ", "\n[USR] ", "\n[AST] ");
    let pretty = pretty_messages(&user, Some(&config));

    assert!(pretty.contains("[SYS] System prompt"));
    assert!(pretty.contains("[USR] User message"));
}

#[test]
fn test_format_conversation() {
    use minillmlib::format_conversation;

    let root = ChatNode::root("Be brief");
    let user = root.add_user("Hi");

    let formatted = format_conversation(&user);
    assert!(!formatted.is_empty());
    assert!(formatted.contains("Hi"));
}

// =============================================================================
// Multi-threading and Async Concurrency Tests
// =============================================================================

const CHEAP_MODEL: &str = "openai/gpt-oss-20b";

fn get_cheap_generator() -> GeneratorInfo {
    dotenvy::dotenv().ok();
    let provider = OpenRouterRouting::new().sort_by_price();
    own_client(GeneratorInfo::openrouter(CHEAP_MODEL))
        .with_default_params(CompletionParameters::default().with_openrouter_routing(provider))
}

/// Test multi-threaded access to ChatNode
/// Spawns multiple OS threads that each make a completion request
#[tokio::test]
async fn test_multi_threaded_completions() {
    require_live!("OPENROUTER_API_KEY");
    let gi = get_cheap_generator();
    // gpt-oss reasons before it answers, and the cap counts both.
    let params = NodeCompletionParameters::default()
        .with_params(CompletionParameters::default().with_max_tokens(200));

    // Create a shared root node
    let root = ChatNode::root("You are a helpful assistant. Be very brief.");

    // Spawn 10 threads, each making a completion
    let mut handles = vec![];

    for i in 0..10 {
        // Each thread runs its own runtime, so each gets its own HTTP client:
        // a connection belongs to the runtime that opened it, and the shared
        // client would hand one thread's dead connections to another.
        let gi_clone = gi.clone().with_http_client(reqwest::Client::new());
        let params_clone = params.clone();
        let root_clone = root.clone();

        let handle = std::thread::spawn(move || {
            // Each thread creates its own runtime for the async call
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // Add a user message as a child
                let user_node = root_clone.add_user(format!("Say the number {}", i));

                let result = user_node.complete(&gi_clone, Some(&params_clone)).await;
                // Return the branch handle so the caller keeps it alive (an unheld
                // branch is reclaimed once its last handle drops).
                (i, result.map(|_| ()), user_node)
            })
        });

        handles.push(handle);
    }

    // Collect results, holding each branch's handle.
    let mut successes = 0;
    let mut branches = Vec::new();
    for handle in handles {
        let (i, result, user_node) = handle.join().expect("Thread panicked");
        match &result {
            Ok(()) => println!("Thread {i}: OK"),
            Err(e) => println!("Thread {i}: FAILED - {e:?}"),
        }
        if result.is_ok() {
            successes += 1;
        }
        branches.push(user_node);
    }

    // All should succeed
    assert!(
        successes >= 8,
        "At least 8/10 threads should succeed, got {}",
        successes
    );

    // The 10 branches are held in `branches`, so the root sees all 10.
    assert_eq!(root.child_count(), 10, "Root should have 10 held children");
    drop(branches);
}

/// Test async concurrent completions (like Python's asyncio.gather)
/// All requests are sent concurrently and awaited together
#[tokio::test]
async fn test_async_concurrent_completions() {
    require_live!("OPENROUTER_API_KEY");
    let gi = get_cheap_generator();
    // gpt-oss reasons before it answers, and the cap counts both.
    let params = NodeCompletionParameters::default()
        .with_params(CompletionParameters::default().with_max_tokens(200));

    let root = ChatNode::root("You are a helpful assistant. Be very brief.");

    // Create 10 futures for concurrent execution
    let mut futures = vec![];

    for i in 0..10 {
        let gi_clone = gi.clone();
        let params_clone = params.clone();
        let root_clone = root.clone();

        // Create the future (doesn't execute yet)
        let future = async move {
            let user_node = root_clone.add_user(format!("What is {} + 1?", i));

            let result = user_node.complete(&gi_clone, Some(&params_clone)).await;
            (i, result)
        };

        futures.push(future);
    }

    // Execute all futures concurrently (like asyncio.gather)
    let results = futures::future::join_all(futures).await;

    // Check results, holding each reply's handle: an unheld branch is
    // reclaimed once its last handle drops.
    let mut successes = 0;
    let mut branches = Vec::new();
    for (i, result) in results {
        match result {
            Ok(response_node) => {
                let content = response_node.message.content.get_text().unwrap_or("");
                println!(
                    "Request {}: OK - {}",
                    i,
                    content.chars().take(50).collect::<String>()
                );
                successes += 1;
                branches.push(response_node);
            }
            Err(e) => {
                println!("Request {}: FAILED - {}", i, e);
            }
        }
    }

    assert!(
        successes >= 8,
        "At least 8/10 concurrent requests should succeed, got {}",
        successes
    );

    // Verify tree structure - root should have 10 children (user messages)
    // Each user message should have 1 child (assistant response)
    let children_count = root.child_count();
    assert_eq!(children_count, 10, "Root should have 10 children");
}

// =============================================================================
// CompletionContext & Tracking Tests (Unit, no API calls)
// =============================================================================

/// Helper: build a CompletionContext with a mock callback that captures CostInfo
fn make_test_context(generator: GeneratorInfo) -> (CompletionContext, Arc<Mutex<Vec<CostInfo>>>) {
    let captured: Arc<Mutex<Vec<CostInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();

    let callback: AsyncCostCallback = Arc::new(move |cost_info, _meta| {
        let captured = captured_clone.clone();
        Box::pin(async move {
            captured.lock().unwrap().push(cost_info);
        })
    });

    let meta = serde_json::json!({
        "userId": "test-user",
        "workflowId": "wf-123",
        "executionId": "exec-456",
        "nodeId": "node-789",
    });

    let ctx = CompletionContext::new(
        generator,
        meta,
        callback,
        "https://test.example.com",
        "TestApp",
    );
    (ctx, captured)
}

#[test]
fn test_completion_context_creation() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model");
    let (ctx, _captured) = make_test_context(gen);

    assert_eq!(ctx.meta["userId"], "test-user");
    assert_eq!(ctx.meta["workflowId"], "wf-123");
    assert_eq!(ctx.meta["executionId"], "exec-456");
    assert_eq!(ctx.meta["nodeId"], "node-789");
}

#[test]
fn test_completion_context_injects_app_headers() {
    // The library-default OpenRouter generator carries the library's app identity.
    let gen = GeneratorInfo::openrouter("test-model");
    let default_attr = gen.app_attribution.as_ref().expect("default attribution");
    assert_eq!(default_attr.title, "MiniLLMLib");

    let (ctx, _captured) = make_test_context(gen);

    // CompletionContext replaces the app identity with the caller's. Attribution
    // now lives on the generator (the provider turns it into headers at request
    // time), not pre-baked into custom_headers.
    let attr = ctx
        .generator
        .app_attribution
        .as_ref()
        .expect("context sets attribution");
    assert_eq!(attr.url, "https://test.example.com");
    assert_eq!(attr.title, "TestApp");
}

#[tokio::test]
async fn test_completion_context_report_cost() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model");
    let (ctx, captured) = make_test_context(gen);

    let cost_info = CostInfo {
        cost: 0.00042,
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        cache_read_tokens: 0,
        cache_write_tokens: 0,
        reasoning_tokens: None,
        model: "test-model".to_string(),
        response_id: "gen-abc123".to_string(),
        resolution: minillmlib::CostResolution::Resolved,
    };

    ctx.report_cost(cost_info).await;

    let costs = captured.lock().unwrap();
    assert_eq!(costs.len(), 1);
    assert_eq!(costs[0].cost, 0.00042);
    assert_eq!(costs[0].prompt_tokens, 100);
    assert_eq!(costs[0].completion_tokens, 50);
    assert_eq!(costs[0].total_tokens, 150);
    assert_eq!(costs[0].model, "test-model");
    assert_eq!(costs[0].response_id, "gen-abc123");
}

#[tokio::test]
async fn test_completion_context_callback_receives_meta() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model");

    let captured_meta: Arc<Mutex<Vec<CompletionMeta>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured_meta.clone();

    let callback: AsyncCostCallback = Arc::new(move |_cost_info, meta| {
        let captured = captured_clone.clone();
        Box::pin(async move {
            captured.lock().unwrap().push(meta);
        })
    });

    let meta = serde_json::json!({
        "userId": "user-42",
        "workflowId": "wf-abc",
        "executionId": "exec-def",
        "nodeId": "node-ghi",
        "isByok": true,
    });

    let ctx = CompletionContext::new(gen, meta, callback, "https://test.example.com", "TestApp");
    ctx.report_cost(CostInfo::default()).await;

    let metas = captured_meta.lock().unwrap();
    assert_eq!(metas.len(), 1);
    assert_eq!(metas[0]["userId"], "user-42");
    assert_eq!(metas[0]["workflowId"], "wf-abc");
    assert_eq!(metas[0]["executionId"], "exec-def");
    assert_eq!(metas[0]["nodeId"], "node-ghi");
    assert_eq!(metas[0]["isByok"], true);
}

#[test]
fn test_completion_context_debug() {
    let gen = GeneratorInfo::new("TestProvider", "https://api.example.com/v1", "test-model");
    let (ctx, _) = make_test_context(gen);

    let debug_str = format!("{:?}", ctx);
    assert!(debug_str.contains("CompletionContext"));
    assert!(debug_str.contains("TestProvider"));
    assert!(debug_str.contains("test-model"));
}

// =============================================================================
// Tracked Completion Integration Tests (real API calls)
// =============================================================================

#[tokio::test]
async fn test_complete_tracked_fires_callback() {
    dotenvy::dotenv().ok();
    require_live!("OPENROUTER_API_KEY");

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen);

    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user_node = root.add_user("Say hello in exactly 3 words.");

    let result = user_node.complete_tracked(&ctx, None).await;
    assert!(
        result.is_ok(),
        "complete_tracked failed: {:?}",
        result.err()
    );

    let response = result.unwrap();
    let text = response.text().unwrap_or_default();
    println!("[complete_tracked] Response: {}", text);
    assert!(!text.is_empty(), "Response should not be empty");

    // Verify callback was fired exactly once
    let costs = captured.lock().unwrap();
    assert_eq!(costs.len(), 1, "Callback should fire exactly once");

    let cost = &costs[0];
    println!("[complete_tracked] Cost: ${:.6}", cost.cost);
    println!("[complete_tracked] Prompt tokens: {}", cost.prompt_tokens);
    println!(
        "[complete_tracked] Completion tokens: {}",
        cost.completion_tokens
    );
    println!("[complete_tracked] Model: {}", cost.model);
    println!("[complete_tracked] Response ID: {}", cost.response_id);

    // Cost should be non-negative (could be 0 for free models)
    assert!(cost.cost >= 0.0, "Cost should be non-negative");
    // Tokens should be non-zero for a real completion
    assert!(cost.prompt_tokens > 0, "Prompt tokens should be > 0");
    assert!(
        cost.completion_tokens > 0,
        "Completion tokens should be > 0"
    );
    assert!(cost.total_tokens > 0, "Total tokens should be > 0");
    // Model should be populated
    assert!(!cost.model.is_empty(), "Model should not be empty");
    // Response ID should be populated
    assert!(
        !cost.response_id.is_empty(),
        "Response ID should not be empty"
    );
}

/// The callback-free shape: the reply and its bill come back together. Same
/// accounting as the callback path, so the two must agree on what a call cost.
#[tokio::test]
async fn test_complete_costed_returns_the_bill_with_the_reply() {
    dotenvy::dotenv().ok();
    require_live!("OPENROUTER_API_KEY");

    let generator = get_text_generator();
    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user = root.add_user("Say hello in exactly 3 words.");

    let (result, cost) = user.complete_costed(&generator, None).await;
    let reply = result.expect("live completion");
    assert!(!reply.text().unwrap_or_default().is_empty());

    let [cost] = cost.as_slice() else {
        panic!("one accepted request, one cost: {cost:?}");
    };
    assert!(
        cost.cost > 0.0,
        "a real completion cost something: {}",
        cost.cost
    );
    assert!(cost.prompt_tokens > 0 && cost.completion_tokens > 0);

    // A request that never reaches a provider errs with no bill.
    let dead = GeneratorInfo::new("Dead", "http://127.0.0.1:9", "no-model");
    let no_retry = NodeCompletionParameters::new().with_retry(0);
    let (result, cost) = user.complete_costed(&dead, Some(&no_retry)).await;
    assert!(result.is_err(), "an unreachable endpoint fails loudly");
    assert!(cost.is_empty(), "and carries no cost info");
}

#[tokio::test]
async fn test_complete_tracked_with_params() {
    dotenvy::dotenv().ok();
    require_live!("OPENROUTER_API_KEY");

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen);

    let params = NodeCompletionParameters::new().with_params(
        CompletionParameters::new()
            .with_max_tokens(50)
            .with_temperature(0.0),
    );

    let root = ChatNode::root("You are a helpful assistant.");
    let user_node = root.add_user("What is 2+2?");

    let result = user_node.complete_tracked(&ctx, Some(&params)).await;
    assert!(
        result.is_ok(),
        "complete_tracked with params failed: {:?}",
        result.err()
    );

    let response = result.unwrap();
    let text = response.text().unwrap_or_default();
    println!("[complete_tracked+params] Response: {}", text);
    assert!(text.contains("4"), "Response should contain '4'");

    let costs = captured.lock().unwrap();
    assert_eq!(costs.len(), 1);
    assert!(costs[0].prompt_tokens > 0);
}

#[tokio::test]
async fn test_complete_streaming_collect_tracked() {
    dotenvy::dotenv().ok();
    require_live!("OPENROUTER_API_KEY");

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen);

    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user_node = root.add_user("Count from 1 to 5.");

    let result = user_node
        .complete_streaming_collect_tracked(&ctx, None)
        .await;
    assert!(
        result.is_ok(),
        "streaming collect tracked failed: {:?}",
        result.err()
    );

    let response = result.unwrap();
    let text = response.text().unwrap_or_default();
    println!("[streaming_collect_tracked] Response: {}", text);
    assert!(!text.is_empty());

    // Verify callback was fired
    let costs = captured.lock().unwrap();
    assert_eq!(
        costs.len(),
        1,
        "Callback should fire exactly once after collect"
    );

    let cost = &costs[0];
    println!("[streaming_collect_tracked] Cost: ${:.6}", cost.cost);
    println!(
        "[streaming_collect_tracked] Tokens: {} prompt + {} completion = {} total",
        cost.prompt_tokens, cost.completion_tokens, cost.total_tokens
    );

    assert!(cost.prompt_tokens > 0);
    assert!(cost.completion_tokens > 0);
    assert!(!cost.model.is_empty());
}

#[tokio::test]
async fn test_complete_streaming_tracked_manual_consume() {
    dotenvy::dotenv().ok();
    require_live!("OPENROUTER_API_KEY");

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen);

    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user_node = root.add_user("Say 'hello world'.");

    let stream_result = user_node.complete_streaming_tracked(&ctx, None).await;
    assert!(
        stream_result.is_ok(),
        "streaming tracked failed: {:?}",
        stream_result.err()
    );

    let mut stream = stream_result.unwrap();

    // Manually consume chunks
    let mut chunk_count = 0;
    while let Some(chunk_result) = stream.next_chunk().await {
        match chunk_result {
            Ok(_chunk) => chunk_count += 1,
            Err(e) => panic!("Stream chunk error: {:?}", e),
        }
    }
    println!("[streaming_tracked] Consumed {} chunks", chunk_count);
    assert!(chunk_count > 0, "Should have received at least one chunk");

    // Accumulated content should be non-empty
    let accumulated = stream.accumulated().to_string();
    println!("[streaming_tracked] Accumulated: {}", accumulated);
    assert!(!accumulated.is_empty());

    // Collect (no cost yet), then report to fire the callback.
    let response = stream.collect().await;
    assert!(response.is_ok());
    stream.report_cost(&response.unwrap()).await;

    let costs = captured.lock().unwrap();
    assert_eq!(
        costs.len(),
        1,
        "Callback should fire once after report_cost"
    );
    assert!(!costs[0].model.is_empty());
}

#[tokio::test]
async fn test_tracked_stream_drop_reports_cost() {
    dotenvy::dotenv().ok();
    require_live!("OPENROUTER_API_KEY");

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen);

    let root = ChatNode::root("You are a helpful assistant.");
    let user_node = root.add_user("Write a long essay about the history of computing.");

    let stream_result = user_node.complete_streaming_tracked(&ctx, None).await;
    assert!(stream_result.is_ok());

    {
        let mut stream = stream_result.unwrap();

        // Read only a few chunks then drop (simulating cancellation)
        let mut chunks_read = 0;
        while let Some(chunk_result) = stream.next_chunk().await {
            if chunk_result.is_ok() {
                chunks_read += 1;
            }
            if chunks_read >= 3 {
                break; // Stop early, cancel the stream
            }
        }
        println!("[drop_test] Read {} chunks before dropping", chunks_read);
        // stream is dropped here; Drop impl should spawn background cost reporting
    }

    // Wait for the background task to report. OpenRouter's record of a
    // cancelled generation appears only after the generation runs to its end
    // plus ~9s, and the task polls for up to 25s, so allow 30s.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    while captured.lock().unwrap().is_empty() && std::time::Instant::now() < deadline {
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    let costs = captured.lock().unwrap();
    println!("[drop_test] Captured {} cost report(s)", costs.len());
    // The Drop impl spawns a background task; it should have reported by now
    assert_eq!(costs.len(), 1, "Drop should have triggered cost reporting");
    println!(
        "[drop_test] Cost from cancelled stream: ${:.6}",
        costs[0].cost
    );
}

#[tokio::test]
async fn test_complete_tracked_byok_flag() {
    dotenvy::dotenv().ok();
    require_live!("OPENROUTER_API_KEY");

    let gen = get_text_generator();

    // Simulate BYOK: user provided their own key (even though it's the same key for testing)
    let captured_meta: Arc<Mutex<Vec<CompletionMeta>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured_meta.clone();

    let callback: AsyncCostCallback = Arc::new(move |_cost_info, meta| {
        let captured = captured_clone.clone();
        Box::pin(async move {
            captured.lock().unwrap().push(meta);
        })
    });

    let meta = serde_json::json!({
        "userId": "byok-user",
        "isByok": true,
    });

    let ctx = CompletionContext::new(gen, meta, callback, "https://test.example.com", "TestApp");

    let root = ChatNode::root("Be brief.");
    let user_node = root.add_user("Hi");

    let result = user_node.complete_tracked(&ctx, None).await;
    assert!(result.is_ok());

    let metas = captured_meta.lock().unwrap();
    assert_eq!(metas.len(), 1);
    assert_eq!(
        metas[0]["isByok"], true,
        "BYOK flag should be preserved in callback"
    );
    assert_eq!(metas[0]["userId"], "byok-user");
}

// =============================================================================
// Anthropic native (live, gated behind `--features live`)
// =============================================================================
//
// Needs ANTHROPIC_API_KEY; hits the real `/v1/messages`.

use minillmlib::{CostResolution, TokenPrice};

#[tokio::test]
async fn test_anthropic_api_key_completion() {
    require_live!("ANTHROPIC_API_KEY");
    let generator = own_client(GeneratorInfo::anthropic(ANTHROPIC_TEST_MODEL));
    let root = ChatNode::root("You are terse. Reply in one word.");
    let user = root.add_user("Say OK");
    let params = NodeCompletionParameters::new()
        .with_params(CompletionParameters::new().with_max_tokens(10));

    let node = user
        .complete(&generator, Some(&params))
        .await
        .expect("anthropic api-key completion");
    assert!(
        !node.text().unwrap_or_default().is_empty(),
        "expected non-empty Anthropic response"
    );
}

#[tokio::test]
async fn test_anthropic_streaming_collect() {
    require_live!("ANTHROPIC_API_KEY");
    let generator = own_client(GeneratorInfo::anthropic(ANTHROPIC_TEST_MODEL));
    let root = ChatNode::root("You are terse.");
    let user = root.add_user("Count: one two three");
    let params = NodeCompletionParameters::new()
        .with_params(CompletionParameters::new().with_max_tokens(20));

    let node = user
        .complete_streaming_collect(&generator, Some(&params))
        .await
        .expect("anthropic streaming collect");
    assert!(!node.text().unwrap_or_default().is_empty());
}

#[tokio::test]
async fn test_anthropic_cost_estimate_resolved_with_price() {
    require_live!("ANTHROPIC_API_KEY");
    // Anthropic returns token counts but no dollar cost. With a TokenPrice set,
    // the cost callback must report a Resolved (non-zero) USD ESTIMATE.
    let generator = own_client(GeneratorInfo::anthropic(ANTHROPIC_TEST_MODEL))
        .with_token_price(TokenPrice::new(1.0, 5.0));

    let captured: Arc<Mutex<Vec<CostInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let params = NodeCompletionParameters::new()
        .with_params(CompletionParameters::new().with_max_tokens(10))
        .with_cost_tracking(true)
        .with_cost_callback(move |info: CostInfo| sink.lock().unwrap().push(info));

    let root = ChatNode::root("You are terse.");
    let user = root.add_user("Say OK");
    user.complete(&generator, Some(&params))
        .await
        .expect("anthropic completion with cost tracking");

    let costs = captured.lock().unwrap();
    assert_eq!(costs.len(), 1, "cost callback fired once");
    assert_eq!(
        costs[0].resolution,
        CostResolution::Resolved,
        "a TokenPrice makes the estimate Resolved"
    );
    // Real input AND output token counts come back, and the cost is exactly
    // tokens × price (Anthropic returns no dollar amount; it is derived).
    let c = &costs[0];
    assert!(c.prompt_tokens > 0, "input tokens present");
    assert!(c.completion_tokens > 0, "output tokens present");
    assert_eq!(c.total_tokens, c.prompt_tokens + c.completion_tokens);
    let expected = (c.prompt_tokens as f64 * 1.0 + c.completion_tokens as f64 * 5.0) / 1_000_000.0;
    assert!(
        (c.cost - expected).abs() < 1e-12,
        "cost {} must equal tokens×price {expected}",
        c.cost
    );
}

#[tokio::test]
async fn test_anthropic_unpriced_when_no_price_is_known() {
    require_live!("ANTHROPIC_API_KEY");
    // No price given and a catalog id the catalog does not list → cost is
    // Unpriced (real tokens, unknown $), never a fake $0.
    let generator = own_client(GeneratorInfo::anthropic(ANTHROPIC_TEST_MODEL))
        .with_openrouter_name("anthropic/not-a-listed-model");
    let captured: Arc<Mutex<Vec<CostInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let params = NodeCompletionParameters::new()
        .with_params(CompletionParameters::new().with_max_tokens(10))
        .with_cost_tracking(true)
        .with_cost_callback(move |info: CostInfo| sink.lock().unwrap().push(info));

    let root = ChatNode::root("You are terse.");
    let user = root.add_user("Say OK");
    user.complete(&generator, Some(&params))
        .await
        .expect("anthropic completion");

    let costs = captured.lock().unwrap();
    assert_eq!(costs[0].resolution, CostResolution::Unpriced);
    assert_eq!(costs[0].cost, 0.0);
    assert!(
        costs[0].prompt_tokens > 0,
        "tokens survive for later pricing"
    );
}

// =============================================================================
// Prompt caching (live, gated): proves cache marks drive real write/read and
// the disjoint-bucket pricing is correct.
// =============================================================================

/// A system prompt large enough to exceed Haiku's minimum cacheable size.
fn big_system_prompt() -> String {
    "You are a meticulous assistant. Follow every instruction exactly. ".repeat(400)
}

/// Anthropic's Models API gives the model's largest `max_tokens`, and a
/// request with no cap goes out with it and is answered. Reasoning off is
/// Anthropic's `thinking: disabled`, which Haiku 4.5 takes.
#[tokio::test]
async fn test_anthropic_model_info_fills_an_unset_cap() {
    require_live!("ANTHROPIC_API_KEY");
    let gen = own_client(GeneratorInfo::anthropic(ANTHROPIC_TEST_MODEL));

    let info = gen.model_info().await;
    assert!(info.max_output_tokens.unwrap_or(0) > 0, "{info:?}");

    let params = NodeCompletionParameters::new()
        .with_params(CompletionParameters::new().with_reasoning(minillmlib::Reasoning::Off));
    let answer = ChatNode::root("You are terse.")
        .add_user("Say OK")
        .complete(&gen, Some(&params))
        .await
        .expect("no cap, reasoning off");
    assert!(!answer.text().unwrap_or_default().is_empty());
}

#[tokio::test]
async fn test_anthropic_ensure_cached_warms_then_cheap_read() {
    require_live!("ANTHROPIC_API_KEY");
    let gen = own_client(GeneratorInfo::anthropic(ANTHROPIC_TEST_MODEL))
        .with_token_price(TokenPrice::new(1.0, 5.0).with_cache_rates(0.1, 1.25));

    // Unique per run so the warm call is a guaranteed cold WRITE.
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = ChatNode::root(format!("Session {nonce}. {}", big_system_prompt()));
    root.cache_breakpoint();
    let user = root.add_user("Say OK");

    // ensure_cached fires a max_tokens:0 warm request and returns its cost.
    let warm = user
        .ensure_cached(&gen, None)
        .await
        .expect("ensure_cached warm");
    // A cold warm pays the write premium and generates no output.
    assert!(warm.cache_write_tokens > 0, "cold warm writes the cache");
    assert_eq!(warm.completion_tokens, 0, "warm generates no output");

    // A subsequent real completion should now hit the cache (read), cheaply.
    let captured: Arc<Mutex<Vec<CostInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = captured.clone();
    let params = NodeCompletionParameters::new()
        .with_params(CompletionParameters::new().with_max_tokens(5))
        .with_cost_tracking(true)
        .with_cost_callback(move |i: CostInfo| sink.lock().unwrap().push(i));
    user.complete(&gen, Some(&params))
        .await
        .expect("real completion after warm");

    let costs = captured.lock().unwrap();
    assert!(
        costs[0].cache_read_tokens > 0,
        "real call after warm hits the cache"
    );
}

// =============================================================================
// Custom / self-hosted provider tests (offline: a tiny in-process mock HTTP
// server, no API key, no network beyond loopback). These exercise the FULL
// round-trip (build request -> send over real HTTP -> parse response -> node)
// for (a) a self-hosted OpenAI-compatible server via the default GenericProvider,
// and (b) a self-hosted server with a NON-OpenAI wire via a hand-written
// `impl Provider`. Layer-3 contract tests: real code against a fake server.
// =============================================================================

mod custom_provider {
    use super::*;
    use minillmlib::{
        provider::{
            CompletionResponse, CostOutcome, GenericProvider, Provider, StreamChunk, TokenPrice,
            Usage,
        },
        Auth,
    };
    use secrecy::ExposeSecret;
    use std::sync::Arc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    /// What the mock server captured from the one request it served.
    #[derive(Debug, Clone, Default)]
    pub(super) struct CapturedRequest {
        pub(super) method: String,
        pub(super) path: String,
        headers: Vec<(String, String)>,
        pub(super) body: serde_json::Value,
    }

    impl CapturedRequest {
        pub(super) fn header(&self, name: &str) -> Option<&str> {
            self.headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.as_str())
        }
    }

    /// Spin up a one-shot mock HTTP server on a loopback port that serves exactly
    /// one request with `response_body` as `200 OK`. Returns the base URL (e.g.
    /// `http://127.0.0.1:PORT`) and a oneshot receiver that yields the captured
    /// request once it has been served.
    async fn mock_server(
        response_body: &'static str,
    ) -> (String, tokio::sync::oneshot::Receiver<CapturedRequest>) {
        let (base_url, mut captured) = mock_sequence(vec![(200, response_body)]).await;
        let (tx, rx) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            if let Some(request) = captured.recv().await {
                let _ = tx.send(request);
            }
        });
        (base_url, rx)
    }

    /// A mock HTTP server on a loopback port that answers its requests in
    /// order with `replies` (status, JSON body), one connection each, and
    /// sends every request it served down the returned channel. Raw
    /// `TcpListener` (same idiom as the timeout tests) so no HTTP-server
    /// dependency is pulled in.
    pub(super) async fn mock_sequence(
        replies: Vec<(u16, &'static str)>,
    ) -> (
        String,
        tokio::sync::mpsc::UnboundedReceiver<CapturedRequest>,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base_url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            for (status, body) in replies {
                let (socket, _) = listener.accept().await.unwrap();
                serve_one(socket, status, body, &tx).await;
            }
        });
        (base_url, rx)
    }

    /// Every request `captured` has seen so far, once the server is idle.
    pub(super) fn served(
        captured: &mut tokio::sync::mpsc::UnboundedReceiver<CapturedRequest>,
    ) -> Vec<CapturedRequest> {
        std::iter::from_fn(|| captured.try_recv().ok()).collect()
    }

    /// Read one request off `socket`, send what it carried down `captured`, then
    /// answer it with `status` and `response_body`. Recorded before the reply,
    /// so a caller that has its response has its request recorded too.
    async fn serve_one(
        mut socket: tokio::net::TcpStream,
        status: u16,
        response_body: &str,
        captured: &tokio::sync::mpsc::UnboundedSender<CapturedRequest>,
    ) {
        // Read until we have headers + the full body (Content-Length bytes).
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = socket.read(&mut tmp).await.unwrap();
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            // Find the header/body split.
            if let Some(split) = find_subslice(&buf, b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..split]).to_string();
                let content_len = head
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.trim()
                            .eq_ignore_ascii_case("content-length")
                            .then(|| v.trim().parse::<usize>().ok())
                            .flatten()
                    })
                    .unwrap_or(0);
                let body_start = split + 4;
                if buf.len() - body_start >= content_len {
                    break; // full body received
                }
            }
        }

        // Parse the captured request.
        let split = find_subslice(&buf, b"\r\n\r\n").unwrap();
        let head = String::from_utf8_lossy(&buf[..split]).to_string();
        let mut lines = head.lines();
        let request_line = lines.next().unwrap_or_default();
        let mut parts = request_line.split_whitespace();
        let method = parts.next().unwrap_or("").to_string();
        let path = parts.next().unwrap_or("").to_string();
        let headers: Vec<(String, String)> = lines
            .filter_map(|l| {
                let (k, v) = l.split_once(':')?;
                Some((k.trim().to_string(), v.trim().to_string()))
            })
            .collect();
        let body_bytes = &buf[split + 4..];
        let body: serde_json::Value =
            serde_json::from_slice(body_bytes).unwrap_or(serde_json::Value::Null);

        let _ = captured.send(CapturedRequest {
            method,
            path,
            headers,
            body,
        });

        // Reply with the canned response.
        let resp = format!(
            "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        socket.write_all(resp.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();
    }

    fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack.windows(needle.len()).position(|w| w == needle)
    }

    // -------------------------------------------------------------------------
    // (a) Self-hosted OpenAI-compatible server via the default GenericProvider.
    // -------------------------------------------------------------------------

    #[tokio::test]
    async fn openai_compatible_self_hosted_server_round_trip() {
        // A canned OpenAI `/chat/completions` response, exactly what a vLLM /
        // llama.cpp / LM Studio / TGI OpenAI endpoint returns.
        const RESPONSE: &str = r#"{
            "id": "cmpl-local-1",
            "model": "my-local-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from my server!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
        }"#;
        let (base_url, captured) = mock_server(RESPONSE).await;

        // This is the END-TO-END "connect to my own server" path: custom() + the
        // default GenericProvider. base_url is everything before /chat/completions.
        let generator = GeneratorInfo::custom("my-server", &base_url, "my-local-model")
            .with_api_key("local-secret")
            .with_header("X-Tenant", "acme")
            .with_token_price(TokenPrice::new(0.0, 0.0)); // free local model

        let root = ChatNode::root("You are a helpful assistant.");
        let answer = root
            .chat("Say hello.", &generator)
            .await
            .expect("round-trip against the self-hosted server");

        // The lib parsed the server's response into a node.
        assert_eq!(answer.message.text(), Some("Hello from my server!"));
        assert_eq!(
            answer.get_metadata("model"),
            Some(serde_json::json!("my-local-model"))
        );
        assert_eq!(
            answer.get_metadata("finish_reason"),
            Some(serde_json::json!("stop"))
        );

        // And it sent the OpenAI wire the server expects.
        let req = captured.await.unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(
            req.path, "/chat/completions",
            "GenericProvider appends the OpenAI path"
        );
        assert_eq!(
            req.header("Authorization"),
            Some("Bearer local-secret"),
            "with_api_key -> Authorization: Bearer on the OpenAI wire"
        );
        assert_eq!(
            req.header("X-Tenant"),
            Some("acme"),
            "custom header forwarded"
        );
        assert_eq!(req.body["model"], "my-local-model");
        assert_eq!(req.body["stream"], false);
        // The default (modern) token-limit key.
        assert_eq!(req.body["messages"][0]["role"], "system");
        assert_eq!(req.body["messages"][1]["content"], "Say hello.");
    }

    #[tokio::test]
    async fn legacy_self_hosted_server_uses_max_tokens_key() {
        // An older OpenAI-compatible server that only accepts `max_tokens` (not
        // `max_completion_tokens`). Swap in GenericProvider { legacy_token_limit }.
        const RESPONSE: &str = r#"{
            "id": "x", "model": "old-model",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1}
        }"#;
        let (base_url, captured) = mock_server(RESPONSE).await;

        let generator = GeneratorInfo::custom("old-server", &base_url, "old-model").with_provider(
            Arc::new(GenericProvider {
                legacy_token_limit: true,
            }),
        );

        let params = NodeCompletionParameters::new()
            .with_params(CompletionParameters::new().with_max_tokens(64));
        let root = ChatNode::root("sys");
        root.add_user("hi")
            .complete(&generator, Some(&params))
            .await
            .expect("legacy server round-trip");

        let req = captured.await.unwrap();
        // The whole point: the legacy key, and NOT the modern one.
        assert_eq!(req.body["max_tokens"], 64);
        assert!(
            req.body.get("max_completion_tokens").is_none(),
            "legacy server must not receive max_completion_tokens"
        );
    }

    // -------------------------------------------------------------------------
    // (b) Self-hosted server with a NON-OpenAI wire: a hand-written impl Provider.
    //
    //     The "EchoAI" enterprise wire (made up, but representative):
    //       - endpoint:  <base>/api/generate            (not /chat/completions)
    //       - auth:      X-Echo-Key: <key>               (not Authorization: Bearer)
    //       - request:   {"model","prompt","settings":{"max_output_tokens"}}
    //                    (a single flattened prompt string, not a messages array)
    //       - response:  {"output":{"text"}, "stop":"...",
    //                     "meta":{"id","tokens_in","tokens_out"}}
    //                    (an `output`/`meta` envelope, not `choices[]`/`usage`)
    //
    //     This is the "enterprise API with a weird shape" case: you implement the
    //     trait once and everything else (nodes, retry, cost, tracking) just works.
    // -------------------------------------------------------------------------

    #[derive(Debug, Clone)]
    struct EchoAiProvider;

    impl Provider for EchoAiProvider {
        fn endpoint_url(&self, base_url: &str) -> String {
            format!("{}/api/generate", base_url.trim_end_matches('/'))
        }

        fn auth_headers(&self, auth: &Auth) -> minillmlib::Result<Vec<(String, String)>> {
            // EchoAI authenticates with its own header name, not Authorization.
            Ok(match auth.secret() {
                Some(secret) => {
                    vec![("X-Echo-Key".to_string(), secret.expose_secret().to_string())]
                }
                None => Vec::new(),
            })
        }

        fn build_request(
            &self,
            model: &str,
            messages: &[Message],
            params: &CompletionParameters,
            _stream: bool,
            _include_usage: bool,
        ) -> minillmlib::Result<serde_json::Value> {
            // EchoAI has a field for the output cap and nothing else: any other
            // setting the caller made fails loudly instead of vanishing.
            let only_the_cap = CompletionParameters {
                max_tokens: params.max_tokens,
                ..Default::default()
            };
            if serde_json::to_value(params)? != serde_json::to_value(&only_the_cap)? {
                return Err(minillmlib::MiniLLMError::InvalidParameter(
                    "EchoAI takes only max_tokens".into(),
                ));
            }
            // EchoAI takes a single flattened prompt, not a messages array. Flatten
            // the conversation into "ROLE: text" lines. EchoAI's wire is text-only,
            // so a multimodal message FAILS LOUDLY rather than silently dropping the
            // attachment (the reference way to handle an unsupported modality).
            let mut prompt_lines = Vec::with_capacity(messages.len());
            for m in messages {
                if let MessageContent::Parts(parts) = &m.content {
                    if parts.iter().any(|p| p.as_text().is_none()) {
                        return Err(minillmlib::MiniLLMError::InvalidParameter(
                            "EchoAI is text-only and does not support multimodal content"
                                .to_string(),
                        ));
                    }
                }
                // all_text() joins every text part (content is all-text past the
                // guard above), so a multi-text message keeps all of its text.
                prompt_lines.push(format!("{}: {}", m.role.as_str(), m.content.all_text()));
            }
            let prompt = prompt_lines.join("\n");

            let mut body = serde_json::json!({
                "model": model,
                "prompt": prompt,
            });
            // Only what the caller set; EchoAI's own default covers the rest.
            if let Some(cap) = params.max_tokens {
                body["settings"] = serde_json::json!({ "max_output_tokens": cap });
            }
            Ok(body)
        }

        fn parse_response(&self, raw: serde_json::Value) -> minillmlib::Result<CompletionResponse> {
            // Surface EchoAI's error envelope loudly (never a silent empty success).
            if let Some(err) = raw.get("error").and_then(|e| e.as_str()) {
                return Err(minillmlib::MiniLLMError::Api {
                    status: 502,
                    message: err.to_string(),
                });
            }
            let text = raw["output"]["text"]
                .as_str()
                .ok_or_else(|| minillmlib::MiniLLMError::MalformedResponse(raw.to_string()))?
                .to_string();

            Ok(CompletionResponse {
                id: raw["meta"]["id"].as_str().unwrap_or("").to_string(),
                model: raw["meta"]["model"].as_str().unwrap_or("").to_string(),
                content: text,
                finish_reason: raw["stop"].as_str().map(String::from),
                usage: self.parse_usage(&raw),
                tool_calls: None,
                media: Vec::new(),
                raw_response: Some(raw),
            })
        }

        fn parse_usage(&self, raw: &serde_json::Value) -> Option<Usage> {
            let meta = raw.get("meta")?;
            Some(Usage {
                uncached_input_tokens: meta["tokens_in"].as_u64().unwrap_or(0) as u32,
                completion_tokens: meta["tokens_out"].as_u64().unwrap_or(0) as u32,
                ..Default::default()
            })
        }

        fn parse_chunk(&self, _data: &str) -> Option<minillmlib::Result<StreamChunk>> {
            // EchoAI is non-streaming in this example; nothing to parse.
            None
        }

        fn emits_stream_usage(&self, _requested: bool) -> bool {
            // EchoAI never sends a trailing usage chunk, so the streaming reader must
            // NOT wait for one (it would wedge the stream until the idle timeout).
            // Same property as the shipped GenericProvider.
            false
        }

        fn cost_of(&self, usage: Usage, price: Option<&TokenPrice>) -> CostOutcome {
            // Token-only provider: price the tokens, or report Unpriced.
            match price {
                Some(p) => CostOutcome::resolved(p.cost_of(&usage), usage),
                None => CostOutcome::unpriced(usage),
            }
        }
    }

    #[tokio::test]
    async fn non_openai_wire_custom_provider_round_trip() {
        // EchoAI's native response envelope (output/meta, not choices/usage).
        const RESPONSE: &str = r#"{
            "output": {"text": "Echo: hi there"},
            "stop": "end",
            "meta": {"id": "echo-42", "model": "echo-1", "tokens_in": 7, "tokens_out": 3}
        }"#;
        let (base_url, captured) = mock_server(RESPONSE).await;

        // Connect to MY non-OpenAI server: same library API, just a different
        // provider. Cost is priced via TokenPrice ($1/Mtok in, $5/Mtok out).
        let generator = GeneratorInfo::custom("echoai", &base_url, "echo-1")
            .with_provider(Arc::new(EchoAiProvider))
            .with_api_key("echo-secret")
            .with_token_price(TokenPrice::new(1.0, 5.0));

        let root = ChatNode::root("You are EchoAI.");

        // Enforced cost tracking through a CompletionContext so we also prove the
        // cost path works against a custom non-OpenAI wire.
        let captured_costs: Arc<Mutex<Vec<CostInfo>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = captured_costs.clone();
        let callback: AsyncCostCallback = Arc::new(move |cost: CostInfo, _meta: CompletionMeta| {
            let sink = sink.clone();
            Box::pin(async move {
                sink.lock().unwrap().push(cost);
            })
        });
        let ctx = CompletionContext::new(
            generator,
            serde_json::json!({}),
            callback,
            "https://app",
            "App",
        );

        // Pass an explicit max_tokens so we prove the normalized param maps into
        // EchoAI's own `settings.max_output_tokens` key (not just the default).
        let params = NodeCompletionParameters::new()
            .with_params(CompletionParameters::new().with_max_tokens(128));
        let user = root.add_user("hi");
        let answer = user
            .complete_tracked(&ctx, Some(&params))
            .await
            .expect("round-trip against the non-OpenAI self-hosted server");

        // Parsed EchoAI's envelope into a node.
        assert_eq!(answer.message.text(), Some("Echo: hi there"));
        assert_eq!(
            answer.get_metadata("model"),
            Some(serde_json::json!("echo-1"))
        );
        assert_eq!(
            answer.get_metadata("finish_reason"),
            Some(serde_json::json!("end"))
        );

        // Sent EchoAI's wire: its endpoint, its auth header, its request shape.
        let req = captured.await.unwrap();
        assert_eq!(
            req.path, "/api/generate",
            "EchoAI endpoint, not /chat/completions"
        );
        assert_eq!(
            req.header("X-Echo-Key"),
            Some("echo-secret"),
            "EchoAI auth header, not Authorization: Bearer"
        );
        assert_eq!(req.body["model"], "echo-1");
        // The per-request max_tokens (128) flows through the normalized params into
        // EchoAI's own `settings.max_output_tokens` key.
        assert_eq!(req.body["settings"]["max_output_tokens"], 128);
        // The flattened prompt carries the whole conversation.
        let prompt = req.body["prompt"].as_str().unwrap();
        assert!(prompt.contains("system: You are EchoAI."), "got: {prompt}");
        assert!(prompt.contains("user: hi"), "got: {prompt}");

        // Cost was tracked from EchoAI's token counts and the configured price:
        // 7 in x $1/Mtok + 3 out x $5/Mtok = (7 + 15) / 1e6 = $0.000022.
        let costs = captured_costs.lock().unwrap();
        assert_eq!(costs.len(), 1, "tracked exactly one completion");
        assert_eq!(costs[0].prompt_tokens, 7);
        assert_eq!(costs[0].completion_tokens, 3);
        assert!(
            (costs[0].cost - 0.000_022).abs() < 1e-12,
            "got {}",
            costs[0].cost
        );
    }

    #[tokio::test]
    async fn echoai_text_only_wire_fails_loudly_on_multimodal() {
        // EchoAI's flat-prompt wire can't carry an image, so a multimodal message
        // must ERROR, never silently flatten to text-only (which would drop the
        // attachment). This locks the reference example's fail-loud behavior; no
        // server is needed because build_request rejects before any request is sent.
        let generator = GeneratorInfo::custom("echoai", "http://127.0.0.1:9", "echo-1")
            .with_provider(Arc::new(EchoAiProvider))
            .with_api_key("echo-secret");

        let root = ChatNode::root("You are EchoAI.");
        let img = ImageData::from_url("https://example.com/x.png");
        let mut msg = Message::user("look at this");
        msg.content = MessageContent::with_images("look at this", &[img]);
        let user = root.add_child(ChatNode::new(msg)).unwrap();

        let result = user.complete(&generator, None).await;
        assert!(
            matches!(result, Err(minillmlib::MiniLLMError::InvalidParameter(_))),
            "multimodal must fail loudly on a text-only wire, got {result:?}"
        );
    }

    #[tokio::test]
    async fn echoai_fails_loudly_on_a_setting_it_cannot_send() {
        // EchoAI has no temperature field: a temperature must ERROR, never be
        // dropped. Refused in build_request, before any request is sent.
        let generator = GeneratorInfo::custom("echoai", "http://127.0.0.1:9", "echo-1")
            .with_provider(Arc::new(EchoAiProvider));
        let params = NodeCompletionParameters::new()
            .with_params(CompletionParameters::new().with_temperature(0.2));
        let result = ChatNode::root("s")
            .add_user("hi")
            .complete(&generator, Some(&params))
            .await;
        assert!(
            matches!(result, Err(minillmlib::MiniLLMError::InvalidParameter(_))),
            "{result:?}"
        );
    }
}

// =============================================================================
// Tool calling (live round trips)
// =============================================================================
//
// The full loop against real wires: definitions out, tool_calls back, results
// in, final answer. OpenRouter exercises the OpenAI wire; the Anthropic test
// exercises the native `/v1/messages` tool_use/tool_result translation.

use minillmlib::{ToolChoice, ToolDefinition};

/// A cheap OpenRouter model with reliable tool support (llama-3.1-8b's tool
/// calling is too flaky to assert on; OpenAI models can be blocked by an
/// account's data-policy settings, gemini-2.5-flash-lite is not).
const TOOL_TEST_MODEL: &str = TEST_MODEL;

fn weather_tool() -> ToolDefinition {
    ToolDefinition::new(
        "get_weather",
        "Get the current weather for a city",
        serde_json::json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"],
        }),
    )
}

fn tool_params() -> NodeCompletionParameters {
    NodeCompletionParameters::new().with_params(
        CompletionParameters::new()
            .with_max_tokens(200)
            .with_tool(weather_tool())
            // Force the call so the assertion is deterministic.
            .with_tool_choice(ToolChoice::Tool("get_weather".into())),
    )
}

/// Follow-up params: same tools, but let the model answer freely.
fn tool_answer_params() -> NodeCompletionParameters {
    NodeCompletionParameters::new().with_params(
        CompletionParameters::new()
            .with_max_tokens(200)
            .with_tool(weather_tool())
            .with_tool_choice(ToolChoice::Auto),
    )
}

async fn run_tool_round_trip(generator: &GeneratorInfo) {
    let root = ChatNode::root("You are helpful. Use the tools when asked about weather.");
    let user = root.add_user("What's the weather in Paris?");

    let node = user
        .complete(generator, Some(&tool_params()))
        .await
        .expect("tool-forcing completion");
    let calls = node.tool_calls().expect("model must call the forced tool");
    assert_eq!(calls[0].name, "get_weather");
    let args = calls[0].arguments_json().expect("valid JSON arguments");
    assert!(
        args["city"].as_str().is_some(),
        "arguments should carry a city, got {args}"
    );

    // Answer every call, then complete again for the final answer.
    let mut current = node.clone();
    for call in &calls {
        current = current.add_tool_result(&call.id, "15 degrees and sunny");
    }
    let answer = current
        .complete(generator, Some(&tool_answer_params()))
        .await
        .expect("post-tool-result completion");
    let text = answer.text().unwrap_or_default().to_lowercase();
    assert!(
        text.contains("15") || text.contains("sunny"),
        "final answer should use the tool result, got: {text}"
    );
}

#[tokio::test]
async fn test_openrouter_tool_calling_round_trip() {
    require_live!("OPENROUTER_API_KEY");
    dotenvy::dotenv().ok();
    run_tool_round_trip(&own_client(GeneratorInfo::openrouter(TOOL_TEST_MODEL))).await;
}

#[tokio::test]
async fn test_anthropic_tool_calling_round_trip() {
    require_live!("ANTHROPIC_API_KEY");
    run_tool_round_trip(&own_client(GeneratorInfo::anthropic(ANTHROPIC_TEST_MODEL))).await;
}

#[tokio::test]
async fn test_openrouter_streaming_tool_calls_assemble() {
    require_live!("OPENROUTER_API_KEY");
    dotenvy::dotenv().ok();
    // Streaming: tool-call fragments must assemble into complete typed calls on
    // the final node.
    let generator = own_client(GeneratorInfo::openrouter(TOOL_TEST_MODEL));
    let root = ChatNode::root("You are helpful.");
    let user = root.add_user("What's the weather in Paris?");
    let node = user
        .complete_streaming_collect(&generator, Some(&tool_params()))
        .await
        .expect("streaming tool completion");
    let calls = node.tool_calls().expect("streamed tool calls assembled");
    assert_eq!(calls[0].name, "get_weather");
    assert!(calls[0].arguments_json().is_ok(), "arguments reassembled");
}

#[tokio::test]
async fn test_anthropic_streaming_tool_calls_assemble() {
    require_live!("ANTHROPIC_API_KEY");
    // Anthropic streams tool calls via content_block_start + input_json_delta;
    // the fragments must assemble despite the shared text/tool index space.
    let generator = own_client(GeneratorInfo::anthropic(ANTHROPIC_TEST_MODEL));
    let root = ChatNode::root("You are helpful.");
    let user = root.add_user("What's the weather in Paris?");
    let node = user
        .complete_streaming_collect(&generator, Some(&tool_params()))
        .await
        .expect("anthropic streaming tool completion");
    let calls = node.tool_calls().expect("streamed tool calls assembled");
    assert_eq!(calls[0].name, "get_weather");
    assert!(calls[0].arguments_json().is_ok(), "arguments reassembled");
}

// ---------------------------------------------------------------------------
// Model catalog (live). The catalog endpoint is public, so these need no key;
// they are still `live`-gated because they hit the network.
// ---------------------------------------------------------------------------

/// Skip a live test that needs no credential, only the network.
macro_rules! require_network {
    () => {
        if !cfg!(feature = "live") {
            eprintln!("Skipping live test (enable with `cargo test --features live`)");
            return;
        }
    };
}

#[tokio::test]
async fn test_catalog_prices_a_first_party_model_with_cache_buckets() {
    require_network!();
    // The catalog needs no credential, so OpenRouter's lookup reads it alone.
    let generator = own_client(GeneratorInfo::openrouter("anthropic/claude-sonnet-4.6"));
    let info = generator.model_info().await;
    let price = info.price.clone().expect("the catalog lists sonnet 4.6");
    assert!(info.context_length.unwrap_or(0) > 0, "{info:?}");

    // Rates are per MILLION tokens once parsed (the wire carries per-token strings).
    assert!(price.input_per_mtok > 0.1, "{price:?}");
    assert!(
        price.output_per_mtok > price.input_per_mtok,
        "output costs more than input"
    );

    // Anthropic models publish both cache buckets: read is a discount, write a premium.
    let read = price
        .cache_read_per_mtok
        .expect("sonnet publishes a cache-read rate");
    let write = price
        .cache_write_per_mtok
        .expect("sonnet publishes a cache-write rate");
    assert!(read < price.input_per_mtok, "cache reads are discounted");
    assert!(write > price.input_per_mtok, "cache writes carry a premium");
}

/// A model the catalog does not list has an unknown price, and so an unknown
/// estimate, rather than a guessed one.
#[cfg(feature = "estimate")]
#[tokio::test]
async fn test_an_unknown_model_has_an_unknown_estimate() {
    require_network!();
    let generator = own_client(GeneratorInfo::openrouter("definitely/not-a-real-model"));
    assert_eq!(generator.model_info().await.price, None);
    let user = ChatNode::root("s").add_user("hi");
    assert_eq!(
        generator
            .estimate_cost_usd(&user.thread(), &CompletionParameters::new())
            .await,
        None
    );
}

/// The property the whole reservation scheme rests on: an estimate made BEFORE a
/// call must never come in under what the call really cost, or a caller can spend
/// past a limit it was told it had room for.
#[cfg(feature = "estimate")]
#[tokio::test]
async fn test_estimate_is_an_upper_bound_on_a_real_completion() {
    require_live!("OPENROUTER_API_KEY");

    let model = "anthropic/claude-haiku-4.5";
    let generator = own_client(GeneratorInfo::openrouter(model));

    let root = ChatNode::root("You are terse.");
    let user = root.add_user("Name three primary colours, comma separated, nothing else.");
    let params = CompletionParameters::new().with_max_tokens(64);

    // OpenRouter may route to any provider of the model, so the estimate is
    // bounded by the dearest endpoint. One call: the generator looks the model
    // up in the catalog, keeps the answer, and prices the prompt.
    let estimate = generator
        .estimate_cost_usd(&user.thread(), &params)
        .await
        .expect("haiku is catalogued");

    let captured: Arc<Mutex<Option<CostInfo>>> = Arc::new(Mutex::new(None));
    let slot = captured.clone();
    let callback: AsyncCostCallback = Arc::new(move |info: CostInfo, _m: CompletionMeta| {
        let slot = slot.clone();
        Box::pin(async move {
            *slot.lock().unwrap() = Some(info);
        })
    });
    let ctx = CompletionContext::new(
        generator,
        serde_json::json!({}),
        callback,
        "https://weavemind.ai",
        "Weft",
    );
    let node_params = NodeCompletionParameters::new().with_params(params);
    user.complete_tracked(&ctx, Some(&node_params))
        .await
        .expect("live completion");

    let actual = captured.lock().unwrap().take().expect("cost reported").cost;
    assert!(actual > 0.0, "a real completion cost something");
    assert!(
        estimate >= actual,
        "the estimate ({estimate}) must never fall below the real cost ({actual})"
    );
}

// =============================================================================
// Retries, billing, and model lookups against a scripted server (offline).
// Layer-3 contract tests: the real completion pipeline over real HTTP to a
// loopback server that answers from a script.
// =============================================================================

mod scripted {
    use super::custom_provider::{mock_sequence, served};
    use super::*;
    use minillmlib::provider::{AnthropicProvider, TokenPrice};
    use minillmlib::MiniLLMError;
    use std::sync::Arc;

    const EMPTY: &str = r#"{"id":"e","model":"m","choices":[{"index":0,
        "message":{"role":"assistant","content":""},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":10,"completion_tokens":0,"total_tokens":10}}"#;
    const GOOD: &str = r#"{"id":"g","model":"m","choices":[{"index":0,
        "message":{"role":"assistant","content":"hi"},"finish_reason":"stop"}],
        "usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}"#;
    const DOWN: &str = r#"{"error":{"message":"overloaded"}}"#;

    fn generator(base_url: &str) -> GeneratorInfo {
        GeneratorInfo::custom("scripted", base_url, "m").with_token_price(TokenPrice::new(1.0, 1.0))
    }

    /// A tracked completion reports to its context, so it refuses a callback
    /// of the caller's own (which it would never fire) before sending
    /// anything.
    #[tokio::test]
    async fn a_tracked_completion_refuses_a_cost_callback() {
        let (ctx, _) = make_test_context(generator("http://127.0.0.1:9"));
        let params = NodeCompletionParameters::new().with_cost_callback(|_: CostInfo| {});
        let user = ChatNode::root("s").add_user("hi");
        let err = user
            .complete_tracked(&ctx, Some(&params))
            .await
            .unwrap_err();
        assert!(matches!(err, MiniLLMError::InvalidParameter(_)), "{err}");
        let err = user
            .complete_streaming_collect_tracked(&ctx, Some(&params))
            .await
            .unwrap_err();
        assert!(matches!(err, MiniLLMError::InvalidParameter(_)), "{err}");
        assert!(user
            .complete_streaming_tracked(&ctx, Some(&params))
            .await
            .is_err());
    }

    /// With no retries a failure is the attempt's own error, after exactly one
    /// request; with retries it is wrapped, after one request per attempt.
    #[tokio::test]
    async fn retries_happen_only_when_asked() {
        let (base_url, mut captured) = mock_sequence(vec![(503, DOWN)]).await;
        let err = ChatNode::root("s")
            .add_user("hi")
            .complete(&generator(&base_url), None)
            .await
            .unwrap_err();
        assert!(
            matches!(err, MiniLLMError::Api { status: 503, .. }),
            "{err:?}"
        );
        assert_eq!(served(&mut captured).len(), 1);

        let (base_url, mut captured) =
            mock_sequence(vec![(503, DOWN), (503, DOWN), (503, DOWN)]).await;
        let params = NodeCompletionParameters::new()
            .with_retry(2)
            .with_back_off_time(0.0);
        let err = ChatNode::root("s")
            .add_user("hi")
            .complete(&generator(&base_url), Some(&params))
            .await
            .unwrap_err();
        assert!(
            matches!(err, MiniLLMError::MaxRetriesExceeded(_)),
            "{err:?}"
        );
        assert_eq!(served(&mut captured).len(), 3);
    }

    /// A response rejected by `crash_on_empty_response` was billed, so its cost
    /// is reported beside the accepted one, and even when the run fails.
    #[tokio::test]
    async fn a_rejected_response_is_still_billed() {
        let (base_url, _captured) = mock_sequence(vec![(200, EMPTY), (200, GOOD)]).await;
        let params = NodeCompletionParameters::new()
            .with_crash_on_empty(true)
            .with_retry(1)
            .with_back_off_time(0.0);
        let (result, costs) = ChatNode::root("s")
            .add_user("hi")
            .complete_costed(&generator(&base_url), Some(&params))
            .await;
        assert_eq!(result.unwrap().text(), Some("hi"));
        assert_eq!(costs.len(), 2, "{costs:?}");

        let (base_url, _captured) = mock_sequence(vec![(200, EMPTY)]).await;
        let params = NodeCompletionParameters::new().with_crash_on_empty(true);
        let (result, costs) = ChatNode::root("s")
            .add_user("hi")
            .complete_costed(&generator(&base_url), Some(&params))
            .await;
        assert!(matches!(result, Err(MiniLLMError::EmptyResponse)));
        assert_eq!(costs.len(), 1, "the rejected answer was billed");
    }

    /// A rejected response followed by a non-retryable error still reports
    /// the rejected one's cost.
    #[tokio::test]
    async fn a_rejected_response_is_billed_when_a_later_attempt_is_refused() {
        let (base_url, _captured) = mock_sequence(vec![(200, EMPTY), (401, DOWN)]).await;
        let params = NodeCompletionParameters::new()
            .with_crash_on_empty(true)
            .with_retry(1)
            .with_back_off_time(0.0);
        let (result, costs) = ChatNode::root("s")
            .add_user("hi")
            .complete_costed(&generator(&base_url), Some(&params))
            .await;
        assert!(
            matches!(result, Err(MiniLLMError::Api { status: 401, .. })),
            "{result:?}"
        );
        assert_eq!(costs.len(), 1);
    }

    /// The cost callback fires once per billed response, rejected ones
    /// included, at the model's price.
    #[tokio::test]
    async fn every_billed_response_fires_the_callback_at_the_models_price() {
        let (base_url, _captured) = mock_sequence(vec![(200, EMPTY), (200, GOOD)]).await;
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sink = seen.clone();
        let generator = GeneratorInfo::custom("scripted", &base_url, "m")
            .with_token_price(TokenPrice::new(1_000_000.0, 1_000_000.0));
        let params = NodeCompletionParameters::new()
            .with_crash_on_empty(true)
            .with_retry(1)
            .with_back_off_time(0.0)
            .with_cost_callback(move |info| sink.lock().unwrap().push(info.cost));
        ChatNode::root("s")
            .add_user("hi")
            .complete(&generator, Some(&params))
            .await
            .unwrap();
        // 10 prompt tokens, then 10 + 2, at $1 per token.
        assert_eq!(*seen.lock().unwrap(), vec![10.0, 12.0]);
    }

    /// The Anthropic provider looks the model up once, with the generator's own
    /// headers, and an unset cap goes out as the model's largest. The price is
    /// given, so the catalog is not read and the test stays offline.
    #[tokio::test]
    async fn the_anthropic_lookup_rides_the_requests_headers() {
        const MODEL: &str = r#"{"id":"claude-x","max_tokens":32000,"capabilities":null}"#;
        const MESSAGE: &str = r#"{"id":"a","model":"claude-x","content":[{"type":"text","text":"ok"}],
            "stop_reason":"end_turn","usage":{"input_tokens":3,"output_tokens":1}}"#;
        let (base_url, mut captured) =
            mock_sequence(vec![(200, MODEL), (200, MESSAGE), (200, MESSAGE)]).await;
        let generator = GeneratorInfo::new("anthropic", &base_url, "claude-x")
            .with_provider(Arc::new(AnthropicProvider))
            .with_api_key("k")
            .with_header("X-Gateway", "g")
            .with_token_price(TokenPrice::new(1.0, 5.0));
        for _ in 0..2 {
            ChatNode::root("s")
                .add_user("hi")
                .complete(&generator, None)
                .await
                .unwrap();
        }
        let requests = served(&mut captured);
        assert_eq!(requests.len(), 3, "one lookup, two messages");
        assert_eq!(requests[0].method, "GET");
        assert_eq!(requests[0].path, "/v1/models/claude-x");
        assert_eq!(requests[0].header("X-Gateway"), Some("g"));
        assert_eq!(requests[0].header("x-api-key"), Some("k"));
        assert_eq!(requests[1].path, "/v1/messages");
        assert_eq!(requests[1].body["max_tokens"], 32000);
    }
}

// =============================================================================
// Every feature, live, on the native Anthropic wire.
// =============================================================================

mod anthropic_features {
    use super::*;
    use minillmlib::{CostResolution, Reasoning};

    fn haiku() -> GeneratorInfo {
        own_client(GeneratorInfo::anthropic(ANTHROPIC_TEST_MODEL))
            .with_openrouter_name("anthropic/claude-haiku-4.5")
    }

    #[tokio::test]
    async fn tool_calling_round_trip() {
        require_live!("ANTHROPIC_API_KEY");
        run_tool_round_trip(&haiku()).await;
    }

    /// An inline image goes out as an Anthropic image block and is answered.
    #[tokio::test]
    async fn an_image_is_seen() {
        require_live!("ANTHROPIC_API_KEY");
        // A 1x1 PNG.
        let pixel = ImageData::from_base64(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg==",
            "image/png",
        );
        let user = ChatNode::root("You are terse.").add_user(MessageContent::with_images(
            "What is this? One word.",
            &[pixel],
        ));
        let params = NodeCompletionParameters::new()
            .with_params(CompletionParameters::new().with_max_tokens(20));
        let answer = user
            .complete(&haiku(), Some(&params))
            .await
            .expect("an image completion");
        assert!(!answer.text().unwrap_or_default().is_empty());
    }

    #[tokio::test]
    async fn json_is_parsed_and_repaired() {
        require_live!("ANTHROPIC_API_KEY");
        let params = NodeCompletionParameters::new()
            .with_params(CompletionParameters::new().with_max_tokens(200))
            .expecting_json()
            .with_crash_on_refusal(true);
        let answer = ChatNode::root("Answer with a JSON object only.")
            .add_user(r#"Give {"city": <a French city>, "population": <a number>}."#)
            .complete(&haiku(), Some(&params))
            .await
            .expect("json completion");
        let value: serde_json::Value =
            serde_json::from_str(answer.text().unwrap()).expect("the reply is repaired JSON");
        assert!(value["city"].is_string(), "{value}");
    }

    /// A raw stream yields text chunks; a collected stream ends in a node; and
    /// both stream and plain completions report a resolved cost.
    #[tokio::test]
    async fn streaming_and_the_cost_of_a_call() {
        require_live!("ANTHROPIC_API_KEY");
        let generator = haiku();
        let user = ChatNode::root("You are terse.").add_user("Count from 1 to 5.");

        let mut stream = user
            .complete_streaming(&generator, None)
            .await
            .expect("raw stream");
        let mut chunks = 0;
        while let Some(chunk) = stream.next_chunk().await {
            chunk.expect("a chunk");
            chunks += 1;
        }
        assert!(chunks > 1, "the reply arrived in pieces");

        let (result, costs) = user.complete_costed(&generator, None).await;
        result.expect("plain completion");
        assert_eq!(costs.len(), 1);
        assert_eq!(costs[0].resolution, CostResolution::Resolved);
        assert!(costs[0].cost > 0.0);

        let seen: Arc<Mutex<Vec<CostInfo>>> = Arc::default();
        let sink = seen.clone();
        let callback: AsyncCostCallback = Arc::new(move |info: CostInfo, _m: CompletionMeta| {
            let sink = sink.clone();
            Box::pin(async move { sink.lock().unwrap().push(info) })
        });
        let ctx = CompletionContext::new(generator, serde_json::json!({}), callback, "u", "t");
        user.complete_streaming_collect_tracked(&ctx, None)
            .await
            .expect("tracked stream");
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert_eq!(seen[0].resolution, CostResolution::Resolved);
        assert!(seen[0].cost > 0.0);
    }

    /// The estimate, made before the call from the catalog's price, never
    /// comes in under what the call cost at that price.
    #[cfg(feature = "estimate")]
    #[tokio::test]
    async fn the_estimate_bounds_the_real_cost() {
        require_live!("ANTHROPIC_API_KEY");
        let generator = haiku();
        let user = ChatNode::root("You are terse.").add_user("Name three colours.");
        let params = CompletionParameters::new().with_max_tokens(64);
        let estimate = generator
            .estimate_cost_usd(&user.thread(), &params)
            .await
            .expect("priced and bounded");
        let (result, costs) = user
            .complete_costed(
                &generator,
                Some(&NodeCompletionParameters::new().with_params(params)),
            )
            .await;
        result.expect("completion");
        assert!(estimate >= costs[0].cost, "{estimate} < {}", costs[0].cost);
    }

    /// Reasoning unset, off, on, and on at a named effort all go through on a
    /// model that takes them; an effort the model lacks is refused by the API
    /// itself, and that refusal is the error.
    #[tokio::test]
    async fn reasoning_is_passed_down() {
        require_live!("ANTHROPIC_API_KEY");
        let sonnet = own_client(GeneratorInfo::anthropic("claude-sonnet-4-6"));
        let ask = |reasoning: Option<Reasoning>| {
            let sonnet = sonnet.clone();
            async move {
                let mut params = CompletionParameters::new().with_max_tokens(4000);
                params.reasoning = reasoning;
                ChatNode::root("You are terse.")
                    .add_user("Say OK")
                    .complete(
                        &sonnet,
                        Some(&NodeCompletionParameters::new().with_params(params)),
                    )
                    .await
            }
        };
        for reasoning in [
            None,
            Some(Reasoning::Off),
            Some(Reasoning::On(None)),
            Some(Reasoning::On(Some("high".into()))),
        ] {
            ask(reasoning.clone())
                .await
                .unwrap_or_else(|e| panic!("{reasoning:?}: {e}"));
        }
        let err = ask(Some(Reasoning::On(Some("xhigh".into()))))
            .await
            .expect_err("sonnet 4.6 has no xhigh effort");
        assert!(
            matches!(err, minillmlib::MiniLLMError::Api { status: 400, .. }),
            "{err}"
        );
    }

    /// A marked prefix is written to the cache once and read after.
    #[tokio::test]
    async fn the_cache_marker_writes_then_reads() {
        require_live!("ANTHROPIC_API_KEY");
        let generator = haiku();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let big = format!("Session {nonce}. {}", big_system_prompt());
        let params = NodeCompletionParameters::new()
            .with_params(CompletionParameters::new().with_max_tokens(5));
        let mut costs = Vec::new();
        for ask in ["Say A", "Say B"] {
            let root = ChatNode::root(big.clone());
            root.cache_breakpoint();
            let (result, cost) = root
                .add_user(ask)
                .complete_costed(&generator, Some(&params))
                .await;
            result.expect("completion");
            costs.extend(cost);
        }
        assert!(costs[0].cache_write_tokens > 0, "{:?}", costs[0]);
        assert!(costs[1].cache_read_tokens > 0, "{:?}", costs[1]);
        assert!(
            costs[1].cost < costs[0].cost,
            "the read ({}) costs less than the write ({})",
            costs[1].cost,
            costs[0].cost
        );
    }
}
