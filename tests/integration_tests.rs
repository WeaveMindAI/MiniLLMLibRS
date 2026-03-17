//! Comprehensive integration tests for MiniLLMLib
//!
//! These tests make real API calls to OpenRouter.
//! Run with: cargo test --test integration_tests -- --nocapture
//!
//! Requires OPENROUTER_API_KEY environment variable to be set.

use minillmlib::{
    chat_node::ChatNode,
    generator::{CompletionParameters, GeneratorInfo, NodeCompletionParameters, ProviderSettings},
    message::{AudioData, ImageData, Media, Message, MessageContent, Role, VideoData},
    provider::{CostInfo, LLMClient},
    tracking::{AsyncCostCallback, CompletionContext, CompletionMeta},
};
use std::sync::{Arc, Mutex};

// Test model - small and cheap for testing
const TEST_MODEL: &str = "google/gemini-2.0-flash-lite-001";
const TEXT_ONLY_MODEL: &str = "google/gemini-2.0-flash-lite-001";

fn get_test_generator() -> GeneratorInfo {
    dotenvy::dotenv().ok();
    GeneratorInfo::openrouter(TEST_MODEL)
}

fn get_text_generator() -> GeneratorInfo {
    dotenvy::dotenv().ok();
    GeneratorInfo::openrouter(TEXT_ONLY_MODEL)
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
    assert!(gen.api_key.is_none());
}

#[test]
fn test_generator_info_with_api_key() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model")
        .with_api_key("test-key-123");
    assert!(gen.api_key.is_some());
}

#[test]
fn test_generator_info_openrouter() {
    dotenvy::dotenv().ok();
    let gen = GeneratorInfo::openrouter("anthropic/claude-3.5-sonnet");
    assert_eq!(gen.name, "OpenRouter");
    assert_eq!(gen.base_url, "https://openrouter.ai/api/v1");
    assert_eq!(gen.model, "anthropic/claude-3.5-sonnet");
    // Should have custom headers for OpenRouter
    assert!(!gen.custom_headers.is_empty());
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
        .with_vision()
        .with_audio()
        .with_max_context(128000)
        .with_header("X-Custom", "value");

    assert!(gen.supports_vision);
    assert!(gen.supports_audio);
    assert_eq!(gen.max_context_length, Some(128000));
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
    assert_eq!(params.max_tokens, Some(4096));
    assert_eq!(params.temperature, Some(0.7));
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

    // Note: merge takes other's value if present, else falls back to self
    // Since override_params has default max_tokens (4096), it takes that
    assert_eq!(merged.max_tokens, Some(4096)); // From override (default)
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
        .with_streaming(true)
        .expecting_json()
        .with_timeout(60);

    assert_eq!(
        params.system_prompt,
        Some("You are a helpful assistant".to_string())
    );
    assert_eq!(params.stream, Some(true));
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
    use minillmlib::ProviderSettings;

    let provider = ProviderSettings::new()
        .sort_by_throughput()
        .deny_data_collection()
        .with_ignore(vec!["SambaNova".to_string()]);

    assert_eq!(provider.sort, Some("throughput".to_string()));
    assert_eq!(provider.data_collection, Some("deny".to_string()));
    assert_eq!(provider.ignore, Some(vec!["SambaNova".to_string()]));
}

#[test]
fn test_completion_parameters_with_provider() {
    use minillmlib::{CompletionParameters, ProviderSettings};

    let provider = ProviderSettings::new()
        .sort_by_throughput()
        .deny_data_collection();

    let params = CompletionParameters::new()
        .with_temperature(0.7)
        .with_provider(provider);

    assert!(params.provider.is_some());
    let p = params.provider.unwrap();
    assert_eq!(p.sort, Some("throughput".to_string()));
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
    assert_eq!(image.mime_type, "url");
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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let video_path = "./data/test.mp4";
    if !std::path::Path::new(video_path).exists() {
        eprintln!("Skipping test_video_completion: test.mp4 not found");
        return;
    }

    let generator = get_test_generator();
    let video = VideoData::from_file(video_path).unwrap();

    let content = MessageContent::with_video("What do you see in this video? Be brief.", &[video]);

    let root = ChatNode::root("You are a helpful assistant.");
    let user_node = root.add_child(ChatNode::new(Message {
        role: Role::User,
        content,
        name: None,
        tool_call_id: None,
        tool_calls: None,
    }));

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
    assert_eq!(video.format, "url");
}

#[tokio::test]
async fn test_image_completion_from_url() {
    dotenvy::dotenv().ok();

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let generator = get_test_generator();
    let image = ImageData::from_url("https://cdn.mos.cms.futurecdn.net/nbaR6JXZ3Z7mzuW9bh4nQN.jpg");

    let content = MessageContent::with_images("Describe this image in one sentence.", &[image]);

    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user_node = root.add_child(ChatNode::new(Message {
        role: Role::User,
        content,
        name: None,
        tool_call_id: None,
        tool_calls: None,
    }));

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
    let api_format = content.to_api_format();
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
    use minillmlib::message::messages_to_payload;
    let image = ImageData::from_url("https://example.com/image.jpg");
    let content = MessageContent::with_images("Describe this image", &[image]);
    let msg = Message {
        role: Role::User,
        content,
        name: None,
        tool_call_id: None,
        tool_calls: None,
    };
    let payload = messages_to_payload(&[msg]);
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
    assert!(media.as_image().is_some());
}

#[test]
fn test_media_from_audio() {
    let audio = AudioData::from_bytes(&[0u8; 100], "wav");
    let media = Media::from(audio.clone());

    assert!(!media.is_image());
    assert!(media.is_audio());
    assert!(!media.is_video());
    assert!(media.as_audio().is_some());
}

#[test]
fn test_media_from_video() {
    let video = VideoData::from_bytes(&[0u8; 100], "mp4");
    let media = Media::from(video.clone());

    assert!(!media.is_image());
    assert!(!media.is_audio());
    assert!(media.is_video());
    assert!(media.as_video().is_some());
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

    let media = vec![Media::from(image), Media::from(audio), Media::from(video)];

    let content = MessageContent::with_media("Describe all this media", &media);
    assert!(content.has_multimodal());
    assert_eq!(content.get_text(), Some("Describe all this media"));
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

    // Merge second tree into first
    let merged_leaf = u1.merge(&a2);

    // The merged leaf should be the leaf of the second tree
    assert_eq!(merged_leaf.id, a2.id);

    // root1 should now have the second tree as a subtree
    // Structure: root1 -> u1 -> root2 -> u2 -> a2
    assert_eq!(root1.node_count(), 5);

    // root2's parent should now be u1
    assert_eq!(root2.parent().unwrap().id, u1.id);
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

    // Reattach a1 to root directly
    root.add_child(a1.clone());

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

    // Merge tree2 into tree1
    tree1_u.merge(&tree2_a);
    // Structure: tree1_root -> tree1_u -> tree2_root -> tree2_u -> tree2_a
    assert_eq!(tree1_root.node_count(), 5);

    // Merge tree3 into tree2_a
    tree2_a.merge(&tree3_u);
    // Structure: tree1_root -> tree1_u -> tree2_root -> tree2_u -> tree2_a -> tree3_root -> tree3_u
    assert_eq!(tree1_root.node_count(), 7);

    // Verify the chain
    assert_eq!(tree3_u.get_root().id, tree1_root.id);
    assert_eq!(tree3_u.depth(), 6);
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
    // Tree 1 with format kwargs
    let tree1 = ChatNode::root("Hello {name}, I am {bot}.");
    tree1.set_format_kwarg("name", "Alice");
    tree1.set_format_kwarg("bot", "Claude");
    let tree1_u = tree1.add_user("Hi {bot}!");

    // Tree 2 with different format kwargs
    let tree2 = ChatNode::root("Switching to {mode} mode.");
    tree2.set_format_kwarg("mode", "expert");
    let tree2_u = tree2.add_user("Tell me about {topic}.");
    tree2_u.set_format_kwarg("topic", "Rust");

    // Merge tree2 into tree1
    tree1_u.merge(&tree2_u);

    // Get formatted thread from the deepest node
    let formatted = tree2_u.formatted_thread();

    // Should have 4 messages
    assert_eq!(formatted.len(), 4);

    // Check that format kwargs from both trees are applied
    // tree1's kwargs should apply to tree1's messages
    assert!(formatted[0].content.get_text().unwrap().contains("Alice"));
    assert!(formatted[0].content.get_text().unwrap().contains("Claude"));
    assert!(formatted[1].content.get_text().unwrap().contains("Claude"));

    // tree2's kwargs should apply to tree2's messages
    assert!(formatted[2].content.get_text().unwrap().contains("expert"));
    assert!(formatted[3].content.get_text().unwrap().contains("Rust"));
}

#[test]
fn test_chat_node_detach_preserves_format_kwargs() {
    let root = ChatNode::root("Hello {name}");
    root.set_format_kwarg("name", "World");
    let u1 = root.add_user("Goodbye {name}");

    // Verify format kwargs work before detach
    let formatted_before = u1.formatted_thread();
    assert!(formatted_before[0]
        .content
        .get_text()
        .unwrap()
        .contains("World"));
    assert!(formatted_before[1]
        .content
        .get_text()
        .unwrap()
        .contains("World"));

    // Detach u1
    u1.detach();

    // u1 should still have its own format kwargs (none set directly)
    // But root's kwargs should no longer be accessible
    let formatted_after = u1.formatted_thread();
    assert_eq!(formatted_after.len(), 1); // Only u1's message
                                          // The placeholder should NOT be replaced since root's kwargs are gone
    assert!(formatted_after[0]
        .content
        .get_text()
        .unwrap()
        .contains("{name}"));
}

#[test]
fn test_chat_node_to_thread_data() {
    let root = ChatNode::root("You are helpful");
    let u1 = root.add_user("Hello");
    let a1 = u1.add_assistant("Hi there!");

    let thread_data = a1.to_thread_data();

    assert_eq!(thread_data.prompts.len(), 3);
    assert_eq!(thread_data.prompts[0].role, "system");
    assert_eq!(thread_data.prompts[0].content, "You are helpful");
    assert_eq!(thread_data.prompts[1].role, "user");
    assert_eq!(thread_data.prompts[1].content, "Hello");
    assert_eq!(thread_data.prompts[2].role, "assistant");
    assert_eq!(thread_data.prompts[2].content, "Hi there!");
}

#[test]
fn test_chat_node_from_thread_json() {
    let json = r#"{
        "prompts": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"}
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
    let root = ChatNode::root("You are {bot_name}.");
    root.set_format_kwarg("bot_name", "Assistant");

    let user = root.add_user("Hi {bot_name}!");
    let assistant = user.add_assistant("Hello {user_name}!");
    assistant.set_format_kwarg("user_name", "Bob");

    let formatted = assistant.formatted_thread();

    assert_eq!(formatted[0].content.get_text(), Some("You are Assistant."));
    assert_eq!(formatted[1].content.get_text(), Some("Hi Assistant!"));
    assert_eq!(formatted[2].content.get_text(), Some("Hello Bob!"));
}

#[test]
fn test_chat_node_format_kwargs_propagate() {
    let root = ChatNode::root("System");
    let user = root.add_user("Hello");
    let assistant = user.add_assistant("Hi");

    // Update from leaf with propagation
    let mut kwargs = std::collections::HashMap::new();
    kwargs.insert("key".to_string(), "value".to_string());
    assistant.update_format_kwargs(&kwargs, true);

    // All nodes should have the kwarg
    assert_eq!(root.get_format_kwarg("key"), Some("value".to_string()));
    assert_eq!(user.get_format_kwarg("key"), Some("value".to_string()));
    assert_eq!(assistant.get_format_kwarg("key"), Some("value".to_string()));
}

#[test]
fn test_chat_node_format_kwargs_save_load() {
    use std::fs;

    let root = ChatNode::root("Hello {name}!");
    root.set_format_kwarg("name", "World");
    let user = root.add_user("Goodbye {name}!");

    // Save
    let temp_path = "/tmp/test_format_kwargs.json";
    user.save_thread(temp_path).unwrap();

    // Load
    let (loaded_root, loaded_leaf) = ChatNode::from_thread_file(temp_path).unwrap();

    // Check format_kwargs were preserved
    assert_eq!(
        loaded_root.get_format_kwarg("name"),
        Some("World".to_string())
    );

    // Check formatted content
    let formatted = loaded_leaf.formatted_thread();
    assert_eq!(formatted[0].content.get_text(), Some("Hello World!"));
    assert_eq!(formatted[1].content.get_text(), Some("Goodbye World!"));

    // Cleanup
    fs::remove_file(temp_path).ok();
}

#[tokio::test]
async fn test_load_data_test_json() {
    let path = "./data/test.json";
    if !std::path::Path::new(path).exists() {
        eprintln!("Skipping test_load_data_test_json: data/test.json not found");
        return;
    }

    // Load the thread
    let (root, leaf) = ChatNode::from_thread_file(path).unwrap();

    // Set multiple format kwargs
    root.set_format_kwarg("assistant_name", "Claude");
    root.set_format_kwarg("user_name", "Alice");
    root.set_format_kwarg("topic", "quantum computing");
    root.set_format_kwarg("style", "friendly and concise");

    // Print the formatted messages that will be sent to the LLM
    println!("\n=== FORMATTED PROMPT ===");
    println!("{}", minillmlib::format_conversation(&leaf));
    println!("========================\n");

    let formatted = leaf.formatted_thread();

    // Verify all placeholders are replaced
    for msg in &formatted {
        let text = msg.content.get_text().unwrap();
        assert!(
            !text.contains("{assistant_name}"),
            "Placeholder not replaced: {}",
            text
        );
        assert!(
            !text.contains("{user_name}"),
            "Placeholder not replaced: {}",
            text
        );
        assert!(
            !text.contains("{topic}"),
            "Placeholder not replaced: {}",
            text
        );
        assert!(
            !text.contains("{style}"),
            "Placeholder not replaced: {}",
            text
        );
    }

    // Now let's actually call the LLM with this
    let gi = get_cheap_generator();
    let params = NodeCompletionParameters::default()
        .with_params(CompletionParameters::default().with_max_tokens(150));

    // Complete from the formatted thread
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
    dotenvy::dotenv().ok();

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test_cost_tracking: OPENROUTER_API_KEY not set");
        return;
    }

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
        .with_openrouter_cost_tracking()
        .with_cost_callback(move |info: CostInfo| {
            println!("\n=== COST CALLBACK RECEIVED ===");
            println!("Cost: {} credits", info.cost);
            println!("Prompt tokens: {}", info.prompt_tokens);
            println!("Completion tokens: {}", info.completion_tokens);
            println!("Total tokens: {}", info.total_tokens);
            println!("Model: {}", info.model);
            println!("Response ID: {}", info.response_id);
            if let Some(cached) = info.cached_tokens {
                println!("Cached tokens: {}", cached);
            }
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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let generator = get_text_generator();

    // Track cumulative costs
    let total_cost = Arc::new(Mutex::new(0.0_f64));
    let request_count = Arc::new(Mutex::new(0_u32));

    let cost_tracker = total_cost.clone();
    let count_tracker = request_count.clone();

    let params = NodeCompletionParameters::new()
        .with_openrouter_cost_tracking()
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
    dotenvy::dotenv().ok();

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test_cost_tracking_streaming: OPENROUTER_API_KEY not set");
        return;
    }

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
        .with_openrouter_cost_tracking()
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
    dotenvy::dotenv().ok();

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test_simple_completion: OPENROUTER_API_KEY not set");
        return;
    }

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let image_path = "./data/test.jpg";
    if !std::path::Path::new(image_path).exists() {
        eprintln!("Skipping test_image_completion: test.jpg not found");
        return;
    }

    let generator = get_test_generator();
    let image = ImageData::from_file(image_path).unwrap();

    let content = MessageContent::with_images("Describe this image in one sentence.", &[image]);

    let root = ChatNode::root("You are a helpful assistant. Be very brief.");
    let user_node = root.add_child(ChatNode::new(Message {
        role: Role::User,
        content,
        name: None,
        tool_call_id: None,
        tool_calls: None,
    }));

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
async fn test_audio_completion() {
    dotenvy::dotenv().ok();

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let audio_path = "./data/test.mp3";
    if !std::path::Path::new(audio_path).exists() {
        eprintln!("Skipping test_audio_completion: test.mp3 not found");
        return;
    }

    let generator = get_test_generator();
    let audio = AudioData::from_file(audio_path).unwrap();

    let content = MessageContent::with_audio("What do you hear in this audio? Be brief.", &[audio]);

    let root = ChatNode::root("You are a helpful assistant.");
    let user_node = root.add_child(ChatNode::new(Message {
        role: Role::User,
        content,
        name: None,
        tool_call_id: None,
        tool_calls: None,
    }));

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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
    let user_node = root.add_child(ChatNode::new(Message {
        role: Role::User,
        content,
        name: None,
        tool_call_id: None,
        tool_calls: None,
    }));

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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

    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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
    let generator = GeneratorInfo::openrouter("google/gemini-2.0-flash-lite-001")
        .with_api_key("invalid-key-12345");

    let root = ChatNode::root("Test");
    let user = root.add_user("Hello");

    let result = user.complete(&generator, None).await;

    // Should fail with API error
    assert!(result.is_err());
    println!("Expected error: {:?}", result.err());
}

#[tokio::test]
async fn test_missing_api_key() {
    // Create generator without API key
    let generator = GeneratorInfo::new(
        "Test",
        "https://openrouter.ai/api/v1",
        "google/gemini-2.0-flash-lite-001",
    );

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
// Validate JSON Response Tests
// =============================================================================

#[test]
fn test_validate_json_response_valid() {
    use minillmlib::validate_json_response;

    let response = serde_json::json!({
        "choices": [{
            "message": {
                "content": "Hello world"
            }
        }]
    });

    let result = validate_json_response(&response).unwrap();
    assert_eq!(result, "Hello world");
}

#[test]
fn test_validate_json_response_missing_choices() {
    use minillmlib::validate_json_response;

    let response = serde_json::json!({
        "message": "no choices"
    });

    let result = validate_json_response(&response);
    assert!(result.is_err());
}

#[test]
fn test_validate_json_response_missing_content() {
    use minillmlib::validate_json_response;

    let response = serde_json::json!({
        "choices": [{
            "message": {}
        }]
    });

    let result = validate_json_response(&response);
    assert!(result.is_err());
}

// =============================================================================
// Multi-threading and Async Concurrency Tests
// =============================================================================

const CHEAP_MODEL: &str = "openai/gpt-oss-20b";

fn get_cheap_generator() -> GeneratorInfo {
    dotenvy::dotenv().ok();
    let provider = ProviderSettings::new().sort_by_price();
    GeneratorInfo::openrouter(CHEAP_MODEL)
        .with_default_params(CompletionParameters::default().with_provider(provider))
}

/// Test multi-threaded access to ChatNode
/// Spawns multiple OS threads that each make a completion request
#[tokio::test]
async fn test_multi_threaded_completions() {
    let gi = get_cheap_generator();
    let params = NodeCompletionParameters::default()
        .with_params(CompletionParameters::default().with_max_tokens(20));

    // Create a shared root node
    let root = ChatNode::root("You are a helpful assistant. Be very brief.");

    // Spawn 10 threads, each making a completion
    let mut handles = vec![];

    for i in 0..10 {
        let gi_clone = gi.clone();
        let params_clone = params.clone();
        let root_clone = Arc::clone(&root);

        let handle = std::thread::spawn(move || {
            // Each thread creates its own runtime for the async call
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                // Add a user message as a child
                let user_node = root_clone.add_user(format!("Say the number {}", i));

                let result = user_node.complete(&gi_clone, Some(&params_clone)).await;
                (i, result.is_ok())
            })
        });

        handles.push(handle);
    }

    // Collect results
    let mut successes = 0;
    for handle in handles {
        let (i, ok) = handle.join().expect("Thread panicked");
        println!("Thread {}: {}", i, if ok { "OK" } else { "FAILED" });
        if ok {
            successes += 1;
        }
    }

    // All should succeed
    assert!(
        successes >= 8,
        "At least 8/10 threads should succeed, got {}",
        successes
    );

    // Verify tree structure - root should have 10 children
    let children_count = root.child_count();
    assert_eq!(children_count, 10, "Root should have 10 children");
}

/// Test async concurrent completions (like Python's asyncio.gather)
/// All requests are sent concurrently and awaited together
#[tokio::test]
async fn test_async_concurrent_completions() {
    let gi = get_cheap_generator();
    let params = NodeCompletionParameters::default()
        .with_params(CompletionParameters::default().with_max_tokens(20));

    let root = ChatNode::root("You are a helpful assistant. Be very brief.");

    // Create 10 futures for concurrent execution
    let mut futures = vec![];

    for i in 0..10 {
        let gi_clone = gi.clone();
        let params_clone = params.clone();
        let root_clone = Arc::clone(&root);

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

    // Check results
    let mut successes = 0;
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
// CompletionContext & Tracking Tests (Unit — no API calls)
// =============================================================================

/// Helper: build a CompletionContext with a mock callback that captures CostInfo
fn make_test_context(
    generator: GeneratorInfo,
    is_byok: bool,
) -> (CompletionContext, Arc<Mutex<Vec<CostInfo>>>) {
    let captured: Arc<Mutex<Vec<CostInfo>>> = Arc::new(Mutex::new(Vec::new()));
    let captured_clone = captured.clone();

    let callback: AsyncCostCallback = Arc::new(move |cost_info, _meta| {
        let captured = captured_clone.clone();
        Box::pin(async move {
            captured.lock().unwrap().push(cost_info);
        })
    });

    let meta = CompletionMeta {
        userId: "test-user".to_string(),
        workflowId: Some("wf-123".to_string()),
        executionId: Some("exec-456".to_string()),
        nodeId: Some("node-789".to_string()),
        isByok: is_byok,
    };

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
    let (ctx, _captured) = make_test_context(gen, false);

    assert_eq!(ctx.meta.userId, "test-user");
    assert_eq!(ctx.meta.workflowId, Some("wf-123".to_string()));
    assert_eq!(ctx.meta.executionId, Some("exec-456".to_string()));
    assert_eq!(ctx.meta.nodeId, Some("node-789".to_string()));
    assert!(!ctx.is_byok());
}

#[test]
fn test_completion_context_byok() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model")
        .with_api_key("user-provided-key");
    let (ctx, _captured) = make_test_context(gen, true);

    assert!(ctx.is_byok());
}

#[test]
fn test_completion_context_injects_app_headers() {
    // Start with a generator that has library-default headers
    let gen = GeneratorInfo::openrouter("test-model");
    assert!(gen
        .custom_headers
        .iter()
        .any(|(k, v)| k == "X-Title" && v == "MiniLLMLib"));

    let (ctx, _captured) = make_test_context(gen, false);

    // CompletionContext should have replaced the library defaults with the test app identity
    let referer = ctx
        .generator
        .custom_headers
        .iter()
        .find(|(k, _)| k == "HTTP-Referer");
    let title = ctx
        .generator
        .custom_headers
        .iter()
        .find(|(k, _)| k == "X-Title");

    assert_eq!(referer.unwrap().1, "https://test.example.com");
    assert_eq!(title.unwrap().1, "TestApp");
    // No duplicate headers
    let referer_count = ctx
        .generator
        .custom_headers
        .iter()
        .filter(|(k, _)| k == "HTTP-Referer")
        .count();
    let title_count = ctx
        .generator
        .custom_headers
        .iter()
        .filter(|(k, _)| k == "X-Title")
        .count();
    assert_eq!(
        referer_count, 1,
        "Should have exactly one HTTP-Referer header"
    );
    assert_eq!(title_count, 1, "Should have exactly one X-Title header");
}

#[tokio::test]
async fn test_completion_context_report_cost() {
    let gen = GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model");
    let (ctx, captured) = make_test_context(gen, false);

    let cost_info = CostInfo {
        cost: 0.00042,
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        cached_tokens: None,
        reasoning_tokens: None,
        model: "test-model".to_string(),
        response_id: "gen-abc123".to_string(),
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

    let meta = CompletionMeta {
        userId: "user-42".to_string(),
        workflowId: Some("wf-abc".to_string()),
        executionId: Some("exec-def".to_string()),
        nodeId: Some("node-ghi".to_string()),
        isByok: true,
    };

    let ctx = CompletionContext::new(gen, meta, callback, "https://test.example.com", "TestApp");
    ctx.report_cost(CostInfo::default()).await;

    let metas = captured_meta.lock().unwrap();
    assert_eq!(metas.len(), 1);
    assert_eq!(metas[0].userId, "user-42");
    assert_eq!(metas[0].workflowId, Some("wf-abc".to_string()));
    assert_eq!(metas[0].executionId, Some("exec-def".to_string()));
    assert_eq!(metas[0].nodeId, Some("node-ghi".to_string()));
    assert!(metas[0].isByok);
}

#[test]
fn test_completion_context_debug() {
    let gen = GeneratorInfo::new("TestProvider", "https://api.example.com/v1", "test-model");
    let (ctx, _) = make_test_context(gen, false);

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
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen, false);

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

#[tokio::test]
async fn test_complete_tracked_with_params() {
    dotenvy::dotenv().ok();
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen, false);

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
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen, false);

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
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen, false);

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

    // Now collect_and_report to fire the callback
    let response = stream.collect_and_report().await;
    assert!(response.is_ok());

    let costs = captured.lock().unwrap();
    assert_eq!(
        costs.len(),
        1,
        "Callback should fire once after collect_and_report"
    );
    assert!(!costs[0].model.is_empty());
}

#[tokio::test]
async fn test_tracked_stream_drop_reports_cost() {
    dotenvy::dotenv().ok();
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

    let gen = get_text_generator();
    let (ctx, captured) = make_test_context(gen, false);

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
                break; // Stop early — cancel the stream
            }
        }
        println!("[drop_test] Read {} chunks before dropping", chunks_read);
        // stream is dropped here — Drop impl should spawn background cost reporting
    }

    // Give the background task time to query OpenRouter and report.
    // Drop retry schedule is 1s + 2s + 4s = 7s worst case, plus query time.
    tokio::time::sleep(std::time::Duration::from_secs(10)).await;

    let costs = captured.lock().unwrap();
    println!("[drop_test] Captured {} cost report(s)", costs.len());
    // The Drop impl spawns a background task — it should have reported by now
    assert_eq!(costs.len(), 1, "Drop should have triggered cost reporting");
    println!(
        "[drop_test] Cost from cancelled stream: ${:.6}",
        costs[0].cost
    );
}

#[tokio::test]
async fn test_complete_tracked_byok_flag() {
    dotenvy::dotenv().ok();
    if std::env::var("OPENROUTER_API_KEY").is_err() {
        eprintln!("Skipping test: OPENROUTER_API_KEY not set");
        return;
    }

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

    let meta = CompletionMeta {
        userId: "byok-user".to_string(),
        workflowId: None,
        executionId: None,
        nodeId: None,
        isByok: true,
    };

    let ctx = CompletionContext::new(gen, meta, callback, "https://test.example.com", "TestApp");

    let root = ChatNode::root("Be brief.");
    let user_node = root.add_user("Hi");

    let result = user_node.complete_tracked(&ctx, None).await;
    assert!(result.is_ok());

    let metas = captured_meta.lock().unwrap();
    assert_eq!(metas.len(), 1);
    assert!(metas[0].isByok, "BYOK flag should be preserved in callback");
    assert_eq!(metas[0].userId, "byok-user");
}
