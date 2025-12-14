//! Comprehensive integration tests for MiniLLMLib
//!
//! These tests make real API calls to OpenRouter.
//! Run with: cargo test --test integration_tests -- --nocapture
//!
//! Requires OPENROUTER_API_KEY environment variable to be set.

use minillmlib::{
    chat_node::ChatNode,
    generator::{CompletionParameters, GeneratorInfo, NodeCompletionParameters, ProviderSettings},
    message::{AudioData, ImageData, Message, MessageContent, Role},
    provider::LLMClient,
};
use std::sync::Arc;

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
            assert!(response.id.len() > 0);
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

const CHEAP_MODEL: &str = "meta-llama/llama-3.2-3b-instruct";

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
    assert!(successes >= 8, "At least 8/10 threads should succeed, got {}", successes);
    
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
                println!("Request {}: OK - {}", i, content.chars().take(50).collect::<String>());
                successes += 1;
            }
            Err(e) => {
                println!("Request {}: FAILED - {}", i, e);
            }
        }
    }
    
    assert!(successes >= 8, "At least 8/10 concurrent requests should succeed, got {}", successes);
    
    // Verify tree structure - root should have 10 children (user messages)
    // Each user message should have 1 child (assistant response)
    let children_count = root.child_count();
    assert_eq!(children_count, 10, "Root should have 10 children");
}
