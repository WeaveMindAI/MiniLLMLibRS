//! HTTP client for LLM API requests

use super::response::{parse_completion_response, CompletionResponse};
use super::streaming::StreamingCompletion;
use crate::error::{MiniLLMError, Result};
use crate::generator::{CompletionParameters, GeneratorInfo};
use crate::message::{messages_to_payload, Message};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest_eventsource::EventSource;
use secrecy::ExposeSecret;
use std::time::Duration;

/// HTTP client for making LLM API requests
#[derive(Clone)]
pub struct LLMClient {
    client: reqwest::Client,
}

impl Default for LLMClient {
    fn default() -> Self {
        Self::new()
    }
}

impl LLMClient {
    /// Create a new LLM client with default settings
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("Failed to create HTTP client");

        Self { client }
    }

    /// Create a client with custom timeout
    pub fn with_timeout(timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("Failed to create HTTP client");

        Self { client }
    }

    /// Build headers for a request
    fn build_headers(&self, generator: &GeneratorInfo) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        // Add API key if present
        if let Some(api_key) = &generator.api_key {
            let auth_value = format!("Bearer {}", api_key.expose_secret());
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&auth_value)
                    .map_err(|e| MiniLLMError::Other(format!("Invalid API key format: {}", e)))?,
            );
        }

        // Add custom headers
        for (name, value) in &generator.custom_headers {
            let header_name = HeaderName::try_from(name.as_str()).map_err(|e| {
                MiniLLMError::Other(format!("Invalid header name '{}': {}", name, e))
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|e| {
                MiniLLMError::Other(format!("Invalid header value for '{}': {}", name, e))
            })?;
            headers.insert(header_name, header_value);
        }

        Ok(headers)
    }

    /// Build request body with optional usage tracking
    pub(crate) fn build_body_with_usage(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
        stream: bool,
        include_usage: bool,
    ) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": generator.model,
            "messages": messages_to_payload(messages),
            "stream": stream,
        });

        // Add OpenRouter usage tracking if requested
        if include_usage {
            body["usage"] = serde_json::json!({ "include": true });
        }

        // Add completion parameters
        // Send both: max_completion_tokens (OpenRouter's preferred name) and
        // max_tokens (for non-OpenRouter OpenAI-compatible providers)
        if let Some(max_tokens) = params.max_tokens {
            body["max_completion_tokens"] = serde_json::json!(max_tokens);
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(temperature) = params.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        if let Some(top_p) = params.top_p {
            body["top_p"] = serde_json::json!(top_p);
        }
        if let Some(top_k) = params.top_k {
            body["top_k"] = serde_json::json!(top_k);
        }
        if let Some(frequency_penalty) = params.frequency_penalty {
            body["frequency_penalty"] = serde_json::json!(frequency_penalty);
        }
        if let Some(presence_penalty) = params.presence_penalty {
            body["presence_penalty"] = serde_json::json!(presence_penalty);
        }
        if let Some(repetition_penalty) = params.repetition_penalty {
            body["repetition_penalty"] = serde_json::json!(repetition_penalty);
        }
        if let Some(stop) = &params.stop {
            body["stop"] = serde_json::json!(stop);
        }
        if let Some(seed) = params.seed {
            body["seed"] = serde_json::json!(seed);
        }
        if let Some(response_format) = &params.response_format {
            body["response_format"] = serde_json::json!(response_format);
        }
        if let Some(tools) = &params.tools {
            body["tools"] = serde_json::json!(tools);
        }
        if let Some(tool_choice) = &params.tool_choice {
            body["tool_choice"] = tool_choice.clone();
        }
        if let Some(provider) = &params.provider {
            body["provider"] = serde_json::to_value(provider).unwrap_or_default();
        }
        if let Some(extra) = &params.extra {
            for (key, value) in extra {
                body[key] = value.clone();
            }
        }

        body
    }

    /// Make a non-streaming completion request
    pub async fn complete(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
    ) -> Result<CompletionResponse> {
        self.complete_with_usage_tracking(generator, messages, params, false)
            .await
    }

    /// Make a non-streaming completion request with usage tracking option
    pub async fn complete_with_usage_tracking(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
        include_usage: bool,
    ) -> Result<CompletionResponse> {
        let url = generator.completions_url();
        let headers = self.build_headers(generator)?;
        let body = self.build_body_with_usage(generator, messages, params, false, include_usage);

        tracing::debug!(url = %url, model = %generator.model, include_usage = %include_usage, "Making completion request");

        let response = self
            .client
            .post(&url)
            .headers(headers)
            .json(&body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            tracing::error!(status = %status, error = %error_text, "API request failed");
            return Err(MiniLLMError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        let raw: serde_json::Value = response.json().await?;
        tracing::debug!("Received completion response");

        parse_completion_response(raw)
    }

    /// Make a streaming completion request
    pub async fn complete_streaming(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
    ) -> Result<StreamingCompletion> {
        self.complete_streaming_with_usage(generator, messages, params, false)
            .await
    }

    /// Make a streaming completion request with usage tracking option
    pub async fn complete_streaming_with_usage(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
        include_usage: bool,
    ) -> Result<StreamingCompletion> {
        let url = generator.completions_url();
        let headers = self.build_headers(generator)?;
        let body = self.build_body_with_usage(generator, messages, params, true, include_usage);

        tracing::debug!(url = %url, model = %generator.model, include_usage = %include_usage, "Starting streaming completion");

        // Build the request builder (EventSource needs RequestBuilder, not Request)
        let request_builder = self.client.post(&url).headers(headers).json(&body);

        // Create EventSource from request builder
        let es = EventSource::new(request_builder)
            .map_err(|e| MiniLLMError::Stream(format!("Failed to create event source: {}", e)))?;

        // Generate a unique ID for this stream
        let id = uuid::Uuid::new_v4().to_string();

        Ok(StreamingCompletion::from_event_source(
            es,
            generator.model.clone(),
            id,
        ))
    }

    /// Make a completion request with optional streaming
    pub async fn complete_with_options(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
        stream: bool,
    ) -> Result<CompletionResponse> {
        if stream {
            self.complete_streaming(generator, messages, params)
                .await?
                .collect()
                .await
        } else {
            self.complete_with_usage_tracking(generator, messages, params, false)
                .await
        }
    }
}

/// Global shared client instance
static GLOBAL_CLIENT: std::sync::OnceLock<LLMClient> = std::sync::OnceLock::new();

/// Get the global shared client
pub fn global_client() -> &'static LLMClient {
    GLOBAL_CLIENT.get_or_init(LLMClient::new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::{CompletionParameters, GeneratorInfo, ProviderSettings};
    use crate::message::Message;

    fn test_generator() -> GeneratorInfo {
        GeneratorInfo::new("Test", "https://api.example.com/v1", "test-model")
    }

    fn test_messages() -> Vec<Message> {
        vec![Message::system("You are helpful."), Message::user("Hello")]
    }

    #[test]
    fn test_body_includes_basic_fields() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new()
            .with_temperature(0.5)
            .with_max_tokens(1024);
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);

        assert_eq!(body["model"], "test-model");
        assert_eq!(body["stream"], false);
        assert_eq!(body["temperature"], 0.5);
        assert_eq!(body["max_tokens"], 1024);
        assert!(body.get("usage").is_none());
    }

    #[test]
    fn test_body_includes_usage_when_requested() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new();
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, true);

        assert_eq!(body["usage"]["include"], true);
    }

    #[test]
    fn test_body_includes_all_sampling_params() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters {
            max_tokens: Some(512),
            temperature: Some(0.9),
            top_p: Some(0.95),
            top_k: Some(40),
            frequency_penalty: Some(0.5),
            presence_penalty: Some(0.3),
            repetition_penalty: Some(1.2),
            stop: Some(vec!["END".to_string()]),
            seed: Some(42),
            stream: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            provider: None,
            extra: None,
        };
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);

        assert_eq!(body["max_tokens"], 512);
        let temp = body["temperature"].as_f64().unwrap();
        assert!((temp - 0.9).abs() < 1e-6, "temperature: {}", temp);
        let top_p = body["top_p"].as_f64().unwrap();
        assert!((top_p - 0.95).abs() < 1e-6, "top_p: {}", top_p);
        assert_eq!(body["top_k"], 40);
        assert_eq!(body["frequency_penalty"], 0.5);
        let presence = body["presence_penalty"].as_f64().unwrap();
        assert!(
            (presence - 0.3).abs() < 1e-6,
            "presence_penalty: {}",
            presence
        );
        let rep = body["repetition_penalty"].as_f64().unwrap();
        assert!((rep - 1.2).abs() < 1e-6, "repetition_penalty: {}", rep);
        assert_eq!(body["stop"][0], "END");
        assert_eq!(body["seed"], 42);
    }

    #[test]
    fn test_body_includes_provider_settings() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new()
            .with_provider(ProviderSettings::new().deny_data_collection());
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);

        assert_eq!(body["provider"]["data_collection"], "deny");
    }

    #[test]
    fn test_body_includes_provider_order_and_sort() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new().with_provider(
            ProviderSettings::new()
                .with_order(vec!["Anthropic".to_string(), "OpenAI".to_string()])
                .sort_by_latency()
                .with_fallbacks(false),
        );
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);

        assert_eq!(body["provider"]["order"][0], "Anthropic");
        assert_eq!(body["provider"]["order"][1], "OpenAI");
        assert_eq!(body["provider"]["sort"], "latency");
        assert_eq!(body["provider"]["allow_fallbacks"], false);
    }

    #[test]
    fn test_body_includes_extra_params() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new()
            .with_extra("custom_field", serde_json::json!("custom_value"))
            .with_extra("custom_number", serde_json::json!(42));
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);

        assert_eq!(body["custom_field"], "custom_value");
        assert_eq!(body["custom_number"], 42);
    }

    #[test]
    fn test_body_includes_tools_and_tool_choice() {
        let client = LLMClient::new();
        let gen = test_generator();
        let tool = serde_json::json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": { "type": "object", "properties": {} }
            }
        });
        let params = CompletionParameters {
            tools: Some(vec![tool.clone()]),
            tool_choice: Some(serde_json::json!("auto")),
            ..CompletionParameters::new()
        };
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);

        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn test_body_includes_response_format() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new().with_json_response();
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);

        assert_eq!(body["response_format"]["type"], "json_object");
    }

    #[test]
    fn test_body_omits_none_fields() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters {
            max_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            repetition_penalty: None,
            stop: None,
            seed: None,
            stream: None,
            response_format: None,
            tools: None,
            tool_choice: None,
            provider: None,
            extra: None,
        };
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);

        // Only model, messages, stream should be present
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
        assert!(body.get("top_p").is_none());
        assert!(body.get("top_k").is_none());
        assert!(body.get("frequency_penalty").is_none());
        assert!(body.get("presence_penalty").is_none());
        assert!(body.get("repetition_penalty").is_none());
        assert!(body.get("stop").is_none());
        assert!(body.get("seed").is_none());
        assert!(body.get("response_format").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none());
        assert!(body.get("provider").is_none());
    }

    #[test]
    fn test_body_stream_flag() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new();

        let body_no_stream =
            client.build_body_with_usage(&gen, &test_messages(), &params, false, false);
        assert_eq!(body_no_stream["stream"], false);

        let body_stream =
            client.build_body_with_usage(&gen, &test_messages(), &params, true, false);
        assert_eq!(body_stream["stream"], true);
    }

    #[test]
    fn test_body_messages_serialization() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new();
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Hello");
    }

    #[test]
    fn test_custom_timeout_creates_working_client() {
        let client = LLMClient::with_timeout(Duration::from_secs(5));
        let gen = test_generator();
        let params = CompletionParameters::new();
        // Verify the client can still build bodies (it's functional)
        let body = client.build_body_with_usage(&gen, &test_messages(), &params, false, false);
        assert_eq!(body["model"], "test-model");
    }

    #[tokio::test]
    async fn test_timeout_is_respected() {
        // Start a TCP listener that accepts but never responds
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Accept connections but never send a response
        tokio::spawn(async move {
            loop {
                let (_socket, _) = listener.accept().await.unwrap();
                // Hold the connection open, never respond
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });

        let client = LLMClient::with_timeout(Duration::from_secs(1));
        let gen = GeneratorInfo::new("Test", &format!("http://{}", addr), "test-model")
            .with_api_key("fake-key");
        let messages = vec![Message::user("Hello")];
        let params = CompletionParameters::new();

        let start = std::time::Instant::now();
        let result = client.complete(&gen, &messages, &params).await;
        let elapsed = start.elapsed();

        // Should fail (timeout or connection error)
        assert!(result.is_err(), "Expected timeout error, got success");
        // Should complete in roughly 1-3 seconds, not 120
        assert!(
            elapsed.as_secs() < 5,
            "Timeout took too long: {:?} (expected ~1s)",
            elapsed
        );
    }
}
