//! HTTP client for LLM API requests

use super::response::{preview_str, CompletionResponse};
use super::streaming::StreamingCompletion;
use crate::error::{MiniLLMError, Result};
use crate::generator::{CompletionParameters, GeneratorInfo};
use crate::message::Message;
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use reqwest_eventsource::EventSource;
use std::time::Duration;

/// Map a transport error from `send()` into a typed error, surfacing timeouts
/// as the distinct `Timeout` variant (consumers and retry logic discriminate on
/// it) rather than collapsing every transport failure into `Http`.
fn map_send_error(e: reqwest::Error) -> MiniLLMError {
    if e.is_timeout() {
        MiniLLMError::Timeout
    } else {
        MiniLLMError::Http(e)
    }
}

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
    /// Create a new LLM client with default settings.
    ///
    /// The 600s timeout is a connection-pool-wide backstop; individual requests
    /// override it via the per-request timeout argument, so a single pooled
    /// client serves both default and custom-timeout requests.
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(600))
            .build()
            .expect("Failed to create HTTP client");

        Self { client }
    }

    /// Build headers for a request.
    ///
    /// The provider owns BOTH the auth headers (OpenAI `Authorization: Bearer`,
    /// Anthropic `x-api-key`/bearer + version) and the app-attribution headers
    /// (e.g. OpenRouter's `HTTP-Referer`/`X-Title`); the generator supplies the
    /// `Auth` strategy and the app identity. Custom headers are applied last and
    /// win on a key collision.
    fn build_headers(&self, generator: &GeneratorInfo) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();

        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

        let auth = generator.provider.auth_headers(&generator.auth)?;
        let attribution = generator
            .provider
            .attribution_headers(generator.app_attribution.as_ref());
        for (name, value) in auth
            .iter()
            .chain(attribution.iter())
            .chain(generator.custom_headers.iter())
        {
            let header_name = HeaderName::try_from(name.as_str()).map_err(|e| {
                MiniLLMError::InvalidParameter(format!("Invalid header name '{}': {}", name, e))
            })?;
            let header_value = HeaderValue::from_str(value).map_err(|e| {
                MiniLLMError::InvalidParameter(format!(
                    "Invalid header value for '{}': {}",
                    name, e
                ))
            })?;
            headers.insert(header_name, header_value);
        }

        Ok(headers)
    }

    /// Build the request body via the generator's provider (the provider owns the
    /// wire shape: OpenAI `/chat/completions` vs Anthropic `/v1/messages`).
    pub(crate) fn build_body_with_usage(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
        stream: bool,
        include_usage: bool,
    ) -> Result<serde_json::Value> {
        generator
            .provider
            .build_request(&generator.model, messages, params, stream, include_usage)
    }

    /// Make a non-streaming completion request
    pub async fn complete(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
    ) -> Result<CompletionResponse> {
        self.complete_with_usage_tracking(generator, messages, params, false, None)
            .await
    }

    /// Make a non-streaming completion request with usage tracking option.
    ///
    /// `timeout` overrides the pooled client's default timeout for this single
    /// request, so callers needing a custom timeout reuse the shared connection
    /// pool instead of building a throwaway client.
    pub async fn complete_with_usage_tracking(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
        include_usage: bool,
        timeout: Option<Duration>,
    ) -> Result<CompletionResponse> {
        let url = generator.completions_url();
        let headers = self.build_headers(generator)?;
        let body = self.build_body_with_usage(generator, messages, params, false, include_usage)?;

        tracing::debug!(url = %url, model = %generator.model, include_usage = %include_usage, "Making completion request");

        let mut request = self.client.post(&url).headers(headers).json(&body);
        if let Some(timeout) = timeout {
            request = request.timeout(timeout);
        }
        let response = request.send().await.map_err(map_send_error)?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            tracing::error!(status = %status, error = %error_text, "API request failed");
            return Err(MiniLLMError::Api {
                status: status.as_u16(),
                message: error_text,
            });
        }

        // A read failure here is a transport error → MiniLLMError::Http via map_send_error.
        let response_bytes = response.bytes().await.map_err(map_send_error)?;

        let raw: serde_json::Value = serde_json::from_slice(&response_bytes).map_err(|e| {
            let preview = preview_str(&String::from_utf8_lossy(&response_bytes));
            tracing::error!(
                "Failed to parse LLM response as JSON: {}. Body: {}",
                e,
                preview
            );
            MiniLLMError::MalformedResponse(format!("non-JSON body: {}", preview))
        })?;

        tracing::debug!("Received completion response");

        generator.provider.parse_response(raw)
    }

    /// Make a streaming completion request
    pub async fn complete_streaming(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
    ) -> Result<StreamingCompletion> {
        self.complete_streaming_with_usage(generator, messages, params, false, None)
            .await
    }

    /// Make a streaming completion request with usage tracking option.
    ///
    /// `timeout` is an **idle** timeout (max silence between SSE events), not a
    /// total-duration cap: a long but live generation must not be killed, but a
    /// dead connection that goes silent should fail loudly. It is enforced
    /// per-event inside the stream, not via the request builder.
    pub async fn complete_streaming_with_usage(
        &self,
        generator: &GeneratorInfo,
        messages: &[Message],
        params: &CompletionParameters,
        include_usage: bool,
        idle_timeout: Option<Duration>,
    ) -> Result<StreamingCompletion> {
        let url = generator.completions_url();
        let headers = self.build_headers(generator)?;
        let body = self.build_body_with_usage(generator, messages, params, true, include_usage)?;

        tracing::debug!(url = %url, model = %generator.model, include_usage = %include_usage, "Starting streaming completion");

        // Note: deliberately NOT `.timeout()` on the request builder, which caps
        // total duration, which wrongly kills long legitimate streams. The idle
        // timeout is applied per-event by the stream task instead.
        let request_builder = self.client.post(&url).headers(headers).json(&body);

        let es = EventSource::new(request_builder)
            .map_err(|e| MiniLLMError::Stream(format!("Failed to create event source: {}", e)))?;

        // Wait for a trailing usage chunk only if the PROVIDER will actually send
        // one (not merely because the caller asked to track cost): a provider with
        // no usage opt-in would otherwise wedge the stream until the idle timeout.
        let expect_usage = generator.provider.emits_stream_usage(include_usage);

        // The real generation id arrives on the SSE chunks (the provider's
        // `gen-...`); the stream starts with an empty id and adopts the first one
        // it sees, so out-of-band cost resolution targets the real generation.
        Ok(StreamingCompletion::from_event_source(
            es,
            generator.model.clone(),
            expect_usage,
            idle_timeout,
            generator.provider.clone(),
        ))
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
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

        assert_eq!(body["model"], "test-model");
        assert_eq!(body["stream"], false);
        assert_eq!(body["temperature"], 0.5);
        // Default generator uses the modern max_completion_tokens key, not max_tokens.
        assert_eq!(body["max_completion_tokens"], 1024);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("usage").is_none());
    }

    #[test]
    fn test_body_legacy_token_field() {
        use crate::provider::GenericProvider;
        use std::sync::Arc;
        let client = LLMClient::new();
        let gen = test_generator().with_provider(Arc::new(GenericProvider {
            legacy_token_limit: true,
        }));
        let params = CompletionParameters::new().with_max_tokens(1024);
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

        // Legacy generator uses max_tokens and never emits max_completion_tokens.
        assert_eq!(body["max_tokens"], 1024);
        assert!(body.get("max_completion_tokens").is_none());
    }

    #[test]
    fn test_body_includes_usage_when_requested() {
        use crate::provider::OpenRouterProvider;
        use std::sync::Arc;
        let client = LLMClient::new();
        // The usage opt-in is provider-specific; OpenRouter uses usage:{include:true}.
        let gen = test_generator().with_provider(Arc::new(OpenRouterProvider));
        let params = CompletionParameters::new();
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, true)
            .unwrap();

        assert_eq!(body["usage"]["include"], true);
    }

    #[test]
    fn test_openai_streaming_usage_opt_in() {
        use crate::provider::OpenAiProvider;
        use std::sync::Arc;
        let client = LLMClient::new();
        let gen = test_generator().with_provider(Arc::new(OpenAiProvider));
        let params = CompletionParameters::new();
        // OpenAI opts into streaming usage via stream_options, only when streaming.
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, true, true)
            .unwrap();
        assert_eq!(body["stream_options"]["include_usage"], true);
        // Non-streaming: no opt-in needed (usage always present).
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, true)
            .unwrap();
        assert!(body.get("stream_options").is_none());
    }

    #[test]
    fn test_extra_param_collision_fails_loudly() {
        let client = LLMClient::new();
        let gen = test_generator();
        // An `extra` key colliding with a typed param must error, not silently clobber.
        let params = CompletionParameters::new().with_extra("temperature", serde_json::json!(2.0));
        assert!(client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .is_err());
        // ...and one colliding with a request-owned key.
        let params = CompletionParameters::new().with_extra("model", serde_json::json!("x"));
        assert!(client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .is_err());
        // A genuinely-extra key is fine.
        let params = CompletionParameters::new().with_extra("logit_bias", serde_json::json!({}));
        assert!(client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .is_ok());
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
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning: None,
            extra: None,
        };
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

        assert_eq!(body["max_completion_tokens"], 512);
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
            .with_openrouter_routing(ProviderSettings::new().deny_data_collection());
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

        assert_eq!(body["provider"]["data_collection"], "deny");
    }

    #[test]
    fn test_body_includes_provider_order_and_sort() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new().with_openrouter_routing(
            ProviderSettings::new()
                .with_order(vec!["Anthropic".to_string(), "OpenAI".to_string()])
                .sort_by_latency()
                .with_fallbacks(false),
        );
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

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
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

        assert_eq!(body["custom_field"], "custom_value");
        assert_eq!(body["custom_number"], 42);
    }

    #[test]
    fn test_body_includes_tools_and_tool_choice() {
        use crate::tools::{ToolChoice, ToolDefinition};
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new()
            .with_tool(ToolDefinition::new(
                "get_weather",
                "Get the weather",
                serde_json::json!({ "type": "object", "properties": {} }),
            ))
            .with_tool_choice(ToolChoice::Auto)
            .with_parallel_tool_calls(false);
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(body["tool_choice"], "auto");
        assert_eq!(body["parallel_tool_calls"], false);
    }

    #[test]
    fn test_body_includes_response_format() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new().with_json_response();
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

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
            response_format: None,
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning: None,
            extra: None,
        };
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

        // Only model, messages, stream should be present
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("max_completion_tokens").is_none());
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

        let body_no_stream = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();
        assert_eq!(body_no_stream["stream"], false);

        let body_stream = client
            .build_body_with_usage(&gen, &test_messages(), &params, true, false)
            .unwrap();
        assert_eq!(body_stream["stream"], true);
    }

    #[test]
    fn test_body_messages_serialization() {
        let client = LLMClient::new();
        let gen = test_generator();
        let params = CompletionParameters::new();
        let body = client
            .build_body_with_usage(&gen, &test_messages(), &params, false, false)
            .unwrap();

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "You are helpful.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "Hello");
    }

    #[tokio::test]
    async fn test_per_request_timeout_is_respected() {
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

        // The pooled client has a 600s default; the per-request timeout overrides it.
        let client = LLMClient::new();
        let gen = GeneratorInfo::new("Test", format!("http://{}", addr), "test-model")
            .with_api_key("fake-key");
        let messages = vec![Message::user("Hello")];
        let params = CompletionParameters::new();

        let start = std::time::Instant::now();
        let result = client
            .complete_with_usage_tracking(
                &gen,
                &messages,
                &params,
                false,
                Some(Duration::from_secs(1)),
            )
            .await;
        let elapsed = start.elapsed();

        // Should fail (timeout or connection error)
        assert!(result.is_err(), "Expected timeout error, got success");
        // Should complete in roughly 1-3 seconds, not 600
        assert!(
            elapsed.as_secs() < 5,
            "Timeout took too long: {:?} (expected ~1s)",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_streaming_idle_timeout_fires_on_silence() {
        // A server that accepts but never sends an SSE event must trip the idle
        // timeout (max silence between chunks), not hang to the pool timeout.
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            loop {
                let (_socket, _) = listener.accept().await.unwrap();
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });

        let client = LLMClient::new();
        let gen = GeneratorInfo::new("Test", format!("http://{}", addr), "test-model")
            .with_api_key("fake-key");
        let messages = vec![Message::user("Hello")];
        let params = CompletionParameters::new();

        let mut stream = client
            .complete_streaming_with_usage(
                &gen,
                &messages,
                &params,
                false,
                Some(Duration::from_secs(1)),
            )
            .await
            .expect("event source should be created");

        let start = std::time::Instant::now();
        // First chunk should be a loud error (timeout/connect), arriving fast.
        let first = stream.next_chunk().await;
        let elapsed = start.elapsed();
        assert!(
            matches!(first, Some(Err(_))),
            "expected a loud error on idle silence, got {:?}",
            first.map(|r| r.map(|c| c.delta))
        );
        assert!(
            elapsed.as_secs() < 5,
            "idle timeout took too long: {:?} (expected ~1s)",
            elapsed
        );
    }

    #[test]
    fn attribution_headers_are_provider_specific() {
        use crate::provider::{OpenAiProvider, OpenRouterProvider};
        use std::sync::Arc;
        let client = LLMClient::new();

        // OpenRouter turns app attribution into HTTP-Referer / X-Title.
        let gen = test_generator()
            .with_provider(Arc::new(OpenRouterProvider))
            .with_app_attribution("https://app.example", "MyApp");
        let headers = client.build_headers(&gen).unwrap();
        assert_eq!(headers.get("HTTP-Referer").unwrap(), "https://app.example");
        assert_eq!(headers.get("X-Title").unwrap(), "MyApp");

        // A non-OpenRouter provider injects NO attribution headers, even with an
        // app identity set (attribution is provider-specific wire).
        let gen = test_generator()
            .with_provider(Arc::new(OpenAiProvider))
            .with_app_attribution("https://app.example", "MyApp");
        let headers = client.build_headers(&gen).unwrap();
        assert!(headers.get("HTTP-Referer").is_none());
        assert!(headers.get("X-Title").is_none());
    }
}
