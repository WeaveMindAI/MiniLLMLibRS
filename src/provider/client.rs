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
    fn build_body_with_usage(
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
        if let Some(max_tokens) = params.max_tokens {
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
