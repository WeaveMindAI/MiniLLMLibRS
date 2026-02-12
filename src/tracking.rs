//! CompletionContext - Enforced cost tracking wrapper for LLM completions
//!
//! CompletionContext wraps a GeneratorInfo and guarantees that every completion
//! call reports cost information via a callback. This is the mechanism WeaveMind
//! uses to track AI usage costs.
//!
//! External library users can still use `ChatNode.complete()` directly with a
//! raw GeneratorInfo (no cost tracking). But WeaveMind nodes must use
//! `ChatNode.complete_tracked()` which requires a CompletionContext.

use crate::generator::GeneratorInfo;
use crate::provider::CostInfo;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Async cost callback that can write to a database or HTTP endpoint.
/// Returns a future so the caller can await the write (or fire-and-forget via spawn).
pub type AsyncCostCallback = Arc<
    dyn Fn(CostInfo, CompletionMeta) -> Pin<Box<dyn Future<Output = ()> + Send>>
        + Send
        + Sync,
>;

/// Metadata about the completion context (passed to the cost callback alongside CostInfo)
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[allow(non_snake_case)]
pub struct CompletionMeta {
    pub userId: String,
    pub workflowId: Option<String>,
    pub executionId: Option<String>,
    pub nodeId: Option<String>,
    pub isByok: bool,
}

/// Wraps a GeneratorInfo with enforced cost tracking.
///
/// The runtime creates this; node authors receive it, not a raw GeneratorInfo.
/// Cost tracking is structurally enforced: `complete_tracked()` requires this type.
pub struct CompletionContext {
    pub generator: GeneratorInfo,
    pub meta: CompletionMeta,
    callback: AsyncCostCallback,
    /// App URL sent as HTTP-Referer to OpenRouter for app attribution/rankings
    pub app_url: String,
    /// App display name sent as X-Title to OpenRouter for app attribution/rankings
    pub app_title: String,
}

impl CompletionContext {
    pub fn new(
        mut generator: GeneratorInfo,
        meta: CompletionMeta,
        callback: AsyncCostCallback,
        app_url: impl Into<String>,
        app_title: impl Into<String>,
    ) -> Self {
        let app_url = app_url.into();
        let app_title = app_title.into();

        // Override any existing HTTP-Referer / X-Title headers on the generator
        // so OpenRouter attributes usage to the calling application, not the library.
        generator.custom_headers.retain(|(name, _)| {
            !name.eq_ignore_ascii_case("HTTP-Referer") && !name.eq_ignore_ascii_case("X-Title")
        });
        generator = generator
            .with_header("HTTP-Referer", &app_url)
            .with_header("X-Title", &app_title);

        Self {
            generator,
            meta,
            callback,
            app_url,
            app_title,
        }
    }

    /// Detect whether this is a BYOK (Bring Your Own Key) setup.
    /// BYOK = the generator has a user-provided API key that differs from the
    /// platform key (OPENROUTER_API_KEY env var).
    pub fn is_byok(&self) -> bool {
        self.meta.isByok
    }

    /// Fire the cost callback. Called internally by complete_tracked().
    /// Also available publicly for testing or manual cost reporting.
    pub async fn report_cost(&self, cost_info: CostInfo) {
        let fut = (self.callback)(cost_info, self.meta.clone());
        fut.await;
    }

    /// Query OpenRouter's /api/v1/generation endpoint to get cost for a
    /// generation that may have been cancelled mid-stream.
    /// Returns CostInfo if the generation is found, None otherwise.
    pub(crate) async fn query_generation_cost(
        &self,
        generation_id: &str,
    ) -> Option<CostInfo> {
        query_generation_cost_static(&self.generator, generation_id).await
    }
}

/// A streaming completion wrapper that reports cost when finished or dropped.
///
/// If the stream completes normally, cost is extracted from the final usage chunk.
/// If the stream is cancelled (dropped before completion), the Drop impl spawns
/// a background task to query OpenRouter's /generation endpoint for the actual cost.
pub struct TrackedStream {
    inner: crate::provider::StreamingCompletion,
    /// Cloned context data needed for cost reporting after the stream ends
    callback: AsyncCostCallback,
    meta: CompletionMeta,
    generator: GeneratorInfo,
    /// Set to true once cost has been reported (prevents double-reporting on drop)
    cost_reported: bool,
}

impl TrackedStream {
    pub(crate) fn new(
        inner: crate::provider::StreamingCompletion,
        ctx: &CompletionContext,
    ) -> Self {
        Self {
            inner,
            callback: ctx.callback.clone(),
            meta: ctx.meta.clone(),
            generator: ctx.generator.clone(),
            cost_reported: false,
        }
    }

    /// Get the next chunk from the underlying stream.
    pub async fn next_chunk(&mut self) -> Option<crate::error::Result<crate::provider::StreamChunk>> {
        self.inner.next_chunk().await
    }

    /// Collect all chunks, report cost, and return the final CompletionResponse.
    pub async fn collect_and_report(&mut self) -> crate::error::Result<crate::provider::CompletionResponse> {
        // Drain the stream
        while let Some(result) = self.inner.next_chunk().await {
            result?;
        }

        let response = crate::provider::CompletionResponse {
            id: self.inner_id().to_string(),
            model: self.inner_model().to_string(),
            content: self.inner.accumulated().to_string(),
            finish_reason: None, // Already consumed
            usage: self.inner.usage().cloned(),
            tool_calls: None,
            raw_response: None,
        };

        // Report cost from usage if available
        let cost_info = if let Some(usage) = &response.usage {
            CostInfo {
                cost: usage.cost.unwrap_or(0.0),
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
                cached_tokens: usage.cached_tokens,
                reasoning_tokens: usage.reasoning_tokens,
                model: response.model.clone(),
                response_id: response.id.clone(),
            }
        } else if !response.id.is_empty() {
            // No usage in stream — query generation endpoint
            tracing::info!(
                "No usage in stream for {}, querying generation endpoint",
                response.id
            );
            query_generation_cost_static(&self.generator, &response.id)
                .await
                .unwrap_or_default()
        } else {
            CostInfo::default()
        };

        let fut = (self.callback)(cost_info, self.meta.clone());
        fut.await;
        self.cost_reported = true;

        Ok(response)
    }

    /// Check if the stream has finished
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Get accumulated content so far
    pub fn accumulated(&self) -> &str {
        self.inner.accumulated()
    }

    fn inner_id(&self) -> &str {
        self.inner.id()
    }

    fn inner_model(&self) -> &str {
        self.inner.model()
    }
}

impl Drop for TrackedStream {
    fn drop(&mut self) {
        if self.cost_reported {
            return;
        }

        // Stream was dropped before collect_and_report() — likely cancelled.
        // Check if we have usage from partial consumption.
        let cost_info_from_usage = self.inner.usage().map(|usage| CostInfo {
            cost: usage.cost.unwrap_or(0.0),
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cached_tokens: usage.cached_tokens,
            reasoning_tokens: usage.reasoning_tokens,
            model: self.inner_model().to_string(),
            response_id: self.inner_id().to_string(),
        });

        let callback = self.callback.clone();
        let meta = self.meta.clone();
        let generator = self.generator.clone();
        let response_id = self.inner_id().to_string();

        // Spawn a background task to report cost.
        // Guard against no tokio runtime (e.g., during shutdown).
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::warn!(
                "TrackedStream dropped outside tokio runtime — cannot report cost for {}",
                response_id
            );
            return;
        };

        handle.spawn(async move {
            let cost_info = if let Some(info) = cost_info_from_usage {
                info
            } else if !response_id.is_empty() {
                // Query OpenRouter for the actual cost of this cancelled generation.
                // Retry with backoff — OpenRouter may not have finalized yet.
                tracing::info!(
                    "Stream cancelled for {}, querying generation cost",
                    response_id
                );
                let mut result = None;
                for delay_secs in [1, 2, 4] {
                    tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
                    if let Some(info) = query_generation_cost_static(&generator, &response_id).await {
                        result = Some(info);
                        break;
                    }
                    tracing::debug!(
                        "Generation {} not found yet, retrying in {}s",
                        response_id, delay_secs * 2
                    );
                }
                result.unwrap_or_default()
            } else {
                CostInfo::default()
            };

            let fut = (callback)(cost_info, meta);
            fut.await;
        });
    }
}

/// Standalone function to query generation cost (used by both CompletionContext and TrackedStream Drop)
async fn query_generation_cost_static(
    generator: &GeneratorInfo,
    generation_id: &str,
) -> Option<CostInfo> {
    use secrecy::ExposeSecret;

    let api_key = generator.api_key.as_ref()?;
    let encoded_id = url::form_urlencoded::byte_serialize(generation_id.as_bytes())
        .collect::<String>();
    let url = format!(
        "https://openrouter.ai/api/v1/generation?id={}",
        encoded_id
    );

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .header(
            "Authorization",
            format!("Bearer {}", api_key.expose_secret()),
        )
        .send()
        .await
        .ok()?;

    if !response.status().is_success() {
        tracing::warn!(
            "Failed to query generation cost for {}: {}",
            generation_id,
            response.status()
        );
        return None;
    }

    let json: serde_json::Value = response.json().await.ok()?;
    let data = json.get("data")?;

    let total_cost = data["total_cost"].as_f64().unwrap_or(0.0);
    let prompt_tokens = data["tokens_prompt"].as_u64().unwrap_or(0) as u32;
    let completion_tokens = data["tokens_completion"].as_u64().unwrap_or(0) as u32;
    let model = data["model"].as_str().unwrap_or("").to_string();
    let gen_id = data["id"].as_str().unwrap_or(generation_id).to_string();

    Some(CostInfo {
        cost: total_cost,
        prompt_tokens,
        completion_tokens,
        total_tokens: prompt_tokens + completion_tokens,
        cached_tokens: data["native_tokens_cached"].as_u64().map(|v| v as u32),
        reasoning_tokens: data["native_tokens_reasoning"]
            .as_u64()
            .map(|v| v as u32),
        model,
        response_id: gen_id,
    })
}

impl std::fmt::Debug for CompletionContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CompletionContext")
            .field("generator", &self.generator.name)
            .field("model", &self.generator.model)
            .field("meta", &self.meta)
            .field("is_byok", &self.is_byok())
            .finish()
    }
}
