//! CompletionContext - Enforced cost tracking wrapper for LLM completions
//!
//! CompletionContext wraps a GeneratorInfo and guarantees that every completion
//! call reports cost information via a callback.
//!
//! Users can still use `ChatNode.complete()` directly with a raw GeneratorInfo
//! (no cost tracking). For tracked usage, use `ChatNode.complete_tracked()`
//! which requires a CompletionContext with opaque metadata passed through
//! to the cost callback.

use crate::generator::GeneratorInfo;
use crate::provider::CostInfo;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

/// Async cost callback that can write to a database or HTTP endpoint.
/// Returns a future so the caller can await the write (or fire-and-forget via spawn).
pub type AsyncCostCallback = Arc<
    dyn Fn(CostInfo, serde_json::Value) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync,
>;

/// Opaque metadata passed through to the cost callback.
/// The library never inspects this, consumers define its shape.
pub type CompletionMeta = serde_json::Value;

/// Wraps a GeneratorInfo with enforced cost tracking.
///
/// The runtime creates this; node authors receive it, not a raw GeneratorInfo.
/// Cost tracking is structurally enforced: `complete_tracked()` requires this type.
pub struct CompletionContext {
    pub generator: GeneratorInfo,
    pub meta: CompletionMeta,
    callback: AsyncCostCallback,
}

impl CompletionContext {
    pub fn new(
        generator: GeneratorInfo,
        meta: CompletionMeta,
        callback: AsyncCostCallback,
        app_url: impl Into<String>,
        app_title: impl Into<String>,
    ) -> Self {
        // Set the calling app's attribution identity on the generator; its
        // provider turns that into whatever headers it uses (e.g. OpenRouter's
        // HTTP-Referer/X-Title). The context no longer hardcodes provider headers.
        let generator = generator.with_app_attribution(app_url, app_title);
        Self {
            generator,
            meta,
            callback,
        }
    }

    /// Detect whether this is a BYOK (Bring Your Own Key) setup.
    /// Reads "isByok" from the metadata JSON (defaults to false if absent).
    pub fn is_byok(&self) -> bool {
        self.meta
            .get("isByok")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }

    /// Fire the cost callback. Called internally by complete_tracked().
    /// Also available publicly for testing or manual cost reporting.
    pub async fn report_cost(&self, cost_info: CostInfo) {
        let fut = (self.callback)(cost_info, self.meta.clone());
        fut.await;
    }

    /// Derive the cost for a completed response (typed usage, or backoff
    /// generation-cost query when absent). The single decision shared with the
    /// streaming path; reports `Unknown` rather than a fake $0 when unresolvable.
    pub(crate) async fn cost_for_response(
        &self,
        response: &crate::provider::CompletionResponse,
    ) -> CostInfo {
        cost_for_response(&self.generator, response).await
    }
}

/// A streaming completion wrapper that reports cost when finished or cancelled.
///
/// Normal completion: cost comes from the final usage chunk (via the generator's
/// provider accounting). Cancellation: call [`cancel`](TrackedStream::cancel) to
/// settle cost reliably (it awaits the resolution); a bare drop falls back to a
/// best-effort detached task that resolves cost out-of-band (e.g. OpenRouter's
/// `/generation` query) and may be lost on runtime shutdown (see the `Drop` impl).
pub struct TrackedStream {
    inner: crate::provider::StreamingCompletion,
    /// Cloned context data needed for cost reporting after the stream ends
    callback: AsyncCostCallback,
    meta: CompletionMeta,
    generator: GeneratorInfo,
    /// Cost has been reported (`report_cost`/`cancel`): suppresses the Drop
    /// fallback and makes `report_cost` idempotent.
    cost_reported: bool,
    /// The caller explicitly rejected this completion (`reject`): Drop books
    /// nothing. The ONLY way to legitimately drop a `TrackedStream` without
    /// booking cost; otherwise a dropped, un-reported stream books (so a
    /// forgotten `report_cost` can't silently lose an accepted generation's cost).
    rejected: bool,
}

/// How a [`TrackedStream::collect_or_cancel`] drain ended. Cost reporting has
/// already happened by the time the caller holds one (booked on `Finished`,
/// resolved out-of-band on `Interrupted`, nothing on `Failed`).
#[derive(Debug)]
pub enum CollectOutcome {
    /// The stream ran to its end; the full response, cost booked from usage.
    Finished(crate::provider::CompletionResponse),
    /// `interrupt` fired first; the stream was cancelled and the actual cost
    /// resolved out-of-band.
    Interrupted,
    /// The transport failed mid-stream; nothing was booked.
    Failed(crate::error::MiniLLMError),
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
            rejected: false,
        }
    }

    /// Get the next chunk from the underlying stream.
    pub async fn next_chunk(
        &mut self,
    ) -> Option<crate::error::Result<crate::provider::StreamChunk>> {
        self.inner.next_chunk().await
    }

    /// Drain until the stream finishes or `interrupt` fires, whichever comes
    /// first, with the cost reported through the context's callback on every
    /// outcome: a finished stream books from its usage, an interrupted one is
    /// cancelled (the actual cost resolves out-of-band, e.g. OpenRouter's
    /// generation ledger, which bills the full generation whether the client
    /// hangs up or not), and a transport-errored one books nothing (not an
    /// accepted generation). The one-call shape for a consumer with a kill
    /// switch: racing the chunks yourself and forgetting `cancel().await` on
    /// the kill path would silently lose the interrupted call's cost.
    ///
    /// Unlike [`collect`](Self::collect) + [`report_cost`](Self::report_cost),
    /// the finished cost is booked BEFORE the caller sees the content: use
    /// this when the call must be paid for regardless of content quality
    /// (metering), not when the caller may still [`reject`](Self::reject).
    pub async fn collect_or_cancel(
        mut self,
        interrupt: impl std::future::Future<Output = ()>,
    ) -> CollectOutcome {
        tokio::pin!(interrupt);
        loop {
            tokio::select! {
                chunk = self.next_chunk() => match chunk {
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        // A failed stream is not an accepted generation:
                        // cancel books nothing, loudly.
                        self.cancel().await;
                        return CollectOutcome::Failed(e);
                    }
                    None => break,
                },
                _ = &mut interrupt => {
                    self.cancel().await;
                    return CollectOutcome::Interrupted;
                }
            }
        }
        let response = match self.collect().await {
            Ok(response) => response,
            Err(e) => {
                self.cancel().await;
                return CollectOutcome::Failed(e);
            }
        };
        self.report_cost(&response).await;
        CollectOutcome::Finished(response)
    }

    /// Drain the stream and return the typed response. Does NOT report cost, so
    /// the caller can post-process (and reject empty/invalid) the content before
    /// any cost is booked, mirroring the non-streaming order. Call
    /// [`report_cost`](Self::report_cost) afterwards.
    pub async fn collect(&mut self) -> crate::error::Result<crate::provider::CompletionResponse> {
        while let Some(result) = self.inner.next_chunk().await {
            result?;
        }
        Ok(self.inner.to_response())
    }

    /// Report cost for an already-collected response: from typed usage when
    /// present, otherwise via the shared backoff generation-cost resolver. Marks
    /// the stream reported so Drop does not re-report. Idempotent: a second call
    /// is a no-op (the callback is the money sink; reporting twice would double-book).
    pub async fn report_cost(&mut self, response: &crate::provider::CompletionResponse) {
        if self.cost_reported {
            tracing::warn!("report_cost called more than once; ignoring the repeat");
            return;
        }
        let cost_info = cost_for_response(&self.generator, response).await;
        (self.callback)(cost_info, self.meta.clone()).await;
        self.cost_reported = true;
    }

    /// Settle the cost of a cancelled (un-collected) stream and report it,
    /// **awaiting** the resolution (which may include a backoff out-of-band query)
    /// on the caller's runtime.
    ///
    /// This is the explicit, reliable cancellation path: prefer it over just
    /// dropping the stream. A bare drop falls back to a detached background task
    /// (see the `Drop` impl), which is best-effort and can be lost if the runtime
    /// shuts down mid-settle; `cancel` guarantees the report completes before it
    /// returns. Reports `Unknown`/`Unpriced` rather than a fake $0 when the cost
    /// can't be determined.
    pub async fn cancel(mut self) {
        // Honor any usage chunk that ALREADY arrived (don't throw away an exact,
        // free, in-hand cost for a slower out-of-band guess). Non-blocking: drains
        // only what is already buffered, never awaits the network.
        self.inner.drain_buffered();

        // A failed stream is not an accepted generation, so book nothing, loudly.
        if self.inner.errored() {
            tracing::warn!(
                "TrackedStream for {} cancelled after a transport error; no cost booked.",
                self.inner.id()
            );
            self.cost_reported = true; // suppress the Drop fallback
            return;
        }

        let response = self.inner.to_response();
        let cost_info = cost_for_response(&self.generator, &response).await;
        (self.callback)(cost_info, self.meta.clone()).await;
        self.cost_reported = true; // suppress the Drop fallback
    }

    /// Explicitly reject this completion: book NO cost and suppress the Drop
    /// fallback. The deliberate "this completion was unacceptable, don't pay for
    /// it" path (mirrors the non-streaming `crash_on_empty`/`crash_on_refusal`
    /// behavior). This is the ONLY way to drop a `TrackedStream` without booking
    /// cost: a plain drop of an un-reported stream books, so forgetting to report
    /// can't silently lose an accepted generation's cost.
    pub fn reject(mut self) {
        self.rejected = true;
    }

    /// Check if the stream has finished
    pub fn is_finished(&self) -> bool {
        self.inner.is_finished()
    }

    /// Get accumulated content so far
    pub fn accumulated(&self) -> &str {
        self.inner.accumulated()
    }

    /// The provider's generation/response id (empty until the first chunk).
    pub fn id(&self) -> &str {
        self.inner.id()
    }
}

impl Drop for TrackedStream {
    fn drop(&mut self) {
        // Already reported (report_cost/cancel) or explicitly rejected (reject):
        // nothing to book. These are the only two ways to drop without booking.
        if self.cost_reported || self.rejected {
            return;
        }

        // Drain any chunk still BUFFERED in the channel (non-blocking) before
        // deciding. A terminal error (transport failure OR an in-band provider
        // `error` frame) can be sitting unread in the channel if the consumer
        // dropped without draining; without this it would be invisible here and we
        // would book a phantom cost for a failed generation. This makes drop
        // symmetric with `cancel`/`next_chunk`, which already observe it.
        self.inner.drain_buffered();

        // A stream that ended in a TRANSPORT ERROR (timeout / SSE failure) or an
        // in-band provider error is not an accepted generation: booking it would
        // charge a phantom cost for a request that failed. Skip the booking and say
        // so loudly, symmetric with the "don't silently lose an accepted cost" rule.
        if self.inner.errored() {
            tracing::warn!(
                "TrackedStream for {} ended in a transport error; no cost booked (failed generation).",
                self.inner.id()
            );
            return;
        }

        // Any other drop (whether the stream was fully drained because the consumer
        // forgot to report, or cancelled mid-flight) is an accepted/used generation
        // whose cost has NOT been reported. We must book it, not silently lose it.
        // Drop can't await the (possibly multi-second, backoff) settle, so this is
        // a best-effort detached task that can be cancelled if the runtime shuts
        // down first; that risk is logged LOUDLY. Callers wanting a guarantee use
        // `cancel().await` (cancellation) or `report_cost().await` (normal end).
        let response = self.inner.to_response();
        let callback = self.callback.clone();
        let meta = self.meta.clone();
        let generator = self.generator.clone();

        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::error!(
                "TrackedStream for {} dropped un-reported outside a tokio runtime: cost CANNOT be settled and is LOST. Use cancel().await or report_cost().await.",
                response.id
            );
            return;
        };

        // Not loud: this fires on the normal drop-without-explicit-settle path,
        // most of which complete fine. The LOUD signal is reserved for an actual
        // loss (the LostGuard below), so it can't cry wolf.
        tracing::debug!(
            "TrackedStream for {} dropped without report_cost()/cancel()/reject(); settling cost on a detached task",
            response.id
        );
        handle.spawn(async move {
            // If this task is cancelled before booking (e.g. the runtime shuts down
            // during the backoff/HTTP in resolve_post_stream), the guard's Drop
            // fires the LOUD loss log. On success we disarm it after the callback.
            let mut guard = LostCostGuard::new(response.id.clone());
            let cost_info = cost_for_response(&generator, &response).await;
            (callback)(cost_info, meta).await;
            guard.settled = true;
        });
    }
}

/// Guards the detached cost-settle task: if it is dropped before `settled` is set
/// (i.e. the task was cancelled before the cost callback completed), it logs the
/// loss LOUDLY, so a runtime-shutdown-induced lost cost report is never silent.
struct LostCostGuard {
    response_id: String,
    settled: bool,
}

impl LostCostGuard {
    fn new(response_id: String) -> Self {
        Self {
            response_id,
            settled: false,
        }
    }
}

impl Drop for LostCostGuard {
    fn drop(&mut self) {
        if !self.settled {
            tracing::error!(
                "Cost settle task for {} was cancelled before booking: cost is LOST (likely runtime shutdown). Use cancel().await or report_cost().await for a guarantee.",
                self.response_id
            );
        }
    }
}

/// Derive the cost for a completed response. The single owner of the
/// "response -> CostInfo" decision, used by every enforced-tracking path. All
/// provider-specific knowledge (cost aggregation, out-of-band resolution) lives
/// behind `generator.provider`; this function only routes:
/// - usage present  -> `accounting.cost_of(usage, price)` (native cost, or
///   token×price, or `Unpriced`),
/// - usage absent    -> `accounting.resolve_post_stream(...)` (the provider's
///   own out-of-band query, or `Unknown` if it has none).
pub(crate) async fn cost_for_response(
    generator: &GeneratorInfo,
    response: &crate::provider::CompletionResponse,
) -> CostInfo {
    let price = generator.token_price.as_ref();
    let outcome = match &response.usage {
        Some(usage) => generator.provider.cost_of(usage.clone(), price),
        None => {
            // The out-of-band query rides the GENERATOR's client, the same
            // one the call it resolves rode: an injected client's routing
            // sees the follow-up too, never a second side-channel client.
            let client = generator.client();
            let ctx = crate::provider::PostStreamCtx {
                client: client.http().clone(),
                base_url: &generator.base_url,
                generation_id: &response.id,
                auth: &generator.auth,
                price,
            };
            generator.provider.resolve_post_stream(ctx).await
        }
    };
    outcome.into_cost_info(response.model.clone(), response.id.clone())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::MiniLLMError;
    use crate::provider::{CostResolution, StreamChunk, StreamingCompletion, Usage};
    use std::sync::Mutex;

    /// A dumb capturing callback: appends every (cost, meta) it receives to a log.
    type CaptureLog = Arc<Mutex<Vec<(CostInfo, serde_json::Value)>>>;

    fn capturing_context(meta: serde_json::Value) -> (CompletionContext, CaptureLog) {
        let log: CaptureLog = Arc::new(Mutex::new(Vec::new()));
        let sink = log.clone();
        let callback: AsyncCostCallback = Arc::new(move |cost, meta| {
            let sink = sink.clone();
            Box::pin(async move {
                sink.lock().unwrap().push((cost, meta));
            })
        });
        // OpenRouter accounting so the streaming tests exercise native USD cost.
        let generator = GeneratorInfo::new("Test", "https://example.test/v1", "test-model")
            .with_provider(std::sync::Arc::new(crate::provider::OpenRouterProvider));
        let ctx = CompletionContext::new(generator, meta, callback, "https://app", "App");
        (ctx, log)
    }

    #[tokio::test]
    async fn report_cost_passes_cost_and_meta_through() {
        let (ctx, log) = capturing_context(serde_json::json!({"userId": "u1"}));
        let cost = CostInfo {
            cost: 0.001,
            model: "test-model".into(),
            response_id: "gen-1".into(),
            ..Default::default()
        };
        ctx.report_cost(cost).await;

        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert!((captured[0].0.cost - 0.001).abs() < 1e-9);
        assert_eq!(captured[0].1["userId"], "u1");
    }

    #[test]
    fn is_byok_reads_metadata() {
        let (byok, _) = capturing_context(serde_json::json!({"isByok": true}));
        assert!(byok.is_byok());
        let (not_byok, _) = capturing_context(serde_json::json!({}));
        assert!(!not_byok.is_byok());
    }

    #[tokio::test]
    async fn collect_then_report_uses_typed_usage_and_sums_byok() {
        let (ctx, log) = capturing_context(serde_json::json!({}));
        let (stream, tx) = StreamingCompletion::from_channel("test-model", "gen-1", true);
        let mut tracked = TrackedStream::new(stream, &ctx);

        // Feed content then a trailing usage chunk (OpenRouter BYOK shape).
        tx.send(Ok(StreamChunk::content("hi"))).await.unwrap();
        tx.send(Ok(StreamChunk {
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                cost: Some(0.001),
                upstream_inference_cost: Some(0.009),
                uncached_input_tokens: 5,
                completion_tokens: 2,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();
        drop(tx);

        // collect() must NOT report cost; report_cost() does.
        let resp = tracked.collect().await.unwrap();
        assert_eq!(resp.content, "hi");
        assert!(log.lock().unwrap().is_empty(), "collect must not book cost");

        tracked.report_cost(&resp).await;

        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1);
        let cost = &captured[0].0;
        // BYOK total = OpenRouter fee + upstream inference cost.
        assert!((cost.cost - 0.010).abs() < 1e-9, "cost was {}", cost.cost);
        assert_eq!(cost.total_tokens, 7);
        assert_eq!(cost.resolution, CostResolution::Resolved);
    }

    #[tokio::test]
    async fn collect_or_cancel_finishes_and_books_once() {
        let (ctx, log) = capturing_context(serde_json::json!({}));
        let (stream, tx) = StreamingCompletion::from_channel("test-model", "gen-1", true);
        let tracked = TrackedStream::new(stream, &ctx);

        tx.send(Ok(StreamChunk::content("hi"))).await.unwrap();
        tx.send(Ok(StreamChunk {
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                cost: Some(0.002),
                uncached_input_tokens: 5,
                completion_tokens: 2,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();
        drop(tx);

        // A never-firing interrupt: the stream must finish on its own.
        let outcome = tracked.collect_or_cancel(std::future::pending()).await;
        let CollectOutcome::Finished(response) = outcome else {
            panic!("expected Finished, got {outcome:?}");
        };
        assert_eq!(response.content, "hi");
        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1, "finished stream books exactly once");
        assert_eq!(captured[0].0.resolution, CostResolution::Resolved);
    }

    #[tokio::test]
    async fn collect_or_cancel_interrupt_cancels_and_still_books() {
        let (ctx, log) = capturing_context(serde_json::json!({}));
        // Empty id: cancel's out-of-band query is skipped and the cost books
        // as an honest Unknown; the point pinned here is that an interrupted
        // drain still fires the callback (never a silent drop).
        let (stream, tx) = StreamingCompletion::from_channel("test-model", "", true);
        let tracked = TrackedStream::new(stream, &ctx);
        tx.send(Ok(StreamChunk::content("partial"))).await.unwrap();

        // Interrupt fires immediately; the channel stays open (a live stream).
        let outcome = tracked.collect_or_cancel(std::future::ready(())).await;
        assert!(matches!(outcome, CollectOutcome::Interrupted), "got {outcome:?}");
        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1, "an interrupted stream still books its cost");
        assert_eq!(captured[0].0.resolution, CostResolution::Unknown);
        drop(tx);
    }

    #[tokio::test]
    async fn collect_or_cancel_transport_error_books_nothing() {
        let (ctx, log) = capturing_context(serde_json::json!({}));
        let (stream, tx) = StreamingCompletion::from_channel("test-model", "gen-1", true);
        let tracked = TrackedStream::new(stream, &ctx);
        tx.send(Ok(StreamChunk::content("hi"))).await.unwrap();
        tx.send(Err(crate::error::MiniLLMError::Stream("wire cut".into()))).await.unwrap();
        drop(tx);

        let outcome = tracked.collect_or_cancel(std::future::pending()).await;
        assert!(matches!(outcome, CollectOutcome::Failed(_)), "got {outcome:?}");
        assert!(log.lock().unwrap().is_empty(), "a failed stream books nothing");
    }

    #[tokio::test]
    async fn report_cost_marks_unknown_when_no_usage_and_no_id() {
        let (ctx, log) = capturing_context(serde_json::json!({}));
        // Empty id => the generation-cost query is skipped (it can't query
        // without an id), so cost must be reported as Unknown, never a fake $0.
        let (stream, tx) = StreamingCompletion::from_channel("test-model", "", true);
        let mut tracked = TrackedStream::new(stream, &ctx);

        tx.send(Ok(StreamChunk::content("hi"))).await.unwrap();
        drop(tx); // close with no usage chunk

        let resp = tracked.collect().await.unwrap();
        tracked.report_cost(&resp).await;

        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1);
        assert_eq!(captured[0].0.resolution, CostResolution::Unknown);
        assert_eq!(captured[0].0.cost, 0.0);
    }

    /// Feed a content chunk + a usage chunk and drain the channel; returns the
    /// tracked stream fully collected (finished).
    async fn drained_stream(ctx: &CompletionContext) -> TrackedStream {
        let (stream, tx) = StreamingCompletion::from_channel("test-model", "gen-1", true);
        let mut tracked = TrackedStream::new(stream, ctx);
        tx.send(Ok(StreamChunk::content("hi"))).await.unwrap();
        tx.send(Ok(StreamChunk {
            finish_reason: Some("stop".into()),
            usage: Some(Usage {
                cost: Some(0.5),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();
        drop(tx);
        let _ = tracked.collect().await.unwrap();
        tracked
    }

    #[tokio::test]
    async fn explicit_reject_books_nothing() {
        // The deliberate "unacceptable completion, don't pay" path.
        let (ctx, log) = capturing_context(serde_json::json!({}));
        drained_stream(&ctx).await.reject();
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        assert!(
            log.lock().unwrap().is_empty(),
            "an explicitly rejected stream must not book cost"
        );
    }

    #[tokio::test]
    async fn drained_then_dropped_without_report_still_books_cost() {
        // Forgetting to report on an accepted (fully-drained) stream must NOT
        // silently lose the cost: Drop books it (only reject() opts out).
        let (ctx, log) = capturing_context(serde_json::json!({}));
        {
            let _tracked = drained_stream(&ctx).await; // dropped here, un-reported
        }
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1, "a forgotten report must still book cost");
        assert!((captured[0].0.cost - 0.5).abs() < 1e-9);
    }

    #[tokio::test]
    async fn genuine_cancellation_books_cost() {
        // A stream dropped mid-flight (never collected, not finished) is a genuine
        // cancellation → Drop books cost. Empty id here → Unknown resolution.
        let (ctx, log) = capturing_context(serde_json::json!({}));
        let (stream, _tx) = StreamingCompletion::from_channel("test-model", "", true);
        let tracked = TrackedStream::new(stream, &ctx);
        assert!(!tracked.is_finished(), "precondition: not collected");
        drop(tracked); // genuine cancel

        // Let the spawned cancel-report task run.
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1, "genuine cancel must book cost");
        assert_eq!(captured[0].0.resolution, CostResolution::Unknown);
    }

    #[tokio::test]
    async fn explicit_cancel_settles_cost_synchronously_and_suppresses_drop() {
        // cancel().await reports cost inline (no detached task to lose), and the
        // subsequent drop must NOT double-report.
        let (ctx, log) = capturing_context(serde_json::json!({}));
        let (mut stream_holder, tx) =
            StreamingCompletion::from_channel("test-model", "gen-1", true);
        // Feed a usage chunk so cancel resolves to a concrete cost synchronously.
        tx.send(Ok(StreamChunk::content("partial"))).await.unwrap();
        tx.send(Ok(StreamChunk {
            usage: Some(Usage {
                cost: Some(0.02),
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();
        // Pull the usage chunk into inner state without fully collecting.
        let _ = stream_holder.next_chunk().await;
        let _ = stream_holder.next_chunk().await;

        let tracked = TrackedStream::new(stream_holder, &ctx);
        tracked.cancel().await; // reports inline, marks reported

        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1, "cancel reports exactly once");
        assert_eq!(captured[0].0.resolution, CostResolution::Resolved);
        assert!((captured[0].0.cost - 0.02).abs() < 1e-9);
    }

    #[tokio::test]
    async fn report_cost_then_drop_books_exactly_once() {
        // The happy path: collect → report_cost → drop must book exactly one cost.
        // A regression that forgot to set cost_reported in report_cost would let
        // Drop re-book and double-charge.
        let (ctx, log) = capturing_context(serde_json::json!({}));
        {
            let mut tracked = drained_stream(&ctx).await;
            let resp = tracked.inner.to_response();
            tracked.report_cost(&resp).await;
        } // dropped here; must NOT re-book.
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            log.lock().unwrap().len(),
            1,
            "report_cost then drop must book exactly once"
        );
    }

    #[tokio::test]
    async fn errored_stream_books_nothing_on_drop() {
        // A stream that ends in a transport error is a FAILED generation, not an
        // accepted one: Drop must NOT book a phantom cost for it.
        let (ctx, log) = capturing_context(serde_json::json!({}));
        let (stream, tx) = StreamingCompletion::from_channel("test-model", "gen-1", true);
        let mut tracked = TrackedStream::new(stream, &ctx);
        tx.send(Ok(StreamChunk::content("partial"))).await.unwrap();
        tx.send(Err(MiniLLMError::Timeout)).await.unwrap();
        // Drive the stream; the error surfaces and marks it errored.
        let err = tracked.collect().await;
        assert!(err.is_err(), "stream surfaces the transport error");
        drop(tracked);
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        assert!(
            log.lock().unwrap().is_empty(),
            "a failed stream must not book cost"
        );
    }

    #[tokio::test]
    async fn buffered_error_left_undrained_then_dropped_books_nothing() {
        // The dangerous shape: a terminal error (transport OR an in-band provider
        // `error` frame) is sitting BUFFERED in the channel, and the consumer drops
        // the TrackedStream WITHOUT draining it (no next_chunk/collect/cancel). Drop
        // must drain the buffered error, see `errored`, and book NOTHING, otherwise
        // it charges a phantom cost for a failed generation.
        //
        // The id is EMPTY on purpose: without the Drop drain, the un-errored path
        // would book an Unknown cost IMMEDIATELY (no out-of-band HTTP for an empty
        // id), so the assertion below deterministically distinguishes "drained →
        // booked nothing" from "not drained → booked a phantom" without racing a
        // slow detached query (a non-empty id would book only after a ~25s poll,
        // making this assertion pass for the wrong reason).
        let (ctx, log) = capturing_context(serde_json::json!({}));
        let (stream, tx) = StreamingCompletion::from_channel("test-model", "", true);
        let tracked = TrackedStream::new(stream, &ctx);
        tx.send(Ok(StreamChunk::content("partial"))).await.unwrap();
        tx.send(Err(MiniLLMError::Stream("in-band provider error".into())))
            .await
            .unwrap();
        // NO next_chunk / collect / cancel: drop straight away with the error buffered.
        drop(tracked);
        for _ in 0..3 {
            tokio::task::yield_now().await;
        }
        assert!(
            log.lock().unwrap().is_empty(),
            "a buffered terminal error must make Drop book nothing"
        );
    }

    #[tokio::test]
    async fn anthropic_split_usage_books_correct_tokens_end_to_end() {
        // The Anthropic split-usage merge (input in message_start, output in
        // message_delta) must reach the booked CostInfo with the right token
        // counts, not just the stream state machine. Priced so cost is Resolved.
        let log: CaptureLog = Arc::new(Mutex::new(Vec::new()));
        let sink = log.clone();
        let callback: AsyncCostCallback = Arc::new(move |cost, meta| {
            let sink = sink.clone();
            Box::pin(async move {
                sink.lock().unwrap().push((cost, meta));
            })
        });
        // Anthropic provider + a token price so cost resolves from token counts.
        let generator = GeneratorInfo::new("Test", "https://example.test", "claude-haiku-4-5")
            .with_provider(std::sync::Arc::new(crate::provider::AnthropicProvider))
            .with_token_price(crate::provider::TokenPrice::new(1.0, 5.0));
        let ctx = CompletionContext::new(generator, serde_json::json!({}), callback, "u", "a");

        let (stream, tx) = StreamingCompletion::from_channel("claude-haiku-4-5", "msg_1", true);
        let mut tracked = TrackedStream::new(stream, &ctx);
        // message_start: input usage only.
        tx.send(Ok(StreamChunk {
            id: Some("msg_1".into()),
            usage: Some(Usage {
                uncached_input_tokens: 1_000_000,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();
        tx.send(Ok(StreamChunk::content("hi"))).await.unwrap();
        // message_delta: stop + output usage (input absent here).
        tx.send(Ok(StreamChunk {
            finish_reason: Some("end_turn".into()),
            usage: Some(Usage {
                completion_tokens: 1_000_000,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();
        drop(tx);

        let resp = tracked.collect().await.unwrap();
        tracked.report_cost(&resp).await;

        let captured = log.lock().unwrap();
        assert_eq!(captured.len(), 1);
        let cost = &captured[0].0;
        assert_eq!(
            cost.prompt_tokens, 1_000_000,
            "input merged from message_start"
        );
        assert_eq!(
            cost.completion_tokens, 1_000_000,
            "output from message_delta"
        );
        assert_eq!(cost.resolution, CostResolution::Resolved);
        // 1M×$1 in + 1M×$5 out = $6.
        assert!((cost.cost - 6.0).abs() < 1e-9, "got {}", cost.cost);
    }
}
