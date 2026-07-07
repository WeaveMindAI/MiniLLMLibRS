//! Streaming completion handling

use super::response::{CompletionResponse, StreamChunk, Usage};
use super::Provider;
use crate::error::{MiniLLMError, Result};
use crate::tools::ToolCallAccumulator;
use futures::StreamExt;
use reqwest_eventsource::{Event, EventSource};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;

/// A streaming completion that yields chunks as they arrive
pub struct StreamingCompletion {
    /// Receiver for stream chunks
    rx: mpsc::Receiver<Result<StreamChunk>>,

    /// Accumulated content so far
    accumulated: String,

    /// Accumulated tool-call fragments (assembled by wire index as they stream in)
    tool_calls: ToolCallAccumulator,

    /// Final usage stats (set when stream completes)
    usage: Option<Usage>,

    /// Model name
    model: String,

    /// Provider generation id, adopted from the first chunk that carries one
    /// (empty until then). Used for out-of-band cost resolution on cancellation.
    id: String,

    /// Whether the stream has finished
    finished: bool,

    /// Whether the stream ended in a TRANSPORT ERROR (timeout, SSE/connection
    /// failure) rather than a clean completion or cancellation. A failed stream is
    /// not an accepted generation, so cost accounting must NOT book it (see
    /// `TrackedStream::Drop`).
    errored: bool,

    /// Finish reason
    finish_reason: Option<String>,

    /// Whether a trailing usage chunk is expected (usage tracking on). When
    /// false, `finish_reason` alone terminates the stream; when true we keep
    /// reading until the usage chunk arrives or the channel closes.
    expect_usage: bool,
}

impl StreamingCompletion {
    /// Create a new streaming completion from an EventSource.
    ///
    /// `idle_timeout` bounds the silence between SSE events (not total duration);
    /// if no event arrives within it, the stream fails loudly with `Timeout`
    /// rather than parking on a dead connection until the pool timeout.
    pub fn from_event_source(
        mut es: EventSource,
        model: String,
        expect_usage: bool,
        idle_timeout: Option<Duration>,
        provider: Arc<dyn Provider>,
    ) -> Self {
        let (tx, rx) = mpsc::channel(100);

        // Spawn task to process SSE events
        tokio::spawn(async move {
            loop {
                // Bound the wait for the next event by the idle timeout (if set).
                let next = match idle_timeout {
                    Some(dur) => match tokio::time::timeout(dur, es.next()).await {
                        Ok(next) => next,
                        Err(_) => {
                            let _ = tx.send(Err(MiniLLMError::Timeout)).await;
                            break;
                        }
                    },
                    None => es.next().await,
                };
                let Some(event) = next else { break };

                match event {
                    Ok(Event::Open) => {
                        tracing::debug!("SSE connection opened");
                    }
                    Ok(Event::Message(msg)) => {
                        // `parse_chunk` returns None for an ignorable frame, Some(Ok)
                        // for a real chunk, and Some(Err) for an in-band PROVIDER
                        // ERROR (a 200 stream that then reports failure). Forward the
                        // error on the channel exactly like a transport error and stop
                        // so it sets `errored` and is never billed as accepted.
                        match provider.parse_chunk(&msg.data) {
                            None => {}
                            Some(Ok(chunk)) => {
                                if tx.send(Ok(chunk)).await.is_err() {
                                    break;
                                }
                            }
                            Some(Err(e)) => {
                                let _ = tx.send(Err(e)).await;
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(MiniLLMError::Stream(e.to_string()))).await;
                        break;
                    }
                }
            }

            es.close();
        });

        Self {
            rx,
            accumulated: String::new(),
            tool_calls: ToolCallAccumulator::default(),
            usage: None,
            model,
            id: String::new(),
            finished: false,
            errored: false,
            finish_reason: None,
            expect_usage,
        }
    }

    /// Test-only constructor: build a stream fed by an in-memory channel so the
    /// state machine can be exercised without a real SSE connection. Returns the
    /// stream and the sender; dropping the sender simulates the stream closing.
    #[cfg(test)]
    pub(crate) fn from_channel(
        model: &str,
        id: &str,
        expect_usage: bool,
    ) -> (Self, mpsc::Sender<Result<StreamChunk>>) {
        let (tx, rx) = mpsc::channel(100);
        let stream = Self {
            rx,
            accumulated: String::new(),
            tool_calls: ToolCallAccumulator::default(),
            usage: None,
            model: model.to_string(),
            id: id.to_string(),
            finished: false,
            errored: false,
            finish_reason: None,
            expect_usage,
        };
        (stream, tx)
    }

    /// Fold a received chunk into the accumulated state.
    ///
    /// This is the single source of truth for the stream's state machine, shared
    /// by both the async `next_chunk` and the `Stream::poll_next` impl so the two
    /// can never diverge.
    ///
    /// Termination is explicit: when usage tracking is on we finish on the
    /// trailing usage chunk; when it is off we finish on `finish_reason` (there is
    /// nothing more to wait for). The channel closing is the backstop in both
    /// cases. This avoids hanging until the connection's pool timeout if a
    /// provider sends `finish_reason` but neither usage nor `[DONE]`.
    fn ingest(&mut self, chunk: &StreamChunk) {
        // Adopt the provider's real generation id from the first chunk that has
        // one, so cancellation cost resolution queries the real generation.
        if self.id.is_empty() {
            if let Some(id) = &chunk.id {
                self.id = id.clone();
            }
        }

        self.accumulated.push_str(&chunk.delta);

        if let Some(reason) = &chunk.finish_reason {
            self.finish_reason = Some(reason.clone());
        }

        if let Some(deltas) = &chunk.tool_calls {
            self.tool_calls.ingest(deltas);
        }

        // Merge usage rather than replace: a provider may split it across events
        // (Anthropic sends input tokens in `message_start`, output in
        // `message_delta`). For single-usage-chunk providers this is a plain set.
        if let Some(usage) = &chunk.usage {
            match &mut self.usage {
                Some(existing) => existing.merge_from(usage),
                None => self.usage = Some(usage.clone()),
            }
        }

        // Termination: when usage is expected, finish once BOTH the finish reason
        // and the usage have arrived (OpenAI: usage chunk trails finish_reason;
        // Anthropic: `message_delta` carries both at once). When usage is not
        // expected, finish on finish_reason alone. The channel close is the
        // backstop in both cases (handled by `next_chunk`).
        if self.finish_reason.is_some() && (!self.expect_usage || self.usage.is_some()) {
            self.finished = true;
        }
    }

    /// Build the final response from the accumulated state. The single place a
    /// `CompletionResponse` is assembled from streamed state.
    pub(crate) fn to_response(&self) -> CompletionResponse {
        CompletionResponse {
            id: self.id.clone(),
            model: self.model.clone(),
            content: self.accumulated.clone(),
            finish_reason: self.finish_reason.clone(),
            usage: self.usage.clone(),
            tool_calls: (!self.tool_calls.is_empty()).then(|| self.tool_calls.finish()),
            raw_response: None,
        }
    }

    /// Get the next chunk from the stream
    pub async fn next_chunk(&mut self) -> Option<Result<StreamChunk>> {
        if self.finished {
            return None;
        }

        match self.rx.recv().await {
            Some(Ok(chunk)) => {
                self.ingest(&chunk);
                Some(Ok(chunk))
            }
            Some(Err(e)) => {
                // A transport error terminates the stream AND marks it failed, so
                // cost accounting can tell a failed stream from a clean one.
                self.finished = true;
                self.errored = true;
                Some(Err(e))
            }
            None => {
                // Channel closed - we're done
                self.finished = true;
                None
            }
        }
    }

    /// Fold every chunk ALREADY buffered in the channel into the accumulated state
    /// WITHOUT awaiting the network (non-blocking). Used by `cancel` so a final
    /// usage chunk that already arrived is honored instead of being thrown away for
    /// a slower out-of-band guess. A buffered transport error marks the stream
    /// errored, same as draining it.
    pub(crate) fn drain_buffered(&mut self) {
        loop {
            match self.rx.try_recv() {
                Ok(Ok(chunk)) => self.ingest(&chunk),
                Ok(Err(_)) => {
                    self.errored = true;
                    self.finished = true;
                    break;
                }
                Err(_) => break, // Empty or Disconnected: nothing more buffered now.
            }
        }
    }

    /// Collect all chunks and return the final response
    pub async fn collect(mut self) -> Result<CompletionResponse> {
        while let Some(result) = self.next_chunk().await {
            result?;
        }
        Ok(self.to_response())
    }

    /// Get the accumulated content so far
    pub fn accumulated(&self) -> &str {
        &self.accumulated
    }

    /// Check if the stream has finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Whether the stream ended in a transport error (timeout / SSE failure)
    /// rather than a clean completion or cancellation. Cost accounting uses this to
    /// avoid booking a failed generation.
    pub fn errored(&self) -> bool {
        self.errored
    }

    /// Get usage stats (only available after stream completes)
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Get the response ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the model name
    pub fn model(&self) -> &str {
        &self.model
    }
}

// NOTE: no `impl futures::Stream`. The async `next_chunk` is the consumption
// API; a `Stream` impl would be a second copy of the receive-and-`ingest` logic
// (via `poll_recv`) with no current consumer. Add it back (delegating to the same
// path) the day a caller actually wants `StreamExt` combinators.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::response::Usage;

    fn content_chunk(delta: &str) -> StreamChunk {
        StreamChunk::content(delta)
    }

    fn tool_delta(index: u64, name: Option<&str>, args: &str) -> StreamChunk {
        StreamChunk {
            tool_calls: Some(vec![crate::tools::ToolCallDelta {
                index,
                id: name.map(|_| format!("call_{index}")),
                name: name.map(String::from),
                arguments_fragment: Some(args.to_string()),
            }]),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn collects_content_and_terminates_on_channel_close() {
        let (stream, tx) = StreamingCompletion::from_channel("m", "gen-1", true);
        tx.send(Ok(content_chunk("Hel"))).await.unwrap();
        tx.send(Ok(content_chunk("lo"))).await.unwrap();
        // No usage chunk; closing the channel must terminate the stream.
        drop(tx);

        let resp = stream.collect().await.unwrap();
        assert_eq!(resp.content, "Hello");
        assert_eq!(resp.id, "gen-1");
        assert_eq!(resp.model, "m");
        assert!(resp.tool_calls.is_none());
    }

    #[tokio::test]
    async fn threads_finish_reason_and_usage() {
        let (stream, tx) = StreamingCompletion::from_channel("m", "gen-1", true);
        tx.send(Ok(content_chunk("hi"))).await.unwrap();
        tx.send(Ok(StreamChunk::finished("stop"))).await.unwrap();
        // Trailing usage chunk (OpenRouter sends usage after finish_reason).
        tx.send(Ok(StreamChunk {
            usage: Some(Usage {
                uncached_input_tokens: 3,
                completion_tokens: 1,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();
        drop(tx);

        let resp = stream.collect().await.unwrap();
        assert_eq!(resp.content, "hi");
        assert_eq!(resp.finish_reason.as_deref(), Some("stop"));
        assert_eq!(resp.usage.unwrap().total_tokens(), 4);
    }

    #[tokio::test]
    async fn accumulates_tool_call_deltas_across_chunks() {
        let (stream, tx) = StreamingCompletion::from_channel("m", "gen-1", true);
        tx.send(Ok(tool_delta(0, Some("search"), "{\"q\":")))
            .await
            .unwrap();
        tx.send(Ok(tool_delta(0, None, "\"rust\"}"))).await.unwrap();
        drop(tx);

        let resp = stream.collect().await.unwrap();
        let tc = resp.tool_calls.expect("tool calls assembled");
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].name, "search");
        assert_eq!(tc[0].arguments, "{\"q\":\"rust\"}");
    }

    #[tokio::test]
    async fn usage_chunk_finishes_stream_early() {
        // Once usage arrives, the stream is finished even if the channel stays open.
        let (mut stream, tx) = StreamingCompletion::from_channel("m", "gen-1", true);
        tx.send(Ok(content_chunk("done"))).await.unwrap();
        tx.send(Ok(StreamChunk {
            finish_reason: Some("stop".into()),
            usage: Some(Usage::default()),
            ..Default::default()
        }))
        .await
        .unwrap();

        // Drain via next_chunk; after the usage chunk, the next call returns None
        // without needing the channel to close.
        while let Some(r) = stream.next_chunk().await {
            r.unwrap();
        }
        assert!(stream.is_finished());
        assert_eq!(stream.accumulated(), "done");
    }

    #[tokio::test]
    async fn finish_reason_terminates_when_not_expecting_usage() {
        // include_usage=false: a finish_reason chunk with no usage and no [DONE]
        // must terminate the stream rather than hang. The channel stays open.
        let (mut stream, _tx) = StreamingCompletion::from_channel("m", "gen-1", false);
        _tx.send(Ok(content_chunk("hi"))).await.unwrap();
        _tx.send(Ok(StreamChunk::finished("stop"))).await.unwrap();

        while let Some(r) = stream.next_chunk().await {
            r.unwrap();
        }
        assert!(stream.is_finished());
        assert_eq!(stream.accumulated(), "hi");
        assert_eq!(stream.to_response().finish_reason.as_deref(), Some("stop"));
    }

    #[tokio::test]
    async fn anthropic_split_usage_accumulates_across_events() {
        // Anthropic sends input tokens in message_start and output tokens +
        // stop_reason in message_delta. The state machine must merge them (not
        // lose the input count) and finish on the message_delta (finish + usage).
        let (stream, tx) = StreamingCompletion::from_channel("claude-haiku-4-5", "msg_1", true);

        // message_start: id + input usage only (no finish, no text).
        tx.send(Ok(StreamChunk {
            id: Some("msg_1".into()),
            usage: Some(Usage {
                uncached_input_tokens: 15,
                completion_tokens: 1,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();
        // content deltas
        tx.send(Ok(content_chunk("Hel"))).await.unwrap();
        tx.send(Ok(content_chunk("lo"))).await.unwrap();
        // message_delta: stop_reason + output usage (input absent here).
        tx.send(Ok(StreamChunk {
            finish_reason: Some("end_turn".into()),
            usage: Some(Usage {
                uncached_input_tokens: 0,
                completion_tokens: 9,
                ..Default::default()
            }),
            ..Default::default()
        }))
        .await
        .unwrap();
        drop(tx);

        let resp = stream.collect().await.unwrap();
        assert_eq!(resp.content, "Hello");
        assert_eq!(resp.finish_reason.as_deref(), Some("end_turn"));
        let u = resp.usage.expect("usage merged across events");
        assert_eq!(
            u.uncached_input_tokens, 15,
            "input from message_start preserved"
        );
        assert_eq!(u.completion_tokens, 9, "output from message_delta applied");
        assert_eq!(u.total_tokens(), 24, "total recomputed from merged buckets");
    }
}
