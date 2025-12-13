//! Streaming completion handling

use super::response::{parse_stream_chunk, CompletionResponse, StreamChunk, Usage};
use crate::error::{MiniLLMError, Result};
use futures::Stream;
use reqwest_eventsource::{Event, EventSource};
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc;

/// A streaming completion that yields chunks as they arrive
pub struct StreamingCompletion {
    /// Receiver for stream chunks
    rx: mpsc::Receiver<Result<StreamChunk>>,

    /// Accumulated content so far
    accumulated: String,

    /// Final usage stats (set when stream completes)
    usage: Option<Usage>,

    /// Model name
    model: String,

    /// Response ID
    id: String,

    /// Whether the stream has finished
    finished: bool,

    /// Finish reason
    finish_reason: Option<String>,
}

impl StreamingCompletion {
    /// Create a new streaming completion from an EventSource
    pub fn from_event_source(mut es: EventSource, model: String, id: String) -> Self {
        let (tx, rx) = mpsc::channel(100);

        // Spawn task to process SSE events
        tokio::spawn(async move {
            use futures::StreamExt;

            while let Some(event) = es.next().await {
                match event {
                    Ok(Event::Open) => {
                        tracing::debug!("SSE connection opened");
                    }
                    Ok(Event::Message(msg)) => {
                        if let Some(chunk) = parse_stream_chunk(&msg.data) {
                            if tx.send(Ok(chunk)).await.is_err() {
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
            usage: None,
            model,
            id,
            finished: false,
            finish_reason: None,
        }
    }

    /// Get the next chunk from the stream
    pub async fn next_chunk(&mut self) -> Option<Result<StreamChunk>> {
        if self.finished {
            return None;
        }

        match self.rx.recv().await {
            Some(Ok(chunk)) => {
                // Accumulate content
                self.accumulated.push_str(&chunk.delta);

                // Check for finish
                if let Some(reason) = &chunk.finish_reason {
                    self.finished = true;
                    self.finish_reason = Some(reason.clone());
                }

                // Store usage if present
                if chunk.usage.is_some() {
                    self.usage = chunk.usage.clone();
                }

                Some(Ok(chunk))
            }
            Some(Err(e)) => {
                self.finished = true;
                Some(Err(e))
            }
            None => {
                self.finished = true;
                None
            }
        }
    }

    /// Collect all chunks and return the final response
    pub async fn collect(mut self) -> Result<CompletionResponse> {
        while let Some(result) = self.next_chunk().await {
            result?;
        }

        Ok(CompletionResponse {
            id: self.id,
            model: self.model,
            content: self.accumulated,
            finish_reason: self.finish_reason,
            usage: self.usage,
            tool_calls: None,
            raw_response: None,
        })
    }

    /// Get the accumulated content so far
    pub fn accumulated(&self) -> &str {
        &self.accumulated
    }

    /// Check if the stream has finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get usage stats (only available after stream completes)
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }
}

impl Stream for StreamingCompletion {
    type Item = Result<StreamChunk>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                self.accumulated.push_str(&chunk.delta);

                if let Some(reason) = &chunk.finish_reason {
                    self.finished = true;
                    self.finish_reason = Some(reason.clone());
                }

                if chunk.usage.is_some() {
                    self.usage = chunk.usage.clone();
                }

                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => {
                self.finished = true;
                Poll::Ready(Some(Err(e)))
            }
            Poll::Ready(None) => {
                self.finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}
