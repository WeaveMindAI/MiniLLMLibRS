//! One answer per key, kept for a while and shared: what makes a side request
//! (a model lookup, a price read) cost one round trip per key for the whole
//! process, however many requests and generators ask.

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use futures::future::{BoxFuture, FutureExt, Shared};

use crate::error::{MiniLLMError, Result};

/// A lookup's outcome, shared by everyone waiting on it. The error is shared
/// too, so it travels behind an `Arc`.
type Outcome<T> = std::result::Result<T, Arc<MiniLLMError>>;

enum Slot<T> {
    /// The outcome, and when it stops being fresh.
    Ready { outcome: Outcome<T>, until: Instant },
    /// A lookup in flight.
    Pending(Shared<BoxFuture<'static, Outcome<T>>>),
}

/// One outcome per key, looked up at most once at a time.
///
/// Concurrent callers for a key whose outcome is missing or stale share one
/// lookup and all get its outcome, so a hanging source costs each of them one
/// wait rather than one wait per caller ahead of them. An outcome is kept for
/// as long as its lookup says. Each key has its own slot, so a lookup for one
/// never waits on another.
pub(crate) struct Memo<T> {
    slots: Mutex<HashMap<String, Slot<T>>>,
}

/// How long an outcome nobody chose a lifetime for (a lookup that panicked or
/// was cancelled) is kept before the next caller asks again.
pub(crate) const FAILURE_TTL: Duration = Duration::from_secs(60);

impl<T: Clone + Send + Sync + 'static> Memo<T> {
    pub(crate) fn new() -> Self {
        Self {
            slots: Mutex::default(),
        }
    }

    /// The outcome for `key`: kept while fresh, else the lookup in flight,
    /// else `lookup`, which returns the outcome and how long it stays fresh.
    ///
    /// A new lookup runs as a task of its own, which settles the slot when it
    /// ends: a caller that gives up (its request timed out) leaves it running
    /// for the others and for the memo, so no slot is ever left pending with
    /// nobody to finish it. A lookup that panics settles as a failure.
    pub(crate) async fn get_or_lookup(
        &'static self,
        key: &str,
        lookup: impl FnOnce() -> BoxFuture<'static, (Result<T>, Duration)>,
    ) -> Outcome<T> {
        let pending = {
            let mut slots = self.lock();
            let now = Instant::now();
            match slots.get(key) {
                Some(Slot::Ready { outcome, until }) if now < *until => return outcome.clone(),
                Some(Slot::Pending(pending)) => pending.clone(),
                _ => {
                    let task = tokio::spawn(
                        self.settle(key.to_string(), AssertUnwindSafe(lookup()).catch_unwind()),
                    );
                    let owned_key = key.to_string();
                    let pending = async move {
                        match task.await {
                            Ok(outcome) => outcome,
                            // The task was cancelled (its runtime shut down)
                            // before it settled: settle it here as a failure,
                            // so the slot is asked again after FAILURE_TTL
                            // instead of staying pending for good.
                            Err(e) => {
                                let outcome = Err(Arc::new(MiniLLMError::Internal(format!(
                                    "the lookup ended without an answer: {e}"
                                ))));
                                self.lock().insert(
                                    owned_key,
                                    Slot::Ready {
                                        outcome: outcome.clone(),
                                        until: Instant::now() + FAILURE_TTL,
                                    },
                                );
                                outcome
                            }
                        }
                    }
                    .boxed()
                    .shared();
                    // Outcomes past their freshness are dropped here, so keys
                    // nobody asks for any more do not pile up.
                    slots.retain(
                        |_, slot| !matches!(slot, Slot::Ready { until, .. } if now >= *until),
                    );
                    // Inserted under the lock the task needs to settle, so the
                    // task always finds its own pending slot.
                    slots.insert(key.to_string(), Slot::Pending(pending.clone()));
                    pending
                }
            }
        };
        pending.await
    }

    /// Run one lookup to its end and keep its outcome for as long as it says
    /// (a panic, [`FAILURE_TTL`]).
    async fn settle(
        &'static self,
        key: String,
        lookup: impl std::future::Future<Output = std::thread::Result<(Result<T>, Duration)>>,
    ) -> Outcome<T> {
        let (outcome, ttl) = match lookup.await {
            Ok((outcome, ttl)) => (outcome.map_err(Arc::new), ttl),
            Err(_) => (
                Err(Arc::new(MiniLLMError::Internal(
                    "the lookup panicked".into(),
                ))),
                FAILURE_TTL,
            ),
        };
        self.lock().insert(
            key,
            Slot::Ready {
                outcome: outcome.clone(),
                until: Instant::now() + ttl,
            },
        );
        outcome
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, Slot<T>>> {
        // Nothing panics while holding this lock, so a poisoned one is a bug.
        self.slots.lock().expect("the memo lock is never poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const HOUR: Duration = Duration::from_secs(3600);

    /// A fresh memo for one test, kept for the process like the real one.
    fn memo() -> &'static Memo<u32> {
        Box::leak(Box::new(Memo::new()))
    }

    fn counting(
        calls: &Arc<AtomicUsize>,
        answer: Result<(u32, Duration)>,
    ) -> BoxFuture<'static, (Result<u32>, Duration)> {
        let calls = calls.clone();
        async move {
            calls.fetch_add(1, Ordering::SeqCst);
            match answer {
                Ok((value, ttl)) => (Ok(value), ttl),
                Err(e) => (Err(e), FAILURE_TTL),
            }
        }
        .boxed()
    }

    /// One lookup per key while fresh; another key is its own lookup.
    #[tokio::test]
    async fn a_fresh_answer_is_remembered_per_key() {
        let memo = memo();
        let calls = Arc::new(AtomicUsize::new(0));
        for _ in 0..3 {
            let got = memo
                .get_or_lookup("a", || counting(&calls, Ok((1, HOUR))))
                .await;
            assert_eq!(got.unwrap(), 1);
        }
        let b = memo
            .get_or_lookup("b", || counting(&calls, Ok((2, HOUR))))
            .await;
        assert_eq!(b.unwrap(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// An answer is kept only as long as its lookup said.
    #[tokio::test]
    async fn a_stale_answer_is_looked_up_again() {
        let memo = memo();
        let calls = Arc::new(AtomicUsize::new(0));
        let first = memo.get_or_lookup("a", || counting(&calls, Ok((1, Duration::ZERO))));
        assert_eq!(first.await.unwrap(), 1);
        let second = memo.get_or_lookup("a", || counting(&calls, Ok((2, HOUR))));
        assert_eq!(second.await.unwrap(), 2);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// A failure reaches every caller of its key and is kept as long as its
    /// lookup said, so a source that is down is not asked once per caller.
    #[tokio::test]
    async fn a_failure_is_kept_briefly() {
        let memo = memo();
        let calls = Arc::new(AtomicUsize::new(0));
        for _ in 0..3 {
            let got = memo
                .get_or_lookup("a", || counting(&calls, Err(MiniLLMError::Timeout)))
                .await;
            assert!(matches!(
                got.as_ref().map_err(|e| &**e),
                Err(MiniLLMError::Timeout)
            ));
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// A lookup that panics fails its callers as an internal error.
    #[tokio::test]
    async fn a_panicking_lookup_fails_as_internal() {
        let memo = memo();
        let got = memo
            .get_or_lookup("a", || async { panic!("boom") }.boxed())
            .await;
        assert!(matches!(
            got.as_ref().map_err(|e| &**e),
            Err(MiniLLMError::Internal(_))
        ));
    }

    /// A caller that gives up leaves the lookup running: it still settles, and
    /// the next caller gets its answer instead of a wait.
    #[tokio::test]
    async fn an_abandoned_lookup_still_settles() {
        let memo = memo();
        let calls = Arc::new(AtomicUsize::new(0));
        let (release, released) = tokio::sync::oneshot::channel::<()>();
        let slow = {
            let calls = calls.clone();
            memo.get_or_lookup("a", move || {
                async move {
                    calls.fetch_add(1, Ordering::SeqCst);
                    released.await.ok();
                    (Ok(5), HOUR)
                }
                .boxed()
            })
        };
        assert!(tokio::time::timeout(Duration::from_millis(10), slow)
            .await
            .is_err());
        release.send(()).unwrap();
        let next = memo
            .get_or_lookup("a", || counting(&calls, Ok((9, HOUR))))
            .await;
        assert_eq!(next.unwrap(), 5);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    /// A lookup whose runtime shuts down before it ends settles as a failure,
    /// so a caller on another runtime later gets a fresh lookup rather than
    /// the cancelled one for good.
    #[test]
    fn a_lookup_cancelled_with_its_runtime_is_asked_again() {
        let memo = memo();
        let calls = Arc::new(AtomicUsize::new(0));
        {
            let runtime = tokio::runtime::Runtime::new().unwrap();
            let calls = calls.clone();
            runtime.block_on(async {
                let hung = memo.get_or_lookup("a", move || {
                    async move {
                        calls.fetch_add(1, Ordering::SeqCst);
                        std::future::pending::<()>().await;
                        (Ok(1), HOUR)
                    }
                    .boxed()
                });
                assert!(tokio::time::timeout(Duration::from_millis(10), hung)
                    .await
                    .is_err());
            });
        }
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let first = runtime.block_on(memo.get_or_lookup("a", || counting(&calls, Ok((2, HOUR)))));
        assert!(
            matches!(
                first.as_ref().map_err(|e| &**e),
                Err(MiniLLMError::Internal(_))
            ),
            "the cancelled lookup settles as a failure: {first:?}"
        );
        // Kept for FAILURE_TTL like any failure, then asked again; the slot is
        // no longer stuck in flight.
        assert!(matches!(memo.lock().get("a"), Some(Slot::Ready { .. })));
    }

    /// Many callers at once, on many threads, for a few keys: each key is
    /// looked up exactly once and everyone gets its answer. Repeated, since a
    /// race here would show only some of the time.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_callers_share_one_lookup_per_key() {
        for _ in 0..50 {
            let memo = memo();
            let calls = Arc::new(AtomicUsize::new(0));
            let callers = (0..32).map(|i| {
                let calls = calls.clone();
                tokio::spawn(async move {
                    let key = (i % 4) as u32;
                    let answer = memo
                        .get_or_lookup(&key.to_string(), || {
                            async move {
                                calls.fetch_add(1, Ordering::SeqCst);
                                tokio::task::yield_now().await;
                                (Ok(key), HOUR)
                            }
                            .boxed()
                        })
                        .await
                        .unwrap();
                    assert_eq!(answer, key);
                })
            });
            for caller in futures::future::join_all(callers).await {
                caller.unwrap();
            }
            assert_eq!(calls.load(Ordering::SeqCst), 4);
        }
    }
}
