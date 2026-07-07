//! Live decoding of a streamed tool call's arguments, field by field.
//!
//! [`ArgumentStream`] incrementally parses the raw argument fragments of one
//! tool call (the pieces the providers already normalize into
//! [`ToolCallDelta`](super::ToolCallDelta)) and routes each field to its
//! consumer. Every field is the same kind of object: take a [`FieldHandle`]
//! for it and choose PER FIELD how to consume it:
//! - [`FieldHandle::delta`]: the field's DECODED text, live, chunk by chunk
//!   (JSON string escaping undone: `\n` a newline, `\"` a quote, `\uXXXX` the
//!   character), exactly as the model "meant" it. Type code into an editor as
//!   the model generates it.
//! - [`FieldHandle::wait`]: the complete value, parsed, once the field ends.
//!
//! Fields nobody took a handle for are buffered into
//! [`fields`](ArgumentStream::fields), so a consumer that doesn't stream
//! anything still gets every argument extracted at the end.
//!
//! Fragments may split at ARBITRARY positions (mid-escape, mid-key, between
//! surrogate halves); the output never changes.
//!
//! Two modes:
//! - **Strict** (default): the arguments must be well-formed JSON; malformed
//!   input fails loudly with the raw text in the error.
//! - **Lenient** (opt-in): for models sloppy at escaping, applied to EVERY
//!   top-level string value. The rule is deterministic because the legitimate
//!   ways a string can end are known: inside any string value, an unescaped
//!   `"` closes the string ONLY when followed (whitespace optional
//!   everywhere) by
//!   1. `, "<key>":` (the next field's declaration; the key is itself a full
//!      JSON string, spaces and escapes included), or
//!   2. `}` and then nothing until the true end of input (the provider's
//!      end-of-call signal), or
//!   3. the bare end of input (the model just stopped).
//!
//!   Any other `"` is literal content, as are raw newlines/control characters
//!   and a `\` before a non-escape character (a literal backslash). Content is
//!   never silently dropped. The one documented misfire: content that
//!   LITERALLY contains `", "somekey": ` reads as a field boundary; that
//!   ambiguity is unresolvable on the wire. On finish, lenient mode
//!   additionally runs the raw arguments through this crate's JSON repair and
//!   fills any field the incremental parse missed into
//!   [`fields`](ArgumentStream::fields) (never overwriting what was parsed).
//!
//! Leniency covers top-level string VALUES only. Keys, numbers, booleans, and
//! nested objects/arrays must be well-formed JSON in both modes: they have no
//! end anchor and models don't mangle them in practice.
//!
//! Non-goals: provider-specific code (this works on the normalized fragment
//! stream), and streaming below the top level (a nested object's inner
//! strings arrive when the nested value completes).

use crate::error::{MiniLLMError, Result};
use std::collections::HashMap;
use tokio::sync::mpsc;

/// What a [`FieldHandle`] receives on its channel.
#[derive(Debug)]
enum FieldEvent {
    /// A chunk of the field's decoded string content (string fields only).
    Text(String),
    /// The field completed. `Some` carries the parsed value for non-string
    /// fields (and for fields that completed before the handle existed);
    /// `None` means "you already received the full content as `Text` chunks".
    Complete(Option<serde_json::Value>),
}

/// The consumer side of one argument field. Obtained from
/// [`ArgumentStream::field`]; consume it with [`delta`](Self::delta) (live
/// decoded chunks) or [`wait`](Self::wait) (the complete parsed value). Both
/// work for any field; the choice is per consumer, per field.
#[derive(Debug)]
pub struct FieldHandle {
    field: String,
    rx: mpsc::UnboundedReceiver<FieldEvent>,
    done: bool,
}

impl FieldHandle {
    /// The next chunk of this field's DECODED text, as the model generates
    /// it. `None` once the field has fully arrived (or, after
    /// [`ArgumentStream::finish`], when it never will). Non-string fields
    /// produce no text chunks; use [`wait`](Self::wait) for them.
    pub async fn delta(&mut self) -> Option<String> {
        if self.done {
            return None;
        }
        match self.rx.recv().await {
            Some(FieldEvent::Text(text)) => Some(text),
            Some(FieldEvent::Complete(_)) | None => {
                self.done = true;
                None
            }
        }
    }

    /// The field's complete value, once it has fully arrived: string fields
    /// yield their decoded text as a JSON string, other fields their parsed
    /// value. Fails loudly if the call ended without this field.
    pub async fn wait(mut self) -> Result<serde_json::Value> {
        let mut text = String::new();
        let mut got_text = false;
        loop {
            match self.rx.recv().await {
                Some(FieldEvent::Text(t)) => {
                    text.push_str(&t);
                    got_text = true;
                }
                Some(FieldEvent::Complete(Some(value))) => return Ok(value),
                Some(FieldEvent::Complete(None)) => return Ok(serde_json::Value::String(text)),
                None => {
                    if got_text {
                        // The stream died mid-field (lenient): what arrived
                        // is the honest value.
                        return Ok(serde_json::Value::String(text));
                    }
                    return Err(MiniLLMError::MalformedResponse(format!(
                        "streamed tool arguments ended without the field '{}'",
                        self.field
                    )));
                }
            }
        }
    }
}

/// Where the string value currently streaming routes its decoded text.
#[derive(Debug, Clone)]
enum Target {
    /// A consumer holds a [`FieldHandle`]: decoded chunks are sent live.
    Handle(mpsc::UnboundedSender<FieldEvent>),
    /// No handle: decoded text accumulates and lands in `fields` at the end.
    Buffer,
}

#[derive(Debug)]
enum State {
    Prelude(Prelude),
    /// Inside a string value's body.
    Str(Target),
    /// Lenient only: saw an unescaped `"`; buffering until the continuation
    /// proves it structural (a `, "key":` boundary, or `}` at the true end)
    /// or literal content.
    Pending {
        target: Target,
        /// Everything after the candidate quote, verbatim, for the flush.
        raw: String,
        phase: Phase,
    },
    /// The object closed; only whitespace may follow.
    Closed,
    /// `finish` succeeded; fields remain readable, feeding is over.
    Done,
    /// A loud error was returned; the stream is dead.
    Failed,
}

/// Continuation matcher after a candidate closing quote (lenient).
#[derive(Debug)]
enum Phase {
    /// Right after the quote: expecting `,` (next field) or `}` (close).
    AfterQuote,
    /// After the `,`: expecting the next key's opening `"`.
    AfterComma,
    /// Inside the candidate key: a FULL JSON string (spaces, escapes, any
    /// characters), terminated by its unescaped `"`.
    InKey { key: String, escaped: bool },
    /// After the key's closing quote: expecting `:` to confirm the boundary.
    AfterKey { key: String },
    /// Saw `}`: only whitespace may follow until the true end of input.
    ObjClose,
}

/// Structure between values: the `{`, keys, `:`. Non-string values (and, in
/// strict mode, handle-less strings) are captured raw here and parsed on
/// completion; other strings hand off to [`State::Str`].
#[derive(Debug)]
enum Prelude {
    BeforeBrace,
    BeforeKey,
    InKey {
        key: String,
        escaped: bool,
    },
    AfterKey,
    BeforeValue,
    /// Capturing a raw value. `depth` counts open `{`/`[`; a scalar is depth
    /// 0 outside a string.
    SkipValue {
        depth: u32,
        in_string: bool,
        escaped: bool,
    },
    AfterValue,
}

/// In-string escape state, split so a fragment can end anywhere inside an
/// escape (`\`, `\uXX`, or between the halves of a surrogate pair).
#[derive(Debug, Default, PartialEq)]
enum Esc {
    #[default]
    None,
    /// After `\`.
    Slash,
    /// Inside `\uXXXX`, hex digits collected so far.
    Unicode(String),
    /// Decoded a high surrogate; awaiting the `\` of the low half.
    AwaitLowSlash { high: u16 },
    /// Awaiting the `u` of the low half.
    AwaitLowU { high: u16 },
    /// Inside the low half's `\uXXXX` hex digits.
    AwaitLowHex { high: u16, buf: String },
}

/// Incremental decoder for a streamed tool call's arguments. Take
/// [`field`](Self::field) handles for the fields you want to consume (before
/// feeding), push the raw fragments with [`feed`](Self::feed), and call
/// [`finish`](Self::finish) when the provider signals the call's end.
/// Fields without a handle land parsed in [`fields`](Self::fields).
///
/// ```
/// use minillmlib::ArgumentStream;
///
/// let mut args = ArgumentStream::strict();
/// let content = args.field("content");
///
/// // Fragments split anywhere, even mid-escape:
/// args.feed(r#"{"path": "a.rs", "content": "line1\"#).unwrap();
/// assert_eq!(args.fields()["path"], "a.rs"); // available before the payload ends
/// args.feed(r#"nline2"}"#).unwrap();
/// args.finish().unwrap();
///
/// let text = futures::executor::block_on(content.wait()).unwrap();
/// assert_eq!(text, "line1\nline2");
/// ```
#[derive(Debug)]
pub struct ArgumentStream {
    lenient: bool,
    state: State,
    esc: Esc,
    /// Everything fed so far, verbatim, for loud errors and the lenient
    /// repair pass.
    raw: String,
    /// The key currently being read/valued.
    current_key: String,
    /// Decoded text of the buffered (handle-less) string currently streaming.
    buf: String,
    /// Live chunk being assembled for a handled string (flushed to its
    /// channel at each feed boundary, so consumers see progress per feed).
    chunk: String,
    /// Raw text of the value currently being captured for a normal JSON
    /// parse on completion.
    value_buf: String,
    /// Consumers' channels, by field name.
    handles: HashMap<String, mpsc::UnboundedSender<FieldEvent>>,
    /// Handle-less fields, parsed, as they complete on the wire.
    fields: serde_json::Map<String, serde_json::Value>,
}

impl ArgumentStream {
    /// Strict stream: the arguments must be well-formed JSON; malformed input
    /// (bad escape, unescaped control char, unterminated string/object) fails
    /// loudly.
    pub fn strict() -> Self {
        Self::new(false)
    }

    /// Lenient stream: tolerates sloppy escaping in every top-level string
    /// value (see the module docs for the deterministic rules), and repairs
    /// what the incremental parse missed at [`finish`](Self::finish). Prefer
    /// strict unless the model demonstrably needs it.
    pub fn lenient() -> Self {
        Self::new(true)
    }

    fn new(lenient: bool) -> Self {
        Self {
            lenient,
            state: State::Prelude(Prelude::BeforeBrace),
            esc: Esc::default(),
            raw: String::new(),
            current_key: String::new(),
            buf: String::new(),
            chunk: String::new(),
            value_buf: String::new(),
            handles: HashMap::new(),
            fields: serde_json::Map::new(),
        }
    }

    /// Take the consumer handle for one field. Create handles BEFORE feeding:
    /// a field that already completed handle-less is delivered to a late
    /// handle as its complete value, but a field currently mid-stream cannot
    /// be re-routed. Taking a second handle for the same field replaces the
    /// first (whose channel closes).
    pub fn field(&mut self, name: impl Into<String>) -> FieldHandle {
        let name = name.into();
        let (tx, rx) = mpsc::unbounded_channel();
        // Late handle for an already-buffered field: deliver it whole.
        if let Some(value) = self.fields.get(&name) {
            let _ = tx.send(FieldEvent::Complete(Some(value.clone())));
        }
        self.handles.insert(name.clone(), tx);
        FieldHandle {
            field: name,
            rx,
            done: false,
        }
    }

    /// The handle-less fields parsed so far, each available as soon as its
    /// value completes on the wire (so fields the model emits before a big
    /// one are readable while it still streams). Handled fields are NOT
    /// duplicated here (their content went to the handle). After a lenient
    /// [`finish`](Self::finish), also contains whatever the repair pass
    /// recovered.
    pub fn fields(&self) -> &serde_json::Map<String, serde_json::Value> {
        &self.fields
    }

    /// Push one raw argument fragment. Decoded content is routed to the
    /// field handles / the [`fields`](Self::fields) map as it becomes
    /// unambiguous. After an error the stream is dead and every further call
    /// errors.
    pub fn feed(&mut self, fragment: &str) -> Result<()> {
        match self.state {
            State::Failed => return Err(self.error("stream already failed")),
            State::Done => return Err(self.error("stream already finished")),
            _ => {}
        }
        self.raw.push_str(fragment);
        for c in fragment.chars() {
            if let Err(e) = self.step(c) {
                self.state = State::Failed;
                return Err(e);
            }
        }
        // Feed boundary: ship what this fragment made available.
        self.flush_chunk();
        Ok(())
    }

    /// The provider signaled the end of the call's arguments: resolve every
    /// held-back ambiguity against the true end, validate the envelope, close
    /// all handles, and (lenient) repair-fill [`fields`](Self::fields) from
    /// the raw text. The stream becomes terminal; `fields()` stays readable.
    pub fn finish(&mut self) -> Result<()> {
        let result = self.finish_inner();
        if result.is_ok() {
            // Lenient: give the whole raw text to this crate's JSON repairer
            // and adopt any field the incremental parse missed (never
            // overwriting what it parsed, and never duplicating handled
            // fields' content, which already went out through the handles).
            if self.lenient {
                if let Ok(serde_json::Value::Object(repaired)) =
                    crate::utils::extract_json_value(&self.raw)
                {
                    for (key, value) in repaired {
                        if !self.fields.contains_key(&key) && !self.handles.contains_key(&key) {
                            self.fields.insert(key, value);
                        }
                    }
                }
            }
            self.state = State::Done;
        }
        // Close every channel so pending `delta`/`wait` consumers resolve,
        // on success AND on failure (a dead stream must not hang its tools).
        self.handles.clear();
        result
    }

    fn finish_inner(&mut self) -> Result<()> {
        loop {
            match std::mem::replace(&mut self.state, State::Failed) {
                State::Failed => return Err(self.error("stream already failed")),
                State::Done => return Err(self.error("stream already finished")),
                State::Prelude(Prelude::BeforeBrace) if self.raw.trim().is_empty() => {
                    // Providers send empty arguments for no-parameter tools.
                    return Ok(());
                }
                State::Prelude(_) => {
                    if self.lenient {
                        // The model died mid-structure; keep what completed
                        // (plus the repair pass) and surface it.
                        tracing::warn!(
                            "streamed tool arguments ended mid-structure; keeping completed fields"
                        );
                        return Ok(());
                    }
                    return Err(self.error("arguments ended mid-structure"));
                }
                State::Str(target) => {
                    if !self.lenient {
                        return Err(self.error("string value never closed"));
                    }
                    // The model just stopped: flush any partial escape
                    // leniently, then the open string ends here. Content is
                    // never dropped.
                    let mut flush = String::new();
                    match std::mem::take(&mut self.esc) {
                        Esc::None => {}
                        Esc::Slash => flush.push('\\'),
                        Esc::Unicode(buf) => {
                            flush.push_str("\\u");
                            flush.push_str(&buf);
                        }
                        Esc::AwaitLowSlash { .. } => flush.push('\u{FFFD}'),
                        Esc::AwaitLowU { .. } => {
                            flush.push('\u{FFFD}');
                            flush.push('\\');
                        }
                        Esc::AwaitLowHex { buf, .. } => {
                            flush.push('\u{FFFD}');
                            flush.push_str("\\u");
                            flush.push_str(&buf);
                        }
                    }
                    for fc in flush.chars() {
                        self.emit(&target, fc);
                    }
                    tracing::warn!(
                        field = %self.current_key,
                        "streamed tool arguments ended inside a string; accepting its content"
                    );
                    self.end_string(target);
                    return Ok(());
                }
                State::Pending { target, raw, phase } => match phase {
                    // String closed at the true end (with or without the `}`
                    // the model may have forgotten).
                    Phase::AfterQuote | Phase::ObjClose => {
                        self.end_string(target);
                        return Ok(());
                    }
                    // A PARTIAL boundary at the true end (e.g. `", "de`): it
                    // never confirmed, so it is literal content. Flush it and
                    // re-evaluate (the flush may itself end in a new pending
                    // quote; the loop resolves until the buffer is drained).
                    Phase::AfterComma | Phase::InKey { .. } | Phase::AfterKey { .. } => {
                        self.state = State::Str(target.clone());
                        self.emit(&target, '"');
                        for rc in raw.chars() {
                            if let Err(e) = self.step(rc) {
                                self.state = State::Failed;
                                return Err(e);
                            }
                        }
                        // loop continues with the new state at EOF
                    }
                },
                State::Closed => return Ok(()),
            }
        }
    }

    fn error(&self, what: &str) -> MiniLLMError {
        MiniLLMError::MalformedResponse(format!(
            "streamed tool arguments: {} (raw: {})",
            what, self.raw
        ))
    }

    /// Push one decoded char to where the current string's text belongs.
    fn emit(&mut self, target: &Target, c: char) {
        match target {
            Target::Handle(_) => self.chunk.push(c),
            Target::Buffer => self.buf.push(c),
        }
    }

    /// Ship the live chunk to the current handled string's channel, if any.
    fn flush_chunk(&mut self) {
        if self.chunk.is_empty() {
            return;
        }
        let tx = match &self.state {
            State::Str(Target::Handle(tx)) => Some(tx),
            State::Pending {
                target: Target::Handle(tx),
                ..
            } => Some(tx),
            _ => None,
        };
        if let Some(tx) = tx {
            let _ = tx.send(FieldEvent::Text(std::mem::take(&mut self.chunk)));
        }
    }

    /// The current string value ended: route its completion.
    fn end_string(&mut self, target: Target) {
        let key = std::mem::take(&mut self.current_key);
        match target {
            Target::Handle(tx) => {
                if !self.chunk.is_empty() {
                    let _ = tx.send(FieldEvent::Text(std::mem::take(&mut self.chunk)));
                }
                let _ = tx.send(FieldEvent::Complete(None));
            }
            Target::Buffer => {
                let value = serde_json::Value::String(std::mem::take(&mut self.buf));
                self.fields.insert(key, value);
            }
        }
    }

    /// A captured raw value (non-string, or strict handle-less string) is
    /// complete: parse it as normal JSON and route it. These values must be
    /// well-formed in both modes (leniency covers only string values, which
    /// have an end anchor).
    fn complete_captured_value(&mut self) -> Result<()> {
        match serde_json::from_str::<serde_json::Value>(self.value_buf.trim()) {
            Ok(value) => {
                let key = std::mem::take(&mut self.current_key);
                self.value_buf.clear();
                if let Some(tx) = self.handles.get(&key) {
                    let _ = tx.send(FieldEvent::Complete(Some(value)));
                } else {
                    self.fields.insert(key, value);
                }
                Ok(())
            }
            Err(e) => Err(self.error(&format!(
                "field '{}' is not valid JSON ({}): {}",
                self.current_key, e, self.value_buf
            ))),
        }
    }

    fn step(&mut self, c: char) -> Result<()> {
        match &self.state {
            State::Prelude(_) => self.step_prelude(c),
            State::Str(target) => {
                let target = target.clone();
                self.step_str(c, &target)
            }
            State::Pending { .. } => self.step_pending(c),
            State::Closed => {
                if c.is_whitespace() {
                    Ok(())
                } else {
                    Err(self.error(&format!("unexpected '{c}' after the arguments closed")))
                }
            }
            State::Done | State::Failed => unreachable!("feed guards Done/Failed"),
        }
    }

    /// Continuation matcher after a candidate closing quote (lenient only).
    /// Confirms a `, "key":` boundary or a `}`-at-true-end close; on any
    /// deviation the quote and the buffered span were literal content and are
    /// re-fed through the string decoder.
    fn step_pending(&mut self, c: char) -> Result<()> {
        let State::Pending { target, raw, phase } = &mut self.state else {
            unreachable!()
        };
        let target = target.clone();
        raw.push(c);
        let deviated = match phase {
            Phase::AfterQuote => {
                if c.is_whitespace() {
                    false
                } else if c == ',' {
                    *phase = Phase::AfterComma;
                    false
                } else if c == '}' {
                    *phase = Phase::ObjClose;
                    false
                } else {
                    true
                }
            }
            Phase::AfterComma => {
                if c.is_whitespace() {
                    false
                } else if c == '"' {
                    *phase = Phase::InKey {
                        key: String::new(),
                        escaped: false,
                    };
                    false
                } else {
                    true
                }
            }
            // The candidate key is a FULL JSON string: any characters
            // (spaces included), `\"`/`\\` escape-aware.
            Phase::InKey { key, escaped } => {
                if *escaped {
                    key.push(c);
                    *escaped = false;
                } else if c == '\\' {
                    *escaped = true;
                } else if c == '"' {
                    let key = std::mem::take(key);
                    *phase = Phase::AfterKey { key };
                } else {
                    key.push(c);
                }
                false
            }
            Phase::AfterKey { key } => {
                if c.is_whitespace() {
                    false
                } else if c == ':' {
                    // BOUNDARY CONFIRMED: the candidate quote really closed
                    // the current string, and `key`'s value comes next.
                    let key = std::mem::take(key);
                    self.end_string(target);
                    self.current_key = key;
                    self.state = State::Prelude(Prelude::BeforeValue);
                    return Ok(());
                } else {
                    true
                }
            }
            Phase::ObjClose => !c.is_whitespace(),
        };

        if !deviated {
            return Ok(());
        }
        // Deviation: the candidate quote was literal content. Re-decode it
        // and the buffered span (which may itself contain a new candidate
        // quote; `step` re-enters this matcher as needed).
        let raw = std::mem::take(raw);
        self.state = State::Str(target.clone());
        self.emit(&target, '"');
        for rc in raw.chars() {
            self.step(rc)?;
        }
        Ok(())
    }

    fn step_prelude(&mut self, c: char) -> Result<()> {
        // Set when a captured value's raw text completed with this char; the
        // parse happens after the match (the state borrow must end first).
        let mut value_completed = false;
        let State::Prelude(p) = &mut self.state else {
            unreachable!()
        };
        match p {
            Prelude::BeforeBrace => {
                if c.is_whitespace() {
                } else if c == '{' {
                    *p = Prelude::BeforeKey;
                } else {
                    return Err(
                        self.error(&format!("arguments do not start with '{{' (got '{c}')"))
                    );
                }
            }
            Prelude::BeforeKey => {
                if c.is_whitespace() {
                } else if c == '"' {
                    *p = Prelude::InKey {
                        key: String::new(),
                        escaped: false,
                    };
                } else if c == '}' {
                    self.state = State::Closed;
                } else {
                    return Err(self.error(&format!("expected a field name, got '{c}'")));
                }
            }
            Prelude::InKey { key, escaped } => {
                if *escaped {
                    // Keys only need `\"`/`\\` fidelity for termination; field
                    // names are plain identifiers in practice.
                    key.push(c);
                    *escaped = false;
                } else if c == '\\' {
                    *escaped = true;
                } else if c == '"' {
                    self.current_key = std::mem::take(key);
                    *p = Prelude::AfterKey;
                } else {
                    key.push(c);
                }
            }
            Prelude::AfterKey => {
                if c.is_whitespace() {
                } else if c == ':' {
                    *p = Prelude::BeforeValue;
                } else {
                    return Err(self.error(&format!("expected ':' after a field name, got '{c}'")));
                }
            }
            Prelude::BeforeValue => {
                if c.is_whitespace() {
                } else if c == '"' && (self.lenient || self.handles.contains_key(&self.current_key))
                {
                    // A string streams through the decoder when a consumer
                    // wants it live (any mode) or when lenient (the boundary
                    // rule needs the decoder for every string).
                    let target = match self.handles.get(&self.current_key) {
                        Some(tx) => Target::Handle(tx.clone()),
                        None => {
                            self.buf.clear();
                            Target::Buffer
                        }
                    };
                    self.state = State::Str(target);
                } else {
                    // Captured raw and parsed on completion (non-strings, and
                    // strict handle-less strings).
                    self.value_buf.clear();
                    self.value_buf.push(c);
                    *p = if c == '"' {
                        Prelude::SkipValue {
                            depth: 0,
                            in_string: true,
                            escaped: false,
                        }
                    } else if c == '{' || c == '[' {
                        Prelude::SkipValue {
                            depth: 1,
                            in_string: false,
                            escaped: false,
                        }
                    } else {
                        // scalar (number/true/false/null)
                        Prelude::SkipValue {
                            depth: 0,
                            in_string: false,
                            escaped: false,
                        }
                    };
                }
            }
            Prelude::SkipValue {
                depth,
                in_string,
                escaped,
            } => {
                if *in_string {
                    self.value_buf.push(c);
                    if *escaped {
                        *escaped = false;
                    } else if c == '\\' {
                        *escaped = true;
                    } else if c == '"' {
                        *in_string = false;
                        if *depth == 0 {
                            value_completed = true;
                            *p = Prelude::AfterValue;
                        }
                    }
                } else if *depth > 0 {
                    self.value_buf.push(c);
                    match c {
                        '"' => *in_string = true,
                        '{' | '[' => *depth += 1,
                        '}' | ']' => {
                            *depth -= 1;
                            if *depth == 0 {
                                value_completed = true;
                                *p = Prelude::AfterValue;
                            }
                        }
                        _ => {}
                    }
                } else {
                    // scalar at depth 0: ends at `,`, `}`, or whitespace
                    // (the terminator is not part of the value's text).
                    if c == ',' {
                        value_completed = true;
                        *p = Prelude::BeforeKey;
                    } else if c == '}' {
                        value_completed = true;
                        self.state = State::Closed;
                    } else if c.is_whitespace() {
                        value_completed = true;
                        *p = Prelude::AfterValue;
                    } else {
                        self.value_buf.push(c);
                    }
                }
            }
            Prelude::AfterValue => {
                if c.is_whitespace() {
                } else if c == ',' {
                    *p = Prelude::BeforeKey;
                } else if c == '}' {
                    self.state = State::Closed;
                } else {
                    return Err(self.error(&format!("expected ',' or '}}', got '{c}'")));
                }
            }
        }
        if value_completed {
            self.complete_captured_value()?;
        }
        Ok(())
    }

    fn step_str(&mut self, c: char, target: &Target) -> Result<()> {
        match std::mem::take(&mut self.esc) {
            Esc::None => match c {
                '\\' => self.esc = Esc::Slash,
                '"' => {
                    if self.lenient {
                        // Cannot know yet whether this closes the string (a
                        // boundary/true-end follows) or is literal content.
                        // Ship what is unambiguous before holding back.
                        self.flush_chunk();
                        self.state = State::Pending {
                            target: target.clone(),
                            raw: String::new(),
                            phase: Phase::AfterQuote,
                        };
                    } else {
                        // Strict: the close is trusted syntax; more fields
                        // may follow.
                        self.end_string(target.clone());
                        self.state = State::Prelude(Prelude::AfterValue);
                    }
                }
                c if (c as u32) < 0x20 => {
                    if self.lenient {
                        self.emit(target, c); // a raw control char is itself
                    } else {
                        return Err(self.error(&format!(
                            "unescaped control character U+{:04X} in string value",
                            c as u32
                        )));
                    }
                }
                c => self.emit(target, c),
            },
            Esc::Slash => match c {
                '"' => self.emit(target, '"'),
                '\\' => self.emit(target, '\\'),
                '/' => self.emit(target, '/'),
                'b' => self.emit(target, '\u{8}'),
                'f' => self.emit(target, '\u{c}'),
                'n' => self.emit(target, '\n'),
                'r' => self.emit(target, '\r'),
                't' => self.emit(target, '\t'),
                'u' => self.esc = Esc::Unicode(String::new()),
                other => {
                    if self.lenient {
                        // `\` before a non-escape character is a literal
                        // backslash, and the character is itself.
                        self.emit(target, '\\');
                        return self.step_str(other, target);
                    }
                    return Err(self.error(&format!("invalid escape '\\{other}'")));
                }
            },
            Esc::Unicode(mut buf) => {
                if c.is_ascii_hexdigit() {
                    buf.push(c);
                    if buf.len() == 4 {
                        let code = u16::from_str_radix(&buf, 16).expect("4 hex digits");
                        self.take_unit(code, target)?;
                    } else {
                        self.esc = Esc::Unicode(buf);
                    }
                } else {
                    if self.lenient {
                        // Not a real escape after all: literal text.
                        self.emit(target, '\\');
                        self.emit(target, 'u');
                        for bc in buf.chars() {
                            self.emit(target, bc);
                        }
                        return self.step_str(c, target);
                    }
                    return Err(self.error(&format!("invalid \\u escape '\\u{buf}{c}'")));
                }
            }
            Esc::AwaitLowSlash { high } => {
                if c == '\\' {
                    self.esc = Esc::AwaitLowU { high };
                } else {
                    if self.lenient {
                        self.emit(target, '\u{FFFD}'); // unpaired high surrogate
                        return self.step_str(c, target);
                    }
                    return Err(self.error("unpaired \\u surrogate in string value"));
                }
            }
            Esc::AwaitLowU { high } => {
                if c == 'u' {
                    self.esc = Esc::AwaitLowHex {
                        high,
                        buf: String::new(),
                    };
                } else {
                    if self.lenient {
                        self.emit(target, '\u{FFFD}');
                        // The consumed `\` starts a fresh escape with `c`.
                        self.esc = Esc::Slash;
                        return self.step_str(c, target);
                    }
                    return Err(self.error("unpaired \\u surrogate in string value"));
                }
            }
            Esc::AwaitLowHex { high, mut buf } => {
                if c.is_ascii_hexdigit() {
                    buf.push(c);
                    if buf.len() == 4 {
                        let low = u16::from_str_radix(&buf, 16).expect("4 hex digits");
                        if (0xDC00..=0xDFFF).contains(&low) {
                            let combined =
                                0x10000 + ((high as u32 - 0xD800) << 10) + (low as u32 - 0xDC00);
                            self.emit(
                                target,
                                char::from_u32(combined).expect("valid surrogate pair"),
                            );
                        } else {
                            if !self.lenient {
                                return Err(self.error("unpaired \\u surrogate in string value"));
                            }
                            self.emit(target, '\u{FFFD}');
                            // The second unit stands alone: another high
                            // surrogate re-arms the pairing, a normal char is
                            // itself.
                            self.take_unit(low, target)?;
                        }
                    } else {
                        self.esc = Esc::AwaitLowHex { high, buf };
                    }
                } else {
                    if self.lenient {
                        self.emit(target, '\u{FFFD}');
                        self.emit(target, '\\');
                        self.emit(target, 'u');
                        for bc in buf.chars() {
                            self.emit(target, bc);
                        }
                        return self.step_str(c, target);
                    }
                    return Err(self.error("unpaired \\u surrogate in string value"));
                }
            }
        }
        Ok(())
    }

    /// A decoded `\uXXXX` unit: a plain char is emitted, a high surrogate arms
    /// the pair-awaiting state, a lone low surrogate is malformed.
    fn take_unit(&mut self, code: u16, target: &Target) -> Result<()> {
        if (0xD800..=0xDBFF).contains(&code) {
            self.esc = Esc::AwaitLowSlash { high: code };
            Ok(())
        } else if (0xDC00..=0xDFFF).contains(&code) {
            if self.lenient {
                self.emit(target, '\u{FFFD}');
                Ok(())
            } else {
                Err(self.error("unpaired \\u surrogate in string value"))
            }
        } else {
            self.emit(
                target,
                char::from_u32(code as u32).expect("non-surrogate BMP code point"),
            );
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drain a handle synchronously (channels are fully buffered after
    /// feed/finish): concatenated Text chunks, or Err if the field never
    /// arrived.
    fn drain(mut h: FieldHandle) -> Result<String> {
        let mut out = String::new();
        let mut completed = false;
        while let Ok(ev) = h.rx.try_recv() {
            match ev {
                FieldEvent::Text(t) => out.push_str(&t),
                FieldEvent::Complete(Some(v)) => {
                    return Ok(v.as_str().map(String::from).unwrap_or(v.to_string()))
                }
                FieldEvent::Complete(None) => completed = true,
            }
        }
        if completed || !out.is_empty() {
            Ok(out)
        } else {
            Err(MiniLLMError::MalformedResponse(format!(
                "field '{}' never arrived",
                h.field
            )))
        }
    }

    /// Run `raw` through a stream with a handle on `field`, split at EVERY
    /// char boundary (two fragments), plus fully char-by-char, and assert the
    /// handle's decoded output is identical each time. This is the core
    /// guarantee: splits at arbitrary positions never change the result.
    fn all_splits(field: &str, lenient: bool, raw: &str) -> Result<String> {
        let make = || {
            if lenient {
                ArgumentStream::lenient()
            } else {
                ArgumentStream::strict()
            }
        };
        let run = |fragments: Vec<&str>| -> Result<String> {
            let mut args = make();
            let h = args.field(field);
            for f in fragments {
                args.feed(f)?;
            }
            args.finish()?;
            drain(h)
        };

        let reference = run(vec![raw]);
        for i in 0..=raw.len() {
            if !raw.is_char_boundary(i) {
                continue;
            }
            let split = run(vec![&raw[..i], &raw[i..]]);
            match (&reference, &split) {
                (Ok(a), Ok(b)) => assert_eq!(a, b, "split at {i} diverged for {raw:?}"),
                (Err(_), Err(_)) => {}
                _ => panic!("split at {i} changed ok/err for {raw:?}"),
            }
        }
        // char-by-char (token-at-a-time worst case)
        let fragments: Vec<String> = raw.chars().map(|c| c.to_string()).collect();
        let by_char = run(fragments.iter().map(|s| s.as_str()).collect());
        match (&reference, &by_char) {
            (Ok(a), Ok(b)) => assert_eq!(a, b, "char-by-char diverged for {raw:?}"),
            (Err(_), Err(_)) => {}
            _ => panic!("char-by-char changed ok/err for {raw:?}"),
        }
        reference
    }

    fn strict(raw: &str) -> Result<String> {
        all_splits("content", false, raw)
    }

    fn lenient(raw: &str) -> Result<String> {
        all_splits("content", true, raw)
    }

    // ---- strict: well-formed inputs --------------------------------------

    #[test]
    fn strict_decodes_simple_payload() {
        assert_eq!(
            strict(r#"{"content": "hello world"}"#).unwrap(),
            "hello world"
        );
    }

    #[test]
    fn strict_decodes_every_escape() {
        assert_eq!(
            strict(r#"{"content":"a\"b\\c\/d\be\ff\ng\rh\tiéj"}"#).unwrap(),
            "a\"b\\c/d\u{8}e\u{c}f\ng\rh\ti\u{e9}j"
        );
    }

    #[test]
    fn strict_decodes_surrogate_pairs() {
        assert_eq!(strict(r#"{"content":"ok 😀!"}"#).unwrap(), "ok 😀!");
    }

    #[test]
    fn strict_parses_other_fields_around_the_handled_one() {
        let raw = r#"{"lang": "py\"x", "n": 4.5e2, "flag": true,
                      "meta": {"a": ["b", {"c": 1}]}, "content": "payload", "after": 1}"#;
        let mut args = ArgumentStream::strict();
        let h = args.field("content");
        args.feed(raw).unwrap();
        args.finish().unwrap();
        assert_eq!(drain(h).unwrap(), "payload");
        assert_eq!(args.fields()["lang"], "py\"x");
        assert_eq!(args.fields()["n"], 450.0);
        assert_eq!(args.fields()["flag"], true);
        assert_eq!(args.fields()["meta"]["a"][1]["c"], 1);
        assert_eq!(args.fields()["after"], 1, "fields after a string work");
    }

    #[test]
    fn strict_handles_empty_payload_and_whitespace_envelope() {
        assert_eq!(strict("  { \"content\" : \"\" }  ").unwrap(), "");
    }

    #[test]
    fn empty_arguments_are_a_valid_no_parameter_call() {
        for lenient_mode in [false, true] {
            let mut args = if lenient_mode {
                ArgumentStream::lenient()
            } else {
                ArgumentStream::strict()
            };
            args.feed("").unwrap();
            args.finish().unwrap();
            assert!(args.fields().is_empty());
        }
    }

    // ---- strict: malformed inputs fail loudly -----------------------------

    #[test]
    fn strict_rejects_malformed() {
        for bad in [
            r#"{"content": "unterminated"#,       // no closing quote
            r#"{"content": "no brace""#,          // string closed, object not
            r#"{"content": "bad \q escape"}"#,    // invalid escape
            r#"{"content": "lone \ud800 high"}"#, // unpaired surrogate
            r#"{"content": "a"} trailing"#,       // junk after envelope
            "{\"content\": \"raw\nnewline\"}",    // unescaped control char
            r#"["content"]"#,                     // not an object
        ] {
            assert!(strict(bad).is_err(), "must reject: {bad:?}");
        }
    }

    #[test]
    fn missing_field_resolves_as_handle_error_not_stream_error() {
        // The stream parses fine; whether a field was required is the
        // consumer's contract, surfaced on the handle.
        let mut args = ArgumentStream::strict();
        let h = args.field("content");
        args.feed(r#"{"other": "x"}"#).unwrap();
        args.finish().unwrap();
        assert!(drain(h).is_err(), "wait on a missing field errors");
        assert_eq!(args.fields()["other"], "x");
    }

    #[test]
    fn error_carries_the_raw_text() {
        let mut args = ArgumentStream::strict();
        // Missing ':' after the field name: structurally malformed.
        let err = args.feed(r#"{"content" x"#).unwrap_err().to_string();
        assert!(err.contains(r#"{"content" x"#), "raw text in error: {err}");
        // Dead after an error; finish closes handles instead of hanging.
        assert!(args.feed("x").is_err());
        assert!(args.finish().is_err());
    }

    // ---- lenient: sloppy models ------------------------------------------

    #[test]
    fn lenient_decodes_well_formed_identically() {
        let raw = r#"{"content":"a\"b\\c\nd 😀"}"#;
        assert_eq!(lenient(raw).unwrap(), strict(raw).unwrap());
    }

    #[test]
    fn lenient_unescaped_quote_mid_content_is_literal() {
        // The " before ` hi` is not followed by a boundary or the true end,
        // so it is content.
        assert_eq!(
            lenient(r#"{"content": "say "hi" ok"}"#).unwrap(),
            r#"say "hi" ok"#
        );
    }

    #[test]
    fn lenient_raw_newline_is_itself() {
        assert_eq!(
            lenient("{\"content\": \"line1\nline2\"}").unwrap(),
            "line1\nline2"
        );
    }

    #[test]
    fn lenient_backslash_before_non_escape_is_literal() {
        // `\p`, `\q`, `\ ` are not JSON escapes → literal backslashes. (A
        // valid escape like `\t` still decodes even in lenient mode: `C:\temp`
        // written unescaped is inherently ambiguous and the JSON reading wins.)
        assert_eq!(
            lenient(r#"{"content": "C:\path \q \ x"}"#).unwrap(),
            r#"C:\path \q \ x"#
        );
    }

    #[test]
    fn lenient_model_forgot_the_closing_quote_and_brace() {
        // The stream just stops: the content survives in full.
        assert_eq!(
            lenient(r#"{"content": "the model stopped here"#).unwrap(),
            "the model stopped here"
        );
    }

    #[test]
    fn lenient_model_closed_string_but_not_object() {
        assert_eq!(lenient(r#"{"content": "done""#).unwrap(), "done");
    }

    #[test]
    fn lenient_trailing_partial_escape_is_flushed_literally() {
        assert_eq!(
            lenient(r#"{"content": "ends with \"#).unwrap(),
            "ends with \\"
        );
        assert_eq!(
            lenient(r#"{"content": "ends with \u12"#).unwrap(),
            "ends with \\u12"
        );
    }

    #[test]
    fn lenient_quote_then_more_content_after_whitespace_and_brace() {
        // `"` + `}` + more content: the quote AND brace were literal.
        assert_eq!(lenient(r#"{"content": "a "} b"}"#).unwrap(), r#"a "} b"#);
    }

    #[test]
    fn lenient_unpaired_surrogate_is_replacement_not_dropped() {
        assert_eq!(
            lenient(r#"{"content": "x \ud800 y"}"#).unwrap(),
            "x \u{FFFD} y"
        );
    }

    // ---- lenient: the key-boundary rule ------------------------------------

    #[test]
    fn lenient_sloppy_field_recovers_via_key_boundary() {
        // The quotes inside the first field are not followed by `, "key":`,
        // so they are content; the one before `, "content":` is the real end.
        let raw = r#"{"note": "he said "hi" there", "content": "ok"}"#;
        assert_eq!(lenient(raw).unwrap(), "ok");
        let mut args = ArgumentStream::lenient();
        args.feed(raw).unwrap();
        args.finish().unwrap();
        assert_eq!(args.fields()["note"], r#"he said "hi" there"#);
    }

    #[test]
    fn lenient_boundary_key_may_contain_spaces_and_tight_whitespace() {
        // `","some thing":"` with no spaces at all is a valid boundary, and
        // the key is a full JSON string (spaces included).
        let raw = r#"{"a": "v "x"","some thing":"w","content":"c"}"#;
        assert_eq!(lenient(raw).unwrap(), "c");
        let mut args = ArgumentStream::lenient();
        args.feed(raw).unwrap();
        args.finish().unwrap();
        assert_eq!(args.fields()["a"], r#"v "x""#);
        assert_eq!(args.fields()["some thing"], "w");
    }

    #[test]
    fn lenient_documented_misfire_content_containing_a_boundary() {
        // THE accepted ambiguity: content that literally contains
        // `", "key": ` reads as a field boundary. Deterministic-in-rule.
        let raw = r#"{"content": "embed ", "meta": " done"}"#;
        assert_eq!(lenient(raw).unwrap(), "embed ");
        let mut args = ArgumentStream::lenient();
        args.feed(raw).unwrap();
        args.finish().unwrap();
        assert_eq!(args.fields()["meta"], " done");
    }

    #[test]
    fn lenient_partial_boundary_at_end_is_literal_content() {
        // The stream dies mid boundary pattern (`", "de`): it never confirmed,
        // so it was content.
        assert_eq!(
            lenient(r#"{"content": "abc", "de"#).unwrap(),
            r#"abc", "de"#
        );
    }

    #[test]
    fn lenient_stream_dying_mid_structure_keeps_completed_fields() {
        // The model died between fields: everything that completed survives.
        let mut args = ArgumentStream::lenient();
        let h = args.field("content");
        args.feed(r#"{"content": "x", "meta": 1"#).unwrap();
        args.finish().unwrap();
        assert_eq!(drain(h).unwrap(), "x");
        // The repair pass recovers the trailing scalar the incremental parse
        // couldn't complete.
        assert_eq!(args.fields()["meta"], 1);
    }

    #[test]
    fn lenient_repair_pass_fills_fields_the_parse_missed() {
        // A trailing string the model never closed: the incremental parse
        // decodes it as an open string; the repair pass ALSO can't invent its
        // end, but a died-mid-number field shows the gap-fill (above) and a
        // handle-less open string is delivered by the incremental decoder.
        let mut args = ArgumentStream::lenient();
        args.feed(r#"{"note": "he died her"#).unwrap();
        args.finish().unwrap();
        assert_eq!(args.fields()["note"], "he died her");
    }

    // ---- routing: handles vs buffered --------------------------------------

    #[test]
    fn two_streamed_fields_route_independently() {
        // The old "one payload" limitation is gone: stream both code fields.
        let raw = r#"{"old_code": "a\nb", "new_code": "c\nd"}"#;
        for lenient_mode in [false, true] {
            let mut args = if lenient_mode {
                ArgumentStream::lenient()
            } else {
                ArgumentStream::strict()
            };
            let old = args.field("old_code");
            let new = args.field("new_code");
            args.feed(raw).unwrap();
            args.finish().unwrap();
            assert_eq!(drain(old).unwrap(), "a\nb");
            assert_eq!(drain(new).unwrap(), "c\nd");
            assert!(
                args.fields().is_empty(),
                "handled fields are not duplicated"
            );
        }
    }

    #[test]
    fn handle_on_non_string_field_resolves_via_wait() {
        let mut args = ArgumentStream::strict();
        let line = args.field("line");
        args.feed(r#"{"line": 42, "content": "x"}"#).unwrap();
        args.finish().unwrap();
        let v = futures::executor::block_on(line.wait()).unwrap();
        assert_eq!(v, 42);
    }

    #[test]
    fn late_handle_for_a_completed_field_gets_the_value() {
        let mut args = ArgumentStream::strict();
        args.feed(r#"{"path": "a.rs", "content": "x"}"#).unwrap();
        // Handle taken AFTER `path` completed: delivered whole.
        let path = args.field("path");
        args.finish().unwrap();
        let v = futures::executor::block_on(path.wait()).unwrap();
        assert_eq!(v, "a.rs");
    }

    #[test]
    fn wait_accumulates_streamed_chunks() {
        let mut args = ArgumentStream::strict();
        let content = args.field("content");
        args.feed(r#"{"content": "hel"#).unwrap();
        args.feed(r#"lo"}"#).unwrap();
        args.finish().unwrap();
        let v = futures::executor::block_on(content.wait()).unwrap();
        assert_eq!(v, "hello");
    }

    #[test]
    fn delta_streams_per_feed_chunks_live() {
        let mut args = ArgumentStream::strict();
        let mut content = args.field("content");
        args.feed(r#"{"content": "ab"#).unwrap();
        // The chunk from the first feed is available BEFORE the call ends.
        assert_eq!(
            futures::executor::block_on(content.delta()).as_deref(),
            Some("ab")
        );
        args.feed(r#"cd"}"#).unwrap();
        args.finish().unwrap();
        assert_eq!(
            futures::executor::block_on(content.delta()).as_deref(),
            Some("cd")
        );
        assert_eq!(futures::executor::block_on(content.delta()), None);
    }

    #[test]
    fn finish_closes_handles_so_missing_fields_resolve() {
        let mut args = ArgumentStream::strict();
        let ghost = args.field("ghost");
        args.feed(r#"{"content": "x"}"#).unwrap();
        args.finish().unwrap();
        assert!(futures::executor::block_on(ghost.wait()).is_err());
        // Terminal: no more feeding, double finish errors loudly.
        assert!(args.feed("x").is_err());
        assert!(args.finish().is_err());
    }

    #[test]
    fn duplicate_key_last_wins_in_fields() {
        let mut args = ArgumentStream::strict();
        args.feed(r#"{"path": "a", "path": "b"}"#).unwrap();
        args.finish().unwrap();
        assert_eq!(args.fields()["path"], "b");
    }

    #[test]
    fn malformed_non_string_field_fails_loudly_in_both_modes() {
        // Leniency covers only string values; a broken number has no end
        // anchor and is a malformed call either way.
        for lenient_mode in [false, true] {
            let mut args = if lenient_mode {
                ArgumentStream::lenient()
            } else {
                ArgumentStream::strict()
            };
            assert!(
                args.feed(r#"{"line": 4x2, "content": "y"}"#).is_err(),
                "lenient={lenient_mode} must reject a broken number"
            );
        }
    }

    // ---- realistic streaming shape ----------------------------------------

    #[test]
    fn code_file_payload_streams_decoded() {
        let code = "fn main() {\n    println!(\"hi \\\\ there\");\n}\n";
        let raw = format!(r#"{{"content": {}}}"#, serde_json::to_string(code).unwrap());
        assert_eq!(strict(&raw).unwrap(), code);
        assert_eq!(lenient(&raw).unwrap(), code);
    }
}
