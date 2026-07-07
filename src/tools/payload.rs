//! Live decoded extraction of a single-string tool argument.
//!
//! For the "the model calls a tool whose argument object holds one big string
//! payload" pattern (`{"content": "<entire code file>"}`), [`PayloadExtractor`]
//! turns the streaming argument fragments into the DECODED payload text, live:
//! JSON string escaping undone (`\n` a newline, `\"` a quote, `\uXXXX` the
//! character), fed one fragment at a time with splits at arbitrary positions
//! (including mid-escape). The consumer receives the text exactly as the model
//! "meant" it, as if the model had spoken raw.
//!
//! Two modes:
//! - **Strict** (default): the arguments must be well-formed JSON; malformed
//!   input fails loudly.
//! - **Lenient** (opt-in): for models sloppy at escaping. Because the payload
//!   is the object's only/last field, its TRUE closing quote is the one at the
//!   very end of the arguments text, and the provider signals that end
//!   explicitly (the fragments stop). That makes leniency deterministic: an
//!   unescaped `"` that is not at the true end is literal content, a raw
//!   newline/control character is itself, a `\` before a non-escape character
//!   is a literal backslash, and a stream that just stops (the model forgot to
//!   close the string or the object) still yields the full payload.
//!
//! Non-goals: general streaming JSON repair, multi-string payloads (fields
//! other than the payload string must PRECEDE it and are skipped as normal
//! JSON), provider-specific code (this works on the normalized fragment
//! stream from [`ToolCallDelta`](super::ToolCallDelta)).

use crate::error::{MiniLLMError, Result};

/// Incremental decoder for one string field of a streamed tool call's
/// arguments. Feed the raw argument fragments with [`feed`](Self::feed) (each
/// call returns the decoded text that became unambiguous), then call
/// [`finish`](Self::finish) when the provider signals the call's end (it
/// returns the final flush).
///
/// ```
/// use minillmlib::PayloadExtractor;
///
/// let mut ex = PayloadExtractor::strict("content");
/// let mut out = String::new();
/// // Fragments split anywhere, even mid-escape:
/// out.push_str(&ex.feed(r#"{"content": "line1\"#).unwrap());
/// out.push_str(&ex.feed(r#"nline2"}"#).unwrap());
/// out.push_str(&ex.finish().unwrap());
/// assert_eq!(out, "line1\nline2");
/// ```
#[derive(Debug)]
pub struct PayloadExtractor {
    field: String,
    lenient: bool,
    state: State,
    esc: Esc,
    /// Everything fed so far, verbatim, for loud errors.
    raw: String,
}

#[derive(Debug)]
enum State {
    Prelude(Prelude),
    /// Inside the payload string's body.
    Body,
    /// Lenient only: saw an unescaped `"`; buffering the raw tail until it is
    /// provably the envelope (end of input) or provably literal content.
    MaybeClosed {
        tail: String,
    },
    /// Strict only: the string closed; validating the `}` envelope.
    Closed {
        saw_brace: bool,
    },
    /// A loud error was returned; the extractor is dead.
    Failed,
}

/// Prelude = everything before the payload string's opening quote: the `{`,
/// any PRECEDING fields (skipped as normal JSON), the payload key and its `:`.
#[derive(Debug)]
enum Prelude {
    BeforeBrace,
    BeforeKey,
    InKey {
        key: String,
        escaped: bool,
    },
    AfterKey {
        matched: bool,
    },
    BeforeValue {
        matched: bool,
    },
    /// Skipping a non-payload value. `depth` counts open `{`/`[`; a scalar is
    /// depth 0 outside a string.
    SkipValue {
        depth: u32,
        in_string: bool,
        escaped: bool,
    },
    AfterValue,
}

/// In-body escape state, split so a fragment can end anywhere inside an
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

impl PayloadExtractor {
    /// Strict extractor: the arguments must be well-formed JSON; malformed
    /// input (bad escape, unescaped control char, unterminated string/object)
    /// fails loudly.
    pub fn strict(field: impl Into<String>) -> Self {
        Self::new(field, false)
    }

    /// Lenient extractor: tolerates sloppy escaping (see the module docs for
    /// the deterministic rules). Prefer strict unless the model demonstrably
    /// needs it.
    pub fn lenient(field: impl Into<String>) -> Self {
        Self::new(field, true)
    }

    fn new(field: impl Into<String>, lenient: bool) -> Self {
        Self {
            field: field.into(),
            lenient,
            state: State::Prelude(Prelude::BeforeBrace),
            esc: Esc::default(),
            raw: String::new(),
        }
    }

    /// Feed one raw argument fragment; returns the decoded payload text that
    /// became unambiguous with it (possibly empty: prelude bytes, or a
    /// held-back partial escape). After an error the extractor is dead and
    /// every further call errors.
    pub fn feed(&mut self, fragment: &str) -> Result<String> {
        if matches!(self.state, State::Failed) {
            return Err(self.error("extractor already failed"));
        }
        self.raw.push_str(fragment);
        let mut out = String::new();
        for c in fragment.chars() {
            if let Err(e) = self.step(c, &mut out) {
                self.state = State::Failed;
                return Err(e);
            }
        }
        Ok(out)
    }

    /// The provider signaled the end of the call's arguments: validate the
    /// envelope and flush anything held back. Consumes the extractor.
    pub fn finish(mut self) -> Result<String> {
        match std::mem::replace(&mut self.state, State::Failed) {
            State::Failed => Err(self.error("extractor already failed")),
            State::Prelude(_) => Err(self.error(&format!(
                "payload field '{}' never started (arguments incomplete or field missing)",
                self.field
            ))),
            State::Body => {
                if self.lenient {
                    // The model just stopped (forgot the closing quote and
                    // brace): the payload is everything decoded so far, plus a
                    // lenient flush of any partial escape. Never drop payload.
                    let mut out = String::new();
                    match std::mem::take(&mut self.esc) {
                        Esc::None => {}
                        Esc::Slash => out.push('\\'),
                        Esc::Unicode(buf) => {
                            out.push_str("\\u");
                            out.push_str(&buf);
                        }
                        Esc::AwaitLowSlash { .. } => out.push('\u{FFFD}'),
                        Esc::AwaitLowU { .. } => {
                            out.push('\u{FFFD}');
                            out.push('\\');
                        }
                        Esc::AwaitLowHex { buf, .. } => {
                            out.push('\u{FFFD}');
                            out.push_str("\\u");
                            out.push_str(&buf);
                        }
                    }
                    Ok(out)
                } else {
                    Err(self.error("payload string never closed"))
                }
            }
            // Lenient: the buffered tail matched the envelope grammar all the
            // way to the true end, so the quote that opened it WAS the closing
            // quote. Accept even without a `}` (model forgot the brace).
            State::MaybeClosed { .. } => Ok(String::new()),
            State::Closed { saw_brace } => {
                if saw_brace {
                    Ok(String::new())
                } else {
                    Err(self.error("arguments object never closed after the payload string"))
                }
            }
        }
    }

    fn error(&self, what: &str) -> MiniLLMError {
        MiniLLMError::MalformedResponse(format!(
            "streamed tool arguments: {} (field '{}', raw: {})",
            what, self.field, self.raw
        ))
    }

    fn step(&mut self, c: char, out: &mut String) -> Result<()> {
        match &mut self.state {
            State::Prelude(_) => self.step_prelude(c),
            State::Body => self.step_body(c, out),
            State::MaybeClosed { tail } => {
                // Envelope grammar after the candidate closing quote:
                // ws* `}`? ws* then end-of-input. A char that still fits keeps
                // the candidacy; one that doesn't proves the quote (and the
                // buffered tail) were literal content.
                let fits = c.is_whitespace() || (c == '}' && !tail.contains('}'));
                if fits {
                    tail.push(c);
                    return Ok(());
                }
                // Violation: the quote was literal. Re-decode it, the buffered
                // tail, and this char as body content.
                let tail = std::mem::take(tail);
                self.state = State::Body;
                out.push('"');
                for tc in tail.chars() {
                    self.step_body(tc, out)?;
                }
                self.step_body(c, out)
            }
            State::Closed { saw_brace } => {
                if c.is_whitespace() {
                    Ok(())
                } else if c == '}' && !*saw_brace {
                    *saw_brace = true;
                    Ok(())
                } else {
                    Err(self.error(&format!("unexpected '{c}' after the payload string closed")))
                }
            }
            State::Failed => unreachable!("feed guards Failed"),
        }
    }

    fn step_prelude(&mut self, c: char) -> Result<()> {
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
                    return Err(self.error(&format!(
                        "arguments object closed before the payload field '{}'",
                        self.field
                    )));
                } else {
                    return Err(self.error(&format!("expected a field name, got '{c}'")));
                }
            }
            Prelude::InKey { key, escaped } => {
                if *escaped {
                    // Keys only need `\"`/`\\` fidelity for termination; the
                    // payload field name itself must be a plain identifier.
                    key.push(c);
                    *escaped = false;
                } else if c == '\\' {
                    *escaped = true;
                } else if c == '"' {
                    let matched = *key == self.field;
                    *p = Prelude::AfterKey { matched };
                } else {
                    key.push(c);
                }
            }
            Prelude::AfterKey { matched } => {
                if c.is_whitespace() {
                } else if c == ':' {
                    *p = Prelude::BeforeValue { matched: *matched };
                } else {
                    return Err(self.error(&format!("expected ':' after a field name, got '{c}'")));
                }
            }
            Prelude::BeforeValue { matched } => {
                if c.is_whitespace() {
                } else if *matched {
                    if c == '"' {
                        self.state = State::Body;
                    } else {
                        return Err(self.error(&format!(
                            "payload field '{}' is not a string (starts with '{c}')",
                            self.field
                        )));
                    }
                } else if c == '"' {
                    *p = Prelude::SkipValue {
                        depth: 0,
                        in_string: true,
                        escaped: false,
                    };
                } else if c == '{' || c == '[' {
                    *p = Prelude::SkipValue {
                        depth: 1,
                        in_string: false,
                        escaped: false,
                    };
                } else {
                    // scalar (number/true/false/null)
                    *p = Prelude::SkipValue {
                        depth: 0,
                        in_string: false,
                        escaped: false,
                    };
                }
            }
            Prelude::SkipValue {
                depth,
                in_string,
                escaped,
            } => {
                if *in_string {
                    if *escaped {
                        *escaped = false;
                    } else if c == '\\' {
                        *escaped = true;
                    } else if c == '"' {
                        *in_string = false;
                        if *depth == 0 {
                            *p = Prelude::AfterValue;
                        }
                    }
                } else if *depth > 0 {
                    match c {
                        '"' => *in_string = true,
                        '{' | '[' => *depth += 1,
                        '}' | ']' => {
                            *depth -= 1;
                            if *depth == 0 {
                                *p = Prelude::AfterValue;
                            }
                        }
                        _ => {}
                    }
                } else {
                    // scalar at depth 0: ends at `,`, `}`, or whitespace.
                    if c == ',' {
                        *p = Prelude::BeforeKey;
                    } else if c == '}' {
                        return Err(self.error(&format!(
                            "arguments object closed before the payload field '{}'",
                            self.field
                        )));
                    } else if c.is_whitespace() {
                        *p = Prelude::AfterValue;
                    }
                }
            }
            Prelude::AfterValue => {
                if c.is_whitespace() {
                } else if c == ',' {
                    *p = Prelude::BeforeKey;
                } else if c == '}' {
                    return Err(self.error(&format!(
                        "arguments object closed before the payload field '{}'",
                        self.field
                    )));
                } else {
                    return Err(self.error(&format!("expected ',' or '}}', got '{c}'")));
                }
            }
        }
        Ok(())
    }

    fn step_body(&mut self, c: char, out: &mut String) -> Result<()> {
        match std::mem::take(&mut self.esc) {
            Esc::None => match c {
                '\\' => self.esc = Esc::Slash,
                '"' => {
                    self.state = if self.lenient {
                        // Cannot know yet whether this is THE closing quote
                        // (the one at the true end) or literal content.
                        State::MaybeClosed {
                            tail: String::new(),
                        }
                    } else {
                        State::Closed { saw_brace: false }
                    };
                }
                c if (c as u32) < 0x20 => {
                    if self.lenient {
                        out.push(c); // a raw newline/control char is itself
                    } else {
                        return Err(self.error(&format!(
                            "unescaped control character U+{:04X} in payload string",
                            c as u32
                        )));
                    }
                }
                c => out.push(c),
            },
            Esc::Slash => match c {
                '"' => out.push('"'),
                '\\' => out.push('\\'),
                '/' => out.push('/'),
                'b' => out.push('\u{8}'),
                'f' => out.push('\u{c}'),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => self.esc = Esc::Unicode(String::new()),
                other => {
                    if self.lenient {
                        // `\` before a non-escape character is a literal
                        // backslash, and the character is itself.
                        out.push('\\');
                        return self.step_body(other, out);
                    }
                    return Err(self.error(&format!("invalid escape '\\{other}'")));
                }
            },
            Esc::Unicode(mut buf) => {
                if c.is_ascii_hexdigit() {
                    buf.push(c);
                    if buf.len() == 4 {
                        let code = u16::from_str_radix(&buf, 16).expect("4 hex digits");
                        self.take_unit(code, out)?;
                    } else {
                        self.esc = Esc::Unicode(buf);
                    }
                } else {
                    if self.lenient {
                        // Not a real escape after all: literal text.
                        out.push_str("\\u");
                        out.push_str(&buf);
                        return self.step_body(c, out);
                    }
                    return Err(self.error(&format!("invalid \\u escape '\\u{buf}{c}'")));
                }
            }
            Esc::AwaitLowSlash { high } => {
                if c == '\\' {
                    self.esc = Esc::AwaitLowU { high };
                } else {
                    if self.lenient {
                        out.push('\u{FFFD}'); // unpaired high surrogate
                        return self.step_body(c, out);
                    }
                    return Err(self.error("unpaired \\u surrogate in payload string"));
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
                        out.push('\u{FFFD}');
                        // The consumed `\` starts a fresh escape with `c`.
                        self.esc = Esc::Slash;
                        return self.step_body(c, out);
                    }
                    return Err(self.error("unpaired \\u surrogate in payload string"));
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
                            out.push(char::from_u32(combined).expect("valid surrogate pair"));
                        } else {
                            if !self.lenient {
                                return Err(self.error("unpaired \\u surrogate in payload string"));
                            }
                            out.push('\u{FFFD}');
                            // The second unit stands alone: another high
                            // surrogate re-arms the pairing, a normal char is
                            // itself.
                            self.take_unit(low, out)?;
                        }
                    } else {
                        self.esc = Esc::AwaitLowHex { high, buf };
                    }
                } else {
                    if self.lenient {
                        out.push('\u{FFFD}');
                        out.push_str("\\u");
                        out.push_str(&buf);
                        return self.step_body(c, out);
                    }
                    return Err(self.error("unpaired \\u surrogate in payload string"));
                }
            }
        }
        Ok(())
    }

    /// A decoded `\uXXXX` unit: a plain char is pushed, a high surrogate arms
    /// the pair-awaiting state, a lone low surrogate is malformed.
    fn take_unit(&mut self, code: u16, out: &mut String) -> Result<()> {
        if (0xD800..=0xDBFF).contains(&code) {
            self.esc = Esc::AwaitLowSlash { high: code };
            Ok(())
        } else if (0xDC00..=0xDFFF).contains(&code) {
            if self.lenient {
                out.push('\u{FFFD}');
                Ok(())
            } else {
                Err(self.error("unpaired \\u surrogate in payload string"))
            }
        } else {
            out.push(char::from_u32(code as u32).expect("non-surrogate BMP code point"));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Run `raw` through an extractor split at EVERY char boundary (two
    /// fragments), plus fully char-by-char, and assert the decoded output is
    /// identical each time. This is the core guarantee: splits at arbitrary
    /// positions (mid-escape included) never change the result.
    fn all_splits(field: &str, lenient: bool, raw: &str) -> Result<String> {
        let make = || {
            if lenient {
                PayloadExtractor::lenient(field)
            } else {
                PayloadExtractor::strict(field)
            }
        };
        let run = |fragments: Vec<&str>| -> Result<String> {
            let mut ex = make();
            let mut out = String::new();
            for f in fragments {
                out.push_str(&ex.feed(f)?);
            }
            out.push_str(&ex.finish()?);
            Ok(out)
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
        let mut ex = make();
        let mut out = String::new();
        let mut failed = false;
        for c in raw.chars() {
            match ex.feed(&c.to_string()) {
                Ok(s) => out.push_str(&s),
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        if !failed {
            match (ex.finish(), &reference) {
                (Ok(s), Ok(r)) => {
                    out.push_str(&s);
                    assert_eq!(&out, r, "char-by-char diverged for {raw:?}");
                }
                (Err(_), Err(_)) => {}
                (a, b) => panic!("char-by-char changed ok/err for {raw:?}: {a:?} vs {b:?}"),
            }
        } else {
            assert!(reference.is_err(), "char-by-char failed but whole ok");
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
    fn strict_skips_preceding_fields() {
        // Preceding string, number, bool, nested object/array values.
        assert_eq!(
            strict(
                r#"{"lang": "py\"x", "n": 4.5e2, "flag": true,
                   "meta": {"a": ["b", {"c": 1}]}, "content": "payload"}"#
            )
            .unwrap(),
            "payload"
        );
    }

    #[test]
    fn strict_handles_empty_payload_and_whitespace_envelope() {
        assert_eq!(strict("  { \"content\" : \"\" }  ").unwrap(), "");
    }

    // ---- strict: malformed inputs fail loudly -----------------------------

    #[test]
    fn strict_rejects_malformed() {
        for bad in [
            r#"{"content": "unterminated"#,       // no closing quote
            r#"{"content": "no brace""#,          // string closed, object not
            r#"{"content": "bad \q escape"}"#,    // invalid escape
            r#"{"content": "lone \ud800 high"}"#, // unpaired surrogate
            r#"{"content": 42}"#,                 // not a string
            r#"{"other": "x"}"#,                  // field missing
            r#"{"content": "a"} trailing"#,       // junk after envelope
            "{\"content\": \"raw\nnewline\"}",    // unescaped control char
            r#"["content"]"#,                     // not an object
        ] {
            assert!(strict(bad).is_err(), "must reject: {bad:?}");
        }
    }

    #[test]
    fn error_carries_the_raw_text() {
        let mut ex = PayloadExtractor::strict("content");
        let err = ex.feed(r#"{"content": 42"#).unwrap_err().to_string();
        assert!(
            err.contains(r#"{"content": 42"#),
            "raw text in error: {err}"
        );
        // Dead after an error.
        assert!(ex.feed("x").is_err());
    }

    // ---- lenient: sloppy models ------------------------------------------

    #[test]
    fn lenient_decodes_well_formed_identically() {
        let raw = r#"{"content":"a\"b\\c\nd 😀"}"#;
        assert_eq!(lenient(raw).unwrap(), strict(raw).unwrap());
    }

    #[test]
    fn lenient_unescaped_quote_mid_content_is_literal() {
        // The " before ` hi` is not at the true end, so it is content.
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
        // The stream just stops: the payload survives in full.
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

    #[test]
    fn lenient_still_rejects_a_broken_prelude() {
        // Leniency is about the string body; a prelude that never reaches the
        // field is a malformed call either way.
        assert!(lenient(r#"{"other": "x"}"#).is_err());
        assert!(lenient(r#"not json at all"#).is_err());
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
