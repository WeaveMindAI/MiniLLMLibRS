//! Parsing context management
//!
//! Tracks where we are in the JSON structure during parsing.
//! This is crucial for handling edge cases like missing quotes.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextValue {
    /// We're parsing an object key (the part before `:`)
    ObjectKey,
    /// We're parsing an object value (the part after `:`)
    ObjectValue,
    /// We're parsing an array element
    Array,
}

#[derive(Debug, Default)]
pub struct JsonContext {
    /// Stack of contexts (most recent is last)
    stack: Vec<ContextValue>,
}

impl JsonContext {
    /// Create a new empty context
    pub fn new() -> Self {
        Self { stack: Vec::new() }
    }

    pub fn current(&self) -> Option<ContextValue> {
        self.stack.last().copied()
    }

    pub fn is_empty(&self) -> bool {
        self.stack.is_empty()
    }

    pub fn contains(&self, value: ContextValue) -> bool {
        self.stack.contains(&value)
    }

    pub fn reset(&mut self) {
        self.stack.pop();
    }

    pub fn set(&mut self, value: ContextValue) {
        self.stack.push(value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_stack() {
        let mut ctx = JsonContext::new();
        assert!(ctx.is_empty());
        assert_eq!(ctx.current(), None);

        ctx.set(ContextValue::ObjectKey);
        assert!(!ctx.is_empty());
        assert_eq!(ctx.current(), Some(ContextValue::ObjectKey));

        ctx.set(ContextValue::Array);
        assert_eq!(ctx.current(), Some(ContextValue::Array));
        assert!(ctx.contains(ContextValue::ObjectKey));

        ctx.reset();
        assert_eq!(ctx.current(), Some(ContextValue::ObjectKey));

        ctx.reset();
        assert!(ctx.is_empty());
    }
}
