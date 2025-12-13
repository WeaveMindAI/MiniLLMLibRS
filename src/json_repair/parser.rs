//! JSON Parser - 1:1 port of Python json_repair library
//!
//! This module contains the main parser and all parsing functions.

use super::context::{ContextValue, JsonContext as Context};
use super::value::JsonValue;
use super::JsonRepairError;

/// Options for JSON repair
#[derive(Debug, Clone)]
pub struct RepairOptions {
    /// Skip trying standard JSON parsing first
    pub skip_json_loads: bool,
    /// Enable strict mode (errors instead of repairs for some cases)
    pub strict: bool,
    /// Keep streaming output stable
    pub stream_stable: bool,
    /// If true (default), escape non-ASCII characters to \uXXXX in output
    /// If false, preserve unicode characters as-is
    pub ensure_ascii: bool,
}

impl Default for RepairOptions {
    fn default() -> Self {
        Self {
            skip_json_loads: false,
            strict: false,
            stream_stable: false,
            ensure_ascii: true, // Match Python's default
        }
    }
}

/// String delimiters that can start a string
pub const STRING_DELIMITERS: &[char] = &['"', '\'', '"', '"'];

/// Characters that can appear in a number
pub const NUMBER_CHARS: &[char] = &[
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', 'e', 'E', '/', ',', '_',
];

/// The main JSON parser struct
pub struct JsonParser<'a> {
    pub chars: Vec<char>,
    pub index: usize,
    pub context: Context,
    pub options: &'a RepairOptions,
}

impl<'a> JsonParser<'a> {
    pub fn new(json_str: &str, options: &'a RepairOptions) -> Self {
        Self {
            chars: json_str.chars().collect(),
            index: 0,
            context: Context::new(),
            options,
        }
    }

    /// Main parse entry point
    pub fn parse(&mut self) -> Result<JsonValue, JsonRepairError> {
        let json = self.parse_json()?;

        if self.index < self.chars.len() {
            let mut results = vec![json];

            while self.index < self.chars.len() {
                self.context.reset();
                if let Ok(j) = self.parse_json() {
                    if !j.is_empty() {
                        if !results.is_empty()
                            && (Self::is_same_object(&results[results.len() - 1], &j)
                                || results[results.len() - 1].is_empty())
                        {
                            results.pop();
                        }
                        results.push(j);
                    } else {
                        self.index += 1;
                    }
                } else {
                    self.index += 1;
                }
            }

            if results.len() == 1 {
                return Ok(results.remove(0));
            } else if self.options.strict {
                return Err(JsonRepairError::ParseError(
                    "Multiple top-level JSON elements found".to_string(),
                ));
            }

            return Ok(JsonValue::Array(results));
        }

        Ok(json)
    }

    /// Parse a single JSON value
    pub fn parse_json(&mut self) -> Result<JsonValue, JsonRepairError> {
        loop {
            let Some(c) = self.get_char_at(0) else {
                return Ok(JsonValue::String(String::new()));
            };

            match c {
                '{' => {
                    self.index += 1;
                    return self.parse_object();
                }
                '[' => {
                    self.index += 1;
                    return self.parse_array();
                }
                '#' | '/' => {
                    return self.parse_comment();
                }
                _ if !self.context.is_empty()
                    && (STRING_DELIMITERS.contains(&c) || c.is_alphabetic()) =>
                {
                    return self.parse_string();
                }
                _ if !self.context.is_empty() && (c.is_ascii_digit() || c == '-' || c == '.') => {
                    return self.parse_number();
                }
                _ => {
                    self.index += 1;
                }
            }
        }
    }

    #[inline]
    pub fn get_char_at(&self, offset: isize) -> Option<char> {
        let idx = self.index as isize + offset;
        if idx >= 0 && (idx as usize) < self.chars.len() {
            Some(self.chars[idx as usize])
        } else {
            None
        }
    }

    pub fn skip_whitespaces(&mut self) {
        while let Some(c) = self.get_char_at(0) {
            if c.is_whitespace() {
                self.index += 1;
            } else {
                break;
            }
        }
    }

    pub fn scroll_whitespaces(&self, start_offset: usize) -> usize {
        let mut offset = start_offset;
        while let Some(c) = self.get_char_at(offset as isize) {
            if c.is_whitespace() {
                offset += 1;
            } else {
                break;
            }
        }
        offset
    }

    pub fn skip_to_character(&self, targets: &[char], start_offset: usize) -> usize {
        let mut i = self.index + start_offset;
        let mut backslashes = 0;

        while i < self.chars.len() {
            let ch = self.chars[i];

            if ch == '\\' {
                backslashes += 1;
                i += 1;
                continue;
            }

            if targets.contains(&ch) && (backslashes % 2 == 0) {
                return i - self.index;
            }

            backslashes = 0;
            i += 1;
        }

        self.chars.len() - self.index
    }

    fn is_same_object(a: &JsonValue, b: &JsonValue) -> bool {
        match (a, b) {
            (JsonValue::Object(obj_a), JsonValue::Object(obj_b)) => {
                if obj_a.is_empty() || obj_b.is_empty() {
                    return false;
                }
                obj_a
                    .iter()
                    .all(|(k, _)| obj_b.iter().any(|(k2, _)| k == k2))
            }
            (JsonValue::Array(arr_a), JsonValue::Array(arr_b)) => {
                if arr_a.is_empty() || arr_b.is_empty() {
                    return false;
                }
                arr_a.len() == arr_b.len()
            }
            _ => false,
        }
    }

    pub fn parse_comment(&mut self) -> Result<JsonValue, JsonRepairError> {
        let char = self.get_char_at(0);
        let mut termination_chars = vec!['\n', '\r'];

        if self.context.contains(ContextValue::Array) {
            termination_chars.push(']');
        }
        if self.context.contains(ContextValue::ObjectValue) {
            termination_chars.push('}');
        }
        if self.context.contains(ContextValue::ObjectKey) {
            termination_chars.push(':');
        }

        match char {
            Some('#') => {
                // Line comment starting with #
                while let Some(c) = self.get_char_at(0) {
                    if termination_chars.contains(&c) {
                        break;
                    }
                    self.index += 1;
                }
            }
            Some('/') => {
                let next_char = self.get_char_at(1);
                match next_char {
                    Some('/') => {
                        // Line comment //
                        self.index += 2;
                        while let Some(c) = self.get_char_at(0) {
                            if termination_chars.contains(&c) {
                                break;
                            }
                            self.index += 1;
                        }
                    }
                    Some('*') => {
                        // Block comment /* */
                        self.index += 2;
                        loop {
                            let c = self.get_char_at(0);
                            if c.is_none() {
                                break;
                            }
                            self.index += 1;
                            if c == Some('*') && self.get_char_at(0) == Some('/') {
                                self.index += 1;
                                break;
                            }
                        }
                    }
                    _ => {
                        self.index += 1;
                    }
                }
            }
            _ => {}
        }

        if self.context.is_empty() {
            self.parse_json()
        } else {
            Ok(JsonValue::String(String::new()))
        }
    }

    pub fn parse_number(&mut self) -> Result<JsonValue, JsonRepairError> {
        let mut number_str = String::new();
        let is_array = self.context.current() == Some(ContextValue::Array);

        while let Some(c) = self.get_char_at(0) {
            if NUMBER_CHARS.contains(&c) && !(is_array && c == ',') {
                if c != '_' {
                    number_str.push(c);
                }
                self.index += 1;
            } else {
                break;
            }
        }

        // Handle trailing invalid characters
        if !number_str.is_empty()
            && matches!(
                number_str.chars().last(),
                Some('-') | Some('e') | Some('E') | Some('/') | Some(',')
            )
        {
            number_str.pop();
            self.index -= 1;
        } else if self.get_char_at(0).is_some_and(|c| c.is_alphabetic()) {
            // This was a string instead
            self.index -= number_str.len();
            return self.parse_string();
        }

        // Try to parse as number
        if number_str.contains(',') {
            return Ok(JsonValue::String(number_str));
        }

        if number_str.contains('.') || number_str.contains('e') || number_str.contains('E') {
            if let Ok(f) = number_str.parse::<f64>() {
                return Ok(JsonValue::Float(f));
            }
        } else if let Ok(i) = number_str.parse::<i64>() {
            return Ok(JsonValue::Integer(i));
        }

        Ok(JsonValue::String(number_str))
    }

    pub fn parse_array(&mut self) -> Result<JsonValue, JsonRepairError> {
        let mut arr: Vec<JsonValue> = Vec::new();
        self.context.set(ContextValue::Array);

        while let Some(c) = self.get_char_at(0) {
            if c == ']' || c == '}' {
                break;
            }

            self.skip_whitespaces();

            let value = if STRING_DELIMITERS.contains(&c) {
                // Check if this is actually an object (string followed by :)
                let i = self.skip_to_character(&[c], 1);
                let j = self.scroll_whitespaces(i + 1);
                if self.get_char_at(j as isize) == Some(':') {
                    self.parse_object()?
                } else {
                    self.parse_string()?
                }
            } else {
                self.parse_json()?
            };

            // Ignore unquoted ellipsis (previous char is .), keep quoted ones
            if value == JsonValue::String("...".to_string()) && self.get_char_at(-1) == Some('.') {
                // Ignore unquoted ellipsis
            } else if !value.is_empty()
                || self.get_char_at(0) == Some(']')
                || self.get_char_at(0) == Some(',')
            {
                // Special case: if parse_object returned an array (due to duplicate key handling),
                // flatten it into our array instead of nesting it
                if let JsonValue::Array(inner) = &value {
                    // Check if this looks like our special duplicate-key result:
                    // [Object, Object] or [Object, Array] where the first element is an object
                    if inner.len() == 2 {
                        if let JsonValue::Object(_) = &inner[0] {
                            // Flatten: add each element separately
                            for elem in inner.clone() {
                                arr.push(elem);
                            }
                            continue;
                        }
                    }
                }
                arr.push(value);
            } else {
                self.index += 1;
            }

            // Skip whitespace and commas
            while let Some(c) = self.get_char_at(0) {
                if c == ']' {
                    break;
                }
                if c.is_whitespace() || c == ',' {
                    self.index += 1;
                } else {
                    break;
                }
            }
        }

        // Skip closing bracket if present
        if self.get_char_at(0) == Some(']') {
            self.index += 1;
        }

        self.context.reset();
        Ok(JsonValue::Array(arr))
    }

    pub fn parse_object(&mut self) -> Result<JsonValue, JsonRepairError> {
        let mut obj: Vec<(String, JsonValue)> = Vec::new();
        let start_index = self.index;

        while self.get_char_at(0).unwrap_or('}') != '}' {
            self.skip_whitespaces();

            // Handle stray colon
            if self.get_char_at(0) == Some(':') {
                self.index += 1;
            }

            self.context.set(ContextValue::ObjectKey);
            let rollback_index = self.index;

            // Parse key
            let mut key = String::new();
            while self.get_char_at(0).is_some() {
                if self.get_char_at(0) == Some('[') && key.is_empty() {
                    // Merge with previous array if exists
                    let prev_key = obj.last().map(|(k, _)| k.clone());
                    let prev_is_array = obj
                        .last()
                        .map(|(_, v)| matches!(v, JsonValue::Array(_)))
                        .unwrap_or(false);

                    if let Some(pk) = prev_key {
                        if prev_is_array {
                            // If the previous key's value is an array, parse the new array and merge
                            self.index += 1;
                            let new_array = self.parse_array()?;

                            if let JsonValue::Array(new_arr) = new_array {
                                // Find the previous key's value and extend it
                                for (k, v) in obj.iter_mut() {
                                    if k == &pk {
                                        if let JsonValue::Array(prev_arr) = v {
                                            // Merge and flatten: new_array[0] if len == 1 and is array, else new_array
                                            if new_arr.len() == 1 {
                                                if let JsonValue::Array(inner) = &new_arr[0] {
                                                    prev_arr.extend(inner.clone());
                                                } else {
                                                    prev_arr.extend(new_arr.clone());
                                                }
                                            } else {
                                                prev_arr.extend(new_arr.clone());
                                            }
                                        }
                                        break;
                                    }
                                }
                            }

                            self.skip_whitespaces();
                            if self.get_char_at(0) == Some(',') {
                                self.index += 1;
                            }
                            self.skip_whitespaces();
                            continue;
                        }
                    }
                }

                key = match self.parse_string()? {
                    JsonValue::String(s) => s,
                    _ => String::new(),
                };

                if key.is_empty() {
                    self.skip_whitespaces();
                }

                if !key.is_empty()
                    || self.get_char_at(0) == Some(':')
                    || self.get_char_at(0) == Some('}')
                {
                    if key.is_empty() && self.options.strict {
                        return Err(JsonRepairError::ParseError("Empty key found".to_string()));
                    }
                    break;
                }
            }

            // Check for duplicate keys
            if self.context.contains(ContextValue::Array) && obj.iter().any(|(k, _)| k == &key) {
                if self.options.strict {
                    return Err(JsonRepairError::ParseError(
                        "Duplicate key found".to_string(),
                    ));
                }
                // Roll back and parse remaining as new object
                self.index = rollback_index;
                self.context.reset();
                let remaining = self.parse_object()?;

                // Return an array containing the current object and the remaining object/array
                let mut result = vec![JsonValue::Object(obj)];
                result.push(remaining);
                return Ok(JsonValue::Array(result));
            }

            self.skip_whitespaces();

            if self.get_char_at(0).unwrap_or('}') == '}' {
                continue;
            }

            // Handle missing colon
            if self.get_char_at(0) != Some(':') && self.options.strict {
                return Err(JsonRepairError::ParseError(
                    "Missing ':' after key".to_string(),
                ));
            }

            self.index += 1;
            self.context.reset();
            self.context.set(ContextValue::ObjectValue);

            self.skip_whitespaces();

            // Parse value
            let value = if self.get_char_at(0) == Some(',') || self.get_char_at(0) == Some('}') {
                if self.options.strict {
                    return Err(JsonRepairError::ParseError(
                        "Parsed value is empty".to_string(),
                    ));
                }
                JsonValue::String(String::new())
            } else {
                self.parse_json()?
            };

            self.context.reset();
            obj.push((key, value));

            // Skip comma or quote
            if matches!(self.get_char_at(0), Some(',') | Some('\'') | Some('"')) {
                self.index += 1;
            }

            self.skip_whitespaces();
        }

        self.index += 1;

        // If object is empty but has content, try parsing as array
        if obj.is_empty() && self.index - start_index > 2 {
            if self.options.strict {
                return Err(JsonRepairError::ParseError(
                    "Parsed object is empty".to_string(),
                ));
            }
            self.index = start_index;
            return self.parse_array();
        }

        // Check for additional key-value pairs after closing brace
        if !self.context.is_empty() {
            return Ok(JsonValue::Object(obj));
        }

        self.skip_whitespaces();
        if self.get_char_at(0) != Some(',') {
            return Ok(JsonValue::Object(obj));
        }
        self.index += 1;
        self.skip_whitespaces();

        if !STRING_DELIMITERS.contains(&self.get_char_at(0).unwrap_or(' ')) {
            return Ok(JsonValue::Object(obj));
        }

        if !self.options.strict {
            if let Ok(JsonValue::Object(additional)) = self.parse_object() {
                for (k, v) in additional {
                    obj.push((k, v));
                }
            }
        }

        Ok(JsonValue::Object(obj))
    }

    pub fn parse_string(&mut self) -> Result<JsonValue, JsonRepairError> {
        // Skip non-delimiter characters at the start
        while let Some(c) = self.get_char_at(0) {
            if STRING_DELIMITERS.contains(&c) || c.is_alphanumeric() {
                break;
            }
            if c == '#' || c == '/' {
                return self.parse_comment();
            }
            self.index += 1;
        }

        let Some(first_char) = self.get_char_at(0) else {
            return Ok(JsonValue::String(String::new()));
        };

        // Determine delimiter and missing_quotes flag
        let (lstring_delimiter, rstring_delimiter, missing_quotes) = if first_char == '\'' {
            ('\'', '\'', false)
        } else if STRING_DELIMITERS.contains(&first_char) && first_char != '\'' {
            ('"', '"', false)
        } else if first_char.is_alphanumeric() {
            // Check for boolean/null first (only if not in OBJECT_KEY context)
            if matches!(first_char.to_ascii_lowercase(), 't' | 'f' | 'n')
                && self.context.current() != Some(ContextValue::ObjectKey)
            {
                if let Some(value) = self.parse_boolean_or_null() {
                    return Ok(value);
                }
            }
            ('"', '"', true)
        } else {
            return Ok(JsonValue::String(String::new()));
        };

        if !missing_quotes {
            self.index += 1;

            // Check for ```json block
            if self.get_char_at(0) == Some('`') {
                if let Some(value) = self.parse_json_llm_block()? {
                    return Ok(value);
                }
            }

            // Handle doubled quotes
            if self.get_char_at(0) == Some(lstring_delimiter) {
                // Empty string case
                if let Some(next) = self.get_char_at(1) {
                    if (self.context.current() == Some(ContextValue::ObjectKey) && next == ':')
                        || (self.context.current() == Some(ContextValue::ObjectValue)
                            && (next == ',' || next == '}'))
                        || (self.context.current() == Some(ContextValue::Array)
                            && (next == ',' || next == ']'))
                    {
                        self.index += 1;
                        return Ok(JsonValue::String(String::new()));
                    }
                    // Tripled quotes
                    if next == lstring_delimiter {
                        if self.options.strict {
                            return Err(JsonRepairError::ParseError(
                                "Found doubled quotes followed by another quote.".to_string(),
                            ));
                        }
                        return Ok(JsonValue::String(String::new()));
                    }
                }

                // Check for ""..."" pattern
                let i = self.skip_to_character(&[rstring_delimiter], 1);
                if self.get_char_at(i as isize).is_some() {
                    if self.get_char_at((i + 1) as isize) == Some(rstring_delimiter) {
                        // Valid doubled quotes pattern
                        self.index += 1;
                    } else {
                        // Check what follows
                        let j = self.scroll_whitespaces(1);
                        if let Some(after_ws) = self.get_char_at(j as isize) {
                            if STRING_DELIMITERS.contains(&after_ws)
                                || after_ws == '{'
                                || after_ws == '['
                            {
                                if self.options.strict {
                                    return Err(JsonRepairError::ParseError("Found doubled quotes followed by another quote while parsing a string.".to_string()));
                                }
                                self.index += 1;
                                return Ok(JsonValue::String(String::new()));
                            } else if after_ws != ',' && after_ws != ']' && after_ws != '}' {
                                self.index += 1;
                            }
                        }
                    }
                }
            }
        }

        // Main string accumulation loop
        let mut string_acc = String::new();
        let mut unmatched_delimiter = false;
        let doubled_quotes = false;

        while let Some(char) = self.get_char_at(0) {
            if char == rstring_delimiter {
                break;
            }

            // Handle missing quotes termination
            if missing_quotes {
                if self.context.current() == Some(ContextValue::ObjectKey)
                    && (char == ':' || char.is_whitespace())
                {
                    break;
                }
                if self.context.current() == Some(ContextValue::Array)
                    && (char == ']' || char == ',')
                {
                    break;
                }
            }

            // Handle comma/brace in OBJECT_VALUE context
            if !self.options.stream_stable
                && self.context.current() == Some(ContextValue::ObjectValue)
                && (char == ',' || char == '}')
                && (string_acc.is_empty() || !string_acc.ends_with(rstring_delimiter))
            {
                let mut rstring_delimiter_missing = true;

                // Check if next char is escaped
                if self.get_char_at(1) == Some('\\') {
                    rstring_delimiter_missing = false;
                }

                let i = self.skip_to_character(&[rstring_delimiter], 1);
                if self.get_char_at(i as isize).is_some() {
                    let j = self.scroll_whitespaces(i + 1);
                    if let Some(next_c) = self.get_char_at(j as isize) {
                        if next_c == ',' || next_c == '}' {
                            rstring_delimiter_missing = false;
                        } else {
                            // Check for new key pattern
                            let k = self.skip_to_character(&[lstring_delimiter], j);
                            if self.get_char_at(k as isize).is_none() {
                                rstring_delimiter_missing = false;
                            } else {
                                let m = self.scroll_whitespaces(k + 1);
                                if self.get_char_at(m as isize) != Some(':') {
                                    rstring_delimiter_missing = false;
                                }
                            }
                        }
                    } else {
                        rstring_delimiter_missing = false;
                    }
                } else {
                    // No delimiter found - check for colon (new key without quotes)
                    let i = self.skip_to_character(&[':'], 1);
                    if self.get_char_at(i as isize).is_some() {
                        // Found colon - break
                    } else {
                        let j = self.scroll_whitespaces(1);
                        let k = self.skip_to_character(&['}'], j);
                        if k - j > 1 {
                            rstring_delimiter_missing = false;
                        } else if self.get_char_at(k as isize).is_some() {
                            // Check for unmatched braces
                            let open = string_acc.chars().filter(|&c| c == '{').count();
                            let close = string_acc.chars().filter(|&c| c == '}').count();
                            if open > close {
                                rstring_delimiter_missing = false;
                            }
                        }
                    }
                }

                if rstring_delimiter_missing {
                    break;
                }
            }

            // Handle ] in ARRAY context
            if !self.options.stream_stable
                && char == ']'
                && self.context.contains(ContextValue::Array)
                && (string_acc.is_empty() || !string_acc.ends_with(rstring_delimiter))
            {
                let i = self.skip_to_character(&[rstring_delimiter], 0);
                if self.get_char_at(i as isize).is_none() {
                    break;
                }
            }

            // Handle } in OBJECT_VALUE context
            if self.context.current() == Some(ContextValue::ObjectValue) && char == '}' {
                let i = self.scroll_whitespaces(1);
                // Check for code fences
                if self.get_char_at(i as isize) == Some('`')
                    && self.get_char_at((i + 1) as isize) == Some('`')
                    && self.get_char_at((i + 2) as isize) == Some('`')
                {
                    break;
                }
                // Check for end of input
                if self.get_char_at(i as isize).is_none() {
                    break;
                }
            }

            // Accumulate character
            string_acc.push(char);
            self.index += 1;

            // Handle stream_stable trailing backslash
            if self.options.stream_stable
                && self.get_char_at(0).is_none()
                && string_acc.ends_with('\\')
            {
                string_acc.pop();
            }

            // Handle escape sequences
            if let Some(next_char) = self.get_char_at(0) {
                if string_acc.ends_with('\\') {
                    if matches!(next_char, '"' | 't' | 'n' | 'r' | 'b' | '\\')
                        || next_char == rstring_delimiter
                    {
                        string_acc.pop();
                        match next_char {
                            't' => string_acc.push('\t'),
                            'n' => string_acc.push('\n'),
                            'r' => string_acc.push('\r'),
                            'b' => string_acc.push('\x08'),
                            _ => string_acc.push(next_char),
                        }
                        self.index += 1;

                        // Handle consecutive escapes
                        while let Some(c) = self.get_char_at(0) {
                            if string_acc.ends_with('\\') && (c == rstring_delimiter || c == '\\') {
                                string_acc.pop();
                                string_acc.push(c);
                                self.index += 1;
                            } else {
                                break;
                            }
                        }
                        continue;
                    } else if next_char == 'u' || next_char == 'x' {
                        // Unicode/hex escape
                        let num_chars = if next_char == 'u' { 4 } else { 2 };
                        if self.index + 1 + num_chars <= self.chars.len() {
                            let hex: String = self.chars
                                [self.index + 1..self.index + 1 + num_chars]
                                .iter()
                                .collect();
                            if hex.len() == num_chars && hex.chars().all(|c| c.is_ascii_hexdigit())
                            {
                                if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                    if let Some(ch) = char::from_u32(code) {
                                        string_acc.pop();
                                        string_acc.push(ch);
                                        self.index += 1 + num_chars;
                                        continue;
                                    }
                                }
                            }
                        }
                    } else if STRING_DELIMITERS.contains(&next_char)
                        && next_char != rstring_delimiter
                    {
                        // Escaped delimiter that shouldn't be escaped
                        string_acc.pop();
                        string_acc.push(next_char);
                        self.index += 1;
                        continue;
                    }
                }
            }

            // Handle colon in OBJECT_KEY context
            if self.get_char_at(0) == Some(':')
                && !missing_quotes
                && self.context.current() == Some(ContextValue::ObjectKey)
            {
                let i = self.skip_to_character(&[lstring_delimiter], 1);
                if self.get_char_at(i as isize).is_some() {
                    let j = self.skip_to_character(&[rstring_delimiter], i + 1);
                    if self.get_char_at(j as isize).is_some() {
                        let k = self.scroll_whitespaces(j + 1);
                        if matches!(self.get_char_at(k as isize), Some(',') | Some('}')) {
                            break;
                        }
                    }
                } else {
                    break;
                }
            }

            // Handle quote inside string
            if let Some(c) = self.get_char_at(0) {
                if c == rstring_delimiter && !string_acc.is_empty() && !string_acc.ends_with('\\') {
                    // Doubled quotes handling
                    if doubled_quotes && self.get_char_at(1) == Some(rstring_delimiter) {
                        self.index += 1;
                        continue;
                    }

                    // Missing quotes in OBJECT_VALUE
                    if missing_quotes && self.context.current() == Some(ContextValue::ObjectValue) {
                        let mut i = 1;
                        while let Some(nc) = self.get_char_at(i as isize) {
                            if nc == rstring_delimiter || nc == lstring_delimiter {
                                break;
                            }
                            i += 1;
                        }
                        if self.get_char_at(i as isize).is_some() {
                            let j = self.scroll_whitespaces(i + 1);
                            if self.get_char_at(j as isize) == Some(':') {
                                self.index -= 1;
                                break;
                            }
                        }
                        continue;
                    }

                    // Unmatched delimiter
                    if unmatched_delimiter {
                        unmatched_delimiter = false;
                        string_acc.push(c);
                        self.index += 1;
                        continue;
                    }

                    // Check for misplaced quote (Python lines 346-458)
                    let mut i = 1;
                    let mut check_comma_in_object_value = true;

                    while let Some(nc) = self.get_char_at(i as isize) {
                        if nc == rstring_delimiter || nc == lstring_delimiter {
                            break;
                        }
                        if check_comma_in_object_value && nc.is_alphabetic() {
                            check_comma_in_object_value = false;
                        }
                        // Check for structural characters
                        if (self.context.contains(ContextValue::ObjectKey)
                            && (nc == ':' || nc == '}'))
                            || (self.context.contains(ContextValue::ObjectValue) && nc == '}')
                            || (self.context.contains(ContextValue::Array)
                                && (nc == ']' || nc == ','))
                            || (check_comma_in_object_value
                                && self.context.current() == Some(ContextValue::ObjectValue)
                                && nc == ',')
                        {
                            break;
                        }
                        i += 1;
                    }

                    let next_c = self.get_char_at(i as isize);

                    // Handle comma in OBJECT_VALUE
                    if next_c == Some(',')
                        && self.context.current() == Some(ContextValue::ObjectValue)
                    {
                        let mut j = i + 1;
                        j = self.skip_to_character(&[rstring_delimiter], j);
                        if self.get_char_at(j as isize).is_some() {
                            let k = self.scroll_whitespaces(j + 1);
                            if matches!(self.get_char_at(k as isize), Some('}') | Some(',')) {
                                string_acc.push(c);
                                self.index += 1;
                                continue;
                            }
                        }
                    }

                    // Handle delimiter found
                    if next_c == Some(rstring_delimiter)
                        && self.get_char_at((i - 1) as isize) != Some('\\')
                    {
                        // Check if only whitespace between quotes
                        let all_whitespace = (1..i).all(|k| {
                            self.get_char_at(k as isize)
                                .map_or(true, |c| c.is_whitespace())
                        });
                        if all_whitespace {
                            break;
                        }

                        if self.context.current() == Some(ContextValue::ObjectValue) {
                            let j = self.scroll_whitespaces(i + 1);
                            if self.get_char_at(j as isize) == Some(',') {
                                // Check for new key pattern
                                let mut k = self.skip_to_character(&[lstring_delimiter], j + 1);
                                k += 1;
                                k = self.skip_to_character(&[rstring_delimiter], k + 1);
                                k += 1;
                                let m = self.scroll_whitespaces(k);
                                if self.get_char_at(m as isize) == Some(':') {
                                    string_acc.push(c);
                                    self.index += 1;
                                    continue;
                                }
                            }

                            // Check if this is a key
                            let mut k = self.skip_to_character(&[rstring_delimiter], j + 1);
                            k += 1;
                            let mut found_colon = false;
                            while let Some(nc) = self.get_char_at(k as isize) {
                                if nc == ':' {
                                    found_colon = true;
                                    break;
                                }
                                if nc == ','
                                    || nc == ']'
                                    || nc == '}'
                                    || (nc == rstring_delimiter
                                        && self.get_char_at((k - 1) as isize) != Some('\\'))
                                {
                                    break;
                                }
                                k += 1;
                            }

                            if !found_colon {
                                unmatched_delimiter = !unmatched_delimiter;
                                string_acc.push(c);
                                self.index += 1;
                                continue;
                            }
                        } else if self.context.current() == Some(ContextValue::Array) {
                            // Check for even delimiters pattern
                            let mut even_delimiters = next_c == Some(rstring_delimiter);
                            let mut j = i;
                            while self.get_char_at(j as isize) == Some(rstring_delimiter) {
                                j = self.skip_to_character(&[rstring_delimiter, ']'], j + 1);
                                if self.get_char_at(j as isize) != Some(rstring_delimiter) {
                                    even_delimiters = false;
                                    break;
                                }
                                j = self.skip_to_character(&[rstring_delimiter, ']'], j + 1);
                            }

                            if even_delimiters {
                                unmatched_delimiter = !unmatched_delimiter;
                                string_acc.push(c);
                                self.index += 1;
                                continue;
                            } else {
                                break;
                            }
                        } else if self.context.current() == Some(ContextValue::ObjectKey) {
                            string_acc.push(c);
                            self.index += 1;
                            continue;
                        }
                    }
                }
            }
        }

        // Handle extreme corner case
        if let Some(c) = self.get_char_at(0) {
            if missing_quotes
                && self.context.current() == Some(ContextValue::ObjectKey)
                && c.is_whitespace()
            {
                self.skip_whitespaces();
                if !matches!(self.get_char_at(0), Some(':') | Some(',')) {
                    return Ok(JsonValue::String(String::new()));
                }
            }
        }

        // Update index for closing quote
        if self.get_char_at(0) != Some(rstring_delimiter) {
            if !self.options.stream_stable {
                string_acc = string_acc.trim_end().to_string();
            }
        } else {
            self.index += 1;
        }

        // Clean trailing whitespace
        if !self.options.stream_stable && (missing_quotes || string_acc.ends_with('\n')) {
            string_acc = string_acc.trim_end().to_string();
        }

        Ok(JsonValue::String(string_acc))
    }

    fn parse_boolean_or_null(&mut self) -> Option<JsonValue> {
        let char = self.get_char_at(0)?.to_ascii_lowercase();

        let (expected, value) = match char {
            't' => ("true", JsonValue::Bool(true)),
            'f' => ("false", JsonValue::Bool(false)),
            'n' => ("null", JsonValue::Null),
            _ => return None,
        };

        let starting_index = self.index;
        let mut i = 0;

        while let Some(c) = self.get_char_at(0) {
            if i >= expected.len() {
                break;
            }
            if c.to_ascii_lowercase() != expected.chars().nth(i).unwrap() {
                break;
            }
            i += 1;
            self.index += 1;
        }

        if i == expected.len() {
            Some(value)
        } else {
            self.index = starting_index;
            None
        }
    }

    fn parse_json_llm_block(&mut self) -> Result<Option<JsonValue>, JsonRepairError> {
        if self.index + 7 <= self.chars.len() {
            let block: String = self.chars[self.index..self.index + 7].iter().collect();
            if block == "```json" {
                let i = self.skip_to_character(&['`'], 7);
                if self.index + i + 3 <= self.chars.len() {
                    let end: String = self.chars[self.index + i..self.index + i + 3]
                        .iter()
                        .collect();
                    if end == "```" {
                        self.index += 7;
                        return Ok(Some(self.parse_json()?));
                    }
                }
            }
        }
        Ok(None)
    }
}
