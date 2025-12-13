//! Comprehensive tests for JSON repair functionality
//!
//! These tests are ported 1:1 from the Python json_repair library's test suite.
//! Source: https://github.com/mangiucugna/json_repair

use super::*;

/// Helper to repair and compare (with ensure_ascii=true, matching Python default)
fn repair(input: &str) -> String {
    repair_json(input, &RepairOptions::default()).unwrap_or_default()
}

/// Helper to repair with ensure_ascii=false (preserve unicode)
fn repair_unicode(input: &str) -> String {
    repair_json(
        input,
        &RepairOptions {
            ensure_ascii: false,
            ..Default::default()
        },
    )
    .unwrap_or_default()
}

/// Helper to repair with skip_json_loads
fn repair_skip(input: &str) -> String {
    repair_json(
        input,
        &RepairOptions {
            skip_json_loads: true,
            ..Default::default()
        },
    )
    .unwrap_or_default()
}

/// Helper to repair with stream_stable
fn repair_stream_stable(input: &str) -> String {
    repair_json(
        input,
        &RepairOptions {
            stream_stable: true,
            ..Default::default()
        },
    )
    .unwrap_or_default()
}

/// Helper to repair with stream_stable = false (explicit)
fn repair_stream_unstable(input: &str) -> String {
    repair_json(
        input,
        &RepairOptions {
            stream_stable: false,
            ..Default::default()
        },
    )
    .unwrap_or_default()
}

/// Helper to load as JsonValue
fn load(input: &str) -> JsonValue {
    loads(input, &RepairOptions::default()).unwrap_or(JsonValue::Null)
}

/// Helper to load with skip_json_loads
fn load_skip(input: &str) -> JsonValue {
    loads(
        input,
        &RepairOptions {
            skip_json_loads: true,
            ..Default::default()
        },
    )
    .unwrap_or(JsonValue::Null)
}

// ============================================================================
// test_json_repair.py - test_valid_json
// ============================================================================

#[test]
fn test_valid_json() {
    assert_eq!(
        repair(r#"{"name": "John", "age": 30, "city": "New York"}"#),
        r#"{"name": "John", "age": 30, "city": "New York"}"#
    );
    assert_eq!(
        repair(r#"{"employees":["John", "Anna", "Peter"]} "#),
        r#"{"employees": ["John", "Anna", "Peter"]}"#
    );
    assert_eq!(
        repair(r#"{"key": "value:value"}"#),
        r#"{"key": "value:value"}"#
    );
    assert_eq!(
        repair(r#"{"text": "The quick brown fox,"}"#),
        r#"{"text": "The quick brown fox,"}"#
    );
    assert_eq!(
        repair(r#"{"text": "The quick brown fox won't jump"}"#),
        r#"{"text": "The quick brown fox won't jump"}"#
    );
    assert_eq!(repair(r#"{"key": ""}"#), r#"{"key": ""}"#);
    assert_eq!(
        repair(r#"{"key1": {"key2": [1, 2, 3]}}"#),
        r#"{"key1": {"key2": [1, 2, 3]}}"#
    );
    // Note: Very large integers may lose precision in Rust due to f64 limitations
    // Python handles arbitrary precision, but Rust i64 max is 9223372036854775807
    assert_eq!(
        repair(r#"{"key": 9223372036854775807}"#),
        r#"{"key": 9223372036854775807}"#
    );
    // Python escapes non-ASCII to \uXXXX by default (ensure_ascii=True)
    assert_eq!(
        repair("{\"key\": \"value\u{263a}\"}"),
        r#"{"key": "value\u263a"}"#
    );
    assert_eq!(
        repair(r#"{"key": "value\nvalue"}"#),
        r#"{"key": "value\nvalue"}"#
    );
}

// ============================================================================
// test_parse_string.py - test_parse_string
// ============================================================================

#[test]
fn test_parse_string() {
    assert_eq!(repair("\""), "");
    assert_eq!(repair("\n"), "");
    assert_eq!(repair(" "), "");
    assert_eq!(repair("string"), "");
    assert_eq!(repair("stringbeforeobject {}"), "{}");
}

// ============================================================================
// test_parse_string.py - test_missing_and_mixed_quotes
// ============================================================================

#[test]
fn test_missing_and_mixed_quotes() {
    assert_eq!(
        repair("{'key': 'string', 'key2': false, \"key3\": null, \"key4\": unquoted}"),
        r#"{"key": "string", "key2": false, "key3": null, "key4": "unquoted"}"#
    );
    assert_eq!(
        repair(r#"{"name": "John", "age": 30, "city": "New York"#),
        r#"{"name": "John", "age": 30, "city": "New York"}"#
    );
    assert_eq!(
        repair(r#"{"name": "John", "age": 30, city: "New York"}"#),
        r#"{"name": "John", "age": 30, "city": "New York"}"#
    );
    assert_eq!(
        repair(r#"{"name": "John", "age": 30, "city": New York}"#),
        r#"{"name": "John", "age": 30, "city": "New York"}"#
    );
    assert_eq!(
        repair(r#"{"name": John, "age": 30, "city": "New York"}"#),
        r#"{"name": "John", "age": 30, "city": "New York"}"#
    );
    assert_eq!(
        repair(r#"{"slanted_delimiter": "value"}"#),
        r#"{"slanted_delimiter": "value"}"#
    );
    assert_eq!(
        repair(r#"{"name": "John", "age": 30, "city": "New"#),
        r#"{"name": "John", "age": 30, "city": "New"}"#
    );
    assert_eq!(repair(r#"{"key": ""value"}"#), r#"{"key": "value"}"#);
    assert_eq!(
        repair(r#"{"key": "value", 5: "value"}"#),
        r#"{"key": "value", "5": "value"}"#
    );
    assert_eq!(repair(r#"{"key": value , }"#), r#"{"key": "value"}"#);
}

// ============================================================================
// test_parse_string.py - test_escaping
// ============================================================================

#[test]
fn test_escaping() {
    assert_eq!(repair(r#"{"key_1\n": "value"}"#), r#"{"key_1": "value"}"#);
    assert_eq!(repair(r#"{"key\t_": "value"}"#), r#"{"key\t_": "value"}"#);
    assert_eq!(repair(r#"{"key": "valu\'e"}"#), r#"{"key": "valu'e"}"#);
    // Unicode literal in single quotes (Python line 73)
    assert_eq!(
        repair("{\"key\": '\u{0076}\u{0061}\u{006c}\u{0075}\u{0065}'}"),
        r#"{"key": "value"}"#
    );
    // Unicode escape sequences (Python line 74)
    assert_eq!(
        repair_skip(r#"{"key": "\u0076\u0061\u006C\u0075\u0065"}"#),
        r#"{"key": "value"}"#
    );
    // Nested JSON string (Python line 76)
    assert_eq!(
        repair(r#"{'key': "{\"key\": 1, \"key2\": 1}"}"#),
        r#"{"key": "{\"key\": 1, \"key2\": 1}"}"#
    );
}

// ============================================================================
// test_parse_string.py - test_markdown
// ============================================================================

#[test]
fn test_markdown() {
    assert_eq!(
        repair(r#"{ "content": "[LINK]("https://google.com")" }"#),
        r#"{"content": "[LINK](\"https://google.com\")"}"#
    );
    assert_eq!(
        repair(r#"{ "content": "[LINK](" }"#),
        r#"{"content": "[LINK]("}"#
    );
    assert_eq!(
        repair(r#"{ "content": "[LINK](", "key": true }"#),
        r#"{"content": "[LINK](", "key": true}"#
    );
}

// ============================================================================
// test_parse_string.py - test_leading_trailing_characters
// ============================================================================

#[test]
fn test_leading_trailing_characters() {
    assert_eq!(
        repair(r#"````{ "key": "value" }```"#),
        r#"{"key": "value"}"#
    );
    assert_eq!(
        repair(
            r#"{    "a": "",    "b": [ { "c": 1} ] 
}```"#
        ),
        r#"{"a": "", "b": [{"c": 1}]}"#
    );
    assert_eq!(
        repair("Based on the information extracted, here is the filled JSON output: ```json { 'a': 'b' } ```"),
        r#"{"a": "b"}"#
    );
}

// ============================================================================
// test_parse_string.py - test_parse_boolean_or_null
// ============================================================================

#[test]
fn test_parse_boolean_or_null() {
    assert_eq!(load("true"), JsonValue::Bool(true));
    assert_eq!(load("false"), JsonValue::Bool(false));
    assert_eq!(load("null"), JsonValue::Null);
    // Capitalized versions are treated as unquoted strings, not booleans
    // Python's return_objects=True returns "" for these
    assert_eq!(repair("True"), "");
    assert_eq!(repair("False"), "");
    assert_eq!(repair("Null"), "");
    assert_eq!(
        repair(r#"  {"key": true, "key2": false, "key3": null}"#),
        r#"{"key": true, "key2": false, "key3": null}"#
    );
    assert_eq!(
        repair(r#"{"key": TRUE, "key2": FALSE, "key3": Null}   "#),
        r#"{"key": true, "key2": false, "key3": null}"#
    );
}

// ============================================================================
// test_parse_object.py - test_parse_object
// ============================================================================

#[test]
fn test_parse_object() {
    assert_eq!(load("{}"), JsonValue::Object(vec![]));
    assert_eq!(repair("{"), "{}");
    assert_eq!(repair("}"), "");
    assert_eq!(repair("{\""), "{}");
    assert_eq!(repair("   {  }   "), "{}");
}

// ============================================================================
// test_parse_object.py - test_parse_object_edge_cases
// ============================================================================

#[test]
fn test_parse_object_edge_cases() {
    assert_eq!(repair("{foo: [}"), r#"{"foo": []}"#);
    assert_eq!(repair(r#"{"": "value"}"#), r#"{"": "value"}"#);
    assert_eq!(
        repair(r#"{"value_1": true, COMMENT "value_2": "data"}"#),
        r#"{"value_1": true, "value_2": "data"}"#
    );
    assert_eq!(
        repair(r#"{"value_1": true, SHOULD_NOT_EXIST "value_2": "data" AAAA }"#),
        r#"{"value_1": true, "value_2": "data"}"#
    );
    assert_eq!(
        repair(r#"{"" : true, "key2": "value2"}"#),
        r#"{"": true, "key2": "value2"}"#
    );
    assert_eq!(
        repair("{key:value,key2:value2}"),
        r#"{"key": "value", "key2": "value2"}"#
    );
    assert_eq!(
        repair(r#"{"key":value, " key2":"value2" }"#),
        r#"{"key": "value", " key2": "value2"}"#
    );
    assert_eq!(
        repair(r#"{"key":value "key2":"value2" }"#),
        r#"{"key": "value", "key2": "value2"}"#
    );
    assert_eq!(
        repair("{'text': 'words{words in brackets}more words'}"),
        r#"{"text": "words{words in brackets}more words"}"#
    );
    assert_eq!(
        repair("{text:words{words in brackets}}"),
        r#"{"text": "words{words in brackets}"}"#
    );
    assert_eq!(
        repair(r#"{"key": "value, value2"```"#),
        r#"{"key": "value, value2"}"#
    );
    assert_eq!(repair(r#"{"key": "value}```"#), r#"{"key": "value"}"#);
    assert_eq!(
        repair(r#"{"key": , "key2": "value2"}"#),
        r#"{"key": "", "key2": "value2"}"#
    );
}

// ============================================================================
// test_parse_object.py - test_parse_object_merge_at_the_end
// ============================================================================

#[test]
fn test_parse_object_merge_at_the_end() {
    assert_eq!(
        repair(r#"{"key": "value"}, "key2": "value2"}"#),
        r#"{"key": "value", "key2": "value2"}"#
    );
    assert_eq!(
        repair(r#"{"key": "value"}, "key2": }"#),
        r#"{"key": "value", "key2": ""}"#
    );
    assert_eq!(repair(r#"{"key": "value"}, []"#), r#"{"key": "value"}"#);
    assert_eq!(repair(r#"{"key": "value"}, {}"#), r#"{"key": "value"}"#);
}

// ============================================================================
// test_parse_array.py - test_parse_array
// ============================================================================

#[test]
fn test_parse_array() {
    assert_eq!(load("[]"), JsonValue::Array(vec![]));
    assert_eq!(
        load("[1, 2, 3, 4]"),
        JsonValue::Array(vec![
            JsonValue::Integer(1),
            JsonValue::Integer(2),
            JsonValue::Integer(3),
            JsonValue::Integer(4),
        ])
    );
    assert_eq!(repair("["), "[]");
    assert_eq!(repair("[\""), "[]");
    assert_eq!(repair("]"), "");
    assert_eq!(repair("[[1\n\n]"), "[[1]]");
}

// ============================================================================
// test_parse_array.py - test_parse_array_edge_cases
// ============================================================================

#[test]
fn test_parse_array_edge_cases() {
    assert_eq!(repair("[{]"), "[]");
    assert_eq!(repair("[1, 2, 3,"), "[1, 2, 3]");
    assert_eq!(repair("[1, 2, 3, ...]"), "[1, 2, 3]");
    assert_eq!(repair("[1, 2, ... , 3]"), "[1, 2, 3]");
    assert_eq!(repair("[1, 2, '...', 3]"), r#"[1, 2, "...", 3]"#);
    assert_eq!(repair("[true, false, null, ...]"), "[true, false, null]");
    assert_eq!(
        repair(r#"{"employees":["John", "Anna","#),
        r#"{"employees": ["John", "Anna"]}"#
    );
    assert_eq!(
        repair(r#"{"employees":["John", "Anna", "Peter"#),
        r#"{"employees": ["John", "Anna", "Peter"]}"#
    );
    assert_eq!(
        repair(r#"{"key1": {"key2": [1, 2, 3"#),
        r#"{"key1": {"key2": [1, 2, 3]}}"#
    );
    assert_eq!(repair(r#"{"key": ["value]}"#), r#"{"key": ["value"]}"#);
}

// ============================================================================
// test_parse_number.py - test_parse_number
// ============================================================================

#[test]
fn test_parse_number() {
    assert_eq!(load("1"), JsonValue::Integer(1));
    assert_eq!(load("1.2"), JsonValue::Float(1.2));
    // Also test repair() for simple numbers
    assert_eq!(repair("1"), "1");
    assert_eq!(repair("1.2"), "1.2");
    // Underscore numbers - test both load and repair
    assert_eq!(
        load(r#"{"value": 82_461_110}"#),
        JsonValue::Object(vec![("value".to_string(), JsonValue::Integer(82461110))])
    );
    assert_eq!(
        load(r#"{"value": 1_234.5_6}"#),
        JsonValue::Object(vec![("value".to_string(), JsonValue::Float(1234.56))])
    );
    assert_eq!(repair(r#"{"value": 82_461_110}"#), r#"{"value": 82461110}"#);
    assert_eq!(repair(r#"{"value": 1_234.5_6}"#), r#"{"value": 1234.56}"#);
}

// ============================================================================
// test_parse_number.py - test_parse_number_edge_cases
// ============================================================================

#[test]
fn test_parse_number_edge_cases() {
    assert_eq!(
        repair(r#" - { "test_key": ["test_value", "test_value2"] }"#),
        r#"{"test_key": ["test_value", "test_value2"]}"#
    );
    assert_eq!(repair(r#"{"key": 1/3}"#), r#"{"key": "1/3"}"#);
    assert_eq!(repair(r#"{"key": .25}"#), r#"{"key": 0.25}"#);
    assert_eq!(
        repair(r#"{"here": "now", "key": 1/3, "foo": "bar"}"#),
        r#"{"here": "now", "key": "1/3", "foo": "bar"}"#
    );
    assert_eq!(
        repair(r#"{"key": 12345/67890}"#),
        r#"{"key": "12345/67890"}"#
    );
    assert_eq!(repair("[105,12"), "[105, 12]");
    assert_eq!(
        repair(r#"{"key": 1/3, "foo": "bar"}"#),
        r#"{"key": "1/3", "foo": "bar"}"#
    );
    assert_eq!(repair(r#"{"key": 10-20}"#), r#"{"key": "10-20"}"#);
    assert_eq!(repair(r#"{"key": 1.1.1}"#), r#"{"key": "1.1.1"}"#);
    assert_eq!(repair("[- "), "[]");
    assert_eq!(repair(r#"{"key": 1. }"#), r#"{"key": 1.0}"#);
    assert_eq!(repair(r#"{"key": 1e10 }"#), r#"{"key": 10000000000.0}"#);
    assert_eq!(repair(r#"{"key": 1e }"#), r#"{"key": 1}"#);
    assert_eq!(
        repair(r#"{"key": 1notanumber }"#),
        r#"{"key": "1notanumber"}"#
    );
    assert_eq!(repair("[1, 2notanumber]"), r#"[1, "2notanumber"]"#);
}

// ============================================================================
// test_parse_comment.py - test_parse_comment
// ============================================================================

#[test]
fn test_parse_comment() {
    assert_eq!(repair("/"), "");
    assert_eq!(
        repair(r#"{ "key": { "key2": "value2" // comment }, "key3": "value3" }"#),
        r#"{"key": {"key2": "value2"}, "key3": "value3"}"#
    );
    assert_eq!(
        repair(r#"{ "key": { "key2": "value2" # comment }, "key3": "value3" }"#),
        r#"{"key": {"key2": "value2"}, "key3": "value3"}"#
    );
    assert_eq!(
        repair(r#"{ "key": { "key2": "value2" /* comment */ }, "key3": "value3" }"#),
        r#"{"key": {"key2": "value2"}, "key3": "value3"}"#
    );
    assert_eq!(
        repair(r#"[ "value", /* comment */ "value2" ]"#),
        r#"["value", "value2"]"#
    );
    assert_eq!(
        repair(r#"{ "key": "value" /* comment"#),
        r#"{"key": "value"}"#
    );
}

// ============================================================================
// test_json_repair.py - test_repair_json_skip_json_loads
// ============================================================================

#[test]
fn test_repair_json_skip_json_loads() {
    assert_eq!(
        repair_skip(r#"{"key": true, "key2": false, "key3": null}"#),
        r#"{"key": true, "key2": false, "key3": null}"#
    );
    assert_eq!(
        repair_skip(r#"{"key": true, "key2": false, "key3": }"#),
        r#"{"key": true, "key2": false, "key3": ""}"#
    );
    assert_eq!(
        load_skip(r#"{"key": true, "key2": false, "key3": }"#),
        JsonValue::Object(vec![
            ("key".to_string(), JsonValue::Bool(true)),
            ("key2".to_string(), JsonValue::Bool(false)),
            ("key3".to_string(), JsonValue::String("".to_string())),
        ])
    );
}

// ============================================================================
// test_json_repair.py - test_stream_stable
// ============================================================================

#[test]
fn test_stream_stable() {
    // stream_stable = false (default)
    assert_eq!(
        repair_stream_unstable(r#"{"key": "val\"#),
        r#"{"key": "val\\"}"#
    );
    assert_eq!(
        repair_stream_unstable(r#"{"key": "val\n"#),
        r#"{"key": "val"}"#
    );
    assert_eq!(
        repair_stream_unstable(r#"{"key": "val\n123,`key2:value2`"}"#),
        r#"{"key": "val\n123,`key2:value2`"}"#
    );

    // stream_stable = true
    assert_eq!(
        repair_stream_stable(r#"{"key": "val\"#),
        r#"{"key": "val"}"#
    );
    assert_eq!(
        repair_stream_stable("{\"key\": \"val\\n"),
        "{\"key\": \"val\\n\"}"
    );
    assert_eq!(
        repair_stream_stable(r#"{"key": "val\n123,`key2:value2"#),
        r#"{"key": "val\n123,`key2:value2"}"#
    );
    assert_eq!(
        repair_stream_stable(r#"{"key": "val\n123,`key2:value2`"}"#),
        r#"{"key": "val\n123,`key2:value2`"}"#
    );
}

// ============================================================================
// test_json_repair.py - test_repair_json_with_objects
// ============================================================================

#[test]
fn test_repair_json_with_objects() {
    assert_eq!(load("[]"), JsonValue::Array(vec![]));
    assert_eq!(load("{}"), JsonValue::Object(vec![]));
    assert_eq!(
        load(r#"{"key": true, "key2": false, "key3": null}"#),
        JsonValue::Object(vec![
            ("key".to_string(), JsonValue::Bool(true)),
            ("key2".to_string(), JsonValue::Bool(false)),
            ("key3".to_string(), JsonValue::Null),
        ])
    );
    assert_eq!(
        load(r#"{"name": "John", "age": 30, "city": "New York"}"#),
        JsonValue::Object(vec![
            ("name".to_string(), JsonValue::String("John".to_string())),
            ("age".to_string(), JsonValue::Integer(30)),
            (
                "city".to_string(),
                JsonValue::String("New York".to_string())
            ),
        ])
    );
    assert_eq!(
        load("[1, 2, 3, 4]"),
        JsonValue::Array(vec![
            JsonValue::Integer(1),
            JsonValue::Integer(2),
            JsonValue::Integer(3),
            JsonValue::Integer(4),
        ])
    );
}

// ============================================================================
// test_json_repair.py - test_multiple_jsons
// ============================================================================

#[test]
fn test_multiple_jsons() {
    assert_eq!(repair("[]{}"), "[]");
    assert_eq!(repair(r#"[]{"key":"value"}"#), r#"{"key": "value"}"#);
}

// ============================================================================
// test_strict_mode.py - strict mode tests
// Note: In Rust we test that strict mode returns errors
// ============================================================================

#[test]
fn test_strict_mode() {
    let strict_opts = RepairOptions {
        strict: true,
        skip_json_loads: true,
        ..Default::default()
    };

    // Multiple top-level values should error
    assert!(repair_json(r#"{"key":"value"}["value"]"#, &strict_opts).is_err());

    // Empty keys should error
    assert!(repair_json(r#"{"" : "value"}"#, &strict_opts).is_err());

    // Missing colon should error
    assert!(repair_json(r#"{"missing" "colon"}"#, &strict_opts).is_err());

    // Empty values should error
    assert!(repair_json(r#"{"key": , "key2": "value2"}"#, &strict_opts).is_err());
}

// ============================================================================
// Additional tests from test_parse_string.py
// ============================================================================

#[test]
fn test_markdown_in_strings() {
    assert_eq!(
        repair(r#"{ "content": "[LINK](" }"#),
        r#"{"content": "[LINK]("}"#
    );
    assert_eq!(
        repair(r#"{ "content": "[LINK](", "key": true }"#),
        r#"{"content": "[LINK](", "key": true}"#
    );
}

#[test]
fn test_string_json_llm_block() {
    assert_eq!(repair(r#"{"key": "``"}"#), r#"{"key": "``"}"#);
    assert_eq!(repair(r#"{"key": "```json"}"#), r#"{"key": "```json"}"#);
}

// ============================================================================
// Additional tests from test_parse_object.py
// ============================================================================

#[test]
fn test_parse_object_more_edge_cases() {
    // Note: {"key": "value}```"} test is in test_parse_object (line 231)
    assert_eq!(repair(r#"{"key:"value"}"#), r#"{"key": "value"}"#);
    assert_eq!(repair(r#"{"key:value}"#), r#"{"key": "value"}"#);
    assert_eq!(
        repair(r#"{"lorem": ipsum, sic, datum.",}"#),
        r#"{"lorem": "ipsum, sic, datum."}"#
    );
}

// ============================================================================
// Additional tests from test_parse_array.py
// ============================================================================

#[test]
fn test_parse_array_more_edge_cases() {
    assert_eq!(repair(r#"["a" "b" "c" 1"#), r#"["a", "b", "c", 1]"#);
    // Note: {"employees":["John", "Anna"," test is in test_parse_array (line 285)
    assert_eq!(
        repair(r#"{"key": ["value" "value1" "value2"]}"#),
        r#"{"key": ["value", "value1", "value2"]}"#
    );
}

// ============================================================================
// Additional tests from test_parse_number.py
// ============================================================================

#[test]
fn test_parse_number_more_edge_cases() {
    assert_eq!(repair(r#"{"key", 105,12,"#), r#"{"key": "105,12"}"#);
}

// ============================================================================
// Additional tests from test_parse_comment.py
// ============================================================================

#[test]
fn test_parse_comment_more() {
    assert_eq!(
        repair("/* comment */ {\"key\": \"value\"}"),
        r#"{"key": "value"}"#
    );
}

// ============================================================================
// Missing tests from test_json_repair.py - test_multiple_jsons
// ============================================================================

#[test]
fn test_multiple_jsons_more() {
    assert_eq!(
        repair(r#"{"key":"value"}[1,2,3,True]"#),
        r#"[{"key": "value"}, [1, 2, 3, true]]"#
    );
    assert_eq!(
        repair(r#"[{"key":"value"}][{"key":"value_after"}]"#),
        r#"[{"key": "value_after"}]"#
    );
}

// ============================================================================
// Missing tests from test_parse_string.py - test_missing_and_mixed_quotes
// ============================================================================

#[test]
fn test_missing_and_mixed_quotes_more() {
    assert_eq!(
        repair(r#"{"name": "John", "age": 30, "city": "New York, "gender": "male"}"#),
        r#"{"name": "John", "age": 30, "city": "New York", "gender": "male"}"#
    );
    assert_eq!(
        repair(r#"[{"key": "value", COMMENT "notes": "lorem "ipsum", sic." }]"#),
        r#"[{"key": "value", "notes": "lorem \"ipsum\", sic."}]"#
    );
    assert_eq!(repair(r#"{"foo": "\"bar\"""#), r#"{"foo": "\"bar\""}"#);
    assert_eq!(repair(r#"{"" key":"val""#), r#"{" key": "val"}"#);
    assert_eq!(
        repair(r#"{"key": value "key2" : "value2" "#),
        r#"{"key": "value", "key2": "value2"}"#
    );
    assert_eq!(
        repair(r#"{"key": "lorem ipsum ... "sic " tamet. ...}"#),
        r#"{"key": "lorem ipsum ... \"sic \" tamet. ..."}"#
    );
    assert_eq!(
        repair(r#"{"comment": "lorem, "ipsum" sic "tamet". To improve"}"#),
        r#"{"comment": "lorem, \"ipsum\" sic \"tamet\". To improve"}"#
    );
    assert_eq!(
        repair(r#"{"key": "v"alu"e"} key:"#),
        r#"{"key": "v\"alu\"e"}"#
    );
    assert_eq!(
        repair(r#"{"key": "v"alue", "key2": "value2"}"#),
        r#"{"key": "v\"alue", "key2": "value2"}"#
    );
    assert_eq!(
        repair(r#"[{"key": "v"alu,e", "key2": "value2"}]"#),
        r#"[{"key": "v\"alu,e", "key2": "value2"}]"#
    );
}

// ============================================================================
// Missing tests from test_parse_string.py - test_escaping
// ============================================================================

#[test]
fn test_escaping_more() {
    assert_eq!(repair(r#"'"'"#), "");
    assert_eq!(
        repair(r#"{"key": 'string"\n\t\\le'"#),
        r#"{"key": "string\"\n\t\\le"}"#
    );
    assert_eq!(
        repair(
            r#"{"real_content": "Some string: Some other string \t Some string <a href=\"https://domain.com\">Some link</a>""#
        ),
        r#"{"real_content": "Some string: Some other string \t Some string <a href=\"https://domain.com\">Some link</a>"}"#
    );
    assert_eq!(
        repair(r#"{'key': "{\"key\": 1, \"key2\": 1}"}"#),
        r#"{"key": "{\"key\": 1, \"key2\": 1}"}"#
    );
}

// ============================================================================
// Missing tests from test_parse_object.py - test_parse_object_edge_cases
// ============================================================================

#[test]
fn test_parse_object_edge_cases_more() {
    assert_eq!(repair(r#"{"key": "v"alue"}"#), r#"{"key": "v\"alue\""}"#);
    assert_eq!(
        repair(r#"{"answer":[{"traits":''Female aged 60+'',""answer1"":"5"}]}"#),
        r#"{"answer": [{"traits": "Female aged 60+", "answer1": "5"}]}"#
    );
    assert_eq!(
        repair(r#"{ "words": abcdef", "numbers": 12345", "words2": ghijkl" }"#),
        r#"{"words": "abcdef", "numbers": 12345, "words2": "ghijkl"}"#
    );
    assert_eq!(
        repair(r#"{"number": 1,"reason": "According...""ans": "YES"}"#),
        r#"{"number": 1, "reason": "According...", "ans": "YES"}"#
    );
    assert_eq!(repair(r#"{ "a" : "{ b": {} }" }"#), r#"{"a": "{ b"}"#);
    assert_eq!(repair(r#"{"b": "xxxxx" true}"#), r#"{"b": "xxxxx"}"#);
    assert_eq!(
        repair(r#"{"key": "Lorem "ipsum" s,"}"#),
        r#"{"key": "Lorem \"ipsum\" s,"}"#
    );
    assert_eq!(
        repair(r#"{"lorem": sic tamet. "ipsum": sic tamet, quick brown fox. "sic": ipsum}"#),
        r#"{"lorem": "sic tamet.", "ipsum": "sic tamet", "sic": "ipsum"}"#
    );
    assert_eq!(
        repair(r#"{"lorem_ipsum": "sic tamet, quick brown fox. }"#),
        r#"{"lorem_ipsum": "sic tamet, quick brown fox."}"#
    );
    assert_eq!(
        repair("{text:words{words in brackets}m}"),
        r#"{"text": "words{words in brackets}m"}"#
    );
}

// ============================================================================
// Missing tests from test_parse_object.py - test_parse_object_merge_at_the_end
// ============================================================================

#[test]
fn test_parse_object_merge_at_the_end_more() {
    assert_eq!(
        repair(r#"{"key": "value"}, ["abc"]"#),
        r#"[{"key": "value"}, ["abc"]]"#
    );
    assert_eq!(
        repair(r#"{"key": "value"}, "" : "value2"}"#),
        r#"{"key": "value", "": "value2"}"#
    );
    assert_eq!(
        repair(r#"{"key": "value"}, "key2" "value2"}"#),
        r#"{"key": "value", "key2": "value2"}"#
    );
    assert_eq!(
        repair(r#"{"key1": "value1"}, "key2": "value2", "key3": "value3"}"#),
        r#"{"key1": "value1", "key2": "value2", "key3": "value3"}"#
    );
}

// ============================================================================
// Missing tests from test_parse_array.py - test_parse_array_edge_cases
// ============================================================================

#[test]
fn test_parse_array_edge_cases_more() {
    assert_eq!(
        repair(r#"["lorem "ipsum" sic"]"#),
        r#"["lorem \"ipsum\" sic"]"#
    );
    assert_eq!(
        repair(r#"{"key1": ["value1", "value2"}, "key2": ["value3", "value4"]}"#),
        r#"{"key1": ["value1", "value2"], "key2": ["value3", "value4"]}"#
    );
    assert_eq!(
        repair(
            r#"{"key": ["lorem "ipsum" dolor "sit" amet, "consectetur" ", "lorem "ipsum" dolor", "lorem"]}"#
        ),
        r#"{"key": ["lorem \"ipsum\" dolor \"sit\" amet, \"consectetur\" ", "lorem \"ipsum\" dolor", "lorem"]}"#
    );
    assert_eq!(repair(r#"{"k"e"y": "value"}"#), r#"{"k\"e\"y": "value"}"#);
    assert_eq!(repair(r#"["key":"value"}]"#), r#"[{"key": "value"}]"#);
    assert_eq!(
        repair(r#"[{"key": "value", "key"#),
        r#"[{"key": "value"}, ["key"]]"#
    );
    assert_eq!(repair("{'key1', 'key2'}"), r#"["key1", "key2"]"#);
}

// ============================================================================
// Missing tests from test_parse_array.py - test_parse_array_missing_quotes
// ============================================================================

#[test]
fn test_parse_array_missing_quotes() {
    assert_eq!(
        repair(r#"["value1" value2", "value3"]"#),
        r#"["value1", "value2", "value3"]"#
    );
    assert_eq!(
        repair(
            r#"{"bad_one":["Lorem Ipsum", "consectetur" comment" ], "good_one":[ "elit", "sed", "tempor"]}"#
        ),
        r#"{"bad_one": ["Lorem Ipsum", "consectetur", "comment"], "good_one": ["elit", "sed", "tempor"]}"#
    );
    assert_eq!(
        repair(
            r#"{"bad_one": ["Lorem Ipsum","consectetur" comment],"good_one": ["elit","sed","tempor"]}"#
        ),
        r#"{"bad_one": ["Lorem Ipsum", "consectetur", "comment"], "good_one": ["elit", "sed", "tempor"]}"#
    );
}

// ============================================================================
// Missing tests from test_json_repair.py - test_stream_stable
// ============================================================================

#[test]
fn test_stream_stable_more() {
    assert_eq!(
        repair_stream_unstable(r#"{"key": "val\n123,`key2:value2"#),
        r#"{"key": "val\n123", "key2": "value2"}"#
    );
    assert_eq!(
        repair_stream_stable(r#"{"key": "val\n123,`key2:value2`"}"#),
        r#"{"key": "val\n123,`key2:value2`"}"#
    );
    assert_eq!(
        repair_stream_stable(r#"{"key": "val\n123,`key2:value2"#),
        r#"{"key": "val\n123,`key2:value2"}"#
    );
}

// ============================================================================
// Missing tests from test_parse_string.py - test_string_json_llm_block
// ============================================================================

#[test]
fn test_string_json_llm_block_more() {
    assert_eq!(
        repair(r#"{"key": "```json {"key": [{"key1": 1},{"key2": 2}]}```"}"#),
        r#"{"key": {"key": [{"key1": 1}, {"key2": 2}]}}"#
    );
    assert_eq!(
        repair(r#"{"response": "```json{}"}"#),
        r#"{"response": "```json{}"}"#
    );
}

// ============================================================================
// Missing tests from test_json_repair.py - test_multiple_jsons
// ============================================================================

#[test]
fn test_multiple_jsons_with_markdown() {
    assert_eq!(
        repair(r#"lorem ```json {"key":"value"} ``` ipsum ```json [1,2,3,True] ``` 42"#),
        r#"[{"key": "value"}, [1, 2, 3, true]]"#
    );
}

// ============================================================================
// Missing tests from test_parse_string.py - test_leading_trailing_characters
// ============================================================================

#[test]
fn test_leading_trailing_characters_multiline() {
    assert_eq!(
        repair(
            r#"
                       The next 64 elements are:
                       ```json
                       { "key": "value" }
                       ```"#
        ),
        r#"{"key": "value"}"#
    );
}

// ============================================================================
// Missing tests from test_parse_string.py - test_escaping (unicode)
// ============================================================================

#[test]
fn test_escaping_unicode() {
    assert_eq!(
        repair(r#"{"key": "\u0076\u0061\u006C\u0075\u0065"}"#),
        r#"{"key": "value"}"#
    );
}

// ============================================================================
// Missing tests from test_parse_object.py - test_parse_object_edge_cases
// ============================================================================

#[test]
fn test_parse_object_array_merging() {
    assert_eq!(
        repair(r#"{ "key": ["arrayvalue"], ["arrayvalue1"], ["arrayvalue2"], "key3": "value3" }"#),
        r#"{"key": ["arrayvalue", "arrayvalue1", "arrayvalue2"], "key3": "value3"}"#
    );
    assert_eq!(
        repair(r#"{ "key": ["arrayvalue"], "key3": "value3", ["arrayvalue1"] }"#),
        r#"{"key": ["arrayvalue"], "key3": "value3", "arrayvalue1": ""}"#
    );
}

#[test]
fn test_parse_object_complex_escaping() {
    assert_eq!(
        repair(r#"{"key": "{\\"key\\\\":[\\"value\\\\\\"],\\"key2\\":\"value2\"}"}"#),
        r#"{"key": "{\"key\":[\"value\"],\"key2\":\"value2\"}"}"#
    );
}

#[test]
fn test_parse_object_duplicate_with_quotes() {
    assert_eq!(
        repair(r#"[{"lorem": {"ipsum": "sic"}, """" "lorem": {"ipsum": "sic"}]"#),
        r#"[{"lorem": {"ipsum": "sic"}}, {"lorem": {"ipsum": "sic"}}]"#
    );
}

// ============================================================================
// Missing tests from test_strict_mode.py
// ============================================================================

#[test]
fn test_strict_duplicate_keys_inside_array() {
    let strict_opts = RepairOptions {
        strict: true,
        skip_json_loads: true,
        ..Default::default()
    };
    assert!(repair_json(r#"[{"key": "first", "key": "second"}]"#, &strict_opts).is_err());
}

#[test]
fn test_strict_rejects_empty_object_with_extra_characters() {
    let strict_opts = RepairOptions {
        strict: true,
        ..Default::default()
    };
    assert!(repair_json(r#"{"dangling"}"#, &strict_opts).is_err());
}

#[test]
fn test_strict_detects_doubled_quotes() {
    let strict_opts = RepairOptions {
        strict: true,
        ..Default::default()
    };
    assert!(repair_json(r#"{"key": """"}"#, &strict_opts).is_err());
    assert!(repair_json(r#"{"key": "" "value"}"#, &strict_opts).is_err());
}

#[test]
fn test_strict_rejects_multiple_top_level_values() {
    let strict_opts = RepairOptions {
        strict: true,
        ..Default::default()
    };
    assert!(repair_json(r#"{"key":"value"}["value"]"#, &strict_opts).is_err());
}

#[test]
fn test_strict_rejects_empty_keys() {
    let strict_opts = RepairOptions {
        strict: true,
        skip_json_loads: true,
        ..Default::default()
    };
    assert!(repair_json(r#"{"" : "value"}"#, &strict_opts).is_err());
}

#[test]
fn test_strict_requires_colon_between_key_and_value() {
    let strict_opts = RepairOptions {
        strict: true,
        ..Default::default()
    };
    assert!(repair_json(r#"{"missing" "colon"}"#, &strict_opts).is_err());
}

#[test]
fn test_strict_rejects_empty_values() {
    let strict_opts = RepairOptions {
        strict: true,
        skip_json_loads: true,
        ..Default::default()
    };
    assert!(repair_json(r#"{"key": , "key2": "value2"}"#, &strict_opts).is_err());
}

// ============================================================================
// test_json_repair.py - test_ensure_ascii
// Note: Rust handles Unicode natively, so this test just verifies
// that non-ASCII characters are preserved correctly
// ============================================================================

#[test]
fn test_ensure_ascii() {
    // Python's ensure_ascii=True (default) escapes non-ASCII to \uXXXX
    assert_eq!(
        repair("{'test_中国人_ascii':'统一码'}"),
        r#"{"test_\u4e2d\u56fd\u4eba_ascii": "\u7edf\u4e00\u7801"}"#
    );

    // Python's ensure_ascii=False preserves unicode characters
    assert_eq!(
        repair_unicode("{'test_中国人_ascii':'统一码'}"),
        r#"{"test_中国人_ascii": "统一码"}"#
    );
}

// ============================================================================
// test_json_repair.py - test_repair_json_with_objects (complex cases)
// ============================================================================

#[test]
fn test_repair_json_with_objects_complex() {
    // HTML embedded in JSON with unescaped quotes
    assert_eq!(
        repair("{\n\"html\": \"<h3 id=\"aaa\">Waarom meer dan 200 Technical Experts - \"Passie voor techniek\"?</h3>\"}"),
        "{\"html\": \"<h3 id=\\\"aaa\\\">Waarom meer dan 200 Technical Experts - \\\"Passie voor techniek\\\"?</h3>\"}"
    );

    // Array with embedded quotes in strings (using # in tag)
    assert_eq!(
        repair("[{\"foo\": \"Foo bar baz\", \"tag\": \"#foo-bar-baz\"}, {\"foo\": \"foo bar \"foobar\" foo bar baz.\", \"tag\": \"#foo-bar-foobar\"}]"),
        "[{\"foo\": \"Foo bar baz\", \"tag\": \"#foo-bar-baz\"}, {\"foo\": \"foo bar \\\"foobar\\\" foo bar baz.\", \"tag\": \"#foo-bar-foobar\"}]"
    );
}

// ============================================================================
// test_json_repair.py - test_repair_json_with_objects (FHIR Bundle case)
// ============================================================================

#[test]
fn test_repair_json_with_objects_fhir_bundle() {
    // Complex nested JSON with FHIR Bundle structure (missing closing brackets)
    let input = r#"
{
  "resourceType": "Bundle",
  "id": "1",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "1",
        "name": [
          {"use": "official", "family": "Corwin", "given": ["Keisha", "Sunny"], "prefix": ["Mrs."},
          {"use": "maiden", "family": "Goodwin", "given": ["Keisha", "Sunny"], "prefix": ["Mrs."]}
        ]
      }
    }
  ]
}
"#;
    let result = load(input);
    // Verify the structure is correctly parsed
    if let JsonValue::Object(obj) = result {
        assert!(obj.iter().any(|(k, _)| k == "resourceType"));
        assert!(obj.iter().any(|(k, _)| k == "entry"));
    } else {
        panic!("Expected object");
    }
}

// ============================================================================
// test_parse_object.py - test_parse_object (additional cases)
// ============================================================================

#[test]
fn test_parse_object_with_return_objects() {
    // { "key": "value", "key2": 1, "key3": True }
    assert_eq!(
        load(r#"{ "key": "value", "key2": 1, "key3": True }"#),
        JsonValue::Object(vec![
            ("key".to_string(), JsonValue::String("value".to_string())),
            ("key2".to_string(), JsonValue::Integer(1)),
            ("key3".to_string(), JsonValue::Bool(true)),
        ])
    );

    // { "key": value, "key2": 1 "key3": null }
    assert_eq!(
        load(r#"{ "key": value, "key2": 1 "key3": null }"#),
        JsonValue::Object(vec![
            ("key".to_string(), JsonValue::String("value".to_string())),
            ("key2".to_string(), JsonValue::Integer(1)),
            ("key3".to_string(), JsonValue::Null),
        ])
    );
}
