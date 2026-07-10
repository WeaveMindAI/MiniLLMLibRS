//! A minimal byte-pair encoder, enough to COUNT the tokens in a prompt.
//!
//! This is the `o200k_base` vocabulary that GPT-4o and GPT-5 use. We count with
//! it and correct its bias for other model families, because no provider will
//! count a prompt for free and every family tokenizes differently. See
//! [`super::estimate`] for the correction.
//!
//! Only counting is implemented. There is no decoding, no special-token handling
//! and no other vocabulary, because a cost estimate needs none of them. The
//! algorithm is the standard greedy merge: split the text into pieces on a fixed
//! pattern, then within each piece repeatedly merge the adjacent byte pair with
//! the lowest rank until no known pair remains. Ranks come from the vocabulary,
//! where a token's rank is its index.
//!
//! The vocabulary is a build-time asset, so this does no I/O and cannot rot.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::OnceLock;

use regex::Regex;

/// The packed `o200k_base` vocabulary: `b"O200K\x01"`, a little-endian `u32`
/// count, then that many `<len: u8><bytes>` entries. A token's rank is the index
/// at which it appears.
const VOCAB: &[u8] = include_bytes!("../../assets/o200k_base.bin");

const MAGIC: &[u8] = b"O200K\x01";

/// A token's position in the merge order. Lower merges first.
type Rank = u32;

/// The piece pattern GPT-4o and GPT-5 split on, ported from OpenAI's published
/// `o200k_base` definition.
///
/// The original ends with `\s+(?!\S)` followed by `\s+`: a whitespace run that
/// reaches the end of the text is taken whole, and otherwise the run gives its
/// LAST character to the following piece, so that a word carries its own leading
/// space (`" leading"`, not `"leading"`). That negative lookahead is the only
/// construct in the pattern that a linear-time regex engine cannot express, so it
/// is dropped here and its meaning applied in [`split_pieces`] instead. The two
/// are equivalent, and `bpe_matches_reference_split` proves it.
const PIECE_PATTERN: &str = concat!(
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
    "|",
    r"\p{N}{1,3}",
    "|",
    r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
    "|",
    r"\s*[\r\n]+",
    "|",
    r"\s+",
);

/// Byte sequence to rank, for every token in the vocabulary.
fn ranks() -> &'static HashMap<&'static [u8], Rank> {
    static RANKS: OnceLock<HashMap<&'static [u8], Rank>> = OnceLock::new();
    RANKS.get_or_init(|| {
        let (magic, rest) = VOCAB.split_at(MAGIC.len());
        assert_eq!(magic, MAGIC, "o200k vocabulary asset is corrupt");
        let (count, mut rest) = rest.split_at(4);
        let count = u32::from_le_bytes(count.try_into().expect("4 bytes")) as usize;

        let mut map = HashMap::with_capacity(count);
        for rank in 0..count {
            let (len, tail) = rest.split_first().expect("vocabulary truncated");
            let (token, tail) = tail.split_at(*len as usize);
            map.insert(token, rank as Rank);
            rest = tail;
        }
        // Hard assert, like every other malformation check here: trailing bytes
        // mean a corrupt asset in release builds too, and this runs once.
        assert!(rest.is_empty(), "vocabulary has trailing bytes");
        map
    })
}

fn pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(PIECE_PATTERN).expect("the piece pattern is valid"))
}

/// Split text the way the reference tokenizer does.
///
/// Everything is the fixed pattern's doing except one rule the pattern cannot
/// state without lookahead: a whitespace run that does NOT end the text hands its
/// final character to whatever follows, so the next word owns its leading space.
fn split_pieces(text: &str) -> impl Iterator<Item = &str> {
    let mut cursor = 0usize;
    std::iter::from_fn(move || {
        if cursor >= text.len() {
            return None;
        }
        let m = pattern().find_at(text, cursor)?;
        let piece = m.as_str();

        // A run of plain spaces or tabs (no newline: those are matched by an
        // earlier alternative and taken whole) gives back its last character when
        // more text follows.
        let is_inline_whitespace = !piece.is_empty()
            && piece.chars().all(char::is_whitespace)
            && !piece.contains(['\r', '\n']);
        if is_inline_whitespace && m.end() < text.len() && piece.chars().count() > 1 {
            let last = piece.char_indices().next_back().expect("non-empty").0;
            cursor = m.start() + last;
            return Some(&piece[..last]);
        }

        cursor = m.end();
        Some(piece)
    })
}

/// How many tokens one piece merges down to.
///
/// Greedy byte-pair merging: while some adjacent pair of parts is a known token,
/// merge the lowest-ranked such pair, leftmost on a tie. The token count is the
/// number of parts that survive.
///
/// The naive shape (a `Vec` of boundaries, rescanned for the minimum and spliced
/// on each merge) is quadratic twice over: the rescan is linear, and so is the
/// splice. The split pattern does not bound a piece's length, so a prompt of one
/// repeated symbol IS one piece, and 100 KB of it took four and a half seconds
/// before any network call. This runs as an affordability gate, so that is a way
/// to stall the caller.
///
/// So parts live in a doubly-linked list over byte offsets, making a merge O(1),
/// and the next pair to merge comes off a heap, making selection logarithmic.
/// A merge invalidates at most two neighbouring pairs; rather than find and delete
/// their heap entries, the new ones are pushed and stale ones are recognised on
/// pop by comparing against the live rank. That is the standard lazy-deletion
/// heap, and it makes the whole merge O(n log n).
fn count_piece(piece: &[u8], ranks: &HashMap<&'static [u8], Rank>) -> usize {
    if piece.len() <= 1 {
        return piece.len();
    }

    // A part is identified by the byte offset it starts at. `next[i]` and `prev[i]`
    // walk the parts that are still alive; `next[i] == piece.len()` marks the end.
    // Both are sized `len + 1` so the trailing sentinel has a slot.
    //
    // `NONE` stands for "no such part": the first part has nothing before it, and a
    // merged-away part has nothing after it. Any index past `len` means the same,
    // which is why every read is guarded by a bound check rather than by equality.
    const NONE: usize = usize::MAX;
    let len = piece.len();
    let mut next: Vec<usize> = (1..=len + 1).collect();
    let mut prev: Vec<usize> = std::iter::once(NONE).chain(0..len).collect();

    // The rank of joining the part at `i` with the one after it, if that join is a
    // known token. `None` once `i` has no successor to merge with.
    let rank_of = |i: usize, next: &[usize]| -> Option<Rank> {
        let mid = *next.get(i)?;
        let end = *next.get(mid)?;
        if end > len {
            return None;
        }
        ranks.get(&piece[i..end]).copied()
    };

    // Ordered by rank then by position, both ascending, so `pop` yields the
    // lowest-ranked leftmost pair. `Reverse` turns the max-heap into a min-heap.
    let mut heap: BinaryHeap<Reverse<(Rank, usize)>> = (0..len)
        .filter_map(|i| rank_of(i, &next).map(|r| Reverse((r, i))))
        .collect();

    let mut parts = len;
    while let Some(Reverse((rank, i))) = heap.pop() {
        // A stale entry: the pair at `i` was merged away, or its rank has changed
        // since this entry was pushed. Recomputing is cheaper than deleting.
        if next[i] > len || rank_of(i, &next) != Some(rank) {
            continue;
        }

        // Splice out the successor, joining it onto `i`.
        let mid = next[i];
        let end = next[mid];
        next[i] = end;
        if end <= len {
            prev[end] = i;
        }
        next[mid] = NONE; // so a stale heap entry for `mid` is recognised on pop
        parts -= 1;

        // Only the pair starting at `i`, and the one ending at `i`, can have
        // changed. Push their new ranks; the old entries die on pop.
        if let Some(r) = rank_of(i, &next) {
            heap.push(Reverse((r, i)));
        }
        let before = prev[i];
        if before != NONE {
            if let Some(r) = rank_of(before, &next) {
                heap.push(Reverse((r, before)));
            }
        }
    }

    parts
}

/// The number of `o200k_base` tokens `text` encodes to.
pub fn count_tokens(text: &str) -> usize {
    let ranks = ranks();
    split_pieces(text)
        .map(|piece| {
            let bytes = piece.as_bytes();
            // A whole piece that is itself a token needs no merging, which is the
            // common case for ordinary words.
            if ranks.contains_key(bytes) {
                1
            } else {
                count_piece(bytes, ranks)
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_vocabulary_asset_loads_and_ranks_are_dense() {
        let r = ranks();
        assert_eq!(r.len(), 199_998, "o200k_base has 199998 entries");
        // Rank 0 is "!" in the published vocabulary.
        assert_eq!(r.get(b"!".as_slice()), Some(&0));
    }

    #[test]
    fn a_single_byte_is_one_token() {
        assert_eq!(count_tokens("a"), 1);
        assert_eq!(count_tokens(""), 0);
    }

    /// The lookahead rule: a word owns the space before it.
    #[test]
    fn an_inline_whitespace_run_hands_its_last_space_to_the_next_word() {
        let pieces: Vec<&str> = split_pieces("  leading").collect();
        assert_eq!(pieces, vec![" ", " leading"]);

        let pieces: Vec<&str> = split_pieces("a b  c").collect();
        assert_eq!(pieces, vec!["a", " b", " ", " c"]);
    }

    /// Trailing whitespace ends the text, so it is taken whole rather than split.
    #[test]
    fn a_whitespace_run_that_ends_the_text_is_taken_whole() {
        let pieces: Vec<&str> = split_pieces("hi   ").collect();
        assert_eq!(pieces, vec!["hi", "   "]);
    }

    /// Newlines are matched by their own alternative, before the inline rule.
    #[test]
    fn newlines_are_taken_whole_with_their_leading_whitespace() {
        let pieces: Vec<&str> = split_pieces("a\n\nb").collect();
        assert_eq!(pieces, vec!["a", "\n\n", "b"]);
    }

    /// Merging is rank-ordered, not longest-match: the count depends on the
    /// vocabulary's merge order, which is what makes a real BPE necessary.
    #[test]
    fn known_strings_count_to_their_published_token_counts() {
        // These counts come from the reference o200k_base tokenizer.
        assert_eq!(count_tokens("hello world"), 2);
        assert_eq!(count_tokens("The quick brown fox"), 4);
        assert_eq!(count_tokens("1234567890"), 4);
    }

    #[test]
    fn multibyte_text_counts_without_panicking_on_char_boundaries() {
        assert!(count_tokens("这是一段中文文本") > 0);
        assert!(count_tokens("emoji 🚀🔥") > 0);
        assert!(count_tokens("café über naïve") > 0);
    }

    /// The split pattern does not bound the length of a symbol run, so a prompt of
    /// nothing but punctuation arrives here as ONE piece. A merge loop that
    /// re-hashed every pair on every step would take seconds on it, which turns a
    /// crafted prompt into a way to stall the caller before any model is reached.
    /// The cached pair ranks keep it linear enough to be uninteresting.
    #[test]
    fn a_pathological_single_piece_does_not_blow_up() {
        // The split pattern does not bound a piece's length, so a prompt of nothing
        // but punctuation arrives at the merge as ONE piece. This runs before any
        // network call, as an affordability gate, so a quadratic merge is a way to
        // stall the caller: the old one took 4.5 seconds on 100 KB.
        //
        // Asserting a wall-clock budget would only measure the machine. Assert the
        // SHAPE instead: quadrupling the input must not multiply the time by
        // sixteen. A generous factor of six still fails a return to quadratic while
        // tolerating a loaded, unoptimised debug build.
        let time = |n: usize| {
            let piece = "!".repeat(n);
            let start = std::time::Instant::now();
            assert!(count_tokens(&piece) > 0);
            start.elapsed()
        };

        // Warm the vocabulary and the regex so neither is charged to the first run.
        time(1_000);

        let small = time(10_000);
        let large = time(40_000);
        assert!(
            large < small * 6,
            "4x the input took {large:?} vs {small:?}: the merge went quadratic"
        );
    }
}

/// Proof that this encoder agrees with OpenAI's reference implementation.
///
/// The expected counts below were produced ONCE by the reference tokenizer and
/// recorded here. `o200k_base` is a frozen artifact and this encoder is
/// deterministic, so the answers cannot drift on their own: if one of these ever
/// fails, this encoder has changed, and every cost estimate built on it is wrong.
///
/// Recording them means the proof carries no dependency of its own. To re-derive
/// a vector, run the string through OpenAI's `tiktoken` (any binding) and take
/// the length of the encoding.
#[cfg(test)]
mod reference_equivalence {
    use super::count_tokens;

    /// Every shape a real prompt is made of, with the reference token count.
    #[test]
    fn it_matches_the_reference_on_the_shapes_a_prompt_is_made_of() {
        let vectors: &[(&str, usize)] = &[
            ("", 0),
            ("a", 1),
            ("hello world", 2),
            ("The quick brown fox jumps over the lazy dog.", 10),
            // Whitespace, where the dropped lookahead lived. These are the cases
            // that break if the give-back-the-last-space rule is wrong.
            ("  leading spaces", 3),
            ("trailing spaces   ", 4),
            ("x  y", 3),
            ("a b  c   d", 6),
            ("\n\n\n", 1),
            ("\t\t", 1),
            ("a\n\nb", 3),
            ("mix \n  \n end", 3),
            ("word  \n", 2),
            ("          ", 1),
            ("             ", 1),
            ("a\t\tb", 3),
            ("end.  ", 3),
            // Code and data, which is most of what a real prompt carries.
            ("def f():\n    return 1\n", 8),
            ("fn main(){let x:Vec<u32>=(0..10).filter(|n|n%2==0).collect();}", 28),
            ("{\"a\": 1, \"bb\": [1, 2, 3], \"ccc\": {\"d\": true, \"e\": null}}", 31),
            ("https://openrouter.ai/api/v1/models?limit=50&offset=0", 17),
            ("3f8a1c2e-9b4d-4f21-8e7a-1c0d5b6e2a94", 34),
            ("SELECT * FROM t WHERE x > 1 GROUP BY 1;", 14),
            ("CamelCaseIdentifier snake_case_name SCREAMING_CASE", 10),
            // The contraction alternatives in the piece pattern.
            ("it's don't we're I've I'm they'll he'd", 7),
            ("IT'S DON'T WE'RE", 6),
            // Numbers, which the pattern caps at three digits per piece.
            ("1234567890", 4),
            ("0.000003", 4),
            ("-42", 2),
            ("1e-9", 4),
            // Merging is rank-ordered, not longest-match: "aaaa" is one token.
            ("aa", 1),
            ("aaa", 1),
            ("aaaa", 1),
            // A long symbol run is a single piece, exercising the merge loop.
            ("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", 3),
            // Non-Latin scripts, multi-byte characters, zero-width, flag emoji.
            ("这是一段中文文本，用于测试分词器的行为差异。", 16),
            ("こんにちは、これは日本語です。", 7),
            ("Здравствуйте, это русский текст.", 6),
            ("café über naïve", 5),
            ("emoji 🚀🔥 and ✨", 7),
            ("🇫🇷🇯🇵", 8),
            ("\u{200b}zero width", 3),
        ];

        for (text, expected) in vectors {
            assert_eq!(count_tokens(text), *expected, "token count diverged on {text:?}");
        }
    }

    /// A realistic block of source, which is what an agent's prompt is mostly
    /// made of and the hardest thing to tokenize: dense punctuation, mixed case,
    /// indentation, and the whitespace runs the piece pattern is fussiest about.
    #[test]
    fn it_matches_the_reference_on_a_real_block_of_source() {
        const SOURCE: &str = r#"fn count_piece(piece: &[u8], ranks: &HashMap<&'static [u8], Rank>) -> usize {
    if piece.len() <= 1 {
        return piece.len();
    }
    let mut parts: Vec<usize> = (0..=piece.len()).collect();
    let mut pair_rank: Vec<Option<Rank>> =
        (0..parts.len() - 1).map(|i| rank_at(piece, &parts, ranks, i)).collect();
    loop {
        let Some((i, _)) = pair_rank
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (i, r)))
            .min_by_key(|&(i, r)| (r, i))
        else {
            break;
        };
        parts.remove(i + 1);
        pair_rank.remove(i + 1);
        pair_rank[i] = rank_at(piece, &parts, ranks, i);
        if i > 0 {
            pair_rank[i - 1] = rank_at(piece, &parts, ranks, i - 1);
        }
    }
    parts.len() - 1
}"#;
        assert_eq!(SOURCE.len(), 800, "the recorded count is for exactly this text");
        assert_eq!(count_tokens(SOURCE), 246);
    }

    /// Five hundred pseudorandom strings of mixed scripts and punctuation, the
    /// byte sequences nobody thinks to write down. Their counts sum to a single
    /// recorded number: any divergence on any one string moves the total.
    #[test]
    fn it_matches_the_reference_across_a_pseudorandom_corpus() {
        /// The reference total for exactly the corpus generated below. Changing
        /// the seed, the alphabet, the length bound or the count invalidates it.
        const REFERENCE_TOTAL: usize = 8266;

        // A deterministic xorshift, so a failure reproduces and no `rand`
        // dependency creeps in.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let alphabet: Vec<char> =
            " \t\n\rabcXYZ0129,.;:'\"/\\_-()[]{}日本語中文🚀é∀×".chars().collect();

        let total: usize = (0..500)
            .map(|_| {
                let len = (next() % 40) as usize;
                let text: String =
                    (0..len).map(|_| alphabet[(next() as usize) % alphabet.len()]).collect();
                count_tokens(&text)
            })
            .sum();

        assert_eq!(total, REFERENCE_TOTAL, "the encoder diverged somewhere in the corpus");
    }
}
