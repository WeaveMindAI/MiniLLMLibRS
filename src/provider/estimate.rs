//! Estimating a completion's cost BEFORE it is sent.
//!
//! A caller who must decide "can this request be afforded?" needs a number ahead
//! of the call, but no provider will count a prompt's tokens for free across
//! model families: OpenRouter exposes no tokenizer endpoint, and each family
//! (GPT, Claude, Llama, Qwen) tokenizes the same bytes differently.
//!
//! So we count with one tokenizer ([`super::bpe`], GPT's `o200k_base`, the most
//! widely deployed) and correct its bias with a fixed multiplier. Measured
//! against Anthropic's authoritative counter over a corpus of real prompts
//! (source code, JSON, prose, URLs, CJK), `o200k_base` undercounts Claude on
//! EVERY sample, by 5.6% to 58.3%.
//! [`SAFETY_MULTIPLIER`] covers the worst observed case, so the estimate leans
//! high rather than flipping a coin. It is deliberately not tuned to the mean:
//! an estimate that is sometimes low is worse than one that is always a little
//! high, because only the low side lets a caller overspend.
//!
//! The estimate is deliberately high, not a best guess, and still an estimate:
//! the multiplier covers every case measured, not every case possible. The true
//! cost is known after the call and should always replace it.

use crate::generator::CompletionParameters;
use crate::generator::ModelRates;
use crate::message::{ContentPart, Message, MessageContent};
use crate::provider::bpe;
use crate::provider::wire::TokenPrice;

/// Corrects `o200k_base`'s systematic undercount on non-GPT tokenizers.
///
/// Derived, not guessed: over a 28-sample corpus of real prompt text priced
/// against Anthropic's `count_tokens` endpoint, the ratio of true tokens to
/// `o200k_base` tokens ranged 1.056 to 1.583 (median 1.228). 1.6 bounds every
/// observed sample. The cost is ~30% average over-estimation, which a reservation
/// tolerates and an undercount does not.
pub const SAFETY_MULTIPLIER: f64 = 1.6;

/// Per-message wire overhead: role tags, delimiters, and the envelope every
/// provider wraps a turn in. Small, but a 40-turn thread would otherwise be
/// undercounted by the whole envelope.
const TOKENS_PER_MESSAGE: u32 = 4;

/// What a still image costs when we cannot see its dimensions.
///
/// Vision models tile an image and charge per tile, so the true count depends on
/// resolution, which a `data:` URI hides behind base64 and a remote URL hides
/// entirely. Anthropic caps a still at roughly this; OpenAI's high-detail 1024px
/// image is about half. So this is the honest high figure for one still of
/// unknown size.
///
/// A video FRAME is a different quantity and costs `TOKENS_PER_VIDEO_FRAME`:
/// the provider resamples every frame to one fixed small size, so a frame is
/// never an arbitrary-resolution still.
const TOKENS_PER_STILL_IMAGE: u32 = 1_600;

/// What one frame sampled out of a video costs. Gemini's published figure at
/// default media resolution.
const TOKENS_PER_VIDEO_FRAME: u32 = 258;

/// A photo of unknown size costs more than a frame the provider has resampled to a
/// fixed small size. Conflating the two is what once made an hour of video look
/// six times its real cost, and therefore unsendable. Checked at compile time, so
/// re-conflating them breaks the build rather than a test.
const _: () = assert!(
    TOKENS_PER_STILL_IMAGE > TOKENS_PER_VIDEO_FRAME,
    "a still of unknown size must cost more than a resampled video frame"
);

/// How many frames a second of video is sampled into. Gemini samples at one frame
/// per second regardless of the source's own frame rate, which is why the
/// source's frame rate never enters the estimate.
const VIDEO_FRAMES_PER_SECOND: f64 = 1.0;

/// How many audio tokens a second of sound becomes. Gemini's published figure,
/// and the only one any provider states.
const AUDIO_TOKENS_PER_SECOND: f64 = 32.0;

/// What to assume a clip lasts when the caller did not say.
///
/// Media carries its duration in a container header this library does not parse,
/// and a remote URL hides it entirely. A caller that opened the file knows the
/// length and should pass it (`with_duration`). Absent that, one minute is a
/// reasonable clip, and anything it understates is caught by the clamp: no
/// estimate may exceed what the model would actually accept as input.
const DEFAULT_MEDIA_SECONDS: f64 = 60.0;

/// The length to price a clip at: what the caller said, or the default.
///
/// A nonsense value (NaN, infinite, negative) is not a length, so it falls back
/// to the default. A real length is used as-is, however long: an over-long clip
/// counts past the model's context window and pricing clamps to the window there,
/// the one place that knows how large the model's input can be.
fn media_seconds(declared: Option<f64>) -> f64 {
    match declared {
        Some(secs) if secs.is_finite() && secs >= 0.0 => secs,
        _ => DEFAULT_MEDIA_SECONDS,
    }
}

// The token arithmetic below saturates instead of overflowing. Saturation sits at
// u64::MAX tokens, nineteen orders of magnitude past any context window, and
// pricing clamps every count to the window long before that, so saturating can
// never change a price; it only keeps an absurd length from panicking the count.

/// Tokens for `secs` of sound, rounded up. (An `as u64` cast saturates.)
fn audio_tokens_for(secs: f64) -> u64 {
    (secs * AUDIO_TOKENS_PER_SECOND).ceil() as u64
}

/// Tokens for the frames `secs` of video samples into, rounded up. A video's
/// soundtrack is counted separately, at the audio rate.
fn video_frame_tokens_for(secs: f64) -> u64 {
    let frames = (secs * VIDEO_FRAMES_PER_SECOND).ceil() as u64;
    frames.saturating_mul(u64::from(TOKENS_PER_VIDEO_FRAME))
}

/// Count the tokens a thread will occupy, biased high, split by how they bill.
///
/// Text is tokenized and scaled by [`SAFETY_MULTIPLIER`]. A still image costs a
/// flat upper-bound tile count. A video costs one frame per second of its length,
/// plus its soundtrack in audio tokens. Audio costs `AUDIO_TOKENS_PER_SECOND`
/// per second. A clip of unstated length is assumed to run
/// `DEFAULT_MEDIA_SECONDS`.
///
/// The counts are unclamped, so they can exceed what a model would accept. Pricing
/// clamps them: see [`estimate_cost_usd`].
pub fn estimate_prompt_tokens(messages: &[Message]) -> PromptEstimate {
    let mut text_tokens = 0u64;
    let mut image_tokens = 0u64;
    let mut audio_tokens = 0u64;

    for message in messages {
        text_tokens += u64::from(TOKENS_PER_MESSAGE);
        match &message.content {
            MessageContent::Text(t) => text_tokens += bpe::count_tokens(t) as u64,
            MessageContent::Parts(parts) => {
                for part in parts {
                    // Every variant is named on purpose: a new content kind must
                    // force a decision here rather than fall into a catch-all and
                    // silently price as zero.
                    match part {
                        ContentPart::Text { text } => text_tokens += bpe::count_tokens(text) as u64,
                        ContentPart::Image { .. } => {
                            image_tokens += u64::from(TOKENS_PER_STILL_IMAGE)
                        }
                        // A video bills as its frames plus its soundtrack. Assume it
                        // has sound: a silent video then over-reserves by an eighth
                        // of a frame per second, which is the safe direction.
                        ContentPart::Video { video_url } => {
                            let secs = media_seconds(video_url.duration_secs);
                            image_tokens =
                                image_tokens.saturating_add(video_frame_tokens_for(secs));
                            audio_tokens = audio_tokens.saturating_add(audio_tokens_for(secs));
                        }
                        ContentPart::Audio { input_audio } => {
                            audio_tokens = audio_tokens.saturating_add(audio_tokens_for(
                                media_seconds(input_audio.duration_secs),
                            ))
                        }
                    }
                }
            }
        }
    }

    PromptEstimate {
        // The tokenizer's bias correction, applied once at the end so it scales
        // the whole text bucket rather than compounding per message.
        text_tokens: (text_tokens as f64 * SAFETY_MULTIPLIER).ceil() as u64,
        image_tokens,
        audio_tokens,
    }
}

/// A deliberately high estimate of a prompt's token count, split by the rate each
/// kind bills at.
///
/// Unclamped: a prompt can be counted larger than any model would accept, because
/// counting does not know which model it is for. [`estimate_cost_usd`] clamps to
/// the model's context window, the largest input that can physically be sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PromptEstimate {
    /// Text tokens, billed at the plain input rate.
    pub text_tokens: u64,
    /// Image tokens: a still costs `TOKENS_PER_STILL_IMAGE`, a video's frames
    /// cost `TOKENS_PER_VIDEO_FRAME` each. Billed at the image rate.
    pub image_tokens: u64,
    /// Audio tokens, including a video's soundtrack. Billed at the audio rate,
    /// which runs from one to a thousand times the input rate.
    pub audio_tokens: u64,
}

impl PromptEstimate {
    /// Every token the prompt occupies, regardless of what each bills at. Use this
    /// for context-window checks, never for money.
    pub fn total_tokens(&self) -> u64 {
        self.text_tokens
            .saturating_add(self.image_tokens)
            .saturating_add(self.audio_tokens)
    }

    /// What this prompt costs to send, in USD, given a model's rates and the
    /// largest input it accepts.
    ///
    /// A count above the context window describes a prompt no provider would take,
    /// so the honest bound is the cost of filling the window instead. Which tokens
    /// fill it decides the price, so the dearest do, judged by the rates this model
    /// actually charges rather than an assumed ordering: audio can bill a thousand
    /// times text, yet on a model that prices no audio separately it bills exactly
    /// as text. Filling with cheap tokens would understate the cost, the one
    /// failure a spend gate cannot tolerate.
    fn input_cost_usd(self, context_length: u32, price: &TokenPrice) -> f64 {
        let mut buckets: [(u64, f64); 3] = [
            (self.audio_tokens, price.audio_rate()),
            (self.image_tokens, price.image_rate()),
            (self.text_tokens, price.input_per_mtok),
        ];
        // Dearest first, so the window fills with the priciest tokens present.
        buckets.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut room = u64::from(context_length);
        let mut cost = 0.0;
        for (tokens, rate) in buckets {
            let billed = tokens.min(room);
            cost += billed as f64 * rate;
            room -= billed;
        }
        cost
    }
}

/// The largest completion this request can produce, in tokens.
///
/// Precedence: an explicit `max_tokens` binds it; otherwise the model's published
/// completion cap; otherwise the context window, which is the only ceiling that
/// exists for the ~17% of catalog models publishing no completion cap.
///
/// Reasoning tokens are billed at the output rate and are NOT counted against
/// `max_tokens` by any provider, so a thinking model bills past the ceiling the
/// caller set. When reasoning is enabled we add its budget on top: an explicit
/// reasoning `max_tokens` when given, else the model's own completion ceiling,
/// since an effort level places no numeric bound on how long a model may think.
fn max_output_tokens(params: &CompletionParameters, rates: &ModelRates) -> u64 {
    let ceiling = rates.max_completion_tokens.unwrap_or(rates.context_length);

    // Both halves are bounded by what the model can emit, so neither a caller's
    // `max_tokens` nor a reasoning budget can inflate the estimate past reality.
    let visible = params.max_tokens.unwrap_or(ceiling).min(ceiling);
    let thinking = match &params.reasoning {
        None => 0,
        Some(r) if r.effort.as_deref() == Some("none") => 0,
        Some(r) => r.max_tokens.unwrap_or(ceiling).min(ceiling),
    };

    u64::from(visible) + u64::from(thinking)
}

/// A deliberately high estimate, in USD, of what one completion will cost.
///
/// Every assumption leans expensive, so the figure works as a ceiling to reserve
/// against; it is still an estimate (tokenizers differ across model families),
/// not a guarantee.
///
/// Each kind of prompt token is priced at the rate it actually bills at: text at
/// the input rate, images (and a video's frames) at the image rate, audio (and a
/// video's soundtrack) at the audio rate, which on some models is a thousand times
/// the others. The largest possible completion is priced at the output rate.
///
/// A prompt counted larger than the model's context window cannot be sent at all,
/// so it is priced as the largest input that model accepts. This is what makes the
/// estimate total: an unknown clip length, or a very long one, yields the cost of
/// filling the window rather than an error the caller has to handle.
///
/// Everything is charged as uncached. Caching only ever makes the real cost lower,
/// so ignoring it keeps the estimate on the high side.
pub fn estimate_cost_usd(
    messages: &[Message],
    params: &CompletionParameters,
    rates: &ModelRates,
) -> f64 {
    let price = &rates.price;
    let input = estimate_prompt_tokens(messages).input_cost_usd(rates.context_length, price);
    let output = max_output_tokens(params, rates) as f64 * price.output_per_mtok;

    (input + output) / 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generator::ReasoningConfig;
    use crate::message::AudioInput;
    use crate::provider::wire::{TokenPrice, AUDIO_RATE_FALLBACK_MULTIPLE};

    fn user(text: &str) -> Message {
        Message::user(text)
    }

    fn parts(parts: Vec<ContentPart>) -> Message {
        Message::user(MessageContent::Parts(parts))
    }

    fn rates(max_completion: Option<u32>, context: u32) -> ModelRates {
        ModelRates {
            price: TokenPrice::new(3.0, 15.0),
            max_completion_tokens: max_completion,
            context_length: context,
        }
    }

    /// The property that matters: the estimate must never fall below the truth.
    /// These are real token counts from Anthropic's `count_tokens` endpoint for
    /// the exact strings below, so a regression in the tokenizer or the
    /// multiplier fails here rather than in a billing report.
    #[test]
    fn the_estimate_is_an_upper_bound_on_real_claude_token_counts() {
        let corpus: &[(&str, u64)] = &[
            ("The quick brown fox jumps over the lazy dog. Tokenization is not standardized across model families, which makes cross-provider estimation inherently approximate.", 39),
            ("fn main(){let x:Vec<u32>=(0..10).filter(|n|n%2==0).map(|n|n*n).collect();println!(\"{:?}\",x);}", 55),
            ("{\"a\": 1, \"bb\": [1, 2, 3], \"ccc\": {\"d\": true, \"e\": null}}", 39),
            ("这是一段中文文本，用于测试分词器的行为差异。", 30),
        ];
        for (text, truth) in corpus {
            let est = estimate_prompt_tokens(&[user(text)]).text_tokens;
            assert!(
                est >= *truth,
                "estimate {est} under-counts {truth} real tokens for {text:?}"
            );
        }
    }

    #[test]
    fn an_explicit_max_tokens_bounds_the_output_but_never_exceeds_the_model_ceiling() {
        let r = rates(Some(8_000), 200_000);
        let p = CompletionParameters {
            max_tokens: Some(500),
            ..Default::default()
        };
        assert_eq!(max_output_tokens(&p, &r), 500);

        // A caller asking for more than the model can emit is bounded by the model.
        let p = CompletionParameters {
            max_tokens: Some(999_999),
            ..Default::default()
        };
        assert_eq!(max_output_tokens(&p, &r), 8_000);
    }

    /// `CompletionParameters::default()` carries `max_tokens: Some(4096)`, so the
    /// estimate prices what will REALLY be sent, not the model's ceiling. A caller
    /// who never sets `max_tokens` is capped at 4096 output tokens on the wire.
    #[test]
    fn the_default_max_tokens_is_what_gets_priced() {
        let r = rates(Some(64_000), 200_000);
        assert_eq!(
            max_output_tokens(&CompletionParameters::default(), &r),
            4_096
        );
    }

    /// Only an explicitly unset `max_tokens` falls through to the model's ceiling,
    /// and with no published completion cap the context window is the last bound.
    #[test]
    fn with_no_max_tokens_and_no_completion_cap_the_context_window_is_the_ceiling() {
        let r = rates(None, 32_768);
        let p = CompletionParameters {
            max_tokens: None,
            ..Default::default()
        };
        assert_eq!(max_output_tokens(&p, &r), 32_768);

        // A published completion cap wins over the context window.
        let capped = rates(Some(8_000), 32_768);
        assert_eq!(max_output_tokens(&p, &capped), 8_000);
    }

    /// Reasoning tokens bill at the output rate and are not charged against
    /// `max_tokens`, so they must be added ON TOP of it, never folded inside.
    #[test]
    fn reasoning_tokens_are_added_on_top_of_the_visible_output_budget() {
        let r = rates(Some(8_000), 200_000);
        let base = CompletionParameters {
            max_tokens: Some(1_000),
            ..Default::default()
        };
        assert_eq!(max_output_tokens(&base, &r), 1_000);

        // An explicit thinking budget adds exactly itself.
        let thinking = CompletionParameters {
            max_tokens: Some(1_000),
            reasoning: Some(ReasoningConfig {
                effort: None,
                max_tokens: Some(4_000),
                exclude: None,
            }),
            ..Default::default()
        };
        assert_eq!(max_output_tokens(&thinking, &r), 5_000);

        // An effort level bounds nothing, so the model's ceiling is the only bound.
        let effort = CompletionParameters {
            max_tokens: Some(1_000),
            reasoning: Some(ReasoningConfig {
                effort: Some("high".into()),
                max_tokens: None,
                exclude: None,
            }),
            ..Default::default()
        };
        assert_eq!(max_output_tokens(&effort, &r), 9_000);

        // "none" disables reasoning entirely, so it costs nothing.
        let off = CompletionParameters {
            max_tokens: Some(1_000),
            reasoning: Some(ReasoningConfig {
                effort: Some("none".into()),
                max_tokens: None,
                exclude: None,
            }),
            ..Default::default()
        };
        assert_eq!(max_output_tokens(&off, &r), 1_000);
    }

    fn video(duration_secs: Option<f64>) -> Message {
        parts(vec![ContentPart::Video {
            video_url: crate::message::VideoUrl {
                url: "https://x/y.mp4".into(),
                duration_secs,
            },
        }])
    }

    fn audio(duration_secs: Option<f64>) -> Message {
        parts(vec![ContentPart::Audio {
            input_audio: AudioInput {
                data: "AAAA".into(),
                format: Some("wav".into()),
                duration_secs,
            },
        }])
    }

    /// A clip of stated length is billed by the second, at the audio rate.
    #[test]
    fn audio_of_known_length_is_counted_by_the_second() {
        let one_second = estimate_prompt_tokens(&[audio(Some(1.0))]);
        assert_eq!(one_second.audio_tokens, AUDIO_TOKENS_PER_SECOND as u64);
        assert_eq!(one_second.image_tokens, 0);

        // A partial second still costs a whole second: the bound never rounds down.
        let sliver = estimate_prompt_tokens(&[audio(Some(0.01))]);
        assert_eq!(sliver.audio_tokens, 1);
    }

    /// A video is its frames PLUS its soundtrack, which bill at different rates.
    /// A frame is not a still: the provider resamples every frame to one fixed
    /// small size, so it costs a sixth of what an arbitrary photo costs.
    #[test]
    fn a_video_is_counted_as_frames_and_a_soundtrack() {
        let ten = estimate_prompt_tokens(&[video(Some(10.0))]);
        assert_eq!(
            ten.image_tokens,
            10 * u64::from(TOKENS_PER_VIDEO_FRAME),
            "one frame a second"
        );
        assert_eq!(
            ten.audio_tokens,
            10 * AUDIO_TOKENS_PER_SECOND as u64,
            "its sound too"
        );

        // A partial second still costs a whole frame.
        let sliver = estimate_prompt_tokens(&[video(Some(0.1))]);
        assert_eq!(sliver.image_tokens, u64::from(TOKENS_PER_VIDEO_FRAME));
    }

    /// The counts must reproduce what providers really charge, or the estimate is
    /// describing a video nobody could send.
    ///
    /// Gemini bills 258 tokens a frame at one frame a second, plus 32 a second for
    /// sound. It publishes a limit of one hour of video WITHOUT audio, and about
    /// forty-five minutes WITH it, on a one-million-token model. These counts land
    /// exactly there, which is the check: silent video just fits the window, and
    /// adding its soundtrack pushes it out, so the fit is what forces the shorter
    /// limit. An earlier constant, six times too large, claimed an hour of video
    /// needed 5.76 million tokens and was therefore unsendable.
    #[test]
    fn the_counts_reproduce_geminis_published_video_limits() {
        const WINDOW: u64 = 1_000_000;

        let hour = estimate_prompt_tokens(&[video(Some(3600.0))]);
        assert_eq!(hour.image_tokens, 3600 * 258, "one frame a second");
        assert_eq!(hour.audio_tokens, 3600 * 32, "its soundtrack");

        // An hour of silent video fits; its frames alone are under the window.
        assert!(hour.image_tokens < WINDOW, "an hour of silent video fits");
        // With sound it does not, which is why the published limit drops to ~45 min.
        assert!(
            hour.total_tokens() > WINDOW,
            "an hour WITH audio does not fit"
        );

        // And forty-five minutes with sound does fit, as Gemini says.
        let three_quarters = estimate_prompt_tokens(&[video(Some(45.0 * 60.0))]);
        assert!(
            three_quarters.total_tokens() < WINDOW,
            "45 min with audio fits"
        );
    }

    /// A clip of unstated length is assumed to run `DEFAULT_MEDIA_SECONDS`, never
    /// refused. Refusing would force every caller to handle an error for a case
    /// with a perfectly good answer.
    #[test]
    fn media_of_unknown_length_is_assumed_not_refused() {
        let assumed = DEFAULT_MEDIA_SECONDS as u64;

        let a = estimate_prompt_tokens(&[audio(None)]);
        assert_eq!(a.audio_tokens, assumed * AUDIO_TOKENS_PER_SECOND as u64);

        let v = estimate_prompt_tokens(&[video(None)]);
        assert_eq!(v.image_tokens, assumed * u64::from(TOKENS_PER_VIDEO_FRAME));
        assert_eq!(v.audio_tokens, assumed * AUDIO_TOKENS_PER_SECOND as u64);
    }

    /// A nonsense duration is not a length, so it falls back to the assumption
    /// rather than casting to a garbage token count.
    #[test]
    fn media_with_a_nonsense_duration_falls_back_to_the_assumption() {
        let assumed = estimate_prompt_tokens(&[audio(None)]).audio_tokens;
        for bad in [f64::INFINITY, f64::NAN, f64::NEG_INFINITY, -1.0] {
            assert_eq!(
                estimate_prompt_tokens(&[audio(Some(bad))]).audio_tokens,
                assumed,
                "{bad}"
            );
            assert_eq!(
                estimate_prompt_tokens(&[video(Some(bad))]).audio_tokens,
                assumed,
                "{bad}"
            );
        }
    }

    /// A declared length is used as-is, however long: the count saturates instead
    /// of overflowing, and pricing clamps to the context window anyway. A few
    /// absurd-but-finite durations once overflowed the accumulation and panicked;
    /// the price was never at stake, because the window bounds it regardless.
    #[test]
    fn an_absurdly_long_clip_counts_saturated_rather_than_overflowing() {
        for absurd in [1e30, f64::MAX] {
            let est = estimate_prompt_tokens(&[video(Some(absurd))]);
            assert!(
                est.image_tokens > 0 && est.audio_tokens > 0,
                "{absurd} still counts"
            );
        }

        // Many such clips still add up without overflowing or panicking.
        let many = vec![
            ContentPart::Video {
                video_url: crate::message::VideoUrl {
                    url: "x".into(),
                    duration_secs: Some(f64::MAX),
                },
            };
            64
        ];
        let est = estimate_prompt_tokens(&[parts(many)]);
        assert_eq!(est.image_tokens, u64::MAX, "saturated, not wrapped");
        assert!(
            est.total_tokens() == u64::MAX,
            "and the total does not panic"
        );
    }

    /// However long the clip, the price is a full context window of the dearest
    /// token present. The cap changes no price; it only keeps the arithmetic sane.
    #[test]
    fn an_absurdly_long_clip_still_prices_as_one_full_window() {
        let window = 1_000u32;
        let r = ModelRates {
            price: TokenPrice::new(1.0, 0.0),
            max_completion_tokens: Some(0),
            context_length: window,
        };
        let params = CompletionParameters {
            max_tokens: Some(0),
            ..Default::default()
        };

        let cost = estimate_cost_usd(&[video(Some(f64::MAX))], &params, &r);
        let full_window_of_audio = f64::from(window) * r.price.audio_rate() / 1e6;
        assert!((cost - full_window_of_audio).abs() < 1e-12, "{cost}");
    }

    /// A prompt counted larger than the model's window cannot be sent, so it is
    /// priced as the largest input the model accepts. That keeps the estimate total
    /// (no error to handle) while never understating the cost.
    #[test]
    fn a_prompt_too_large_to_send_is_priced_as_the_full_window() {
        let window = 1_000u32;
        let r = ModelRates {
            price: TokenPrice::new(1.0, 0.0),
            max_completion_tokens: Some(0),
            context_length: window,
        };
        let params = CompletionParameters {
            max_tokens: Some(0),
            ..Default::default()
        };

        // A day of video is millions of tokens; the window is a thousand.
        let huge = estimate_prompt_tokens(&[video(Some(86_400.0))]);
        assert!(huge.total_tokens() > u64::from(window) * 1_000);

        // Priced at exactly a full window of the dearest token present.
        let cost = estimate_cost_usd(&[video(Some(86_400.0))], &params, &r);
        let full_window_of_audio = f64::from(window) * r.price.audio_rate() / 1e6;
        assert!(
            (cost - full_window_of_audio).abs() < 1e-12,
            "{cost} vs {full_window_of_audio}"
        );
    }

    /// The window fills with the DEAREST tokens present. Filling it with cheap ones
    /// and discarding the dear ones would understate the cost.
    #[test]
    fn an_over_long_prompt_fills_the_window_with_its_dearest_tokens() {
        let window = 100u32;
        // Audio at $1000/Mtok, text at $1/Mtok.
        let price = TokenPrice::new(1.0, 0.0).with_media_rates(Some(1000.0), None);
        let r = ModelRates {
            price,
            max_completion_tokens: Some(0),
            context_length: window,
        };
        let params = CompletionParameters {
            max_tokens: Some(0),
            ..Default::default()
        };

        // Ten seconds of audio is 320 tokens, well over the window, plus text.
        let long_text = "word ".repeat(500);
        let messages = [audio(Some(10.0)), user(&long_text)];

        let cost = estimate_cost_usd(&messages, &params, &r);
        let window_of_audio = f64::from(window) * 1000.0 / 1e6;
        assert!(
            (cost - window_of_audio).abs() < 1e-12,
            "audio must crowd out the cheap text"
        );

        // Had text won the window, it would have been a thousand times cheaper.
        let window_of_text = f64::from(window) * 1.0 / 1e6;
        assert!(cost > window_of_text * 500.0);
    }

    /// The whole point of splitting the buckets: audio can cost a thousand times
    /// what text costs, so folding it into the input count would under-reserve by
    /// three orders of magnitude.
    #[test]
    fn audio_is_billed_at_the_audio_rate_not_the_input_rate() {
        let r = ModelRates {
            // Voxtral's real shape: $0.10 input, $100 audio, per million tokens.
            price: TokenPrice::new(0.1, 0.2).with_media_rates(Some(100.0), None),
            max_completion_tokens: Some(0),
            context_length: 1_000,
        };
        let params = CompletionParameters {
            max_tokens: Some(0),
            ..Default::default()
        };

        // One second = 32 audio tokens at $100/Mtok = $0.0032, plus the message
        // envelope's few text tokens at $0.10/Mtok (negligible but nonzero).
        let cost = estimate_cost_usd(&[audio(Some(1.0))], &params, &r);
        let audio_only = 32.0 * 100.0 / 1e6;
        assert!(
            cost > audio_only,
            "the envelope's text tokens count too: {cost}"
        );
        assert!(
            cost < audio_only * 1.01,
            "but audio dominates: {cost} vs {audio_only}"
        );

        // Had we billed it as input, it would have been a thousand times cheaper.
        let as_input = 32.0 * 0.1 / 1e6;
        assert!(
            cost > as_input * 500.0,
            "audio must not be billed at the input rate"
        );
    }

    /// An unpublished image rate is exactly the text rate. An unpublished AUDIO
    /// rate is not: every model that publishes one charges a premium, so assuming
    /// the text rate would under-charge.
    #[test]
    fn an_unpublished_media_rate_falls_back_to_a_premium_over_text() {
        let plain = TokenPrice::new(3.0, 15.0);
        assert_eq!(plain.image_rate(), 3.0, "image is billed exactly as text");
        assert_eq!(plain.audio_rate(), 3.0 * AUDIO_RATE_FALLBACK_MULTIPLE);
        assert!(
            plain.audio_rate() > plain.input_per_mtok,
            "audio always costs more"
        );

        // A published rate always wins over the assumption, high or low.
        let priced = plain.clone().with_media_rates(Some(30.0), Some(4.0));
        assert_eq!(priced.audio_rate(), 30.0);
        assert_eq!(priced.image_rate(), 4.0);
    }

    #[test]
    fn a_still_image_is_charged_a_bounded_tile_cost_rather_than_nothing() {
        let with_image = parts(vec![ContentPart::Image {
            image_url: crate::message::ImageUrl {
                url: "https://x/y.png".into(),
                detail: None,
            },
        }]);
        let est = estimate_prompt_tokens(&[with_image]);
        assert_eq!(
            est.image_tokens,
            u64::from(TOKENS_PER_STILL_IMAGE),
            "a still is not free"
        );
        assert_eq!(est.audio_tokens, 0, "a still image has no soundtrack");
    }

    #[test]
    fn cost_prices_the_bounded_output_at_the_output_rate() {
        // 1M output tokens at $15/Mtok = $15, plus a small prompt at $3/Mtok.
        let r = ModelRates {
            price: TokenPrice::new(3.0, 15.0),
            max_completion_tokens: Some(1_000_000),
            context_length: 2_000_000,
        };
        let p = CompletionParameters {
            max_tokens: Some(1_000_000),
            ..Default::default()
        };
        let cost = estimate_cost_usd(&[user("hi")], &p, &r);
        assert!((cost - 15.0).abs() < 0.001, "{cost}");
    }
}
