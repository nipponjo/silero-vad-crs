//! Safe Rust wrapper around the vendored Silero VAD C implementation.
//!
//! The C files are compiled into this crate by `build.rs`, so a Rust program
//! using this crate does not need to ship or locate a separate `.dll`, `.so`,
//! or `.dylib` at runtime.

use std::error::Error;
use std::ffi::c_void;
use std::fmt;
use std::ptr::NonNull;

/// The sample rate supported by the wrapped 16 kHz Silero VAD C model.
pub const SAMPLE_RATE: usize = 16_000;

/// Number of previous samples prepended to each streaming chunk.
pub const DEFAULT_CONTEXT_SAMPLES: usize = 64;

/// Number of fresh samples consumed per normal streaming step.
pub const DEFAULT_CHUNK_SAMPLES: usize = 512;

/// Total input samples expected by the low-level C chunk function.
pub const DEFAULT_INPUT_SAMPLES: usize = DEFAULT_CONTEXT_SAMPLES + DEFAULT_CHUNK_SAMPLES;

/// A detected speech segment.
///
/// `start` and `end` are sample indices in the original 16 kHz audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechTimestamp {
    /// Inclusive start sample of the speech segment.
    pub start: usize,
    /// Exclusive end sample of the speech segment.
    pub end: usize,
}

/// Tuning options for [`get_timestamps_from_probs`].
#[derive(Debug, Clone, Copy)]
pub struct TimestampConfig {
    /// Probability at or above this value starts a speech segment.
    pub threshold: f32,
    /// Minimum speech duration kept in the output.
    pub min_speech_duration_ms: usize,
    /// Maximum speech segment duration before forcing a split.
    pub max_speech_duration_s: f32,
    /// Silence duration needed before ending a speech segment.
    pub min_silence_duration_ms: usize,
    /// Padding added around detected speech segments.
    pub speech_pad_ms: usize,
    /// Optional lower threshold used to confirm silence.
    pub neg_threshold: Option<f32>,
    /// Number of audio samples represented by each probability.
    pub window_size_samples: usize,
    /// Minimum silence used when splitting at maximum speech duration.
    pub min_silence_at_max_speech_ms: usize,
    /// Prefer the longest possible silence when splitting at maximum speech duration.
    pub use_max_possible_silence_at_max_speech: bool,
}

impl Default for TimestampConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_speech_duration_ms: 250,
            max_speech_duration_s: f32::INFINITY,
            min_silence_duration_ms: 100,
            speech_pad_ms: 30,
            neg_threshold: None,
            window_size_samples: DEFAULT_CHUNK_SAMPLES,
            min_silence_at_max_speech_ms: 98,
            use_max_possible_silence_at_max_speech: true,
        }
    }
}

/// Errors returned by the safe Rust wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SileroVadError {
    /// The native model could not be allocated or initialized.
    CreateFailed,
    /// A caller passed the wrong number of samples for the selected operation.
    InvalidInputLength {
        /// Length required by the operation.
        expected: usize,
        /// Length actually supplied by the caller.
        actual: usize,
    },
    /// A caller supplied a context size that is impossible for this model.
    InvalidContextLength {
        /// Context length supplied by the caller.
        context_samples: usize,
        /// Model input size configured at creation time.
        input_samples: usize,
    },
    /// The C library returned a non-OK status code.
    NativeStatus(SileroVadStatus),
}

impl fmt::Display for SileroVadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CreateFailed => write!(formatter, "failed to create the native Silero VAD model"),
            Self::InvalidInputLength { expected, actual } => {
                write!(formatter, "expected {expected} audio samples, got {actual}")
            }
            Self::InvalidContextLength {
                context_samples,
                input_samples,
            } => write!(
                formatter,
                "context length {context_samples} is invalid for model input length {input_samples}"
            ),
            Self::NativeStatus(status) => write!(formatter, "native Silero VAD error: {status:?}"),
        }
    }
}

impl Error for SileroVadError {}

/// Status values returned by the C API.
#[repr(i32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SileroVadStatus {
    /// The native call completed successfully.
    Ok = 0,
    /// A null pointer or invalid scalar argument was passed to C.
    InvalidArgument = 1,
    /// The C code could not allocate an internal buffer.
    AllocationFailed = 2,
    /// The supplied input shape does not match the model.
    InvalidShape = 3,
    /// A status value not known to this Rust wrapper.
    Unknown = -1,
}

impl SileroVadStatus {
    fn from_raw(value: i32) -> Self {
        match value {
            0 => Self::Ok,
            1 => Self::InvalidArgument,
            2 => Self::AllocationFailed,
            3 => Self::InvalidShape,
            _ => Self::Unknown,
        }
    }

    fn into_result(value: i32) -> Result<(), SileroVadError> {
        match Self::from_raw(value) {
            Self::Ok => Ok(()),
            status => Err(SileroVadError::NativeStatus(status)),
        }
    }
}

/// Owns one native Silero VAD model instance.
///
/// The model keeps recurrent state between chunk calls. Use [`reset`](Self::reset)
/// before starting a new stream or call [`forward_audio`](Self::forward_audio)
/// for whole-file inference, which resets internally in the C implementation.
pub struct SileroVad {
    model: NonNull<c_void>,
    input_samples: usize,
    context: Vec<f32>,
}

impl SileroVad {
    /// Create a model with the standard 576-sample input size.
    ///
    /// This uses the embedded weights compiled from `native/silero_vad_weights.c`.
    pub fn new() -> Result<Self, SileroVadError> {
        Self::with_input_samples(DEFAULT_INPUT_SAMPLES)
    }

    /// Create a model using a custom native input length.
    ///
    /// Most users should prefer [`new`](Self::new). The current C model is
    /// designed around 64 samples of context plus 512 fresh samples.
    pub fn with_input_samples(input_samples: usize) -> Result<Self, SileroVadError> {
        let weights = unsafe { ffi::silero_vad_get_embedded_weights() };
        let model = unsafe { ffi::silero_vad_model_create(weights, input_samples) };
        let model = NonNull::new(model).ok_or(SileroVadError::CreateFailed)?;

        Ok(Self {
            model,
            input_samples,
            context: Vec::new(),
        })
    }

    /// Return the native input size configured for this model.
    pub fn input_samples(&self) -> usize {
        self.input_samples
    }

    /// Reset the native recurrent state and the Rust-side rolling context.
    pub fn reset(&mut self) {
        unsafe {
            ffi::silero_vad_model_reset(self.model.as_ptr());
        }
        self.context.clear();
    }

    /// Run one low-level model step on a chunk that already includes context.
    ///
    /// The slice length must equal [`input_samples`](Self::input_samples),
    /// normally 576 samples. The returned value is the speech probability for
    /// the current 512-sample step.
    pub fn forward_chunk_with_context(&mut self, chunk: &[f32]) -> Result<f32, SileroVadError> {
        if chunk.len() != self.input_samples {
            return Err(SileroVadError::InvalidInputLength {
                expected: self.input_samples,
                actual: chunk.len(),
            });
        }

        let mut speech_probability = 0.0_f32;
        let status = unsafe {
            ffi::silero_vad_model_forward(
                self.model.as_ptr(),
                chunk.as_ptr(),
                &mut speech_probability,
            )
        };
        SileroVadStatus::into_result(status)?;
        Ok(speech_probability)
    }

    /// Run one streaming step on fresh audio and let Rust prepend context.
    ///
    /// With the default context length, pass 512 new samples. The wrapper
    /// prepends 64 samples of remembered context for the C model and stores the
    /// tail for the next call.
    pub fn forward_chunk(&mut self, chunk: &[f32]) -> Result<f32, SileroVadError> {
        self.forward_chunk_with_rolling_context(chunk, DEFAULT_CONTEXT_SAMPLES)
    }

    /// Run one streaming step with a caller-selected rolling context size.
    pub fn forward_chunk_with_rolling_context(
        &mut self,
        chunk: &[f32],
        context_samples: usize,
    ) -> Result<f32, SileroVadError> {
        if context_samples > self.input_samples {
            return Err(SileroVadError::InvalidContextLength {
                context_samples,
                input_samples: self.input_samples,
            });
        }

        let expected = self.input_samples - context_samples;
        if chunk.len() != expected {
            return Err(SileroVadError::InvalidInputLength {
                expected,
                actual: chunk.len(),
            });
        }

        if self.context.len() != context_samples {
            self.context = vec![0.0; context_samples];
        }

        let mut native_input = Vec::with_capacity(self.input_samples);
        native_input.extend_from_slice(&self.context);
        native_input.extend_from_slice(chunk);

        self.context.clear();
        self.context
            .extend_from_slice(&native_input[native_input.len() - context_samples..]);

        self.forward_chunk_with_context(&native_input)
    }

    /// Run full-audio inference and return one speech probability per step.
    ///
    /// The input audio must be 16 kHz mono `f32` samples. The C implementation
    /// handles chunking and resets model state at the start of the call.
    pub fn forward_audio(&mut self, audio: &[f32]) -> Result<Vec<f32>, SileroVadError> {
        let capacity = audio_probability_count(audio.len());
        let mut probabilities = vec![0.0_f32; capacity];
        let mut written = 0_usize;

        let status = unsafe {
            ffi::silero_vad_model_forward_audio(
                self.model.as_ptr(),
                audio.as_ptr(),
                audio.len(),
                probabilities.as_mut_ptr(),
                probabilities.len(),
                &mut written,
            )
        };
        SileroVadStatus::into_result(status)?;
        probabilities.truncate(written);
        self.context.clear();
        Ok(probabilities)
    }
}

impl Drop for SileroVad {
    fn drop(&mut self) {
        unsafe {
            ffi::silero_vad_model_destroy(self.model.as_ptr());
        }
    }
}

/// Return how many probability values the C model will produce for `audio_samples`.
pub fn audio_probability_count(audio_samples: usize) -> usize {
    unsafe { ffi::silero_vad_model_audio_prob_count(audio_samples) }
}

/// Convert frame-level speech probabilities into speech timestamps.
///
/// This ports the timestamp post-processing used by the Python Silero VAD
/// helper. The returned `start` and `end` values are sample indices in the
/// original audio. Use [`get_timestamps_from_probs_with_config`] when you need
/// custom thresholds, durations, or padding.
pub fn get_timestamps_from_probs(
    speech_probs: &[f32],
    audio_length_samples: usize,
) -> Vec<SpeechTimestamp> {
    get_timestamps_from_probs_with_config(
        speech_probs,
        audio_length_samples,
        TimestampConfig::default(),
    )
}

/// Convert frame-level speech probabilities into speech timestamps with custom options.
pub fn get_timestamps_from_probs_with_config(
    speech_probs: &[f32],
    audio_length_samples: usize,
    config: TimestampConfig,
) -> Vec<SpeechTimestamp> {
    if speech_probs.is_empty() || config.window_size_samples == 0 {
        return Vec::new();
    }

    let threshold = config.threshold;
    let neg_threshold = config
        .neg_threshold
        .unwrap_or_else(|| (threshold - 0.15).max(0.01));
    let min_speech_samples = SAMPLE_RATE as f64 * config.min_speech_duration_ms as f64 / 1000.0;
    let speech_pad_samples = SAMPLE_RATE as f64 * config.speech_pad_ms as f64 / 1000.0;
    let max_speech_samples = SAMPLE_RATE as f64 * config.max_speech_duration_s as f64
        - config.window_size_samples as f64
        - 2.0 * speech_pad_samples;
    let min_silence_samples = SAMPLE_RATE as f64 * config.min_silence_duration_ms as f64 / 1000.0;
    let min_silence_samples_at_max_speech =
        SAMPLE_RATE as f64 * config.min_silence_at_max_speech_ms as f64 / 1000.0;

    let mut triggered = false;
    let mut speeches = Vec::new();
    let mut current_speech: Option<SpeechTimestamp> = None;
    let mut temp_end = 0_usize;
    let mut prev_end = 0_usize;
    let mut next_start = 0_usize;
    let mut possible_ends: Vec<(usize, usize)> = Vec::new();

    for (index, speech_prob) in speech_probs.iter().copied().enumerate() {
        let cur_sample = config.window_size_samples * index;

        if speech_prob >= threshold && temp_end != 0 {
            let silence_duration = cur_sample.saturating_sub(temp_end);
            if silence_duration as f64 > min_silence_samples_at_max_speech {
                possible_ends.push((temp_end, silence_duration));
            }
            temp_end = 0;
            if next_start < prev_end {
                next_start = cur_sample;
            }
        }

        if speech_prob >= threshold && !triggered {
            triggered = true;
            current_speech = Some(SpeechTimestamp {
                start: cur_sample,
                end: 0,
            });
            continue;
        }

        if triggered
            && current_speech.is_some_and(|speech| {
                cur_sample.saturating_sub(speech.start) as f64 > max_speech_samples
            })
        {
            if config.use_max_possible_silence_at_max_speech && !possible_ends.is_empty() {
                let (best_end, best_duration) = possible_ends
                    .iter()
                    .copied()
                    .max_by_key(|(_, duration)| *duration)
                    .expect("possible_ends is not empty");
                prev_end = best_end;

                if let Some(mut speech) = current_speech.take() {
                    speech.end = prev_end;
                    speeches.push(speech);
                }

                next_start = prev_end + best_duration;
                if next_start < prev_end + cur_sample {
                    current_speech = Some(SpeechTimestamp {
                        start: next_start,
                        end: 0,
                    });
                } else {
                    triggered = false;
                }

                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
            } else if prev_end != 0 {
                if let Some(mut speech) = current_speech.take() {
                    speech.end = prev_end;
                    speeches.push(speech);
                }

                if next_start < prev_end {
                    triggered = false;
                } else {
                    current_speech = Some(SpeechTimestamp {
                        start: next_start,
                        end: 0,
                    });
                }

                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                possible_ends.clear();
            } else {
                if let Some(mut speech) = current_speech.take() {
                    speech.end = cur_sample;
                    speeches.push(speech);
                }

                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
                possible_ends.clear();
                continue;
            }
        }

        if speech_prob < neg_threshold && triggered {
            if temp_end == 0 {
                temp_end = cur_sample;
            }
            let silence_duration_now = cur_sample.saturating_sub(temp_end);

            if !config.use_max_possible_silence_at_max_speech
                && silence_duration_now as f64 > min_silence_samples_at_max_speech
            {
                prev_end = temp_end;
            }

            if (silence_duration_now as f64) < min_silence_samples {
                continue;
            }

            if let Some(mut speech) = current_speech.take() {
                speech.end = temp_end;
                if speech.end.saturating_sub(speech.start) as f64 > min_speech_samples {
                    speeches.push(speech);
                }
            }

            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
            possible_ends.clear();
        }
    }

    if let Some(mut speech) = current_speech {
        if audio_length_samples.saturating_sub(speech.start) as f64 > min_speech_samples {
            speech.end = audio_length_samples;
            speeches.push(speech);
        }
    }

    apply_speech_padding(&mut speeches, audio_length_samples, speech_pad_samples);
    speeches
}

fn apply_speech_padding(
    speeches: &mut [SpeechTimestamp],
    audio_length_samples: usize,
    speech_pad_samples: f64,
) {
    let speech_pad_samples = speech_pad_samples as usize;

    for index in 0..speeches.len() {
        if index == 0 {
            speeches[index].start = speeches[index].start.saturating_sub(speech_pad_samples);
        }

        if index != speeches.len() - 1 {
            let silence_duration = speeches[index + 1]
                .start
                .saturating_sub(speeches[index].end);

            if silence_duration < 2 * speech_pad_samples {
                let half_silence = silence_duration / 2;
                speeches[index].end += half_silence;
                speeches[index + 1].start = speeches[index + 1].start.saturating_sub(half_silence);
            } else {
                speeches[index].end =
                    audio_length_samples.min(speeches[index].end + speech_pad_samples);
                speeches[index + 1].start =
                    speeches[index + 1].start.saturating_sub(speech_pad_samples);
            }
        } else {
            speeches[index].end =
                audio_length_samples.min(speeches[index].end + speech_pad_samples);
        }
    }
}

mod ffi {
    use std::ffi::c_void;

    unsafe extern "C" {
        pub fn silero_vad_get_embedded_weights() -> *const c_void;

        pub fn silero_vad_model_create(weights: *const c_void, input_samples: usize)
            -> *mut c_void;

        pub fn silero_vad_model_reset(model: *mut c_void);

        pub fn silero_vad_model_destroy(model: *mut c_void);

        pub fn silero_vad_model_forward(
            model: *mut c_void,
            input: *const f32,
            speech_probability: *mut f32,
        ) -> i32;

        pub fn silero_vad_model_audio_prob_count(audio_samples: usize) -> usize;

        pub fn silero_vad_model_forward_audio(
            model: *mut c_void,
            audio: *const f32,
            audio_samples: usize,
            speech_probabilities: *mut f32,
            speech_probabilities_capacity: usize,
            speech_probabilities_written: *mut usize,
        ) -> i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_probability_count_for_empty_audio() {
        assert_eq!(audio_probability_count(0), 0);
    }

    #[test]
    fn can_run_silence_through_static_c_model() {
        let mut vad = SileroVad::new().unwrap();
        let probabilities = vad.forward_audio(&vec![0.0; SAMPLE_RATE]).unwrap();

        assert!(!probabilities.is_empty());
        assert!(probabilities
            .iter()
            .all(|probability| probability.is_finite()));
    }
}
