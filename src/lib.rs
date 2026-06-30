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
        assert!(
            probabilities
                .iter()
                .all(|probability| probability.is_finite())
        );
    }
}
