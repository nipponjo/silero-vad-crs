Changelog
=========

0.3.0
-----
- Added full-audio and streaming support for non-16 kHz input by resampling
  internally to 16 kHz before VAD inference.
- Added sample-rate-aware probability counts and source window size helpers.
- Added `SileroVad::push` for streaming buffers that are not exactly one VAD
  chunk long.
- Added seconds-based timestamp helpers for callers that want speech segment
  times instead of sample indices.

0.2.0
-----
- Added probability-to-timestamp post-processing with `SpeechTimestamp` and
  `get_timestamps_from_probs`.

0.1.0
-----

Initial release.

- Added a safe Rust wrapper around the `silero-vad-c` 16 kHz voice activity
  detector.
- Added full-audio inference with `SileroVad::forward_audio`.
- Added streaming chunk inference with `SileroVad::forward_chunk` and
  `SileroVad::forward_chunk_with_context`.
- Embedded the generated model weights and vendored C source so the crate does
  not depend on ONNX Runtime, `ort`, Python, or external model files at runtime.
- Added optional native build features for `sse`, `avx2`, `neon`, and
  `fast-math`.
- Added reference-probability integration tests using a WAV fixture and CSV
  output from the C/Python reference path.
- Added full-audio and chunked benchmark examples that report runtime and
  real-time factor.
