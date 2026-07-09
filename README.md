# silero-vad-crs

Silero voice activity detection (VAD) for Rust, built small and fast.

`silero-vad-crs` wraps [`silero-vad-c`](https://github.com/nipponjo/silero-vad-c),
a C port of the [Silero VAD](https://github.com/snakers4/silero-vad) model. It
embeds the model weights and compiles the native code with Cargo, giving you
full-audio and streaming VAD without ONNX Runtime, `ort`, or runtime
model downloads.

## Features

- wraps the `silero-vad-c` implementation
- **no ONNX** Runtime / `ort` dependency
- **embedded** model **weights**
- simple **full-audio** and **streaming** APIs
- full-audio and streaming **resampling** for non-16 kHz input
- optional **SIMD** build features: `auto`, `sse`, `avx2`, `neon`, `wasm-simd`

## Install

```toml
[dependencies]
silero-vad-crs = "0.4"
```

## Basic Use

```rust
use silero_vad_crs::{SileroVad, SAMPLE_RATE, get_timestamps_from_probs};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vad = SileroVad::new()?;

    // 16 kHz mono f32 audio samples.
    let audio = vec![0.0_f32; SAMPLE_RATE];
    let speech_probabilities = vad.forward_audio(&audio)?;
    let timestamps = get_timestamps_from_probs(&speech_probabilities, audio.len());

    println!("{timestamps:?}");
    Ok(())
}
```

For full-audio input at another sample rate, create the model with that rate:

```rust
use silero_vad_crs::{SileroVad, TimestampConfig, get_timestamps_from_probs_with_config};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vad = SileroVad::with_sample_rate(48_000)?;

    // 48 kHz mono f32 audio samples. The model resamples internally.
    let audio = vec![0.0_f32; 48_000];
    let speech_probabilities = vad.forward_audio(&audio)?;
    let timestamps = get_timestamps_from_probs_with_config(
        &speech_probabilities,
        audio.len(),
        TimestampConfig {
            sampling_rate: vad.sampling_rate(),
            window_size_samples: vad.source_window_samples(),
            ..Default::default()
        },
    );

    println!("{timestamps:?}");
    Ok(())
}
```

`get_timestamps_from_probs` returns sample-index timestamps with `start` and
`end` fields. Use `get_timestamps_from_probs_seconds` if you want `start` and
`end` in seconds. Use the `_with_config` variants to customize thresholds,
minimum durations, padding, and non-16 kHz timestamp conversion.

## Streaming Usage

For 16 kHz streaming, pass 512 new samples at a time:

```rust
use silero_vad_crs::{SileroVad, DEFAULT_CHUNK_SAMPLES};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vad = SileroVad::new()?;
    let chunk = vec![0.0_f32; DEFAULT_CHUNK_SAMPLES];

    let speech_probability = vad.forward_chunk(&chunk)?;
    println!("{speech_probability}");

    Ok(())
}
```

For streaming at another sample rate, use `source_window_samples()` as the
fresh chunk size:

```rust
use silero_vad_crs::SileroVad;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vad = SileroVad::with_sample_rate(48_000)?;
    let chunk = vec![0.0_f32; vad.source_window_samples()];

    let speech_probability = vad.forward_chunk(&chunk)?;
    println!("{speech_probability}");

    Ok(())
}
```

`forward_chunk` keeps the rolling 16 kHz model context for you and resamples
fresh chunks when needed. If your input already includes 16 kHz context, use
`forward_chunk_with_context`.

If incoming buffers are not exactly one VAD chunk long, use `push`:

```rust
use silero_vad_crs::SileroVad;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vad = SileroVad::with_sample_rate(48_000)?;

    let small_buffer = vec![0.0_f32; 500];
    let probabilities = vad.push(&small_buffer)?;

    // `probabilities` may be empty until enough samples have accumulated.
    println!("{probabilities:?}");

    Ok(())
}
```

`push` buffers incomplete chunks internally and returns one probability for each
complete chunk it can process. Short buffers may return no probabilities yet;
longer buffers may return multiple probabilities from one call.

## Input Format

The wrapped model expects:

- mono audio
- `f32` samples, normally in `[-1.0, 1.0]`

`SileroVad::new()` expects 16 kHz audio. `SileroVad::with_sample_rate(rate)`
accepts full-audio and streaming input at `rate` and resamples internally to
16 kHz.

When converting 16-bit PCM WAV samples yourself:

```rust
let sample_f32 = sample_i16 as f32 / 32768.0;
```

## CPU Features

The default build uses the portable scalar C path.

```toml
[dependencies]
silero-vad-crs = { version = "0.4", features = ["sse"] }
```

Use `auto` to enable SSE on supported x86/x86_64 targets or NEON on supported
ARM/AArch64 targets:

```toml
[dependencies]
silero-vad-crs = { version = "0.4", features = ["auto"] }
```

Use `avx2` only when the target CPU supports AVX2/FMA. Use `neon` for supported
ARM targets. Use `wasm-simd` for WebAssembly targets that support SIMD.

## Examples

Run the full-audio benchmark example:

```bash
cargo run --release --example full_audio_benchmark
```

Run the chunked streaming benchmark example:

```bash
cargo run --release --example chunked_benchmark
```

Both examples generate dummy 16 kHz audio, run 10 iterations, and print runtime
plus real-time factor. RTF is `processing_seconds / audio_seconds`, so values
below `1.0` are faster than real time.

## Notes

This crate is intended as a lightweight Rust wrapper around the C port, not a
Rust reimplementation of the neural network. For implementation details,
including the FFI boundary and native build flow, see
[`FFI_NOTES.md`](FFI_NOTES.md).
