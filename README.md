# silero-vad-crs

Rust bindings for [`silero-vad-c`](https://github.com/nipponjo/silero-vad-c), a
C port of the 16 kHz [Silero VAD](https://github.com/snakers4/silero-vad)
model.

This crate provides a small safe Rust API over the C implementation. The model
weights are embedded, and the native code is compiled by Cargo through a build
script.

## Features

- wraps the `silero-vad-c` implementation
- no ONNX Runtime / `ort` dependency
- embedded model weights
- simple full-audio and streaming APIs
- optional SIMD build features: `sse`, `avx2`, `neon`

## Install

```toml
[dependencies]
silero-vad-crs = "0.1"
```

For a local checkout:

```toml
[dependencies]
silero-vad-crs = { path = "../silero-vad-crs" }
```

## Basic Use

```rust
use silero_vad_crs::{SileroVad, SAMPLE_RATE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vad = SileroVad::new()?;

    // 16 kHz mono f32 audio samples.
    let audio = vec![0.0_f32; SAMPLE_RATE];
    let speech_probabilities = vad.forward_audio(&audio)?;

    println!("{speech_probabilities:?}");
    Ok(())
}
```

For streaming, pass 512 new samples at a time:

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

`forward_chunk` keeps the 64-sample rolling left context for you. If your input
already includes context, use `forward_chunk_with_context`.

## Input Format

The wrapped model expects:

- 16 kHz sample rate
- mono audio
- `f32` samples, normally in `[-1.0, 1.0]`

When converting 16-bit PCM WAV samples yourself:

```rust
let sample_f32 = sample_i16 as f32 / 32768.0;
```

## CPU Features

The default build uses the portable scalar C path.

```toml
[dependencies]
silero-vad-crs = { version = "0.1", features = ["sse"] }
```

Use `avx2` only when the target CPU supports AVX2/FMA. Use `neon` for supported
ARM targets.

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
