# silero-vad-crs

Rust wrapper around a vendored C implementation of the 16 kHz Silero VAD model.

This crate uses Rust FFI, but it does not link to a prebuilt `.dll`, `.so`, or
`.dylib`, and it does not run a Python script during `cargo build`. Instead,
`build.rs` compiles the vendored C source and the already generated C weight
file into a static native library.

## How to Build

From this folder:

```powershell
cargo test
```

What happens during the build:

1. Cargo runs `build.rs`.
2. The `cc` crate compiles:
   - `native/silero_vad.c`
   - `native/silero_vad_weights.c`
3. Cargo links that compiled C code into the Rust crate.

The final Rust program therefore contains the VAD implementation and weights in
its own binary. There is no runtime shared-library path to manage and no
dependency on a neighboring `silero-vad-c` checkout.

## Reference Test Fixtures

The integration test fixtures live in `tests/fixtures/`:

- `tests_data_test.wav`
- `ref_probs.csv`

`tests/reference_probs.rs` loads the WAV, runs `SileroVad::forward_audio`, and
compares every output probability against the CSV reference with a small float
tolerance.

## Vendored Native Files

The native implementation lives in `native/`:

- `silero_vad.c`
- `silero_vad.h`
- `silero_vad_simd.h`
- `silero_vad_weights.c`
- `silero_vad_weights.h`
- `SILERO_VAD_C_LICENSE`

`silero_vad_weights.c` is intentionally checked in. It contains the embedded
model weights as C arrays, so building this Rust crate does not need Python,
`safetensors`, `tinygrad`, or the original `silero-vad-c` folder.

## Optional CPU Features

The default build is the portable scalar C path:

```powershell
cargo build --release
```

You can enable the same SIMD compile switches exposed by the C project:

```powershell
cargo build --release --features sse
cargo build --release --features avx2
cargo build --release --features neon
```

Only enable CPU-specific features when the target machine supports them.

## Basic Use

```rust
use silero_vad_crs::{SileroVad, SAMPLE_RATE};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut vad = SileroVad::new()?;

    // 16 kHz mono f32 audio samples. This example is one second of silence.
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

`forward_chunk` remembers the previous 64 samples for you. If you already have a
576-sample buffer containing 64 context samples plus 512 fresh samples, use
`forward_chunk_with_context`.
