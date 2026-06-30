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

## What FFI Means

FFI means "foreign function interface." It is how Rust calls functions written
in another language.

The C header declares a function like this:

```c
SileroVadStatus silero_vad_model_forward_audio(
    SileroVadModel *model,
    const float *audio,
    size_t audio_samples,
    float *speech_probabilities,
    size_t speech_probabilities_capacity,
    size_t *speech_probabilities_written
);
```

The Rust FFI declaration mirrors the C signature:

```rust
unsafe extern "C" {
    fn silero_vad_model_forward_audio(
        model: *mut std::ffi::c_void,
        audio: *const f32,
        audio_samples: usize,
        speech_probabilities: *mut f32,
        speech_probabilities_capacity: usize,
        speech_probabilities_written: *mut usize,
    ) -> i32;
}
```

Important pieces:

- `extern "C"` tells Rust to use the C calling convention.
- Raw pointers such as `*const f32` and `*mut f32` match C pointers.
- The block is `unsafe` because Rust cannot prove the C function obeys Rust's
  memory and lifetime rules.
- The wrapper keeps these declarations private inside `mod ffi`.

## The Safe Wrapper Pattern

The public type is `SileroVad`.

It owns the native model pointer:

```rust
pub struct SileroVad {
    model: NonNull<c_void>,
    input_samples: usize,
    context: Vec<f32>,
}
```

That pointer is created by C:

```rust
let model = unsafe { ffi::silero_vad_model_create(weights, input_samples) };
```

Then Rust checks the pointer:

```rust
let model = NonNull::new(model).ok_or(SileroVadError::CreateFailed)?;
```

This is the core FFI pattern:

1. Keep raw C calls in a small private module.
2. Call them in tiny `unsafe` sections.
3. Immediately convert raw pointers and status codes into normal Rust types.
4. Expose safe methods like `forward_audio(&[f32]) -> Result<Vec<f32>, ...>`.
5. Use `Drop` to release the C object automatically.

`Drop` is the Rust version of "clean this up when the owner goes away":

```rust
impl Drop for SileroVad {
    fn drop(&mut self) {
        unsafe {
            ffi::silero_vad_model_destroy(self.model.as_ptr());
        }
    }
}
```

That means callers do not have to remember to manually call a C destroy
function.

## Why `build.rs` Exists

Cargo knows how to compile Rust. It does not automatically know how to compile
C files. A build script fills that gap.

This crate's `build.rs` asks the `cc` crate to compile the vendored C files into
a static library:

```rust
build.compile("silero_vad_static");
```

Cargo then links that static library into the Rust build.
