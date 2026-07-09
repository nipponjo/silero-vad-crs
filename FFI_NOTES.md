# FFI Notes

This file explains how `silero-vad-crs` connects Rust to the vendored C port.
The README is kept short for crates.io readers; this file is the place for build
details and the Rust FFI learning notes.

## Native Code Layout

The native implementation lives in `native/`:

- `silero_vad.c`
- `silero_vad.h`
- `silero_vad_simd.h`
- `silero_vad_weights.c`
- `silero_vad_weights.h`
- `SILERO_VAD_C_LICENSE`

`silero_vad_weights.c` contains the embedded model weights as C arrays. Because
the generated weights are checked in, normal Rust builds do not need Python,
`safetensors`, `tinygrad`, ONNX Runtime, or `ort`.

## What Happens During `cargo build`

Cargo runs `build.rs` before compiling the Rust library.

`build.rs` asks the `cc` crate to compile:

- `native/silero_vad.c`
- `native/silero_vad_weights.c`
- `native/dsp/resample.c`

The important line is:

```rust
build.compile("silero_vad_static");
```

That creates a static native library and tells Cargo to link it into the Rust
crate. A final Rust binary using this crate contains the C implementation and
weights in the binary itself, rather than loading a `.dll`, `.so`, or `.dylib`
at runtime.

## Optional Native Features

The crate exposes Cargo features that map to the C port's compile-time switches:

- `sse`
- `avx2`
- `neon`
- `wasm-simd`
- `auto`
- `fast-math`

The build script rejects `sse` and `avx2` together because they select different
x86 SIMD paths. The `auto` feature enables SSE on supported x86/x86_64 targets
or NEON on supported ARM/AArch64 targets. If `avx2` is explicitly enabled,
`auto` does not also enable the SSE backend.
The `wasm-simd` feature defines the C SIMD switch and passes
`-msimd128` when compiling for `wasm32`. Because this crate compiles C code that
uses standard C headers such as `math.h`, wasm builds also need a C toolchain
with a matching wasm sysroot, for example a WASI SDK setup. The default feature
set uses the portable scalar path.

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
    sampling_rate: usize,
    source_window_samples: usize,
    context: Vec<f32>,
}
```

That pointer is created by C:

```rust
let model = unsafe {
    ffi::silero_vad_model_create_with_sample_rate(weights, input_samples, sampling_rate)
};
```

Then Rust checks the pointer:

```rust
let model = NonNull::new(model).ok_or(SileroVadError::CreateFailed)?;
```

This is the core FFI pattern:

1. Keep raw C calls in a small private module.
2. Call them in small `unsafe` sections.
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

## Reference Test Fixtures

The integration test fixtures live in `tests/fixtures/`:

- `tests_data_test.wav`
- `ref_probs.csv`
- `ref_timestamps.csv`

`tests/reference_probs.rs` loads the WAV, runs `SileroVad::forward_audio`, and
compares every output probability and timestamp against the CSV references.
This checks that the Rust wrapper, native build, embedded weights, audio sample
conversion, and timestamp post-processing all still agree with the reference
outputs.
