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
