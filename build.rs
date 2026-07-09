use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let native_dir = manifest_dir.join("native");
    let c_source = native_dir.join("silero_vad.c");
    let weights_source = native_dir.join("silero_vad_weights.c");
    let resample_source = native_dir.join("dsp").join("resample.c");

    println!("cargo:rerun-if-changed={}", c_source.display());
    println!("cargo:rerun-if-changed={}", weights_source.display());
    println!("cargo:rerun-if-changed={}", resample_source.display());
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("silero_vad.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("silero_vad_simd.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("silero_vad_weights.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        native_dir.join("dsp").join("resample.h").display()
    );

    compile_static_c_library(&native_dir, &c_source, &weights_source, &resample_source);
}

fn compile_static_c_library(
    native_dir: &PathBuf,
    c_source: &PathBuf,
    weights_source: &PathBuf,
    resample_source: &PathBuf,
) {
    if cfg!(feature = "sse") && cfg!(feature = "avx2") {
        panic!("features `sse` and `avx2` cannot both be enabled");
    }

    let mut build = cc::Build::new();
    build
        .file(c_source)
        .file(weights_source)
        .file(resample_source)
        .include(native_dir)
        .warnings(true);

    if cfg!(feature = "sse") {
        build.define("SILERO_VAD_ENABLE_SSE", "1");
        if !is_msvc() && target_arch_is_32_bit_x86() {
            build.flag("-msse");
        }
    }

    if cfg!(feature = "avx2") {
        build.define("SILERO_VAD_ENABLE_AVX2", "1");
        if is_msvc() {
            build.flag("/arch:AVX2");
        } else {
            build.flag("-mavx2").flag("-mfma");
        }
    }

    if cfg!(feature = "neon") {
        build.define("SILERO_VAD_ENABLE_NEON", "1");
    }

    if cfg!(feature = "wasm-simd") {
        build.define("SILERO_VAD_ENABLE_WASM_SIMD", "1");
        if target_arch_is_wasm32() && !is_msvc() {
            build.flag("-msimd128");
        }
    }

    if cfg!(feature = "fast-math") {
        if is_msvc() {
            build.flag("/fp:fast");
        } else {
            build.flag("-ffast-math");
        }
    }

    if !is_msvc() && !target_arch_is_wasm32() {
        println!("cargo:rustc-link-lib=m");
    }

    build.compile("silero_vad_static");
}

fn is_msvc() -> bool {
    env::var("TARGET").is_ok_and(|target| target.contains("msvc"))
}

fn target_arch_is_32_bit_x86() -> bool {
    env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|arch| arch == "x86")
}

fn target_arch_is_wasm32() -> bool {
    env::var("CARGO_CFG_TARGET_ARCH").is_ok_and(|arch| arch == "wasm32")
}
