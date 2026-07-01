use silero_vad_crs::{SileroVad, SAMPLE_RATE};
use std::error::Error;
use std::time::Instant;

const ITERATIONS: usize = 10;
const AUDIO_SECONDS: usize = 10;

fn main() -> Result<(), Box<dyn Error>> {
    let audio = make_dummy_audio(AUDIO_SECONDS);
    let audio_duration_seconds = audio.len() as f64 / SAMPLE_RATE as f64;
    let mut vad = SileroVad::new()?;

    println!("full-audio benchmark: {AUDIO_SECONDS}s dummy audio, {ITERATIONS} iterations");

    let mut total_seconds = 0.0;
    for iteration in 1..=ITERATIONS {
        let started = Instant::now();
        let probabilities = vad.forward_audio(&audio)?;
        let elapsed_seconds = started.elapsed().as_secs_f64();
        total_seconds += elapsed_seconds;

        print_iteration(iteration, elapsed_seconds, audio_duration_seconds);
        println!("  probabilities: {}", probabilities.len());
    }

    print_average(total_seconds / ITERATIONS as f64, audio_duration_seconds);
    Ok(())
}

fn make_dummy_audio(seconds: usize) -> Vec<f32> {
    let sample_count = seconds * SAMPLE_RATE;
    let frequency_hz = 220.0_f32;

    (0..sample_count)
        .map(|sample_index| {
            let time_seconds = sample_index as f32 / SAMPLE_RATE as f32;
            (time_seconds * frequency_hz * std::f32::consts::TAU).sin() * 0.1
        })
        .collect()
}

fn print_iteration(iteration: usize, elapsed_seconds: f64, audio_duration_seconds: f64) {
    println!(
        "iteration {iteration:02}: {:8.3} ms, rtf {:.6}",
        elapsed_seconds * 1000.0,
        elapsed_seconds / audio_duration_seconds
    );
}

fn print_average(mean_seconds: f64, audio_duration_seconds: f64) {
    println!(
        "average:      {:8.3} ms, rtf {:.6}",
        mean_seconds * 1000.0,
        mean_seconds / audio_duration_seconds
    );
}
