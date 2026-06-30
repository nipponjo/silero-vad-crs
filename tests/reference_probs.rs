use silero_vad_crs::{SAMPLE_RATE, SileroVad};
use std::error::Error;
use std::fs;
use std::path::Path;

const TEST_AUDIO_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/tests_data_test.wav"
);
const REFERENCE_PROBS_PATH: &str =
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/ref_probs.csv");
const PROBABILITY_TOLERANCE: f32 = 1.0e-5;

#[test]
fn full_audio_probabilities_match_reference_csv() -> Result<(), Box<dyn Error>> {
    let audio = read_mono_16khz_pcm_wav(Path::new(TEST_AUDIO_PATH))?;
    let reference_probabilities = read_reference_probabilities(Path::new(REFERENCE_PROBS_PATH))?;

    let mut vad = SileroVad::new()?;
    let actual_probabilities = vad.forward_audio(&audio)?;

    assert_eq!(
        actual_probabilities.len(),
        reference_probabilities.len(),
        "probability count should match the reference CSV"
    );

    let (worst_index, worst_difference) = actual_probabilities
        .iter()
        .zip(reference_probabilities.iter())
        .enumerate()
        .map(|(index, (actual, expected))| (index, (actual - expected).abs()))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("the reference fixture should not be empty");

    assert!(
        worst_difference <= PROBABILITY_TOLERANCE,
        "largest probability difference was {worst_difference} at index {worst_index}; \
         actual={}, expected={}",
        actual_probabilities[worst_index],
        reference_probabilities[worst_index]
    );

    Ok(())
}

fn read_reference_probabilities(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let csv = fs::read_to_string(path)?;
    let probabilities = csv
        .split([',', '\n', '\r'])
        .filter_map(|field| {
            let field = field.trim();
            (!field.is_empty()).then_some(field)
        })
        .map(str::parse)
        .collect::<Result<Vec<f32>, _>>()?;

    Ok(probabilities)
}

fn read_mono_16khz_pcm_wav(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path)?;
    if bytes.get(0..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err("expected a RIFF/WAVE file".into());
    }

    let mut cursor = 12;
    let mut format: Option<WavFormat> = None;
    let mut data: Option<&[u8]> = None;

    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_size = read_u32_le(&bytes[cursor + 4..cursor + 8]) as usize;
        let chunk_start = cursor + 8;
        let chunk_end = chunk_start
            .checked_add(chunk_size)
            .ok_or("WAV chunk size overflowed")?;

        if chunk_end > bytes.len() {
            return Err("WAV chunk extends past the end of the file".into());
        }

        match chunk_id {
            b"fmt " => format = Some(parse_format_chunk(&bytes[chunk_start..chunk_end])?),
            b"data" => data = Some(&bytes[chunk_start..chunk_end]),
            _ => {}
        }

        cursor = chunk_end + (chunk_size % 2);
    }

    let format = format.ok_or("missing WAV fmt chunk")?;
    let data = data.ok_or("missing WAV data chunk")?;

    if format.audio_format != 1 {
        return Err(format!("expected PCM WAV format 1, got {}", format.audio_format).into());
    }
    if format.channels != 1 {
        return Err(format!("expected mono audio, got {} channels", format.channels).into());
    }
    if format.sample_rate != SAMPLE_RATE as u32 {
        return Err(format!(
            "expected {SAMPLE_RATE} Hz audio, got {} Hz",
            format.sample_rate
        )
        .into());
    }
    if format.bits_per_sample != 16 {
        return Err(format!(
            "expected 16-bit PCM audio, got {} bits per sample",
            format.bits_per_sample
        )
        .into());
    }

    if data.len() % 2 != 0 {
        return Err("16-bit PCM data chunk has an odd byte length".into());
    }

    let samples = data
        .chunks_exact(2)
        .map(|sample_bytes| i16::from_le_bytes([sample_bytes[0], sample_bytes[1]]) as f32 / 32768.0)
        .collect();

    Ok(samples)
}

fn parse_format_chunk(chunk: &[u8]) -> Result<WavFormat, Box<dyn Error>> {
    if chunk.len() < 16 {
        return Err("WAV fmt chunk is too short".into());
    }

    Ok(WavFormat {
        audio_format: read_u16_le(&chunk[0..2]),
        channels: read_u16_le(&chunk[2..4]),
        sample_rate: read_u32_le(&chunk[4..8]),
        bits_per_sample: read_u16_le(&chunk[14..16]),
    })
}

fn read_u16_le(bytes: &[u8]) -> u16 {
    u16::from_le_bytes([bytes[0], bytes[1]])
}

fn read_u32_le(bytes: &[u8]) -> u32 {
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

struct WavFormat {
    audio_format: u16,
    channels: u16,
    sample_rate: u32,
    bits_per_sample: u16,
}
