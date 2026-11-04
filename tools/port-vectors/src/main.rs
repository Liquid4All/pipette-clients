//! Writes the Android prompt-seed test fixture from the Rust implementation.
//! Run from the repo root: the output path is relative.

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use pipette_ops::prompt_seed::{build_prompt_text, repeat_token_sequence};

#[derive(Serialize, Deserialize, Debug)]
struct PromptSeedTestCase {
    tokenizer: String,
    target_tokens: u32,
    expected_text: String,
    token_count: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct RepeatTokenTestCase {
    seed: Vec<u32>,
    target_len: usize,
    expected: Vec<u32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct PromptSeedVectors {
    prompt_seed_cases: Vec<PromptSeedTestCase>,
    repeat_token_cases: Vec<RepeatTokenTestCase>,
}

fn generate_prompt_seed_vectors() -> anyhow::Result<PromptSeedVectors> {
    let targets = [0u32, 1, 5, 17, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    let mut prompt_cases = Vec::new();

    for &target in &targets {
        let text = build_prompt_text(target, |s| Ok(s.len()))?;
        prompt_cases.push(PromptSeedTestCase {
            tokenizer: "byte_count".into(),
            target_tokens: target,
            expected_text: text.clone(),
            token_count: text.len(),
        });
    }

    for &target in &targets {
        let text = build_prompt_text(target, |s| Ok(s.chars().count()))?;
        prompt_cases.push(PromptSeedTestCase {
            tokenizer: "char_count".into(),
            target_tokens: target,
            expected_text: text.clone(),
            token_count: text.chars().count(),
        });
    }

    let repeat_token_cases = vec![
        RepeatTokenTestCase {
            seed: vec![1, 2, 3],
            target_len: 7,
            expected: repeat_token_sequence(&[1, 2, 3], 7)?,
        },
        RepeatTokenTestCase {
            seed: vec![1, 2],
            target_len: 0,
            expected: repeat_token_sequence(&[1, 2], 0)?,
        },
        RepeatTokenTestCase {
            seed: vec![1, 2, 3, 4, 5],
            target_len: 3,
            expected: repeat_token_sequence(&[1, 2, 3, 4, 5], 3)?,
        },
        RepeatTokenTestCase {
            seed: vec![42, 99, 101],
            target_len: 10,
            expected: repeat_token_sequence(&[42, 99, 101], 10)?,
        },
    ];

    Ok(PromptSeedVectors {
        prompt_seed_cases: prompt_cases,
        repeat_token_cases,
    })
}

fn main() -> anyhow::Result<()> {
    let prompt_seed_vectors = generate_prompt_seed_vectors()?;

    let out_dir = Path::new("android/Pipette/app/src/test/resources/fixtures");
    fs::create_dir_all(out_dir)?;

    let prompt_seed_path = out_dir.join("prompt_seed_vectors.json");
    fs::write(
        &prompt_seed_path,
        serde_json::to_string_pretty(&prompt_seed_vectors)?,
    )?;

    println!(
        "Wrote {} prompt seed cases + {} repeat token cases to {:?}",
        prompt_seed_vectors.prompt_seed_cases.len(),
        prompt_seed_vectors.repeat_token_cases.len(),
        prompt_seed_path
    );

    Ok(())
}
