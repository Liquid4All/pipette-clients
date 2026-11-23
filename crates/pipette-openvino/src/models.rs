//! The IR directory of a bound OpenVINO model, resolved from the plan's
//! [`Model`] and checked to be complete.
//!
//! Install is `pipette_artifacts::ensure_model`.
//!
//! Nothing here decides whether a model may run on a device: a wrong
//! support-matrix guard produces no number rather than a bad one, and the
//! plan, not a release, decides which pairs to attempt. `docs/openvino-ir.md`
//! ("Why there are no preflight guards") has the measurements.

use std::ops::Deref;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

use pipette_ops::models::require_bound_model_dir;
use pipette_plan_types::{Model, ModelType};

/// Files `openvino_genai.LLMPipeline` needs. The tokenizer pair is the one
/// people miss: GenAI runs the tokenizer as a compiled model rather than
/// through the `tokenizers` library, so a directory without it loads and then
/// cannot generate.
const REQUIRED_FILES: &[&str] = &[
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
];

/// A bound `openvino` model's IR directory, holding the pipeline's required
/// files.
#[derive(Debug)]
pub struct ResolvedOpenvinoModel(PathBuf);

impl TryFrom<&Model> for ResolvedOpenvinoModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        // Flattened rather than `.context`, which would put the advice in front
        // of the missing filename, the part that identifies the problem.
        require_bound_model_dir(model, ModelType::Openvino, REQUIRED_FILES)
            .map(Self)
            .map_err(|e| {
                anyhow::anyhow!(
                    "{e}; an IR export needs the model and the tokenizer/detokenizer pair \
                     (re-export with openvino-tokenizers installed)"
                )
            })
    }
}

impl Deref for ResolvedOpenvinoModel {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

/// Files `openvino_genai.VLMPipeline` needs. It also reads the image processor's
/// settings, which [`image_processor`] requires: without them GenAI encodes with
/// its defaults and fails on the first image, after the compile.
const VL_REQUIRED_FILES: &[&str] = &[
    "openvino_language_model.xml",
    "openvino_language_model.bin",
    "openvino_text_embeddings_model.xml",
    "openvino_text_embeddings_model.bin",
    "openvino_vision_embeddings_model.xml",
    "openvino_vision_embeddings_model.bin",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
    "config.json",
];

/// LFM3-VL's chat template wraps a user turn in 10 tokens with thinking at its
/// default, as llama-server renders it: BOS, `<|im_start|>user\n`, `<|im_end|>\n`,
/// `<|im_start|>assistant\n` and `<think>`.
const LFM3_TURN_TOKENS: u32 = 10;

/// A bound `openvino` model's VL IR directory, holding the pipeline's required
/// files, with what an image costs where this client knows the model's rule.
#[derive(Debug)]
pub struct ResolvedOpenvinoVlModel {
    dir: PathBuf,
    images: Option<Lfm3Images>,
}

impl TryFrom<&Model> for ResolvedOpenvinoVlModel {
    type Error = anyhow::Error;

    fn try_from(model: &Model) -> anyhow::Result<Self> {
        let dir = require_bound_model_dir(model, ModelType::Openvino, VL_REQUIRED_FILES)?;
        let processor = image_processor(&dir)?;
        let config = read_json(&dir.join("config.json"))?
            .with_context(|| format!("{} has no config.json", dir.display()))?;
        let images = match config.get("model_type").and_then(Value::as_str) {
            Some("lfm3_vl") => Some(Lfm3Images::from_processor(&processor)?),
            _ => None,
        };
        Ok(Self { dir, images })
    }
}

impl ResolvedOpenvinoVlModel {
    /// The image tokens `count` images of `width` x `height` put in the prompt,
    /// without the start and end marker each gets: what llama.cpp reports.
    /// `None` for a model whose rule this client does not know.
    pub fn image_tokens(&self, width: u32, height: u32, count: u32) -> Option<u32> {
        Some(self.images?.tokens(width, height) * count)
    }

    /// The prompt GenAI builds for `text` tokens followed by `count` images.
    pub fn prompt_tokens(&self, text: u32, width: u32, height: u32, count: u32) -> Option<u32> {
        let images = self.image_tokens(width, height, count)?;
        Some(LFM3_TURN_TOKENS + text + images + 2 * count)
    }
}

impl Deref for ResolvedOpenvinoVlModel {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.dir
    }
}

/// The image processor's settings where GenAI looks for them: the
/// `image_processor` of `processor_config.json`, else `preprocessor_config.json`.
fn image_processor(dir: &Path) -> anyhow::Result<Value> {
    if let Some(image) = read_json(&dir.join("processor_config.json"))?
        .and_then(|mut processor| processor.get_mut("image_processor").map(Value::take))
    {
        return Ok(image);
    }
    read_json(&dir.join("preprocessor_config.json"))?.with_context(|| {
        format!(
            "{} has no image processor settings (processor_config.json or \
             preprocessor_config.json), so GenAI would encode images with its defaults",
            dir.display()
        )
    })
}

/// `None` for a file that does not exist.
fn read_json(path: &Path) -> anyhow::Result<Option<Value>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("reading {}", path.display())),
    };
    serde_json::from_slice(&bytes)
        .map(Some)
        .with_context(|| format!("{} is not JSON", path.display()))
}

/// LFM3-VL's image cost as GenAI computes it (`lfm3_vl/classes.cpp`): each side
/// rounded to whole merged patches, scaled into the pixel bounds, then a token
/// per merged patch.
#[derive(Debug, Clone, Copy)]
struct Lfm3Images {
    patch: u64,
    merge: u64,
    min_pixels: u64,
    max_pixels: u64,
}

impl Lfm3Images {
    /// Under GenAI's keys: the `size` edges stand in for both pixel bounds when
    /// neither is set.
    fn from_processor(processor: &Value) -> anyhow::Result<Self> {
        let number = |path: &[&str]| {
            path.iter()
                .try_fold(processor, |value, key| value.get(key))
                .and_then(Value::as_u64)
        };
        let required = |path: &[&str]| {
            number(path)
                .with_context(|| format!("the image processor settings have no {}", path.join(".")))
        };
        let (min, max): (&[&str], &[&str]) =
            if number(&["min_pixels"]).is_none() && number(&["max_pixels"]).is_none() {
                (&["size", "shortest_edge"], &["size", "longest_edge"])
            } else {
                (&["min_pixels"], &["max_pixels"])
            };
        let images = Self {
            patch: required(&["patch_size"])?,
            merge: required(&["merge_size"])?,
            min_pixels: required(min)?,
            max_pixels: required(max)?,
        };
        // GenAI's own check, which it runs only at the first image, after the compile.
        anyhow::ensure!(
            images.patch > 0
                && images.merge > 0
                && images.min_pixels > 0
                && images.max_pixels >= images.min_pixels,
            "the image processor settings need a positive patch_size, merge_size and \
             pixel floor, and a ceiling at least the floor: {images:?}"
        );
        Ok(images)
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "GenAI's own casts, of whole and positive pixel counts"
    )]
    fn tokens(self, width: u32, height: u32) -> u32 {
        let factor = self.patch * self.merge;
        let f = factor as f64;
        let (mut h, mut w) = (f64::from(height), f64::from(width));
        if h < f || w < f {
            let scale = (f / h).max(f / w);
            (h, w) = ((h * scale).trunc(), (w * scale).trunc());
        }
        let round = |side: f64| (side / f).round_ties_even() as u64 * factor;
        let (mut th, mut tw) = (round(h), round(w));
        if th * tw > self.max_pixels {
            let beta = (h * w / self.max_pixels as f64).sqrt();
            th = factor.max((h / beta / f).floor() as u64 * factor);
            tw = factor.max((w / beta / f).floor() as u64 * factor);
        } else if th * tw < self.min_pixels {
            let beta = (self.min_pixels as f64 / (h * w)).sqrt();
            th = (h * beta / f).ceil() as u64 * factor;
            tw = (w * beta / f).ceil() as u64 * factor;
        }
        ((th / self.patch) * (tw / self.patch) / (self.merge * self.merge)) as u32
    }
}

/// A VL IR directory as [`ResolvedOpenvinoVlModel`] reads one: the files, a
/// `config.json` naming `model_type`, and LFM3-VL's image settings.
#[cfg(test)]
pub(crate) fn write_vl_ir(dir: &Path, model_type: &str) -> anyhow::Result<()> {
    VL_REQUIRED_FILES
        .iter()
        .try_for_each(|file| std::fs::write(dir.join(file), b"ir"))?;
    let config = serde_json::json!({ "model_type": model_type });
    std::fs::write(dir.join("config.json"), config.to_string())?;
    let image = serde_json::json!({
        "patch_size": 16,
        "merge_size": 2,
        "size": { "shortest_edge": 65536, "longest_edge": 16777216 },
    });
    std::fs::write(dir.join("preprocessor_config.json"), image.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::{AbsolutePath, ModelSource, Openvino};

    use super::*;

    fn ir_model(model_dir: &Path) -> anyhow::Result<Model> {
        Ok(Model::Openvino(Openvino {
            source: ModelSource::AbsoluteDir {
                dir: AbsolutePath::try_new(model_dir.to_string_lossy().into_owned())?,
            },
        }))
    }

    /// The tokenizer pair is the omission people actually hit, so the error has
    /// to name the file rather than say the directory is incomplete.
    #[test]
    fn resolved_openvino_model_names_the_missing_tokenizer() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        std::fs::write(tmp.path().join("openvino_model.xml"), "<net/>")?;
        std::fs::write(tmp.path().join("openvino_model.bin"), [0u8])?;
        let Err(err) = ResolvedOpenvinoModel::try_from(&ir_model(tmp.path())?) else {
            anyhow::bail!("expected a missing-tokenizer rejection");
        };
        assert!(
            err.to_string().contains("openvino_tokenizer.xml"),
            "got {err}"
        );
        Ok(())
    }

    #[test]
    fn a_complete_directory_is_accepted() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        REQUIRED_FILES
            .iter()
            .try_for_each(|f| std::fs::write(tmp.path().join(f), [0u8]))?;
        let resolved = ResolvedOpenvinoModel::try_from(&ir_model(tmp.path())?)?;
        assert_eq!(resolved.to_path_buf(), tmp.path());
        Ok(())
    }

    fn vl_model(dir: &Path, model_type: &str) -> anyhow::Result<ResolvedOpenvinoVlModel> {
        write_vl_ir(dir, model_type)?;
        ResolvedOpenvinoVlModel::try_from(&ir_model(dir)?)
    }

    /// GenAI's own counts for one image on LFM3-VL (2026.5 nightly,
    /// `get_num_input_tokens` less the template and the markers): each branch
    /// of its resize under the pixel ceiling, with ties rounding to even, and
    /// the ladder's squares.
    #[rstest::rstest]
    #[case::photo(640, 480, 300)]
    #[case::under_the_pixel_floor(224, 224, 64)]
    #[case::odd_sides(300, 200, 70)]
    #[case::tie_then_floor(48, 48, 64)]
    #[case::tie_down_to_even(528, 512, 256)]
    #[case::tie_up_to_even(560, 512, 288)]
    #[case::side_under_a_merged_patch(20, 40, 72)]
    #[case::narrow(16, 600, 98)]
    #[case::short(1000, 33, 90)]
    #[case::ladder_256(256, 256, 64)]
    #[case::ladder_768(768, 768, 576)]
    #[case::ladder_1280(1280, 1280, 1600)]
    #[case::ladder_2048(2048, 2048, 4096)]
    fn an_lfm3_image_costs_what_genai_counts(
        #[case] width: u32,
        #[case] height: u32,
        #[case] tokens: u32,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let model = vl_model(dir.path(), "lfm3_vl")?;
        assert_eq!(model.image_tokens(width, height, 1), Some(tokens));
        Ok(())
    }

    /// Ladder prompts as GenAI counted them, three of them also as llama.cpp
    /// b11988 did on the same model's GGUF (1300, 1364 and 1612).
    #[rstest::rstest]
    #[case(0, 512, 1, 268)]
    #[case(0, 512, 5, 1300)]
    #[case(1024, 256, 5, 1364)]
    #[case(0, 1280, 1, 1612)]
    #[case(1024, 512, 16, 5162)]
    fn an_lfm3_prompt_counts_as_genai_and_llama_cpp_do(
        #[case] text: u32,
        #[case] side: u32,
        #[case] images: u32,
        #[case] prompt: u32,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let model = vl_model(dir.path(), "lfm3_vl")?;
        assert_eq!(model.prompt_tokens(text, side, side, images), Some(prompt));
        Ok(())
    }

    #[test]
    fn another_model_has_no_image_rule() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let model = vl_model(dir.path(), "qwen2_vl")?;
        assert_eq!(model.prompt_tokens(0, 512, 512, 1), None);
        Ok(())
    }

    /// GenAI reads `processor_config.json`'s settings before
    /// `preprocessor_config.json`'s, and pixel bounds before the `size` edges.
    #[test]
    fn the_image_settings_come_from_where_genai_reads_them() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        write_vl_ir(dir.path(), "lfm3_vl")?;
        let processor = serde_json::json!({ "image_processor": {
            "patch_size": 16, "merge_size": 2, "min_pixels": 262144, "max_pixels": 16777216,
            "size": { "shortest_edge": 65536, "longest_edge": 16777216 },
        }});
        std::fs::write(
            dir.path().join("processor_config.json"),
            processor.to_string(),
        )?;
        let model = ResolvedOpenvinoVlModel::try_from(&ir_model(dir.path())?)?;
        // Scaled up to the 512x512 floor; the `size` edges or
        // `preprocessor_config.json` would leave it at 64.
        assert_eq!(model.image_tokens(256, 256, 1), Some(256));
        Ok(())
    }

    /// GenAI refuses these only at the first image; a zero would also divide
    /// by zero here.
    #[test]
    fn image_settings_genai_refuses_are_refused() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        write_vl_ir(dir.path(), "lfm3_vl")?;
        let image = serde_json::json!({
            "patch_size": 16,
            "merge_size": 0,
            "size": { "shortest_edge": 65536, "longest_edge": 16777216 },
        });
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            image.to_string(),
        )?;
        let Err(err) = ResolvedOpenvinoVlModel::try_from(&ir_model(dir.path())?) else {
            anyhow::bail!("expected a zero merge_size to be refused");
        };
        assert!(err.to_string().contains("merge_size"), "{err:#}");
        Ok(())
    }

    /// GenAI's counts with the IR's pixel ceiling lowered to 262144, so these
    /// sizes scale down to it: the branch no benchmark image reaches.
    #[rstest::rstest]
    #[case(640, 480, 234)]
    #[case(1000, 1000, 256)]
    #[case(700, 300, 198)]
    fn an_lfm3_image_over_the_ceiling_scales_down_as_genai_does(
        #[case] width: u32,
        #[case] height: u32,
        #[case] tokens: u32,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        write_vl_ir(dir.path(), "lfm3_vl")?;
        let image = serde_json::json!({
            "patch_size": 16,
            "merge_size": 2,
            "size": { "shortest_edge": 65536, "longest_edge": 262144 },
        });
        std::fs::write(
            dir.path().join("preprocessor_config.json"),
            image.to_string(),
        )?;
        let model = ResolvedOpenvinoVlModel::try_from(&ir_model(dir.path())?)?;
        assert_eq!(model.image_tokens(width, height, 1), Some(tokens));
        Ok(())
    }

    /// Without them GenAI would encode with its defaults and fail on the first
    /// image, after the compile.
    #[test]
    fn a_vl_ir_without_image_settings_is_refused() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        write_vl_ir(dir.path(), "lfm3_vl")?;
        std::fs::remove_file(dir.path().join("preprocessor_config.json"))?;
        let Err(err) = ResolvedOpenvinoVlModel::try_from(&ir_model(dir.path())?) else {
            anyhow::bail!("expected a VL IR without image settings to be refused");
        };
        assert!(err.to_string().contains("image processor"), "{err:#}");
        Ok(())
    }

    /// A text IR has no language model of a VL pipeline's.
    #[test]
    fn a_text_ir_is_not_a_vl_ir() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        REQUIRED_FILES
            .iter()
            .try_for_each(|f| std::fs::write(tmp.path().join(f), [0u8]))?;
        let Err(err) = ResolvedOpenvinoVlModel::try_from(&ir_model(tmp.path())?) else {
            anyhow::bail!("expected a text IR to be refused");
        };
        assert!(
            err.to_string().contains("openvino_language_model.xml"),
            "{err:#}"
        );
        Ok(())
    }
}
