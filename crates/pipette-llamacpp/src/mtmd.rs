//! VL image-token calibration for the llama-server path, to size `n_ctx` and
//! report `image_tokens`.
//!
//! The tiling-aware count for a `(mmproj, resolution)` is not a formula, so it
//! comes from upstream's `llama-mtmd-cli`, which ships version-matched next to
//! `llama-server`. Run with `-v` on a solid `W×H` image, it logs
//! `decoding image batch …, n_tokens_batch = N`; the sum is memoized in the
//! per-build runtime dir, as the count does not depend on the device.
//!
//! Best-effort: any failure returns `None`, and the caller falls back to the
//! `(w/14)²` over-estimate and reports no `image_tokens`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use pipette_plan_types::{
    ContentPart, EnvForward, EvalSample, LlamacppCliStockToolsExtraRuntimeConfiguration,
    MessageContent,
};
use pipette_subprocess::start_process;

use crate::runtimes::{
    CommonKnobs, ResolvedLlamaServer, canonical_flag_name, canonicalize_flag_order,
};

/// Headroom in tokens for the chat-template scaffolding around the prompt: the
/// calibration counts the raw image content only.
const CTX_MARGIN: u32 = 64;

/// Per-image headroom for the template's media markers. Measured: the LFM
/// template adds ~2.5 tokens per image, ~200 past a flat 64 at 80 frames. 8 is
/// generous across the template families benchmarked.
const CTX_MARGIN_PER_IMAGE: u32 = 8;

/// Bound on one `llama-mtmd-cli` calibration run (a full model+mmproj load, the
/// image encode and a prefill of every image token). Generous so a slow
/// big-model load on a weak device is not a false timeout — that would drop the
/// run to the looser fallback numbers.
pub(crate) const VL_CALIBRATION_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(600);

/// The least context a one-frame calibration runs with. The frame's estimate
/// runs close to a tiled count (InternVL3.5 tiles 512² into 1280 tokens against
/// an estimate of 1296), so the floor keeps room for a tiler it misses, and
/// still cuts the helper's peak 2-3x against llama.cpp's default, the model's
/// training context (Qwen3.5-2B: 6.5 to 3.4 GB).
pub(crate) const HELPER_CTX_FLOOR: u32 = 4096;

/// The most a frame's size adds to the one-frame helper's context: `(w/14)²`
/// grows without bound, while the projectors in the vendored llama.cpp cap a
/// frame at about 4096 tokens by default.
const HELPER_PER_IMAGE_CAP: u32 = 16384;

/// Upper bound on `parameter_num_images`, an unrestricted u32: each frame is a
/// cloned data-URL part in the request, so an absurd value would abort on
/// allocation. 4096 is ~50x the largest fleet workload (80).
const MAX_NUM_IMAGES: u32 = 4096;

/// The VL workload shape `n_ctx` has to cover.
#[derive(Debug, Clone, Copy)]
pub struct CtxInputs {
    pub width: u32,
    pub height: u32,
    pub text_tokens: u32,
    pub decode_tokens: u32,
    /// At least one.
    pub num_images: u32,
}

impl CtxInputs {
    pub fn new(
        width: u32,
        height: u32,
        text_tokens: u32,
        decode_tokens: u32,
        num_images: u32,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            num_images <= MAX_NUM_IMAGES,
            "parameter_num_images {num_images} exceeds the supported maximum {MAX_NUM_IMAGES}"
        );
        Ok(Self {
            width,
            height,
            text_tokens,
            decode_tokens,
            num_images: num_images.max(1),
        })
    }

    /// `n_ctx` for the workload: the calibrated per-image count, else the
    /// estimate, times the frames, plus text, decode and the same margins either
    /// way. `fallback_floor` applies to the estimate alone. The count is a
    /// parameter so a runner calibrates once, before the server holds the model.
    pub fn ctx_size(&self, calibrated: Option<u32>, raw: &[String], fallback_floor: u32) -> u32 {
        let (per_image, floor) = match calibrated {
            Some(per_image) => (per_image, 0),
            None => {
                let count_flags = raw_flags(raw, &COUNT_FLAGS, &[]);
                (
                    fallback_per_image(self.width, self.height)
                        .max(min_tokens_override(&count_flags)),
                    fallback_floor,
                )
            }
        };
        per_image
            .saturating_mul(self.num_images)
            .saturating_add(self.text_tokens)
            .saturating_add(self.decode_tokens)
            .saturating_add(CTX_MARGIN)
            .saturating_add(CTX_MARGIN_PER_IMAGE.saturating_mul(self.num_images))
            .max(floor)
    }
}

/// One `llama-mtmd-cli` calibration run, placed like the server it calibrates
/// for: the helper loads the same model, so it takes the cell's GPU layers,
/// offload flags and envs. On a shared box that load belongs on the run's card,
/// and on the Snapdragon build llama.cpp's default placement gets the helper
/// killed.
#[derive(Debug, Clone)]
pub struct MtmdCommandBuilder {
    llama_server: ResolvedLlamaServer,
    model: PathBuf,
    mmproj: PathBuf,
    number_gpu_layers: Option<u32>,
    raw: Vec<String>,
    ctx_size: Option<u32>,
    envs: Vec<EnvForward>,
    client_config: Option<LlamacppCliStockToolsExtraRuntimeConfiguration>,
    without_inherited_llama_args: bool,
}

impl MtmdCommandBuilder {
    /// `llama_server` locates the per-build runtime dir, which holds both the
    /// cache and the `llama-mtmd-cli` helper, so the cache is build-scoped.
    pub fn new(
        llama_server: ResolvedLlamaServer,
        model: impl Into<PathBuf>,
        mmproj: impl Into<PathBuf>,
    ) -> Self {
        Self {
            llama_server,
            model: model.into(),
            mmproj: mmproj.into(),
            number_gpu_layers: None,
            raw: Vec::new(),
            ctx_size: None,
            envs: Vec::new(),
            client_config: None,
            without_inherited_llama_args: false,
        }
    }

    /// No llama.cpp option variable inherited from pipette reaches the helper,
    /// as none reaches the vlm_eval server it calibrates for.
    pub fn without_inherited_llama_args(mut self) -> Self {
        self.without_inherited_llama_args = true;
        self
    }

    pub fn number_gpu_layers(mut self, number_gpu_layers: Option<u32>) -> Self {
        self.number_gpu_layers = number_gpu_layers;
        self
    }

    /// Only the [`COUNT_FLAGS`] and the placement flags reach the helper.
    pub fn raw(mut self, raw: Vec<String>) -> Self {
        self.raw = raw;
        self
    }

    /// The helper's `--ctx-size`. `None` leaves [`calibrate_frames`] on
    /// llama.cpp's default and [`calibrate`] on [`helper_ctx_size`]. Not in the
    /// cache key: a context that holds the frames does not change their count.
    pub fn ctx_size(mut self, ctx_size: Option<u32>) -> Self {
        self.ctx_size = ctx_size;
        self
    }

    pub fn envs(mut self, envs: Vec<EnvForward>) -> Self {
        self.envs = envs;
        self
    }

    pub fn client_config(
        mut self,
        client_config: Option<LlamacppCliStockToolsExtraRuntimeConfiguration>,
    ) -> Self {
        self.client_config = client_config;
        self
    }

    pub(crate) fn helper(&self) -> PathBuf {
        self.llama_server.with_file_name(HELPER_NAME)
    }

    /// The per-build runtime dir, which holds the helper and its caches.
    pub(crate) fn runtime_dir(&self) -> Option<&Path> {
        self.llama_server.parent()
    }

    pub(crate) fn mmproj(&self) -> &Path {
        &self.mmproj
    }

    pub(crate) fn forwarded_envs(&self) -> &[EnvForward] {
        &self.envs
    }

    /// The [`COUNT_FLAGS`] the cell sets, which key the cache.
    fn count_flags(&self) -> Vec<String> {
        raw_flags(&self.raw, &COUNT_FLAGS, &[])
    }

    /// The cell's placement, image-token overrides, batch sizes and the context,
    /// in the server's order, so a flag the cell repeats resolves the same way.
    /// The overrides and batch sizes make the helper count what the server will
    /// prefill (b10142: LFM 512² measures 64 under `--image-max-tokens 64`, 256
    /// without). A helper too old for these flags exits non-zero and falls back.
    pub fn args(&self) -> Vec<String> {
        self.args_with(self.count_flags())
    }

    /// [`Self::args`] without the image-token overrides and batch sizes, for an
    /// audio count: the overrides are image-only, and the batch sizes only split
    /// the audio batches the count sums.
    pub fn placement_args(&self) -> Vec<String> {
        self.args_with(Vec::new())
    }

    fn args_with(&self, count_flags: Vec<String>) -> Vec<String> {
        let placement = CommonKnobs {
            number_gpu_layers: self.number_gpu_layers,
            ..CommonKnobs::default()
        }
        .scalar_tokens()
        .into_iter()
        .chain(raw_flags(&self.raw, &PLACEMENT_FLAGS, &PLACEMENT_SWITCHES));
        let ctx_size = self
            .ctx_size
            .into_iter()
            .flat_map(|n| ["--ctx-size".to_owned(), n.to_string()]);
        canonicalize_flag_order(
            &placement
                .chain(count_flags)
                .chain(ctx_size)
                .collect::<Vec<_>>(),
        )
    }

    /// The helper's [`Command`] for one run on `frames` copies of `image`.
    pub fn build(&self, image: &Path, frames: u32) -> Command {
        let mut command = self.command(self.args());
        for _ in 0..frames.max(1) {
            command.arg("--image").arg(image);
        }
        command
    }

    /// The helper's [`Command`] for one run on the WAV file `clip`.
    pub fn build_audio(&self, clip: &Path) -> Command {
        let mut command = self.command(self.placement_args());
        command.arg("--audio").arg(clip);
        command
    }

    fn command(&self, args: Vec<String>) -> Command {
        let mut command = Command::new(self.helper());
        if self.without_inherited_llama_args {
            crate::thinking_controls::remove_inherited_llama_args(&mut command);
        }
        crate::runtimes::apply_launch_env(
            &mut command,
            &self.llama_server,
            &self.envs,
            self.client_config.as_ref(),
        );
        command
            .arg("--model")
            .arg(&self.model)
            .arg("--mmproj")
            .arg(&self.mmproj)
            .arg("--prompt")
            .arg("hi")
            .arg("--n-predict")
            .arg("1")
            .args(args)
            // Use the model's embedded jinja chat template. Some models (e.g.
            // gemma-4) ship a template llama-mtmd-cli's default path rejects with
            // "this custom template is not supported, try using --jinja", aborting
            // before the image is processed; --jinja fixes those and is a no-op for
            // the others (verified: LFM/Qwen/InternVL counts unchanged).
            .arg("--jinja")
            // The image-batch lines are in the verbose log, on stderr.
            .arg("--verbose");
        command
    }
}

/// The per-image token count, from the cache or else the helper. A `launch`
/// without a context gets [`helper_ctx_size`] rather than the
/// training context llama.cpp would allocate: about 5 GB of peak against 2 GB
/// on a phone.
pub fn calibrate(inputs: &CtxInputs, launch: &MtmdCommandBuilder) -> Option<u32> {
    let one = CtxInputs {
        num_images: 1,
        ..*inputs
    };
    let ctx_size = launch
        .ctx_size
        .unwrap_or_else(|| helper_ctx_size(inputs.width, inputs.height, &launch.raw));
    calibrate_frames(&one, &launch.clone().ctx_size(Some(ctx_size)))
}

/// The one-frame helper's context: the frame's estimate, at most
/// [`HELPER_PER_IMAGE_CAP`], or the cell's `--image-min-tokens` if higher, plus
/// the margins, at least [`HELPER_CTX_FLOOR`]. The helper's prompt is `hi` and
/// the frame, not the cell's text.
fn helper_ctx_size(width: u32, height: u32, raw: &[String]) -> u32 {
    fallback_per_image(width, height)
        .min(HELPER_PER_IMAGE_CAP)
        .max(min_tokens_override(&raw_flags(raw, &COUNT_FLAGS, &[])))
        .saturating_add(CTX_MARGIN)
        .saturating_add(CTX_MARGIN_PER_IMAGE)
        .max(HELPER_CTX_FLOOR)
}

/// [`calibrate`] over all `inputs.num_images` frames in one helper run, since
/// some llama.cpp builds merge frames.
pub fn calibrate_frames(inputs: &CtxInputs, launch: &MtmdCommandBuilder) -> Option<u32> {
    // The cache key leaves out the placement and the rest of `client_config`:
    // the count depends on the cell (its `GGML_*` knobs), not on the device or
    // host, so two clients pinned to different cards share one memoised count.
    let fp = mmproj_fingerprint(&launch.mmproj)?;
    let runtime_dir = launch.llama_server.parent()?;
    let cache = runtime_dir.join("vl_calibration.json");
    let key = cache_key(
        &fp,
        inputs.width,
        inputs.height,
        &launch.count_flags(),
        &launch.envs,
    );
    let key = if inputs.num_images > 1 {
        format!("{key}|frames:{}", inputs.num_images)
    } else {
        key
    };
    // The count option variables in the helper's own environment key it,
    // whichever source sets them: pipette's (unless the helper drops it), the
    // client configuration or the cell.
    let key = match count_envs(|name| helper_env(launch, name)) {
        Some(set) => format!("{key}|count-env:{set}"),
        None => key,
    };

    if let Some(v) = cache_lookup(&cache, &key) {
        return Some(v);
    }
    if !launch.helper().exists() {
        return None;
    }
    let v = run_helper(
        launch,
        inputs.width,
        inputs.height,
        inputs.num_images,
        VL_CALIBRATION_TIMEOUT,
    )?;
    cache_store(&cache, &key, v);
    Some(v)
}

/// The `raw` flags that change how many tokens an image becomes: the image-token
/// levers, and the batch sizes, whose smaller caps the image tokens of a
/// projector the text model attends to non-causally. The helper runs under them
/// and the cache keys by them, or it would count what the server does not prefill.
const COUNT_FLAGS: [&str; 6] = [
    "--image-min-tokens",
    "--image-max-tokens",
    "-ub",
    "--ubatch-size",
    "-b",
    "--batch-size",
];

/// llama.cpp's option variables for the [`COUNT_FLAGS`]: set in pipette's own
/// environment, they reach a helper that inherits it and change its count.
const COUNT_ENVS: [&str; 4] = [
    "LLAMA_ARG_IMAGE_MIN_TOKENS",
    "LLAMA_ARG_IMAGE_MAX_TOKENS",
    "LLAMA_ARG_BATCH",
    "LLAMA_ARG_UBATCH",
];

/// The [`COUNT_ENVS`] set in a helper's environment, as `NAME=value`, for the
/// cache key; `None` when none is, so a clean environment keeps the original
/// key and its cache entries.
fn count_envs(lookup: impl Fn(&str) -> Option<String>) -> Option<String> {
    let set: Vec<String> = COUNT_ENVS
        .iter()
        .filter_map(|name| lookup(name).map(|value| format!("{name}={value}")))
        .collect();
    (!set.is_empty()).then(|| set.join(" "))
}

/// The value `name` takes in the environment `launch`'s helper runs with, read
/// from the command [`MtmdCommandBuilder::build`] makes, so the precedence is
/// the launch's own: a variable the command sets or drops, else pipette's.
fn helper_env(launch: &MtmdCommandBuilder, name: &str) -> Option<String> {
    let command = launch.build(Path::new(""), 1);
    let same = |key: &std::ffi::OsStr| {
        if cfg!(windows) {
            key.eq_ignore_ascii_case(name)
        } else {
            key == name
        }
    };
    let value = match command.get_envs().find(|(key, _)| same(key)) {
        Some((_, value)) => value.map(std::ffi::OsStr::to_owned),
        None => std::env::var_os(name),
    };
    value.map(|value| value.to_string_lossy().into_owned())
}

/// The `raw` flags that place the model, its projector and its KV cache. They
/// reach the helper but not its cache key: the count does not depend on the
/// device. A pinned `--n-gpu-layers` turns llama.cpp's fit off, so the helper
/// needs every offload flag the server's fit-free load relies on.
const PLACEMENT_FLAGS: [&str; 14] = [
    "--device",
    "-dev",
    "--mmproj-device",
    "-mmdev",
    "--override-tensor",
    "-ot",
    "--n-cpu-moe",
    "-ncmoe",
    "--tensor-split",
    "-ts",
    "--main-gpu",
    "-mg",
    "--split-mode",
    "-sm",
];

/// The valueless `raw` switches of [`PLACEMENT_FLAGS`]' kind.
const PLACEMENT_SWITCHES: [&str; 7] = [
    "--mmproj-offload",
    "--no-mmproj-offload",
    "--cpu-moe",
    "-cmoe",
    "--no-kv-offload",
    "-nkvo",
    "--no-op-offload",
];

/// The `names` flags (`--flag value` or `--flag=value`) and the `switches` in a
/// cell's `raw`, in order, each flag split into its name and value; every other
/// raw flag is ignored. A long name matches with `_` for `-`, as llama.cpp
/// parses it, and comes out with `-`.
fn raw_flags(raw: &[String], names: &[&str], switches: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    let mut tokens = raw.iter();
    while let Some(token) = tokens.next() {
        let (name, value) = match token.split_once('=') {
            Some((name, value)) => (canonical_flag_name(name), Some(value)),
            None => (canonical_flag_name(token), None),
        };
        if value.is_none() && switches.contains(&name.as_str()) {
            out.push(name);
        } else if names.contains(&name.as_str())
            && let Some(value) = value.map(str::to_owned).or_else(|| tokens.next().cloned())
        {
            out.push(name);
            out.push(value);
        }
    }
    out
}

/// A vlm_eval calibration's helper context: room for one image at its fallback
/// bound, capped by the cell's `--image-max-tokens` and raised to its
/// `--image-min-tokens`, plus the prompt, between [`HELPER_CTX_FLOOR`] and the
/// vlm_eval ceiling.
///
/// The cap matters for a budgeted projector on a huge page: Gemma's 1120-token
/// budget on a 4961×7016 OCRBench scan needs about 1.3k tokens, where the
/// uncapped grid asks for the 131072 ceiling. A projector that ignores the
/// budget overflows the smaller context, and the helper's non-zero exit falls
/// back rather than caching a count.
///
/// Not [`helper_ctx_size`]: its [`HELPER_PER_IMAGE_CAP`] holds for the VL
/// cells' synthetic frames, but a dataset's own extreme shapes tile past it.
/// SmolVLM2 (8192) tiles a 35-megapixel OCRBench page into ~16k tokens; under
/// the cap the count fails, and the fallback estimate sizes vlm_eval past its
/// ceiling.
fn helper_ctx(width: u32, height: u32, raw: &[String]) -> u32 {
    let count_flags = raw_flags(raw, &COUNT_FLAGS, &[]);
    helper_ctx_for(
        max_tokens_override(&count_flags).unwrap_or(u32::MAX),
        width,
        height,
        &count_flags,
    )
}

/// [`helper_ctx`] without the `--image-max-tokens` cap: the context a
/// calibration retries at when the capped one fails, for a projector that
/// ignores the budget.
fn helper_ctx_uncapped(width: u32, height: u32, raw: &[String]) -> u32 {
    helper_ctx_for(u32::MAX, width, height, &raw_flags(raw, &COUNT_FLAGS, &[]))
}

fn helper_ctx_for(cap: u32, width: u32, height: u32, count_flags: &[String]) -> u32 {
    fallback_per_image(width, height)
        .min(cap)
        .max(min_tokens_override(count_flags))
        .saturating_add(1024)
        .clamp(HELPER_CTX_FLOOR, VLM_EVAL_CTX_CEILING)
}

/// The per-image estimate without calibration: the `(w/14)²` patch grid,
/// floored at the flat count some encoders emit at any size. Per
/// `docs/methodology/vl-image-tokens.md`, InternVL is one 256-token tile below
/// 448px, where the grid gives 81 at 128², and gemma-4 a flat 256 on `b8814`. The floor
/// over-sizes LFM (64 at its smallest) by up to 4x and Qwen (49 at 224², fewer below)
/// by more, only where the helper is absent.
fn fallback_per_image(width: u32, height: u32) -> u32 {
    const PATCH_DIVISOR: u32 = 14;
    /// InternVL's single tile, gemma-4's flat count on b8814.
    const ENCODER_FLOOR: u32 = 256;
    // The dimensions are unrestricted u32 plan fields, so the product can overflow.
    (width / PATCH_DIVISOR)
        .saturating_mul(height / PATCH_DIVISOR)
        .max(ENCODER_FLOOR)
}

/// A cell's `--image-min-tokens` can push the real count above the patch grid,
/// so the estimate honors it.
fn min_tokens_override(count_flags: &[String]) -> u32 {
    count_override(count_flags, "--image-min-tokens").map_or(0, with_rounding_slack)
}

/// A cell's `--image-max-tokens`, with the slack [`min_tokens_override`] adds:
/// the most tokens a projector that honours the budget gives one image. Only
/// the helper's context uses it; the estimate needs no cap to stay an upper
/// bound.
fn max_tokens_override(count_flags: &[String]) -> Option<u32> {
    count_override(count_flags, "--image-max-tokens").map(with_rounding_slack)
}

/// The last positive value of `flag`, as the CLI's last value wins.
fn count_override(count_flags: &[String], flag: &str) -> Option<u32> {
    count_flags
        .chunks(2)
        .filter(|pair| pair.first().map(String::as_str) == Some(flag))
        .filter_map(|pair| pair.get(1)?.parse::<u32>().ok())
        .next_back()
        .filter(|&tokens| tokens > 0)
}

/// `tokens` plus the overshoot of an encoder that rounds both sides of a frame
/// up: about 2√M on a square frame and more on a long one, so 4√M.
fn with_rounding_slack(tokens: u32) -> u32 {
    tokens
        .saturating_add(tokens.isqrt().saturating_mul(4))
        .saturating_add(1)
}

#[cfg(windows)]
pub(crate) const HELPER_NAME: &str = "llama-mtmd-cli.exe";
#[cfg(not(windows))]
pub(crate) const HELPER_NAME: &str = "llama-mtmd-cli";

fn cache_key(
    fingerprint: &str,
    width: u32,
    height: u32,
    count_flags: &[String],
    envs: &[EnvForward],
) -> String {
    let base = if count_flags.is_empty() {
        // The original format, so existing cache entries stay valid.
        format!("{fingerprint}|{width}x{height}")
    } else {
        format!("{fingerprint}|{width}x{height}|{}", count_flags.join(" "))
    };
    with_envs(base, envs)
}

/// `base` keyed by the env forwards the helper gets too: without them two cells
/// differing only in `envs` would share one memoized count. Appended rather than
/// folded into the existing segment so a cell that forwards nothing keeps the
/// format above, and its cache entries stay valid.
fn with_envs(base: String, envs: &[EnvForward]) -> String {
    if envs.is_empty() {
        return base;
    }
    // Env forwards reach the helper, so they key the count. Appended, so a cell
    // that forwards nothing keeps the format above.
    let joined = envs
        .iter()
        .map(AsRef::as_ref)
        .collect::<Vec<&str>>()
        .join(" ");
    format!("{base}|env:{joined}")
}

/// Content fingerprint of the mmproj: `<size>:<sha256(first 1 MiB)[..8]>`.
/// Distinguishes same-named projectors (five models ship `mmproj-F16.gguf`
/// with different vision encoders). Stable across toolchains and platforms,
/// so a cache produced on one host stays valid on another for the same build.
pub(crate) fn mmproj_fingerprint(path: &Path) -> Option<String> {
    use sha2::{Digest, Sha256};
    let size = std::fs::metadata(path).ok()?.len();
    let f = std::fs::File::open(path).ok()?;
    // A bare read() may return short (network FS, EINTR) and mint another
    // fingerprint for the same file, missing the cache.
    let mut head = Vec::with_capacity(1 << 20);
    f.take(1 << 20).read_to_end(&mut head).ok()?;
    let digest: [u8; 32] = Sha256::digest(&head).into();
    // SHA-256 rather than `DefaultHasher`, which a toolchain change can alter.
    Some(format!("{size}:{}", hex::encode(&digest[..8])))
}

/// `vl_calibration.json`. A file that fails to parse reads as empty and is
/// rebuilt on the next store.
#[derive(Default, serde::Serialize, serde::Deserialize)]
struct Cache {
    #[serde(default)]
    entries: std::collections::BTreeMap<String, u32>,
}

impl Cache {
    fn load(path: &Path) -> Cache {
        std::fs::read_to_string(path)
            .ok()
            .and_then(|t| serde_json::from_str(&t).ok())
            .unwrap_or_default()
    }
}

pub(crate) fn cache_lookup(cache: &Path, key: &str) -> Option<u32> {
    Cache::load(cache).entries.get(key).copied()
}

pub(crate) fn cache_store(cache: &Path, key: &str, value: u32) {
    // Read-modify-write; best-effort (ignore errors — the value was still
    // computed and returned, we just failed to memoize it). Unlocked: two
    // processes calibrating concurrently race here and one update may be lost.
    // Benign — every key recomputes to the same value for a given
    // (mmproj, build, W×H), so a lost write only costs a later cache miss.
    let mut c = Cache::load(cache);
    c.entries.insert(key.to_string(), value);
    if let Ok(text) = serde_json::to_string_pretty(&c) {
        let _ = std::fs::write(cache, text);
    }
}

/// For a count the server did not confirm, so the next cell measures it again;
/// best-effort and unlocked as [`cache_store`] is.
pub(crate) fn cache_evict(cache: &Path, key: &str) {
    let mut c = Cache::load(cache);
    if c.entries.remove(key).is_some()
        && let Ok(text) = serde_json::to_string_pretty(&c)
    {
        let _ = std::fs::write(cache, text);
    }
}

/// Run the helper `launch` describes on `frames` solid `W×H` images and parse
/// the marker-free image-token count from its verbose log. Cost is one
/// model+mmproj load, the encode and a prefill of every frame's image tokens;
/// paid once per resolution and frame count, then memoized.
///
/// `deadline` bounds the whole helper run — production passes
/// [`VL_CALIBRATION_TIMEOUT`]; it is a parameter so tests can exercise the
/// kill path without waiting out the real 600s.
fn run_helper(
    launch: &MtmdCommandBuilder,
    width: u32,
    height: u32,
    frames: u32,
    deadline: std::time::Duration,
) -> Option<u32> {
    let png = pipette_ops::png::gray(width, height).ok()?;
    let img_path =
        std::env::temp_dir().join(format!("vlcal-{}-{width}x{height}.png", std::process::id()));
    std::fs::write(&img_path, &png).ok()?;
    run_counting_helper(
        launch.build(&img_path, frames),
        &img_path,
        IMAGE_BATCH_LINE,
        deadline,
    )
}

/// Sums the `batch_line` batches the helper's verbose log prints, and removes
/// `media` afterwards.
pub(crate) fn run_counting_helper(
    command: Command,
    media: &Path,
    batch_line: &'static str,
    deadline: std::time::Duration,
) -> Option<u32> {
    let Ok(mut helper) = start_process("llama-mtmd-cli", command, Duration::ZERO) else {
        let _ = std::fs::remove_file(media);
        return None;
    };
    // Collected as they arrive: the `-v` log outgrows the output `Process`
    // keeps, and a batch line cut from its head would undercount.
    let batch_lines = Arc::new(Mutex::new(String::new()));
    let sink = Arc::clone(&batch_lines);
    helper.on_line(move |stream, line| {
        pipette_subprocess::log_line(stream, line);
        if line.contains(batch_line) {
            let mut kept = sink.lock().unwrap_or_else(PoisonError::into_inner);
            kept.push_str(line);
            kept.push('\n');
        }
    });
    // Bounded, so a wedged helper (a GPU-driver hang) cannot stall the benchmark.
    let exited = helper.wait_success(deadline);
    // Exited or stopped by now, so its media file can be removed.
    let _ = std::fs::remove_file(media);

    // Trust only a clean exit: a failed run can leave partial batch lines (a tiled
    // model that emitted tile 1/2 then died), which would undercount.
    if let Err(err) = exited.map_err(anyhow::Error::from) {
        tracing::warn!(
            error = %format_args!("{err:#}"),
            "calibration helper failed; falling back"
        );
        return None;
    }
    parse_batch_tokens(
        &batch_lines.lock().unwrap_or_else(PoisonError::into_inner),
        batch_line,
    )
}

/// Marks the lines [`parse_batch_tokens`] counts for an image.
const IMAGE_BATCH_LINE: &str = "decoding image batch";

/// Sum `n_tokens_batch = N` across the `batch_line` lines `llama-mtmd-cli -v`
/// emits: the marker-free media content only. A tiled image prints one line per
/// batch and a clip one per chunk; they sum to the full count.
///
/// **All-or-nothing.** The one thing this must never do is return a *partial*
/// (too-small) sum: the caller sizes `n_ctx` from it, and the fallback
/// (`(w/14)²` for an image, the clip's upper bound for audio) is an
/// OVER-estimate, so `None` is always safe (n_ctx over-sized) but an
/// undercount would tip `n_ctx` short. Therefore any batch line
/// we cannot turn into a count — a missing/garbled `n_tokens_batch`, or a sum
/// that overflows `u32` — fails the whole parse to `None` rather than dropping
/// that batch and summing the rest. Returns `None` when no batch line is present
/// at all. (Lines without `batch_line`, e.g. a stray `n_tokens_batch`
/// in a summary line, are not batch lines and are ignored.)
pub(crate) fn parse_batch_tokens(log: &str, batch_line: &str) -> Option<u32> {
    let mut total: u32 = 0;
    let mut found = false;
    for line in log.lines() {
        if !line.contains(batch_line) {
            continue; // not a batch line — ignore
        }
        let n = parse_n_tokens_batch(line)?;
        total = total.checked_add(n)?;
        found = true;
    }
    found.then_some(total)
}

/// The `n_tokens_batch = N` count on one line. Reads the leading digit run after
/// the `=`, so a trailing field (`= 256, foo = 3`) or a `\r` still parses `256`.
fn parse_n_tokens_batch(line: &str) -> Option<u32> {
    let (_, after_marker) = line.split_once("n_tokens_batch")?;
    let (_, after_eq) = after_marker.split_once('=')?;
    let after_eq = after_eq.trim_start();
    let digits: &str = after_eq
        .split(|c: char| !c.is_ascii_digit())
        .next()
        .unwrap_or("");
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u32>().ok()
}

/// The least text term in a vlm_eval `n_ctx`, for a dataset whose scan finds no
/// text.
const VLM_EVAL_MIN_TEXT_HEADROOM: u32 = 256;

/// The vlm_eval `n_ctx` validated end to end: small-image datasets keep it, and
/// only large images push the size higher.
pub(super) const VLM_EVAL_CTX_FLOOR: u32 = 8192;

/// Ceiling on the derived vlm_eval `n_ctx`, the total across the slots. The
/// runner never reads `n_ctx_train`, so this is a named cap: 128K is at or above
/// the trained context of every model family benchmarked, and already a KV
/// cache past what the target edge devices allocate. [`bound_vlm_eval_ctx`]
/// fails past it rather than clamping.
pub(super) const VLM_EVAL_CTX_CEILING: u32 = 131_072;

/// Decoded bytes the image-header scan reads: covers a PNG IHDR and any
/// realistic JPEG SOF offset (EXIF/APPn/comment segments ahead of it).
const HEADER_SCAN_DECODED_BYTES: usize = 8192;

/// Base64 chars that decode to at least [`HEADER_SCAN_DECODED_BYTES`], in whole
/// 4-char groups so the prefix stays decodable. 8192 chars would decode only
/// 6144 bytes.
const HEADER_SCAN_B64_CHARS: usize = HEADER_SCAN_DECODED_BYTES.div_ceil(3) * 4;

/// One slot's `n_ctx` for a vlm_eval run: the server starts once, so it bounds
/// every sample.
///
/// ```text
///   n_ctx = max_over_candidates( image_tokens(candidate dims) )
///           × max_over_samples( images_per_sample )
///         + max_over_samples( text tokens )
///         + max output tokens
///         + margins,
///   floored at VLM_EVAL_CTX_FLOOR + max_tokens,
///   n_ctx x slots over VLM_EVAL_CTX_CEILING is a hard error.
/// ```
///
/// The image term is the max over [`scan_max_image_dims`]' candidates: the
/// count is not coordinate-wise monotonic for an aspect-ratio tiler (InternVL
/// can need more tokens for 6000×500 than for 6000×6000), so one synthetic
/// `max_w × max_h` point is not enough. That bounds the dataset's own extreme
/// shapes, not every shape between them. A candidate past the helper's PNG
/// bound is calibrated at [`clamp_to_calibration_bound`]. If any candidate
/// fails, the whole term falls back to the patch grid at `max_w × max_h`, which
/// is monotonic: a partial max could undercount. `Ok(None)` when no image
/// dimension is parseable.
pub(super) fn vlm_eval_ctx_size(
    launch: &MtmdCommandBuilder,
    samples: &[EvalSample],
    max_tokens: u32,
    slots: u32,
) -> anyhow::Result<Option<u32>> {
    let Some((inputs, candidates)) = vlm_eval_ctx_inputs(samples, max_tokens) else {
        return Ok(None);
    };
    let per_image = calibrated_image_term(launch, &inputs, &candidates);
    // No floor here: the vlm_eval floor is applied per slot, below.
    let sized = inputs.ctx_size(per_image, &launch.raw, 0);
    let source = if per_image.is_some() {
        "calibrated"
    } else {
        "fallback"
    };
    bound_vlm_eval_ctx(sized, &inputs, max_tokens, source, slots).map(Some)
}

/// The image term [`vlm_eval_ctx_size`] and [`vlm_eval_sample_need`] measure:
/// the max of the helper's count over the candidate set, `None` (the
/// patch-grid fallback) when any candidate fails to calibrate.
fn calibrated_image_term(
    launch: &MtmdCommandBuilder,
    inputs: &CtxInputs,
    candidates: &[(u32, u32)],
) -> Option<u32> {
    candidates.iter().try_fold(0u32, |best, &(w, h)| {
        let (cw, ch) = clamp_to_calibration_bound(w, h);
        let at = CtxInputs {
            width: cw,
            height: ch,
            ..*inputs
        };
        // A projector that ignores the cell's `--image-max-tokens` overflows
        // the capped context, so a failure retries once uncapped.
        let capped = helper_ctx(cw, ch, &launch.raw);
        let uncapped = helper_ctx_uncapped(cw, ch, &launch.raw);
        let measured = calibrate(&at, &launch.clone().ctx_size(Some(capped))).or_else(|| {
            (uncapped > capped)
                .then(|| calibrate(&at, &launch.clone().ctx_size(Some(uncapped))))
                .flatten()
        });
        match measured {
            Some(n) if (cw, ch) == (w, h) => {
                tracing::debug!("vlm_eval ctx: candidate {w}x{h}: calibrated {n} image tokens");
                Some(best.max(n))
            }
            Some(n) => {
                tracing::debug!(
                    "vlm_eval ctx: candidate {w}x{h}: calibrated {n} image tokens at the \
                     aspect-preserving clamp {cw}x{ch} (saturated count)"
                );
                Some(best.max(n))
            }
            None => {
                tracing::debug!(
                    "vlm_eval ctx: candidate {w}x{h} (calibrated at {cw}x{ch}): calibration \
                     failed (helper absent, non-zero exit, or unparseable log); the whole \
                     image term drops to the patch-grid fallback"
                );
                None
            }
        }
    })
}

/// What one sample needs at most, unfloored and unbounded: the dataset's
/// largest image term, its longest text, the output budget and the margins.
/// Measured by `launch`'s helper over the candidate set when given (a model
/// load on a cache miss), else the patch-grid over-estimate, which ignores an
/// `--image-max-tokens` budget. `None` when no image dimension is parseable.
pub(super) fn vlm_eval_sample_need(
    samples: &[EvalSample],
    max_tokens: u32,
    raw: &[String],
    launch: Option<&MtmdCommandBuilder>,
) -> Option<(u32, &'static str)> {
    let (inputs, candidates) = vlm_eval_ctx_inputs(samples, max_tokens)?;
    let per_image = launch.and_then(|launch| calibrated_image_term(launch, &inputs, &candidates));
    let raw = launch.map_or(raw, |launch| launch.raw.as_slice());
    let source = if per_image.is_some() {
        "calibrated"
    } else {
        "estimated"
    };
    Some((inputs.ctx_size(per_image, raw, 0), source))
}

/// Aspect-preserving clamp of a calibration candidate to
/// [`pipette_ops::png::MAX_PIXELS`], the helper's synthetic-PNG bound. OCRBench
/// holds 4961×7016 scans and a 620×14495 receipt, whose per-axis maxima make a
/// 72 MP point.
///
/// The clamp does not change the count: a VLM preprocessor downscales or tiles
/// an oversized image to its cap before encoding, so the count saturates, and
/// the 64 MP bound sits far past every preprocessor cap shipped.
///
/// Each axis scales by `√(bound/area)` via `isqrt`, rounding down, so
/// `w'·h' ≤ bound`; each axis is floored at 1 px.
#[allow(
    clippy::cast_possible_truncation,
    reason = "w and h are proven <= BOUND (2^26) above, so the u32 cast never truncates"
)]
fn clamp_to_calibration_bound(width: u32, height: u32) -> (u32, u32) {
    const BOUND: u64 = pipette_ops::png::MAX_PIXELS;
    let area = u64::from(width) * u64::from(height);
    if area <= BOUND {
        return (width, height);
    }
    // floor(dim·√(bound/area)) = isqrt(dim²·bound/area), in u128 (dim²·bound
    // reaches 2^90 for u32 dims). Division and isqrt both round down, so the
    // clamped area can only land under the bound, never over.
    let scale = |dim: u32| -> u64 {
        let scaled = u128::from(dim) * u128::from(dim) * u128::from(BOUND) / u128::from(area);
        u64::try_from(scaled.isqrt()).unwrap_or(u64::MAX)
    };
    let w = scale(width).clamp(1, BOUND);
    // The 1-px floor on a needle shape (one axis rounding to 0) can push the
    // area back over; capping the other axis at BOUND/w (≥1, as w ≤ BOUND)
    // restores it.
    let h = scale(height).clamp(1, BOUND / w);
    (w as u32, h as u32)
}

/// [`vlm_eval_ctx_size`] without the helper, for a cell that pins its own
/// `ctx_size`: the derived number is then never used, so it is not worth a
/// model load (up to 600s).
pub(super) fn vlm_eval_fallback_ctx_size(
    samples: &[EvalSample],
    max_tokens: u32,
    raw: &[String],
    slots: u32,
) -> anyhow::Result<Option<u32>> {
    // The patch grid is coordinate-wise monotonic, so the synthetic point
    // bounds every candidate.
    let Some((inputs, _candidates)) = vlm_eval_ctx_inputs(samples, max_tokens) else {
        return Ok(None);
    };
    let sized = inputs.ctx_size(None, raw, 0);
    bound_vlm_eval_ctx(sized, &inputs, max_tokens, "fallback", slots).map(Some)
}

/// The floor and ceiling of both vlm_eval sizings, and the one-line summary of
/// the derivation.
///
/// The floor is on one slot's budget: every sample still needs the validated
/// context, so a scaled floor would give a 4-slot run a quarter of it per slot.
/// The ceiling is on the total llama-server is handed, as it stands for a KV
/// cache the device allocates whole. It is an error, not a clamp: past it is
/// either an exploded estimate (an absurd header dimension on the unbounded
/// patch grid) or a genuine need (the floor alone, a huge multi-image
/// calibration, too many slots), and a clamp would turn the genuine case into
/// ctx overflows deep in the run. A giant-image dataset without the helper
/// lands here by design: without it, where the count saturates is unknown.
fn bound_vlm_eval_ctx(
    sized: u32,
    inputs: &CtxInputs,
    max_tokens: u32,
    source: &str,
    slots: u32,
) -> anyhow::Result<u32> {
    let floor = VLM_EVAL_CTX_FLOOR.saturating_add(max_tokens);
    let floored = sized.max(floor);
    // `u32::MAX` is past the ceiling, so saturating cannot pass an overflow.
    let total = floored.saturating_mul(slots);
    let margins =
        CTX_MARGIN.saturating_add(CTX_MARGIN_PER_IMAGE.saturating_mul(inputs.num_images.max(1)));
    // The image term reconstructed by subtraction, so the summary stays exact
    // for both sources (the fallback folds an --image-min-tokens override
    // into it that the inputs alone don't carry).
    let image_term = sized
        .saturating_sub(inputs.text_tokens)
        .saturating_sub(inputs.decode_tokens)
        .saturating_sub(margins);
    tracing::info!(
        "vlm_eval n_ctx: image {image_term} + text {} + output {} + margins {margins} = {sized}; \
         floor {floor} {}; per-slot n_ctx {floored} x {slots} slot(s) = {total}; \
         ceiling {VLM_EVAL_CTX_CEILING} {} (source: {source})",
        inputs.text_tokens,
        inputs.decode_tokens,
        if floored > sized {
            "applied"
        } else {
            "not applied"
        },
        if total > VLM_EVAL_CTX_CEILING {
            "EXCEEDED"
        } else {
            "ok"
        },
    );
    // Both numbers are named: the per-slot budget is the dataset's to change,
    // the multiplier the cell's.
    anyhow::ensure!(
        total <= VLM_EVAL_CTX_CEILING,
        "vlm_eval derived n_ctx {floored} per slot x {slots} slot(s) = {total}, past the \
         {VLM_EVAL_CTX_CEILING} ceiling; refusing to clamp silently (an undersized context \
         would fail oversized samples at the server). Lower `parallel`; or, if the workload \
         is genuinely this large, pin --ctx-size explicitly in the cell's runtime flags \
         (it is the total across all slots); otherwise check the dataset for a mis-read or \
         absurd image-header dimension",
    );
    Ok(floored)
}

/// The worst-case workload of a vlm_eval dataset, and the candidate dims to
/// calibrate. `None` when no image dimension is parseable.
fn vlm_eval_ctx_inputs(
    samples: &[EvalSample],
    max_tokens: u32,
) -> Option<(CtxInputs, Vec<(u32, u32)>)> {
    let scan = scan_max_image_dims(samples)?;
    let text_tokens = scan_max_text_tokens(samples).max(VLM_EVAL_MIN_TEXT_HEADROOM);
    tracing::debug!(
        "vlm_eval ctx scan: max_w {} max_h {}, {} candidate dims {:?}, max_text_tokens \
         {text_tokens}, images_per_sample {}",
        scan.max_w,
        scan.max_h,
        scan.candidates.len(),
        scan.candidates,
        scan.max_images,
    );
    let inputs = CtxInputs {
        width: scan.max_w,
        height: scan.max_h,
        text_tokens,
        decode_tokens: max_tokens,
        num_images: scan.max_images.max(1),
    };
    Some((inputs, scan.candidates))
}

/// `max_w` and `max_h` are maximized independently, the worst case for the
/// coordinate-wise monotonic patch grid; the tiled count is not, hence
/// `candidates`.
struct ImageScan {
    max_w: u32,
    max_h: u32,
    max_images: u32,
    /// At most 4, deduped: the synthetic `(max_w, max_h)` point, then the
    /// max-area, widest and tallest images in the dataset.
    candidates: Vec<(u32, u32)>,
}

/// The per-axis maxima, the most images in one sample, and the calibration
/// candidates (see [`vlm_eval_ctx_size`]). An image whose dimensions cannot be
/// read is skipped with a warning, as it could be the largest; `None` when none
/// is readable. Only headers are parsed, so the scan is cheap on hundreds of MB
/// of base64.
fn scan_max_image_dims(samples: &[EvalSample]) -> Option<ImageScan> {
    let mut max_w = 0u32;
    let mut max_h = 0u32;
    let mut max_images = 0u32;
    // The dataset's extreme shapes, tracked without floats: area in u64,
    // aspect (w/h) ordered by cross-multiplication.
    let mut max_area: Option<(u32, u32)> = None;
    let mut widest: Option<(u32, u32)> = None;
    let mut tallest: Option<(u32, u32)> = None;
    for sample in samples {
        let sample_id = sample.id.as_str();
        let mut images_here = 0u32;
        for msg in &sample.messages {
            let MessageContent::Parts(parts) = &msg.content else {
                continue;
            };
            for part in parts {
                let ContentPart::ImageUrl { image_url } = part else {
                    continue;
                };
                images_here += 1;
                let Some((w, h)) = data_uri_dims(&image_url.url) else {
                    // `data_uri_dims` already retried on the full payload.
                    tracing::warn!(
                        "vlm_eval ctx scan: sample `{sample_id}` image {images_here}: could not \
                         read dimensions from the image header; if this is the dataset's largest \
                         image the derived n_ctx will not cover it"
                    );
                    continue;
                };
                max_w = max_w.max(w);
                max_h = max_h.max(h);
                if max_area.is_none_or(|(aw, ah)| {
                    u64::from(w) * u64::from(h) > u64::from(aw) * u64::from(ah)
                }) {
                    max_area = Some((w, h));
                }
                // w/h > ww/wh ⇔ w·wh > ww·h (all positive here).
                if widest.is_none_or(|(ww, wh)| {
                    u64::from(w) * u64::from(wh) > u64::from(ww) * u64::from(h)
                }) {
                    widest = Some((w, h));
                }
                if tallest.is_none_or(|(tw, th)| {
                    u64::from(w) * u64::from(th) < u64::from(tw) * u64::from(h)
                }) {
                    tallest = Some((w, h));
                }
            }
        }
        max_images = max_images.max(images_here);
    }
    if max_w == 0 || max_h == 0 {
        return None;
    }
    let mut candidates = vec![(max_w, max_h)];
    for dims in [max_area, widest, tallest].into_iter().flatten() {
        if !candidates.contains(&dims) {
            candidates.push(dims);
        }
    }
    Some(ImageScan {
        max_w,
        max_h,
        max_images,
        candidates,
    })
}

/// Pixel dimensions from a `data:image/<fmt>;base64,<payload>` URL, decoding
/// only a [`HEADER_SCAN_B64_CHARS`] prefix; the alphabet is ASCII, so the byte
/// slice cannot split a char. A JPEG can carry its SOF past the prefix (behind
/// an APP1/EXIF segment over 8 KiB), and the server reads it, so a failed
/// prefix walk retries on the full payload, still with no pixel decode.
fn data_uri_dims(url: &str) -> Option<(u32, u32)> {
    use base64::Engine;
    let b64 = url.split(";base64,").nth(1)?;
    let window = b64.as_bytes().get(..b64.len().min(HEADER_SCAN_B64_CHARS))?;
    let prefix = base64::engine::general_purpose::STANDARD
        .decode(window)
        .ok();
    let from_prefix = prefix.as_deref().and_then(image_dims);
    // Only a JPEG's SOF can sit past the prefix; any other format would decode
    // the whole payload just to return `None` again.
    let jpeg = prefix
        .as_deref()
        .is_some_and(|b| b.starts_with(&[0xFF, 0xD8]));
    if from_prefix.is_some() || b64.len() <= HEADER_SCAN_B64_CHARS || !jpeg {
        return from_prefix;
    }
    base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .ok()
        .and_then(|bytes| image_dims(&bytes))
}

/// The largest per-sample text-token estimate, over every text part of a
/// sample's messages, so a text-heavy prompt (a long OCR passage, many MCQ
/// options) cannot tip `n_ctx` short.
fn scan_max_text_tokens(samples: &[EvalSample]) -> u32 {
    let mut max_tokens = 0u32;
    for sample in samples {
        let mut sample_tokens = 0u32;
        for msg in &sample.messages {
            match &msg.content {
                MessageContent::Parts(parts) => {
                    for part in parts {
                        if let ContentPart::Text { text } = part {
                            sample_tokens =
                                sample_tokens.saturating_add(estimate_text_tokens(text));
                        }
                    }
                }
                MessageContent::Text(text) => {
                    sample_tokens = sample_tokens.saturating_add(estimate_text_tokens(text));
                }
            }
        }
        max_tokens = max_tokens.max(sample_tokens);
    }
    max_tokens
}

/// One token per UTF-8 byte, with no server round-trip: an upper bound even for
/// digit-dense OCR text, where BPE vocabularies emit about one token per digit.
/// Prose averages ~4 bytes per token, so it over-reserves ~4x there, cheap next
/// to the image budget.
fn estimate_text_tokens(text: &str) -> u32 {
    // Saturating: a truncating `as u32` could undercount.
    u32::try_from(text.len()).unwrap_or(u32::MAX)
}

/// `None` for a format other than PNG or JPEG, or a malformed header.
fn image_dims(bytes: &[u8]) -> Option<(u32, u32)> {
    const PNG_SIG: &[u8] = &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if bytes.starts_with(PNG_SIG) {
        // IHDR is the first chunk: 8-byte sig, 4-byte len, "IHDR", then
        // width (4) + height (4), all big-endian.
        let w = u32::from_be_bytes(bytes.get(16..20)?.try_into().ok()?);
        let h = u32::from_be_bytes(bytes.get(20..24)?.try_into().ok()?);
        return (w > 0 && h > 0).then_some((w, h));
    }
    if bytes.starts_with(&[0xFF, 0xD8]) {
        return jpeg_dims(bytes);
    }
    None
}

/// The size in the JPEG's Start-Of-Frame segment: a marker in 0xC0-0xCF but
/// the non-frame 0xC4 DHT, 0xC8 JPG and 0xCC DAC.
fn jpeg_dims(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut i = 2; // skip SOI (0xFF 0xD8)
    while i + 1 < bytes.len() {
        if *bytes.get(i)? != 0xFF {
            i += 1;
            continue;
        }
        // Skip any fill bytes (runs of 0xFF).
        let mut marker = *bytes.get(i + 1)?;
        let mut j = i + 1;
        while marker == 0xFF && j + 1 < bytes.len() {
            j += 1;
            marker = *bytes.get(j)?;
        }
        i = j;
        match marker {
            0xC0..=0xCF if !matches!(marker, 0xC4 | 0xC8 | 0xCC) => {
                // marker, then 2-byte segment length, 1-byte precision,
                // 2-byte height, 2-byte width.
                let h = u16::from_be_bytes(bytes.get(i + 4..i + 6)?.try_into().ok()?);
                let w = u16::from_be_bytes(bytes.get(i + 6..i + 8)?.try_into().ok()?);
                return (w > 0 && h > 0).then_some((w as u32, h as u32));
            }
            // Standalone markers (no length field): RSTn, SOI, EOI, TEM.
            0xD0..=0xD9 | 0x01 => i += 1,
            // Everything else has a 2-byte length; skip the segment.
            _ => {
                let len = u16::from_be_bytes(bytes.get(i + 1..i + 3)?.try_into().ok()?) as usize;
                i += 1 + len;
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use rstest::rstest;

    use pipette_plan_types::{ChatMessage, ChatRole, ImageUrl};

    use super::*;
    use crate::runtimes::test_llama_server;

    /// The template adds markers per image: 80 frames of 64 tokens measured ~330
    /// scaffolding tokens, past a flat 64-token margin.
    #[test]
    fn ctx_margin_scales_with_frame_count() {
        let at = |num_images| {
            CtxInputs {
                width: 256,
                height: 256,
                text_tokens: 0,
                decode_tokens: 64,
                num_images,
            }
            .ctx_size(Some(64), &vl_flags(&[]), 0)
        };
        assert_eq!(at(1), 64 + 64 + CTX_MARGIN + CTX_MARGIN_PER_IMAGE);
        let f80 = at(80);
        assert_eq!(f80, 64 * 80 + 64 + CTX_MARGIN + CTX_MARGIN_PER_IMAGE * 80);
        assert!(f80 - (64 * 80 + 64) >= 330, "margin too small: {f80}");
    }

    /// (128/14)² = 81 would undersize gemma-4/InternVL (~256 real).
    #[rstest::rstest]
    #[case::small_floors(128, 128, 256)]
    #[case::sub_patch_floors(8, 8, 256)]
    #[case::grid_above_floor(256, 256, 324)]
    #[case::grid_rules_large(512, 512, 1296)]
    fn fallback_per_image_floors_flat_encoders(
        #[case] w: u32,
        #[case] h: u32,
        #[case] expect: u32,
    ) {
        assert_eq!(fallback_per_image(w, h), expect);
    }

    #[rstest::rstest]
    #[case::min_above_grid(&["--image-min-tokens", "2048"], 2048 + 4 * 45 + 1)]
    #[case::min_below_grid(&["--image-min-tokens", "100"], 324)]
    #[case::last_wins(&["--image-min-tokens", "4000", "--image-min-tokens", "2048"], 2048 + 4 * 45 + 1)]
    #[case::max_ignored(&["--image-max-tokens", "9999"], 324)]
    fn fallback_ctx_honors_min_tokens_override(#[case] raw: &[&str], #[case] per_image: u32) {
        let inputs = CtxInputs {
            width: 256,
            height: 256,
            text_tokens: 0,
            decode_tokens: 1,
            num_images: 1,
        };
        assert_eq!(
            inputs.ctx_size(None, &vl_flags(raw), 0),
            per_image + 1 + CTX_MARGIN + CTX_MARGIN_PER_IMAGE
        );
    }

    /// The floor applies to the fallback alone.
    #[rstest::rstest]
    #[case::fallback(None, 0, 324 * 5 + 10 + 64 + CTX_MARGIN + CTX_MARGIN_PER_IMAGE * 5)]
    #[case::fallback_floored(None, 8192, 8192)]
    #[case::calibrated_unfloored(Some(64), 8192, 64 * 5 + 10 + 64 + CTX_MARGIN + CTX_MARGIN_PER_IMAGE * 5)]
    fn ctx_size_applies_shared_margins(
        #[case] calibrated: Option<u32>,
        #[case] floor: u32,
        #[case] expect: u32,
    ) {
        let inputs = CtxInputs {
            width: 256,
            height: 256,
            text_tokens: 10,
            decode_tokens: 64,
            num_images: 5,
        };
        assert_eq!(inputs.ctx_size(calibrated, &vl_flags(&[]), floor), expect);
    }

    // A temp dir per test tag, best-effort.
    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("vlcal-{}-{}", std::process::id(), tag));
        let _ = std::fs::create_dir_all(&d);
        d
    }

    /// A fake runtime dir with `script_body` as `llama-mtmd-cli`. Each test must
    /// use a distinct `W×H`: the helper's temp PNG path is keyed by pid and
    /// resolution, and the tests share one process.
    #[cfg(unix)]
    fn fake_runtime(tag: &str, script_body: &str) -> anyhow::Result<(MtmdCommandBuilder, PathBuf)> {
        use std::os::unix::fs::PermissionsExt;

        use pipette_plan_types::{
            AbsolutePath, LlamaCppFlavor, LlamacppCliStockTools, LlamacppCliStockToolsSource,
            Runtime,
        };

        let dir = scratch(tag);
        let _ = std::fs::remove_file(dir.join("vl_calibration.json"));
        let helper = dir.join(HELPER_NAME);
        std::fs::write(&helper, format!("#!/bin/sh\n{script_body}\n"))?;
        let mut perms = std::fs::metadata(&helper)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&helper, perms)?;
        // Never run: it only places the helper.
        std::fs::write(dir.join("llama-server"), "")?;
        let server = ResolvedLlamaServer::try_from(&Runtime::LlamacppCliStockTools(
            LlamacppCliStockTools {
                source: LlamacppCliStockToolsSource::AbsoluteDir {
                    dir: AbsolutePath::try_new(dir.display().to_string())?,
                },
                flavor: LlamaCppFlavor::MacosArm64,
            },
        ))?;
        let mmproj = dir.join("mmproj.gguf");
        std::fs::write(&mmproj, format!("mmproj-bytes-{tag}"))?;
        let launch = MtmdCommandBuilder::new(server, dir.join("model.gguf"), mmproj);
        Ok((launch, dir))
    }

    /// A memoized failure would pin the fallback for every later run at that
    /// resolution.
    #[cfg(unix)]
    #[rstest::rstest]
    #[case::exits_nonzero("helper-fail", "exit 3", 96)]
    #[case::garbled_output(
        "helper-garbled",
        "echo 'decoding image batch 1/1, n_tokens_batch = garbage' >&2",
        112
    )]
    fn failed_helper_yields_none_and_caches_nothing(
        #[case] tag: &str,
        #[case] script: &str,
        #[case] dim: u32,
    ) -> anyhow::Result<()> {
        let (launch, dir) = fake_runtime(tag, script)?;
        let got = calibrate(&CtxInputs::new(dim, dim, 0, 0, 1)?, &launch);
        assert_eq!(got, None);
        assert!(
            !dir.join("vl_calibration.json").exists(),
            "a failed helper run must not create a cache row"
        );
        Ok(())
    }

    /// The second call is served by the cache: the helper is swapped for one
    /// that fails.
    #[cfg(unix)]
    #[test]
    fn clean_helper_output_is_parsed_and_cached() -> anyhow::Result<()> {
        let (launch, dir) = fake_runtime(
            "helper-ok",
            "echo 'decoding image batch 1/2, n_tokens_batch = 100' >&2\n\
             echo 'decoding image batch 2/2, n_tokens_batch = 28' >&2",
        )?;
        let inputs = CtxInputs::new(140, 140, 0, 0, 1)?;
        let got = calibrate(&inputs, &launch);
        assert_eq!(got, Some(128), "tiled batches must sum");
        assert!(dir.join("vl_calibration.json").exists());

        std::fs::write(dir.join(HELPER_NAME), "#!/bin/sh\nexit 9\n")?;
        let cached = calibrate(&inputs, &launch);
        assert_eq!(cached, Some(128), "second call must be served by the cache");
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn the_vlm_eval_helper_context_is_sized_to_the_image() -> anyhow::Result<()> {
        let (launch, dir) = fake_runtime(
            "helper-ctx",
            "printf '%s\\n' \"$@\" > \"$(dirname \"$0\")/argv\"\n\
             echo 'decoding image batch 1/1, n_tokens_batch = 64' >&2",
        )?;
        let samples = vec![sample_with(&[data_uri("jpeg", &tiny_jpeg(1414, 1428))])];
        let (inputs, _) =
            vlm_eval_ctx_inputs(&samples, 64).ok_or_else(|| anyhow::anyhow!("must scan"))?;
        let expect = inputs
            .ctx_size(Some(64), &[], 0)
            .max(VLM_EVAL_CTX_FLOOR + 64);
        assert_eq!(vlm_eval_ctx_size(&launch, &samples, 64, 1)?, Some(expect));
        let argv = std::fs::read_to_string(dir.join("argv"))?;
        let argv: Vec<&str> = argv.lines().collect();
        // (1414/14)·(1428/14) + 1024 = 101·102 + 1024.
        assert!(
            argv.windows(2).any(|pair| pair == ["--ctx-size", "11326"]),
            "{argv:?}"
        );
        Ok(())
    }

    /// Gemma's 1120 budget on the page: 1120 + 4 x 33 + 1 + 1024 = 2277, under
    /// the floor.
    #[rstest::rstest]
    #[case::floored(64, 64, &[], 4096)]
    #[case::past_the_vl_cap(1904, 1904, &[], 136 * 136 + 1024)]
    #[case::ocrbench_page(4961, 7016, &[], VLM_EVAL_CTX_CEILING)]
    #[case::raised_minimum(238, 238, &["--image-min-tokens", "4000"], 4000 + 4 * 63 + 1 + 1024)]
    #[case::budgeted_ocrbench_page(
        4961, 7016, &["--image-min-tokens", "70", "--image-max-tokens", "1120"], 4096
    )]
    #[case::budget_over_a_small_grid(1904, 1904, &["--image-max-tokens", "65536"], 136 * 136 + 1024)]
    #[case::budget_past_the_floor(
        4961, 7016, &["--image-max-tokens", "8192"], 8192 + 4 * 90 + 1 + 1024
    )]
    fn the_vlm_eval_helper_context_bounds(
        #[case] width: u32,
        #[case] height: u32,
        #[case] raw: &[&str],
        #[case] expected: u32,
    ) {
        assert_eq!(helper_ctx(width, height, &vl_flags(raw)), expected);
    }

    #[cfg(unix)]
    #[test]
    fn frames_are_calibrated_together_and_cached_apart() -> anyhow::Result<()> {
        let (launch, dir) = fake_runtime(
            "helper-frames",
            "n=0; ctx=none; prev=\n\
             for a in \"$@\"; do\n\
             [ \"$a\" = --image ] && n=$((n+1))\n\
             [ \"$prev\" = --ctx-size ] && ctx=$a\n\
             prev=$a\n\
             done\n\
             [ $n -gt 1 ] && [ \"$ctx\" != 9000 ] && exit 3\n\
             echo \"decoding image batch 1/1, n_tokens_batch = $((n*10))\" >&2",
        )?;
        let one = CtxInputs::new(150, 150, 0, 0, 1)?;
        let three = CtxInputs::new(150, 150, 0, 0, 3)?;
        let frames = |inputs: &CtxInputs, ctx: Option<u32>| {
            calibrate_frames(inputs, &launch.clone().ctx_size(ctx))
        };
        assert_eq!(frames(&three, Some(9000)), Some(30));
        assert_eq!(frames(&one, None), Some(10), "one frame keys apart");
        assert_eq!(calibrate(&three, &launch), Some(10));
        std::fs::write(dir.join(HELPER_NAME), "#!/bin/sh\nexit 9\n")?;
        assert_eq!(frames(&three, Some(9000)), Some(30));
        assert_eq!(frames(&one, None), Some(10));
        Ok(())
    }

    /// A one-frame calibration bounds the helper's context: the frame's estimate,
    /// at most 16384, or the cell's minimum with room for alignment, plus the
    /// margins, at least 4096, unless the launch sets one. The cell's text,
    /// decode and frames do not reach the helper's prompt, so not its context
    /// either.
    #[cfg(unix)]
    #[rstest::rstest]
    #[case::floored("helper-ctx-floor", 196, &[], None, "4096")]
    #[case::estimated("helper-ctx-estimate", 1792, &[], None, "16456")]
    #[case::capped("helper-ctx-cap", 1904, &[], None, "16456")]
    #[case::raised_minimum(
        "helper-ctx-min",
        238,
        &["--image-min-tokens", "4000"],
        None,
        "4325"
    )]
    #[case::set_by_the_launch("helper-ctx-set", 210, &[], Some(9000), "9000")]
    fn a_one_frame_calibration_bounds_its_context(
        #[case] tag: &str,
        #[case] dim: u32,
        #[case] raw: &[&str],
        #[case] ctx_size: Option<u32>,
        #[case] expected: &str,
    ) -> anyhow::Result<()> {
        let (launch, dir) = fake_runtime(
            tag,
            "printf '%s\\n' \"$@\" > \"$(dirname \"$0\")/argv\"\n\
             echo 'decoding image batch 1/1, n_tokens_batch = 64' >&2",
        )?;
        let inputs = CtxInputs::new(dim, dim, 1024, 64, 3)?;
        let launch = launch.raw(vl_flags(raw)).ctx_size(ctx_size);
        assert_eq!(calibrate(&inputs, &launch), Some(64));
        let argv = std::fs::read_to_string(dir.join("argv"))?;
        let argv: Vec<&str> = argv.lines().collect();
        assert!(
            argv.windows(2).any(|pair| pair == ["--ctx-size", expected]),
            "{argv:?}"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn hung_helper_is_killed_at_the_deadline() -> anyhow::Result<()> {
        let (launch, _) = fake_runtime("helper-hang", "sleep 60")?;
        let started = std::time::Instant::now();
        let got = run_helper(&launch, 84, 84, 1, std::time::Duration::from_millis(300));
        assert_eq!(got, None);
        assert!(
            started.elapsed() < std::time::Duration::from_secs(20),
            "the deadline kill must not wait out the child's sleep"
        );
        Ok(())
    }

    /// A real `-v` log outgrows the output `Process` keeps, so a batch printed
    /// before that much more log must still be counted.
    #[cfg(unix)]
    #[test]
    fn a_batch_further_back_than_the_kept_output_is_counted() -> anyhow::Result<()> {
        let (launch, _) = fake_runtime(
            "helper-long-log",
            "echo 'decoding image batch 1/2, n_tokens_batch = 100' >&2\n\
             yes 'ggml_metal_library_compile_pipeline: loaded kernel' | head -n 20000 >&2\n\
             echo 'decoding image batch 2/2, n_tokens_batch = 28' >&2",
        )?;
        let got = run_helper(&launch, 126, 126, 1, std::time::Duration::from_secs(20));
        assert_eq!(got, Some(128));
        Ok(())
    }

    #[test]
    fn fingerprint_is_stable_and_distinguishes_content() -> anyhow::Result<()> {
        let dir = scratch("fp");
        let a = dir.join("a.gguf");
        let b = dir.join("b.gguf");
        std::fs::write(&a, b"vision-encoder-A-bytes")?;
        std::fs::write(&b, b"vision-encoder-B-different")?;
        let fa = mmproj_fingerprint(&a).unwrap_or_default();
        assert_eq!(fa, mmproj_fingerprint(&a).unwrap_or_default()); // stable
        assert_ne!(fa, mmproj_fingerprint(&b).unwrap_or_default()); // content-distinct
        Ok(())
    }

    #[test]
    fn cache_round_trips_and_looks_up_by_key() -> anyhow::Result<()> {
        let cache = scratch("cache").join("vl_calibration.json");
        let _ = std::fs::remove_file(&cache);
        let key = cache_key("123:abc", 512, 512, &[], &[]);
        assert_eq!(cache_lookup(&cache, &key), None); // no file yet
        cache_store(&cache, &key, 1280);
        assert_eq!(cache_lookup(&cache, &key), Some(1280));
        // a second key coexists
        let key2 = cache_key("123:abc", 256, 256, &[], &[]);
        cache_store(&cache, &key2, 256);
        assert_eq!(cache_lookup(&cache, &key), Some(1280));
        assert_eq!(cache_lookup(&cache, &key2), Some(256));
        Ok(())
    }

    #[test]
    fn parses_image_tokens_sums_batch_lines() {
        // Single batch (continuous / small image): the count is that batch.
        let log = "loading model...\nencoding image slice...\n\
                   decoding image batch 1/1, n_tokens_batch = 64\n\
                   image decoded (batch 1/1) in 3 ms\n";
        assert_eq!(parse_batch_tokens(log, IMAGE_BATCH_LINE), Some(64));
        // Multiple batches (tiled / large image) sum to the full count.
        let tiled = "decoding image batch 1/2, n_tokens_batch = 256\n\
                     decoding image batch 2/2, n_tokens_batch = 1024\n";
        assert_eq!(parse_batch_tokens(tiled, IMAGE_BATCH_LINE), Some(1280));
        // A stray n_tokens_batch outside an image-batch line is ignored.
        assert_eq!(
            parse_batch_tokens("prompt eval: n_tokens_batch = 999", IMAGE_BATCH_LINE),
            None
        );
        assert_eq!(parse_batch_tokens("no image here", IMAGE_BATCH_LINE), None);
    }

    #[test]
    fn parse_batch_tokens_bails_rather_than_undercount() {
        // THE invariant: a tiled run whose 2nd batch is garbled must NOT return
        // the 1st batch's 256 (a partial sum → undersized n_ctx). All-or-nothing.
        let garbled = "decoding image batch 1/2, n_tokens_batch = 256\n\
                       decoding image batch 2/2, n_tokens_batch = oops\n";
        assert_eq!(parse_batch_tokens(garbled, IMAGE_BATCH_LINE), None);
        // A batch marker with no `n_tokens_batch` at all → bail (not skip+sum).
        let missing_marker = "decoding image batch 1/2, n_tokens_batch = 256\n\
                              decoding image batch 2/2 (served from cache)\n";
        assert_eq!(parse_batch_tokens(missing_marker, IMAGE_BATCH_LINE), None);
        // Marker present but empty value → bail.
        assert_eq!(
            parse_batch_tokens(
                "decoding image batch 1/1, n_tokens_batch = \n",
                IMAGE_BATCH_LINE
            ),
            None
        );
    }

    #[test]
    fn parse_batch_tokens_tolerates_trailing_field_and_crlf() {
        // A future appended field parses the count (256), not the extra number
        // and not a fallback.
        assert_eq!(
            parse_batch_tokens(
                "decoding image batch 1/1, n_tokens_batch = 256, extra = 3\n",
                IMAGE_BATCH_LINE
            ),
            Some(256)
        );
        // CRLF: the digit run stops at '\r'.
        assert_eq!(
            parse_batch_tokens(
                "decoding image batch 1/1, n_tokens_batch = 64\r\n",
                IMAGE_BATCH_LINE
            ),
            Some(64)
        );
    }

    #[test]
    fn parse_batch_tokens_bails_on_overflow() {
        // Two in-range batches whose sum overflows u32 → None (never wrapped-small
        // or saturated-huge).
        let overflow = "decoding image batch 1/2, n_tokens_batch = 3000000000\n\
                        decoding image batch 2/2, n_tokens_batch = 2000000000\n";
        assert_eq!(parse_batch_tokens(overflow, IMAGE_BATCH_LINE), None);
        // A single value that doesn't fit u32 → None.
        assert_eq!(
            parse_batch_tokens(
                "decoding image batch 1/1, n_tokens_batch = 5000000000\n",
                IMAGE_BATCH_LINE
            ),
            None
        );
    }

    #[test]
    fn parse_n_tokens_batch_unit_cases() {
        assert_eq!(parse_n_tokens_batch("x n_tokens_batch = 64"), Some(64));
        assert_eq!(parse_n_tokens_batch("n_tokens_batch=64"), Some(64)); // no spaces
        assert_eq!(parse_n_tokens_batch("n_tokens_batch = 64, y = 1"), Some(64));
        assert_eq!(parse_n_tokens_batch("n_tokens_batch = "), None); // marker, no digits
        assert_eq!(parse_n_tokens_batch("n_tokens_batch 64"), None); // no '='
        assert_eq!(parse_n_tokens_batch("nothing here"), None); // no marker
        assert_eq!(parse_n_tokens_batch("n_tokens_batch = -5"), None); // sign → no leading digit
    }

    fn vl_flags(raw: &[&str]) -> Vec<String> {
        raw.iter().map(|s| s.to_string()).collect()
    }

    /// Image-token overrides and the batch sizes reach calibration and its key,
    /// and the placement flags reach the helper alone: a missed count flag means
    /// a wrong cached count and a mis-sized n_ctx.
    #[rstest::rstest]
    #[case::nothing(&[], &[], &[])]
    #[case::unrelated(&["--slots", "--verbose-prompt"], &[], &[])]
    #[case::spaced(&["--image-max-tokens", "64", "--slots"], &["--image-max-tokens", "64"], &[])]
    #[case::equals_and_order(
        &["--image-min-tokens=16", "--image-max-tokens", "256"],
        &["--image-min-tokens", "16", "--image-max-tokens", "256"],
        &[]
    )]
    #[case::batch_sizes(
        &["-ub", "1024", "-b", "256", "--batch-size=128"],
        &["-ub", "1024", "-b", "256", "--batch-size", "128"],
        &[]
    )]
    #[case::devices(
        &["--device", "HTP0", "--no-mmproj-offload", "-mmdev=GPUOpenCL", "--slots"],
        &[],
        &["--device", "HTP0", "--no-mmproj-offload", "-mmdev", "GPUOpenCL"]
    )]
    #[case::other_spellings(
        &["-dev", "CPU", "--mmproj-device", "GPUOpenCL", "--mmproj-offload", "--ubatch-size=256"],
        &["--ubatch-size", "256"],
        &["-dev", "CPU", "--mmproj-device", "GPUOpenCL", "--mmproj-offload"]
    )]
    #[case::offload(
        &["-ot", "exps=CPU", "--tensor-split=1,1", "-nkvo", "--cpu-moe", "-mg", "1"],
        &[],
        &["-ot", "exps=CPU", "--tensor-split", "1,1", "-nkvo", "--cpu-moe", "-mg", "1"]
    )]
    #[case::underscores(
        &["--ubatch_size", "512", "--no_mmproj_offload", "--n_cpu_moe=4"],
        &["--ubatch-size", "512"],
        &["--no-mmproj-offload", "--n-cpu-moe", "4"]
    )]
    fn raw_flags_split_into_count_and_placement(
        #[case] raw: &[&str],
        #[case] count: &[&str],
        #[case] placement: &[&str],
    ) {
        let raw = vl_flags(raw);
        assert_eq!(raw_flags(&raw, &COUNT_FLAGS, &[]), count);
        assert_eq!(
            raw_flags(&raw, &PLACEMENT_FLAGS, &PLACEMENT_SWITCHES),
            placement
        );
    }

    /// The helper is placed like the cell's server, in the server's flag order:
    /// its GPU layers, offload flags and batch sizes reach the helper and
    /// nothing else of `raw` does, and a cell that sets none adds nothing.
    #[rstest::rstest]
    #[case::placed(
        Some(99),
        &["-ub", "1024", "--device", "HTP0", "--slots"],
        &["--device", "HTP0", "--n-gpu-layers", "99", "-ub", "1024"]
    )]
    #[case::repeated_like_the_server(
        None,
        &["-ub", "1024", "--ubatch-size", "256"],
        &["--ubatch-size", "256", "-ub", "1024"]
    )]
    #[case::defaults(None, &[], &[])]
    fn the_helper_takes_the_cells_placement(
        #[case] number_gpu_layers: Option<u32>,
        #[case] raw: &[&str],
        #[case] args: &[&str],
    ) {
        let launch = MtmdCommandBuilder::new(test_llama_server(), "model.gguf", "mmproj.gguf")
            .number_gpu_layers(number_gpu_layers)
            .raw(vl_flags(raw));
        assert_eq!(launch.args(), args);
    }

    /// The fixed flags wrap the cell's, and the image comes last.
    #[test]
    fn the_helper_command_wraps_the_cells_flags() {
        let command = MtmdCommandBuilder::new(test_llama_server(), "model.gguf", "mmproj.gguf")
            .number_gpu_layers(Some(99))
            .build(Path::new("frame.png"), 1);
        assert_eq!(command.get_program(), Path::new(HELPER_NAME));
        let argv: Vec<_> = command.get_args().collect();
        assert_eq!(
            argv,
            [
                "--model",
                "model.gguf",
                "--mmproj",
                "mmproj.gguf",
                "--prompt",
                "hi",
                "--n-predict",
                "1",
                "--n-gpu-layers",
                "99",
                "--jinja",
                "--verbose",
                "--image",
                "frame.png",
            ]
        );
    }

    /// Placement reaches the helper but not the cache key, and the micro-batch
    /// keys it: a cached count is shared across devices, not across batches.
    #[cfg(unix)]
    #[test]
    fn placement_shares_a_cached_count_and_the_micro_batch_does_not() -> anyhow::Result<()> {
        let (launch, dir) = fake_runtime(
            "helper-keyed",
            "echo 'decoding image batch 1/1, n_tokens_batch = 64' >&2",
        )?;
        let inputs = CtxInputs::new(182, 182, 0, 0, 1)?;
        let calibrated = |number_gpu_layers, raw: &[&str]| {
            let launch = launch
                .clone()
                .number_gpu_layers(number_gpu_layers)
                .raw(vl_flags(raw));
            calibrate(&inputs, &launch)
        };
        assert_eq!(calibrated(Some(99), &["--device", "HTP0"]), Some(64));
        std::fs::write(dir.join(HELPER_NAME), "#!/bin/sh\nexit 9\n")?;
        assert_eq!(calibrated(None, &["--no-mmproj-offload"]), Some(64));
        assert_eq!(
            calibrated(Some(99), &["--device", "HTP0", "-ub", "1024"]),
            None
        );
        Ok(())
    }

    /// The no-override key keeps the original format, so existing caches stay valid.
    #[test]
    fn cache_key_separates_image_flag_overrides() {
        let plain = cache_key("1:ab", 512, 512, &[], &[]);
        assert_eq!(plain, "1:ab|512x512"); // legacy format preserved
        let flagged = cache_key(
            "1:ab",
            512,
            512,
            &["--image-max-tokens".into(), "64".into()],
            &[],
        );
        assert_ne!(plain, flagged);
        assert_eq!(flagged, "1:ab|512x512|--image-max-tokens 64");
        // The micro-batch keys the count; the device does not.
        let raw = vl_flags(&["-ub", "1024", "--device", "HTP0"]);
        let batched = cache_key("1:ab", 512, 512, &raw_flags(&raw, &COUNT_FLAGS, &[]), &[]);
        assert_eq!(batched, "1:ab|512x512|-ub 1024");
    }

    #[test]
    fn count_variables_key_the_cache() {
        let env = |name: &str| match name {
            "LLAMA_ARG_UBATCH" => Some("128".to_owned()),
            "LLAMA_ARG_IMAGE_MAX_TOKENS" => Some("64".to_owned()),
            "LLAMA_ARG_THREADS" => Some("8".to_owned()),
            _ => None,
        };
        assert_eq!(
            count_envs(env).as_deref(),
            Some("LLAMA_ARG_IMAGE_MAX_TOKENS=64 LLAMA_ARG_UBATCH=128")
        );
        assert_eq!(count_envs(|_| None), None);
    }

    /// A forwarded variable counts whether or not the helper drops what it
    /// inherits.
    #[cfg(unix)]
    #[test]
    fn the_helpers_own_environment_is_what_counts() -> anyhow::Result<()> {
        let (launch, _dir) = fake_runtime("helper-env", "true")?;
        let forwarded = launch.envs(vec![EnvForward::try_new(
            "LLAMA_ARG_UBATCH=128".to_owned(),
        )?]);
        assert_eq!(
            helper_env(&forwarded, "LLAMA_ARG_UBATCH").as_deref(),
            Some("128")
        );
        assert_eq!(
            helper_env(
                &forwarded.clone().without_inherited_llama_args(),
                "LLAMA_ARG_UBATCH"
            )
            .as_deref(),
            Some("128")
        );
        Ok(())
    }

    /// The APP0 segment exercises segment skipping.
    fn tiny_jpeg(w: u16, h: u16) -> Vec<u8> {
        let mut b = vec![0xFF, 0xD8]; // SOI
        b.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x04, 0x00, 0x00]); // APP0, len=4, 2 payload bytes
        b.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]); // SOF0, len=17, precision=8
        b.extend_from_slice(&h.to_be_bytes());
        b.extend_from_slice(&w.to_be_bytes());
        b.extend_from_slice(&[0x03]); // 3 components (payload not parsed for dims)
        b.extend_from_slice(&[0xFF, 0xD9]); // EOI
        b
    }

    fn data_uri(fmt: &str, bytes: &[u8]) -> String {
        use base64::Engine;
        format!(
            "data:image/{fmt};base64,{}",
            base64::engine::general_purpose::STANDARD.encode(bytes)
        )
    }

    fn text(text: &str) -> ContentPart {
        ContentPart::Text {
            text: text.to_owned(),
        }
    }

    fn image(url: &str) -> ContentPart {
        ContentPart::ImageUrl {
            image_url: ImageUrl {
                url: url.to_owned(),
            },
        }
    }

    fn sample(parts: Vec<ContentPart>) -> EvalSample {
        EvalSample {
            id: "s".to_owned(),
            messages: vec![ChatMessage {
                role: ChatRole::User,
                content: MessageContent::Parts(parts),
            }],
            chat_template_kwargs: None,
        }
    }

    fn sample_with(urls: &[String]) -> EvalSample {
        sample(urls.iter().map(|u| image(u)).chain([text("Q?")]).collect())
    }

    #[test]
    fn image_dims_reads_png_ihdr() -> anyhow::Result<()> {
        let png = pipette_ops::png::gray(320, 200)?;
        assert_eq!(image_dims(&png), Some((320, 200)));
        Ok(())
    }

    #[test]
    fn image_dims_reads_jpeg_sof() {
        assert_eq!(image_dims(&tiny_jpeg(640, 480)), Some((640, 480)));
    }

    #[test]
    fn image_dims_rejects_unknown_format() {
        assert_eq!(image_dims(b"GIF89a....."), None);
    }

    #[test]
    fn scan_max_image_dims_takes_the_largest_and_counts_images() -> anyhow::Result<()> {
        let big_png = data_uri("png", &pipette_ops::png::gray(1024, 768)?);
        let small_jpeg = data_uri("jpeg", &tiny_jpeg(320, 240));
        let two_imgs = data_uri("jpeg", &tiny_jpeg(200, 900));
        let samples = vec![
            sample_with(&[small_jpeg]),
            sample_with(&[big_png]),
            sample_with(&[two_imgs.clone(), two_imgs]),
        ];
        // Each axis is maximized on its own.
        let scan = scan_max_image_dims(&samples).ok_or_else(|| anyhow::anyhow!("must scan"))?;
        assert_eq!((scan.max_w, scan.max_h, scan.max_images), (1024, 900, 2));
        Ok(())
    }

    #[test]
    fn scan_candidates_cover_aspect_extremes_beyond_the_synthetic_max() -> anyhow::Result<()> {
        let wide = data_uri("jpeg", &tiny_jpeg(6000, 500));
        let tall = data_uri("jpeg", &tiny_jpeg(500, 6000));
        let samples = vec![sample_with(&[wide]), sample_with(&[tall])];
        let scan =
            scan_max_image_dims(&samples).ok_or_else(|| anyhow::anyhow!("dims must scan"))?;
        assert_eq!((scan.max_w, scan.max_h), (6000, 6000));
        assert_eq!(scan.candidates[0], (6000, 6000), "synthetic point first");
        assert!(
            scan.candidates.contains(&(6000, 500)),
            "panorama (max aspect) must be a candidate: {:?}",
            scan.candidates
        );
        assert!(
            scan.candidates.contains(&(500, 6000)),
            "tall (min aspect) must be a candidate: {:?}",
            scan.candidates
        );
        assert!(scan.candidates.len() <= 4, "{:?}", scan.candidates);
        Ok(())
    }

    #[test]
    fn scan_candidates_dedup_to_one_for_a_single_shape() -> anyhow::Result<()> {
        let samples = vec![sample_with(&[data_uri("jpeg", &tiny_jpeg(640, 480))])];
        let scan =
            scan_max_image_dims(&samples).ok_or_else(|| anyhow::anyhow!("dims must scan"))?;
        assert_eq!(scan.candidates, vec![(640, 480)]);
        Ok(())
    }

    /// The fake tiler reports 20000 tokens for the 608×48 panorama and 3000 for
    /// everything else, the 608×608 synthetic max included.
    #[cfg(unix)]
    #[test]
    fn vlm_eval_ctx_size_takes_the_max_over_aspect_candidates() -> anyhow::Result<()> {
        let (launch, _dir) = fake_runtime(
            "helper-aspect",
            "case \"$*\" in\n\
             *608x48.png*) echo 'decoding image batch 1/1, n_tokens_batch = 20000' >&2 ;;\n\
             *) echo 'decoding image batch 1/1, n_tokens_batch = 3000' >&2 ;;\n\
             esac",
        )?;
        let wide = data_uri("jpeg", &tiny_jpeg(608, 48));
        let tall = data_uri("jpeg", &tiny_jpeg(48, 608));
        let samples = vec![sample_with(&[wide]), sample_with(&[tall])];
        let (inputs, _) =
            vlm_eval_ctx_inputs(&samples, 64).ok_or_else(|| anyhow::anyhow!("must scan"))?;
        let expect = inputs.ctx_size(Some(20000), &[], 0);
        assert_eq!(
            vlm_eval_ctx_size(&launch, &samples, 64, 1)?,
            Some(expect),
            "n_ctx must be sized from the panorama candidate, not the synthetic max"
        );
        Ok(())
    }

    #[rstest::rstest]
    #[case::ocrbench_synthetic(4961, 14495)]
    #[case::giant_a4(9922, 14032)]
    #[case::wide_panorama(40000, 2000)]
    fn clamp_fits_the_png_bound_and_preserves_aspect(#[case] w: u32, #[case] h: u32) {
        const BOUND: u64 = pipette_ops::png::MAX_PIXELS;
        let (cw, ch) = clamp_to_calibration_bound(w, h);
        assert!(cw >= 1 && ch >= 1);
        let area = u64::from(cw) * u64::from(ch);
        assert!(area <= BOUND, "{w}x{h} -> {cw}x{ch} still over the bound");
        // Not over-shrunk: a sqrt-vs-linear scaling bug would land far below.
        assert!(area > BOUND / 2, "{w}x{h} -> {cw}x{ch} shrunk too far");
        // Aspect preserved within rounding: each axis floors by < 1 unit, so
        // the cross products can differ by at most max(w, h).
        let drift = (u64::from(cw) * u64::from(h)).abs_diff(u64::from(ch) * u64::from(w));
        assert!(
            drift <= u64::from(w.max(h)),
            "aspect drifted: {w}x{h} -> {cw}x{ch}"
        );
    }

    #[test]
    fn clamp_is_identity_under_the_bound() {
        assert_eq!(clamp_to_calibration_bound(4961, 7016), (4961, 7016));
        assert_eq!(clamp_to_calibration_bound(1, 1), (1, 1));
    }

    /// OCRBench's A4 scan and receipt make a 72 MP candidate past the helper's
    /// PNG bound; unclamped, it would fall to a ~366k-token patch grid past the
    /// ceiling.
    #[cfg(unix)]
    #[test]
    fn vlm_eval_ctx_size_clamps_oversized_candidates_and_calibrates() -> anyhow::Result<()> {
        const BOUND: u64 = pipette_ops::png::MAX_PIXELS;
        let (launch, _dir) = fake_runtime(
            "helper-clamp",
            "echo 'decoding image batch 1/1, n_tokens_batch = 3000' >&2",
        )?;
        let a4 = data_uri("jpeg", &tiny_jpeg(4961, 7016));
        let receipt = data_uri("jpeg", &tiny_jpeg(620, 14495));
        let samples = vec![sample_with(&[a4]), sample_with(&[receipt])];
        let (inputs, candidates) =
            vlm_eval_ctx_inputs(&samples, 64).ok_or_else(|| anyhow::anyhow!("must scan"))?;
        assert_eq!(candidates[0], (4961, 14495));
        assert!(u64::from(candidates[0].0) * u64::from(candidates[0].1) > BOUND);
        assert!(inputs.ctx_size(None, &[], 0) > VLM_EVAL_CTX_CEILING);
        let got = vlm_eval_ctx_size(&launch, &samples, 64, 1)?;
        let expect = inputs
            .ctx_size(Some(3000), &[], 0)
            .max(VLM_EVAL_CTX_FLOOR + 64);
        assert_eq!(got, Some(expect));
        assert!(expect <= VLM_EVAL_CTX_CEILING);
        Ok(())
    }

    /// The fake helper exits non-zero below a 100k context, as a projector that
    /// ignores the budget would on OCRBench's 4961x7016 page.
    #[cfg(unix)]
    #[test]
    fn a_budget_the_projector_ignores_is_retried_uncapped() -> anyhow::Result<()> {
        let (launch, _dir) = fake_runtime(
            "helper-ignores-budget",
            r#"ctx=0; prev=""; for a in "$@"; do [ "$prev" = "--ctx-size" ] && ctx=$a; prev=$a; done
[ "$ctx" -ge 100000 ] || exit 1
echo 'decoding image batch 1/1, n_tokens_batch = 3000' >&2"#,
        )?;
        let launch = launch.raw(vec!["--image-max-tokens".into(), "64".into()]);
        let page = data_uri("jpeg", &tiny_jpeg(4961, 7016));
        let samples = vec![sample_with(&[page])];
        let (inputs, _) =
            vlm_eval_ctx_inputs(&samples, 64).ok_or_else(|| anyhow::anyhow!("must scan"))?;
        let got = vlm_eval_ctx_size(&launch, &samples, 64, 1)?;
        let expect = inputs
            .ctx_size(Some(3000), &launch.raw, 0)
            .max(VLM_EVAL_CTX_FLOOR + 64);
        assert_eq!(got, Some(expect), "calibrated at the uncapped retry");
        Ok(())
    }

    #[test]
    fn header_scan_window_is_decoded_bytes_not_b64_chars() -> anyhow::Result<()> {
        let mut jpeg = vec![0xFF, 0xD8]; // SOI
        let payload_len: usize = 7000; // past 6144, inside 8192
        jpeg.extend_from_slice(&[0xFF, 0xFE]); // COM segment
        jpeg.extend_from_slice(&u16::try_from(payload_len + 2)?.to_be_bytes());
        jpeg.extend(std::iter::repeat_n(0u8, payload_len));
        jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]); // SOF0
        jpeg.extend_from_slice(&300u16.to_be_bytes()); // height
        jpeg.extend_from_slice(&500u16.to_be_bytes()); // width
        jpeg.extend_from_slice(&[0x03, 0xFF, 0xD9]);
        let samples = vec![sample_with(&[data_uri("jpeg", &jpeg)])];
        let scan =
            scan_max_image_dims(&samples).ok_or_else(|| anyhow::anyhow!("dims must scan"))?;
        assert_eq!((scan.max_w, scan.max_h, scan.max_images), (500, 300, 1));
        Ok(())
    }

    #[test]
    fn header_scan_retries_the_full_payload_when_sof_is_past_the_prefix() -> anyhow::Result<()> {
        let mut jpeg = vec![0xFF, 0xD8]; // SOI
        let payload_len: usize = 16 * 1024; // APP1 payload, well past the 8 KiB window
        jpeg.extend_from_slice(&[0xFF, 0xE1]); // APP1 (EXIF)
        jpeg.extend_from_slice(&u16::try_from(payload_len + 2)?.to_be_bytes());
        jpeg.extend(std::iter::repeat_n(0u8, payload_len));
        jpeg.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]); // SOF0
        jpeg.extend_from_slice(&300u16.to_be_bytes()); // height
        jpeg.extend_from_slice(&500u16.to_be_bytes()); // width
        jpeg.extend_from_slice(&[0x03, 0xFF, 0xD9]);
        let samples = vec![sample_with(&[data_uri("jpeg", &jpeg)])];
        let scan =
            scan_max_image_dims(&samples).ok_or_else(|| anyhow::anyhow!("dims must scan"))?;
        assert_eq!((scan.max_w, scan.max_h, scan.max_images), (500, 300, 1));
        Ok(())
    }

    #[test]
    fn header_scan_does_not_retry_a_format_it_cannot_read() {
        let mut gif = b"GIF89a".to_vec();
        gif.extend_from_slice(&500u16.to_le_bytes());
        gif.extend_from_slice(&300u16.to_le_bytes());
        gif.extend(std::iter::repeat_n(0u8, 16 * 1024));
        assert_eq!(data_uri_dims(&data_uri("gif", &gif)), None);
    }

    #[test]
    fn vlm_eval_fallback_ctx_size_scans_and_floors_without_the_helper() -> anyhow::Result<()> {
        let uri = data_uri("jpeg", &tiny_jpeg(640, 480));
        let samples = vec![sample_with(&[uri])];
        let (inputs, _) =
            vlm_eval_ctx_inputs(&samples, 64).ok_or_else(|| anyhow::anyhow!("dims must scan"))?;
        let expect = inputs.ctx_size(None, &[], 0).max(VLM_EVAL_CTX_FLOOR + 64);
        assert_eq!(
            vlm_eval_fallback_ctx_size(&samples, 64, &[], 1)?,
            Some(expect)
        );
        // No parseable image anywhere.
        let textless = vec![EvalSample {
            id: "s".to_owned(),
            messages: Vec::new(),
            chat_template_kwargs: None,
        }];
        assert_eq!(vlm_eval_fallback_ctx_size(&textless, 64, &[], 1)?, None);
        Ok(())
    }

    #[rstest]
    // Well under the floor, so the floor decides.
    #[case::floor_is_per_slot_at_one(64, 1, Ok(VLM_EVAL_CTX_FLOOR + 64))]
    #[case::floor_is_per_slot_at_eight(64, 8, Ok(VLM_EVAL_CTX_FLOOR + 64))]
    // 8256 x 15 = 123_840, still inside the 131_072 ceiling.
    #[case::the_product_may_reach_the_ceiling(64, 15, Ok(VLM_EVAL_CTX_FLOOR + 64))]
    // 8256 x 16 = 132_096, past it: refused, though each slot alone is tiny.
    #[case::the_product_is_what_is_refused(64, 16, Err(()))]
    #[case::an_overflowing_product_refuses(64, u32::MAX, Err(()))]
    fn the_ctx_floor_is_per_slot_and_the_ceiling_is_the_total(
        #[case] max_tokens: u32,
        #[case] slots: u32,
        #[case] want: Result<u32, ()>,
    ) -> anyhow::Result<()> {
        let inputs = CtxInputs {
            width: 64,
            height: 64,
            text_tokens: 16,
            decode_tokens: max_tokens,
            num_images: 1,
        };
        match (
            bound_vlm_eval_ctx(1024, &inputs, max_tokens, "test", slots),
            want,
        ) {
            (Ok(got), Ok(expected)) => {
                anyhow::ensure!(got == expected, "got {got}, want {expected}")
            }
            (Err(e), Err(())) => {
                let msg = format!("{e:#}");
                // Both numbers, so the operator can see which to change.
                anyhow::ensure!(
                    msg.contains(&slots.to_string()),
                    "must name the slots: {msg}"
                );
                anyhow::ensure!(
                    msg.contains(&(VLM_EVAL_CTX_FLOOR + max_tokens).to_string()),
                    "must name the per-slot budget: {msg}"
                );
                anyhow::ensure!(msg.contains("parallel"), "must name the knob: {msg}");
            }
            (Ok(got), Err(())) => anyhow::bail!("expected a refusal, got {got}"),
            (Err(e), Ok(expected)) => anyhow::bail!("expected {expected}, got error {e:#}"),
        }
        Ok(())
    }

    /// A JPEG header claiming 65535×65535 gives a ~21.9M-token patch grid.
    #[test]
    fn vlm_eval_ctx_errors_past_the_ceiling_instead_of_clamping() -> anyhow::Result<()> {
        let uri = data_uri("jpeg", &tiny_jpeg(65535, 65535));
        let samples = vec![sample_with(&[uri])];
        let err = match vlm_eval_fallback_ctx_size(&samples, 64, &[], 1) {
            Ok(sized) => anyhow::bail!("an over-ceiling requirement must fail, got {sized:?}"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        let (inputs, _) =
            vlm_eval_ctx_inputs(&samples, 64).ok_or_else(|| anyhow::anyhow!("must scan"))?;
        let computed = inputs.ctx_size(None, &[], 0);
        assert!(
            msg.contains(&computed.to_string()),
            "error must name the computed value {computed}: {msg}"
        );
        assert!(
            msg.contains(&VLM_EVAL_CTX_CEILING.to_string()),
            "error must name the ceiling: {msg}"
        );
        assert!(
            msg.contains("--ctx-size"),
            "error must name the escape hatch: {msg}"
        );
        Ok(())
    }

    #[test]
    fn scan_max_image_dims_none_when_no_images() {
        let samples = vec![sample(vec![text("hi")])];
        assert!(scan_max_image_dims(&samples).is_none());
    }

    #[test]
    fn estimate_text_tokens_is_a_conservative_upper_bound() {
        // 1 token/byte: covers digit-dense OCR text (~1 token per digit on
        // common BPE vocabs), which a 2-bytes/token rate would undercount.
        assert_eq!(estimate_text_tokens(""), 0);
        assert_eq!(estimate_text_tokens("abcd"), 4);
        assert_eq!(estimate_text_tokens("3.14159"), 7);
    }

    #[test]
    fn scan_max_text_tokens_takes_the_largest_sample_and_sums_text_parts() {
        let short = "short";
        let long = "a considerably longer question with several MCQ options A B C D E";
        let samples = vec![
            // Multimodal sample: text part counts, image part does not.
            sample(vec![text(short), image("data:image/png;base64,AAAA")]),
            // Text-heavy sample drives the max.
            sample(vec![text(long)]),
        ];
        assert_eq!(scan_max_text_tokens(&samples), estimate_text_tokens(long));
    }

    #[test]
    fn scan_max_text_tokens_sums_multiple_text_parts_in_one_sample() {
        let samples = vec![sample(vec![
            text("aaaa"),
            image("data:image/png;base64,AAAA"),
            text("bbbb"),
        ])];
        // 4 + 4 bytes at 1 token/byte -> 8 tokens.
        assert_eq!(scan_max_text_tokens(&samples), 8);
    }
}
