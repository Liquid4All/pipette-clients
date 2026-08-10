//! Typed runtime flags for a plan's cells.
//!
//! Design (stable):
//! - One [`RuntimeFlags`] variant per runnable `(benchmark, runtime_type, model)`
//!   cell. The variant name encodes that triple; knobs are fields on the
//!   variant (absent field ⇒ not accepted).
//! - Authored flat as [`RuntimeFlagRef`]; `TryFrom` routes by the triple and
//!   rejects knobs the target variant does not carry. Unknown triples are
//!   [`RuntimeFlagError::NoSuchCombination`].
//! - Knob values are logical, not tool argv. Clients map them; plan-types does
//!   not own flag spellings. Optional `raw` is an argv escape hatch only where
//!   a cell has that surface, and is denylisted against typed/reserved names.
//! - Host/transport resolution (binaries, paths, ssh/adb) is out of scope here.

use serde::{Deserialize, Serialize};

use crate::{
    BenchmarkType, Model, ModelType, NonEmptyString, OpenvinoDevice, Runtime, RuntimeType,
};

/// Error resolving a [`RuntimeFlagRef`] into a typed [`RuntimeFlags`].
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum RuntimeFlagError {
    /// No variant for this `(benchmark, runtime, model)` triple.
    #[error("no runtime flags defined for {benchmark} \u{d7} {runtime:?} \u{d7} {model:?}")]
    NoSuchCombination {
        benchmark: BenchmarkType,
        runtime: RuntimeType,
        model: ModelType,
    },
    /// A typed knob was set that this cell doesn't accept.
    #[error("knob `{knob}` is not accepted by {benchmark} \u{d7} {runtime:?} \u{d7} {model:?}")]
    KnobNotAllowed {
        knob: &'static str,
        benchmark: BenchmarkType,
        runtime: RuntimeType,
        model: ModelType,
    },
    /// A `raw` entry uses the spelling of a knob this cell types — the author
    /// should set the typed field instead.
    #[error("raw flag {flag:?} aliases a typed knob for this cell; set the typed field instead")]
    RawAliasesTypedKnob { flag: String },
    /// A `raw` entry uses a flag the benchmark/tool fixes internally.
    #[error("raw flag {flag:?} is reserved by the benchmark/tool for this cell")]
    RawReservedFlag { flag: String },
}

/// OpenVINO's NPU compile/performance tradeoff (`GENERATE_HINT`).
///
/// Closed because the property is: OpenVINO accepts exactly `BEST_PERF` and
/// `FAST_COMPILE` and rejects anything else, **case-sensitively** — a plan
/// writing `best_perf`, which the lowercase spelling of every other value in
/// these files invites, is refused by the plugin. Modelling it as free text
/// would move that failure from plan parsing to a device compile, after the
/// weights had downloaded and the venv had been built.
///
/// Wire form is kebab-case (`best-perf`) to match the rest of the plan
/// vocabulary; [`Self::as_property`] renders what OpenVINO wants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum OpenvinoGenerateHint {
    BestPerf,
    FastCompile,
}

impl OpenvinoGenerateHint {
    /// The exact `GENERATE_HINT` value OpenVINO accepts.
    pub fn as_property(self) -> &'static str {
        match self {
            Self::BestPerf => "BEST_PERF",
            Self::FastCompile => "FAST_COMPILE",
        }
    }
}

/// llama.cpp flash-attention mode. `auto` lets llama.cpp decide per build/model;
/// `on`/`off` force it. Wire form is the lowercase word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum LlamacppFlashAttention {
    Auto,
    On,
    Off,
}

impl LlamacppFlashAttention {
    /// The wire word, shared by the argv renderers and the result record.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::On => "on",
            Self::Off => "off",
        }
    }
}

// ---------------------------------------------------------------------------
// raw-flag denylists
//
// A `raw` escape-hatch entry may not smuggle in a flag we already type
// (aliases below) or one the benchmark/tool owns (reserved sets). Each cell's
// denylist is `family typed aliases ++ benchmark-reserved`, composed from the
// shared `const` pieces so a name is written once.
// ---------------------------------------------------------------------------

/// Concatenate two `&'static [&'static str]` consts into one, at compile time.
macro_rules! concat_str_slices {
    ($a:expr, $b:expr) => {{
        const A: &[&str] = $a;
        const B: &[&str] = $b;
        const LEN: usize = A.len() + B.len();
        const OUT: [&str; LEN] = {
            let mut out = [""; LEN];
            let mut i = 0;
            while i < A.len() {
                out[i] = A[i];
                i += 1;
            }
            let mut j = 0;
            while j < B.len() {
                out[A.len() + j] = B[j];
                j += 1;
            }
            out
        };
        &OUT
    }};
}

/// Long/short spellings of the llama.cpp knobs we type. A `raw` entry using any
/// of these must go through the typed field instead.
const LLAMA_TYPED_ALIASES: &[&str] = &[
    "-t",
    "--threads",
    "-ngl",
    "--n-gpu-layers",
    "--gpu-layers",
    "--mmap",
    "--no-mmap",
    "-fa",
    "--flash-attn",
];
/// vLLM typed-knob aliases (see [`LLAMA_TYPED_ALIASES`]). Both prefix-cache
/// spellings are here because the cell types the setting, not the flag: the
/// client renders whichever spelling `prefix_caching` calls for.
const VLLM_TYPED_ALIASES: &[&str] = &[
    "--tensor-parallel-size",
    "-tp",
    "--dtype",
    "--max-model-len",
    "--enable-prefix-caching",
    "--no-enable-prefix-caching",
];
/// sglang typed-knob aliases (see [`LLAMA_TYPED_ALIASES`]). The radix cache is
/// sglang's prefix cache — on by default, `--disable-radix-cache` the lever.
const SGLANG_TYPED_ALIASES: &[&str] = &["--tensor-parallel-size", "-tp", "--disable-radix-cache"];

/// Flags each runtime's benchmarks fix internally — the single source of truth
/// for that policy. The executor crates consume these to reject operator
/// overrides in `--runtime-flags`, and plan-types folds them into its `raw`
/// denylists. They live here (not next to the executor code) because those
/// crates depend on plan-types, not the reverse. One submodule per runtime type.
pub mod reserved_flags {
    /// Flags the llama.cpp stock CLI tools (`llama-bench` / `llama-server`) fix
    /// per benchmark. Only the `llamacpp_cli_stock_tools` runtime uses them —
    /// the apk/iOS llama runtimes run in-process, not through these binaries.
    pub mod llamacpp_cli_stock_tools {
        /// Flags every llama-bench benchmark fixes: output format, model, the
        /// token counts that define the workload, and the repetition count —
        /// all set by the harness for cross-run comparability.
        ///
        /// The depth belongs here rather than on the decode list alone. It
        /// feeds llama-bench's context sizing (`n_ctx = n_prompt + n_gen +
        /// n_depth`) on any bench cell, and a prefill cell selects its result
        /// row on the prompt and gen counts alone — so a `-d` override there
        /// measures prefill against a deep KV cache and is recorded as an
        /// ordinary prefill, with nothing in the row to give it away.
        const BENCH: &[&str] = &[
            "--output",
            "-o",
            "--model",
            "-m",
            "--n-prompt",
            "-p",
            "--n-gen",
            "-n",
            "--n-depth",
            "-d",
            "--repetitions",
            "-r",
        ];
        /// The bench flags. Prefill leaves the depth at llama-bench's zero
        /// default, which is the measurement it names.
        pub const PREFILL: &[&str] = BENCH;
        /// The bench flags. Decode passes `--n-depth` itself, from the cell's
        /// prefill parameter.
        pub const DECODE: &[&str] = BENCH;
        /// The bench flags plus the context size, to which peak memory is
        /// sensitive. This llama-bench has no `--ctx-size` option at all —
        /// reserving it turns an unknown-argument exit deep inside the tool
        /// into a refusal at plan time. What actually sizes the context is the
        /// token counts, every one of which is reserved above.
        pub const MAX_MEMORY: &[&str] = concat_str_slices!(BENCH, &["--ctx-size", "-c"]);
        /// Flags llama-server fixes for the latency/eval/VL benchmarks.
        pub const SERVER: &[&str] = &[
            "--model",
            "-m",
            "--mmproj",
            "--host",
            "--port",
            "--no-warmup",
        ];
    }
}

/// llama-server knobs we now type (`ctx_size`, `no_cache`). Server-only, so they
/// belong on the server denylist but not the bench ones.
const LLAMA_SERVER_TYPED_ALIASES: &[&str] = &["-c", "--ctx-size", "--no-cache-prompt"];
/// vLLM launch flags the runtime fixes itself.
const VLLM_RESERVED: &[&str] = &["--model", "--host", "--port"];
/// sglang launch flags the runtime fixes itself.
const SGLANG_RESERVED: &[&str] = &["--model-path", "--host", "--port"];

const DENY_LLAMA_PREFILL: &[&str] = concat_str_slices!(
    LLAMA_TYPED_ALIASES,
    reserved_flags::llamacpp_cli_stock_tools::PREFILL
);
const DENY_LLAMA_DECODE: &[&str] = concat_str_slices!(
    LLAMA_TYPED_ALIASES,
    reserved_flags::llamacpp_cli_stock_tools::DECODE
);
const DENY_LLAMA_MAXMEM: &[&str] = concat_str_slices!(
    LLAMA_TYPED_ALIASES,
    reserved_flags::llamacpp_cli_stock_tools::MAX_MEMORY
);
const DENY_LLAMA_SERVER: &[&str] = concat_str_slices!(
    concat_str_slices!(
        LLAMA_TYPED_ALIASES,
        reserved_flags::llamacpp_cli_stock_tools::SERVER
    ),
    LLAMA_SERVER_TYPED_ALIASES
);
const DENY_VLLM: &[&str] = concat_str_slices!(VLLM_TYPED_ALIASES, VLLM_RESERVED);
const DENY_SGLANG: &[&str] = concat_str_slices!(SGLANG_TYPED_ALIASES, SGLANG_RESERVED);

// ---------------------------------------------------------------------------
// RuntimeFlags — one variant per concrete cell.
// ---------------------------------------------------------------------------

/// Runtime flags for one cell. See the module docs for the closed-enum / flat-wire
/// design; variants and fields are the inventory.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Deserialize, Serialize, strum::EnumCount)]
#[serde(try_from = "RuntimeFlagRef", into = "RuntimeFlagRef")]
pub enum RuntimeFlags {
    // llama.cpp CLI (stock tools) × gguf-text — bench cells (llama-bench).
    PrefillLlamacppCliStockToolsGgufText {
        threads: Option<u32>,
        number_gpu_layers: Option<u32>,
        mmap: Option<bool>,
        flash_attention: Option<LlamacppFlashAttention>,
        raw: Vec<String>,
    },
    DecodeLlamacppCliStockToolsGgufText {
        threads: Option<u32>,
        number_gpu_layers: Option<u32>,
        mmap: Option<bool>,
        flash_attention: Option<LlamacppFlashAttention>,
        raw: Vec<String>,
    },
    MaxMemoryLlamacppCliStockToolsGgufText {
        threads: Option<u32>,
        number_gpu_layers: Option<u32>,
        mmap: Option<bool>,
        flash_attention: Option<LlamacppFlashAttention>,
        raw: Vec<String>,
    },
    // llama.cpp CLI × gguf-text — server cells (llama-server): + ctx_size/no_cache.
    EndToEndLlamacppCliStockToolsGgufText {
        threads: Option<u32>,
        number_gpu_layers: Option<u32>,
        mmap: Option<bool>,
        flash_attention: Option<LlamacppFlashAttention>,
        ctx_size: Option<u32>,
        no_cache: Option<bool>,
        raw: Vec<String>,
    },
    EvalLlamacppCliStockToolsGgufText {
        threads: Option<u32>,
        number_gpu_layers: Option<u32>,
        mmap: Option<bool>,
        flash_attention: Option<LlamacppFlashAttention>,
        ctx_size: Option<u32>,
        no_cache: Option<bool>,
        raw: Vec<String>,
    },
    // llama.cpp CLI × gguf-vision — server cell.
    VlLlamacppCliStockToolsGgufVision {
        threads: Option<u32>,
        number_gpu_layers: Option<u32>,
        mmap: Option<bool>,
        flash_attention: Option<LlamacppFlashAttention>,
        ctx_size: Option<u32>,
        no_cache: Option<bool>,
        raw: Vec<String>,
    },
    // Docker vLLM: server flags (tensor_parallel_size + dtype + the context and
    // prefix-cache settings the client derives) plus the `docker run` launcher
    // settings (gpus/shm_size/ipc) and env forwards.
    EndToEndDockerVllmTorch {
        tensor_parallel_size: Option<u32>,
        dtype: Option<NonEmptyString>,
        max_model_len: Option<u32>,
        prefix_caching: Option<bool>,
        gpus: Option<String>,
        shm_size: Option<String>,
        ipc: Option<String>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    EvalDockerVllmTorch {
        tensor_parallel_size: Option<u32>,
        dtype: Option<NonEmptyString>,
        max_model_len: Option<u32>,
        prefix_caching: Option<bool>,
        gpus: Option<String>,
        shm_size: Option<String>,
        ipc: Option<String>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    MaxMemoryDockerVllmTorch {
        tensor_parallel_size: Option<u32>,
        dtype: Option<NonEmptyString>,
        max_model_len: Option<u32>,
        prefix_caching: Option<bool>,
        gpus: Option<String>,
        shm_size: Option<String>,
        ipc: Option<String>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    // Uv vLLM: server flags + env forwards (no `docker run` launcher settings).
    EndToEndUvVllmTorch {
        tensor_parallel_size: Option<u32>,
        dtype: Option<NonEmptyString>,
        max_model_len: Option<u32>,
        prefix_caching: Option<bool>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    EvalUvVllmTorch {
        tensor_parallel_size: Option<u32>,
        dtype: Option<NonEmptyString>,
        max_model_len: Option<u32>,
        prefix_caching: Option<bool>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    MaxMemoryUvVllmTorch {
        tensor_parallel_size: Option<u32>,
        dtype: Option<NonEmptyString>,
        max_model_len: Option<u32>,
        prefix_caching: Option<bool>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    // Docker SGLang: tensor_parallel_size + prefix caching (the radix cache) +
    // the `docker run` launcher settings. Context length is left to the operator
    // (`--context-length`), so there's no `max_model_len` here.
    EndToEndDockerSglangTorch {
        tensor_parallel_size: Option<u32>,
        prefix_caching: Option<bool>,
        gpus: Option<String>,
        shm_size: Option<String>,
        ipc: Option<String>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    EvalDockerSglangTorch {
        tensor_parallel_size: Option<u32>,
        prefix_caching: Option<bool>,
        gpus: Option<String>,
        shm_size: Option<String>,
        ipc: Option<String>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    MaxMemoryDockerSglangTorch {
        tensor_parallel_size: Option<u32>,
        prefix_caching: Option<bool>,
        gpus: Option<String>,
        shm_size: Option<String>,
        ipc: Option<String>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    // Uv SGLang: tensor_parallel_size + prefix caching + env forwards.
    EndToEndUvSglangTorch {
        tensor_parallel_size: Option<u32>,
        prefix_caching: Option<bool>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    EvalUvSglangTorch {
        tensor_parallel_size: Option<u32>,
        prefix_caching: Option<bool>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    MaxMemoryUvSglangTorch {
        tensor_parallel_size: Option<u32>,
        prefix_caching: Option<bool>,
        envs: Vec<String>,
        raw: Vec<String>,
    },
    // iOS llama.cpp (in-process Metal) × gguf-text. Knobs match the app load path
    // (ngl/ctx/ubatch/threads/swa). No `raw`: there is no CLI argv to splice onto.
    //
    // `threads` is llama's `n_threads`/`n_threads_batch`, which the app otherwise derives
    // from the P-core count. It matters even at `number_gpu_layers = 99`: sampling and any
    // op ggml keeps on the CPU run there, and a cell offloading fewer layers runs most of
    // the model on it.
    //
    // `swa_full` is llama's sliding-window cache policy. On a SWA model it decides whether
    // the windowed layers allocate KV for the whole context or for the window alone, so two
    // runs agreeing on `ctx_size` can still allocate very differently. The stock CLI tools
    // pin it off and the library defaults it on, which is why the cell states it rather than
    // inheriting: an unstated value here is a memory result no one can account for.
    PrefillLlamacppIosPipetteGgufText {
        number_gpu_layers: Option<u32>,
        ctx_size: Option<u32>,
        n_ubatch: Option<u32>,
        threads: Option<u32>,
        swa_full: Option<bool>,
    },
    DecodeLlamacppIosPipetteGgufText {
        number_gpu_layers: Option<u32>,
        ctx_size: Option<u32>,
        n_ubatch: Option<u32>,
        threads: Option<u32>,
        swa_full: Option<bool>,
    },
    MaxMemoryLlamacppIosPipetteGgufText {
        number_gpu_layers: Option<u32>,
        ctx_size: Option<u32>,
        n_ubatch: Option<u32>,
        threads: Option<u32>,
        swa_full: Option<bool>,
    },
    EndToEndLlamacppIosPipetteGgufText {
        number_gpu_layers: Option<u32>,
        ctx_size: Option<u32>,
        n_ubatch: Option<u32>,
        threads: Option<u32>,
        swa_full: Option<bool>,
    },
    EvalLlamacppIosPipetteGgufText {
        number_gpu_layers: Option<u32>,
        ctx_size: Option<u32>,
        n_ubatch: Option<u32>,
        threads: Option<u32>,
        swa_full: Option<bool>,
    },
    VlLlamacppIosPipetteGgufVision {
        number_gpu_layers: Option<u32>,
        ctx_size: Option<u32>,
        n_ubatch: Option<u32>,
        threads: Option<u32>,
        swa_full: Option<bool>,
    },
    // iOS MLX × mlx — only prefill-chunk / ubatch; no raw argv surface.
    PrefillMlxIosPipetteMlx {
        n_ubatch: Option<u32>,
    },
    DecodeMlxIosPipetteMlx {
        n_ubatch: Option<u32>,
    },
    MaxMemoryMlxIosPipetteMlx {
        n_ubatch: Option<u32>,
    },
    EndToEndMlxIosPipetteMlx {
        n_ubatch: Option<u32>,
    },
    EvalMlxIosPipetteMlx {
        n_ubatch: Option<u32>,
    },
    // OpenVINO x IR. No `raw`: the runtime takes typed `LLMPipeline`
    // properties, not a command line, so there is no escape hatch to validate.
    // `min_response_len` is the one that bites — GenAI reserves 128 output
    // tokens by default, which truncates a 256-token cell on NPU.
    //
    // `device` lives here rather than on the runtime because it is substituted
    // at the call, not at the install: one wheel serves CPU, GPU and NPU, so it
    // configures a cell instead of identifying an artifact. It is `Option` only
    // because every flag field is — an OpenVINO cell that names no device is
    // refused rather than defaulted, since OpenVINO would silently pick CPU.
    PrefillUvOpenvinoOpenvino {
        device: Option<OpenvinoDevice>,
        max_prompt_len: Option<u32>,
        min_response_len: Option<u32>,
        generate_hint: Option<OpenvinoGenerateHint>,
    },
    DecodeUvOpenvinoOpenvino {
        device: Option<OpenvinoDevice>,
        max_prompt_len: Option<u32>,
        min_response_len: Option<u32>,
        generate_hint: Option<OpenvinoGenerateHint>,
    },
    EndToEndUvOpenvinoOpenvino {
        device: Option<OpenvinoDevice>,
        max_prompt_len: Option<u32>,
        min_response_len: Option<u32>,
        generate_hint: Option<OpenvinoGenerateHint>,
    },
    MaxMemoryUvOpenvinoOpenvino {
        device: Option<OpenvinoDevice>,
        max_prompt_len: Option<u32>,
        min_response_len: Option<u32>,
        generate_hint: Option<OpenvinoGenerateHint>,
    },
}

impl RuntimeFlags {
    /// The `(benchmark, runtime, model)` triple this variant encodes — its cell
    /// identity, used to match cells and reject duplicate entries in a variant.
    pub fn axes(&self) -> (BenchmarkType, RuntimeType, ModelType) {
        use BenchmarkType as B;
        use ModelType as M;
        use RuntimeType as R;
        match self {
            RuntimeFlags::PrefillLlamacppCliStockToolsGgufText { .. } => {
                (B::PrefillThroughput, R::LlamacppCliStockTools, M::GgufText)
            }
            RuntimeFlags::DecodeLlamacppCliStockToolsGgufText { .. } => {
                (B::DecodeThroughput, R::LlamacppCliStockTools, M::GgufText)
            }
            RuntimeFlags::MaxMemoryLlamacppCliStockToolsGgufText { .. } => {
                (B::MaxMemoryUsage, R::LlamacppCliStockTools, M::GgufText)
            }
            RuntimeFlags::EndToEndLlamacppCliStockToolsGgufText { .. } => {
                (B::EndToEndLatency, R::LlamacppCliStockTools, M::GgufText)
            }
            RuntimeFlags::EvalLlamacppCliStockToolsGgufText { .. } => {
                (B::Eval, R::LlamacppCliStockTools, M::GgufText)
            }
            RuntimeFlags::VlLlamacppCliStockToolsGgufVision { .. } => {
                (B::VlThroughput, R::LlamacppCliStockTools, M::GgufVision)
            }
            RuntimeFlags::PrefillUvOpenvinoOpenvino { .. } => {
                (B::PrefillThroughput, R::UvOpenvino, M::Openvino)
            }
            RuntimeFlags::DecodeUvOpenvinoOpenvino { .. } => {
                (B::DecodeThroughput, R::UvOpenvino, M::Openvino)
            }
            RuntimeFlags::EndToEndUvOpenvinoOpenvino { .. } => {
                (B::EndToEndLatency, R::UvOpenvino, M::Openvino)
            }
            RuntimeFlags::MaxMemoryUvOpenvinoOpenvino { .. } => {
                (B::MaxMemoryUsage, R::UvOpenvino, M::Openvino)
            }
            RuntimeFlags::EndToEndDockerVllmTorch { .. } => {
                (B::EndToEndLatency, R::DockerVllm, M::Torch)
            }
            RuntimeFlags::EvalDockerVllmTorch { .. } => (B::Eval, R::DockerVllm, M::Torch),
            RuntimeFlags::MaxMemoryDockerVllmTorch { .. } => {
                (B::MaxMemoryUsage, R::DockerVllm, M::Torch)
            }
            RuntimeFlags::EndToEndUvVllmTorch { .. } => (B::EndToEndLatency, R::UvVllm, M::Torch),
            RuntimeFlags::EvalUvVllmTorch { .. } => (B::Eval, R::UvVllm, M::Torch),
            RuntimeFlags::MaxMemoryUvVllmTorch { .. } => (B::MaxMemoryUsage, R::UvVllm, M::Torch),
            RuntimeFlags::EndToEndDockerSglangTorch { .. } => {
                (B::EndToEndLatency, R::DockerSglang, M::Torch)
            }
            RuntimeFlags::EvalDockerSglangTorch { .. } => (B::Eval, R::DockerSglang, M::Torch),
            RuntimeFlags::MaxMemoryDockerSglangTorch { .. } => {
                (B::MaxMemoryUsage, R::DockerSglang, M::Torch)
            }
            RuntimeFlags::EndToEndUvSglangTorch { .. } => {
                (B::EndToEndLatency, R::UvSglang, M::Torch)
            }
            RuntimeFlags::EvalUvSglangTorch { .. } => (B::Eval, R::UvSglang, M::Torch),
            RuntimeFlags::MaxMemoryUvSglangTorch { .. } => {
                (B::MaxMemoryUsage, R::UvSglang, M::Torch)
            }
            RuntimeFlags::PrefillLlamacppIosPipetteGgufText { .. } => {
                (B::PrefillThroughput, R::LlamacppIosPipette, M::GgufText)
            }
            RuntimeFlags::DecodeLlamacppIosPipetteGgufText { .. } => {
                (B::DecodeThroughput, R::LlamacppIosPipette, M::GgufText)
            }
            RuntimeFlags::MaxMemoryLlamacppIosPipetteGgufText { .. } => {
                (B::MaxMemoryUsage, R::LlamacppIosPipette, M::GgufText)
            }
            RuntimeFlags::EndToEndLlamacppIosPipetteGgufText { .. } => {
                (B::EndToEndLatency, R::LlamacppIosPipette, M::GgufText)
            }
            RuntimeFlags::EvalLlamacppIosPipetteGgufText { .. } => {
                (B::Eval, R::LlamacppIosPipette, M::GgufText)
            }
            RuntimeFlags::VlLlamacppIosPipetteGgufVision { .. } => {
                (B::VlThroughput, R::LlamacppIosPipette, M::GgufVision)
            }
            RuntimeFlags::PrefillMlxIosPipetteMlx { .. } => {
                (B::PrefillThroughput, R::MlxIosPipette, M::Mlx)
            }
            RuntimeFlags::DecodeMlxIosPipetteMlx { .. } => {
                (B::DecodeThroughput, R::MlxIosPipette, M::Mlx)
            }
            RuntimeFlags::MaxMemoryMlxIosPipetteMlx { .. } => {
                (B::MaxMemoryUsage, R::MlxIosPipette, M::Mlx)
            }
            RuntimeFlags::EndToEndMlxIosPipetteMlx { .. } => {
                (B::EndToEndLatency, R::MlxIosPipette, M::Mlx)
            }
            RuntimeFlags::EvalMlxIosPipetteMlx { .. } => (B::Eval, R::MlxIosPipette, M::Mlx),
        }
    }

    /// True iff this entry applies to a cell running `benchmark` on `runtime`
    /// with `model` — all three axes match. Plan validation rejects an entry
    /// that matches no cell in its variant.
    pub fn matches(&self, benchmark: BenchmarkType, runtime: &Runtime, model: &Model) -> bool {
        self.axes() == (benchmark, RuntimeType::of(runtime), ModelType::of(model))
    }

    /// This variant's own settings, as a result submission carries them: a JSON
    /// object of the fields it sets, without the `(runtime, model, benchmark)`
    /// axis keys — the payload already names the cell through its descriptors
    /// and benchmark id, so repeating them here would be a second source of
    /// truth to keep honest.
    ///
    /// Env forwards keep their names and lose their values: a forward may hold
    /// a token and this record leaves the device. `raw` rides verbatim — it is
    /// operator-authored argv, already on the launch command line — so a secret
    /// belongs in the bare-`K` `envs` inherit form, never in `raw`.
    ///
    /// The engines report the flags a cell *ran with* (plan entry plus whatever
    /// the run overlaid), so this records the launch, not the plan. Flags a
    /// benchmark fixes for every run (llama.cpp's `-r 1` and `--no-warmup`)
    /// can't appear — [`RuntimeFlags`] carries no reserved flag, by design.
    /// They stay in that engine's recorded argv (`RunResponse::command`).
    pub fn submission_value(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self.without_env_values())
            // `RuntimeFlags` serializes infallibly (no maps, floats, or fallible
            // `Serialize` impls), so this is the impossible branch. Fall closed
            // on an empty object rather than dropping the record silently.
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
        if let Some(fields) = value.as_object_mut() {
            AXIS_KEYS.iter().for_each(|key| {
                fields.remove(*key);
            });
        }
        value
    }

    /// These flags with env *values* stripped to their names. Mirrors
    /// [`Model::without_auth_token`](crate::Model::without_auth_token); see
    /// [`Self::submission_value`] for why the record can't carry them.
    fn without_env_values(&self) -> Self {
        let mut out = self.clone();
        out.envs_mut()
            .iter_mut()
            .for_each(|e| *e = e.split('=').next().unwrap_or(e).to_string());
        out
    }

    /// The `docker run --gpus` value this cell sets, if any. Only the docker
    /// cells carry it; `None` for uv/llama.cpp cells (which have no such field).
    pub fn gpus(&self) -> Option<&str> {
        match self {
            RuntimeFlags::EndToEndDockerVllmTorch { gpus, .. }
            | RuntimeFlags::EvalDockerVllmTorch { gpus, .. }
            | RuntimeFlags::MaxMemoryDockerVllmTorch { gpus, .. }
            | RuntimeFlags::EndToEndDockerSglangTorch { gpus, .. }
            | RuntimeFlags::EvalDockerSglangTorch { gpus, .. }
            | RuntimeFlags::MaxMemoryDockerSglangTorch { gpus, .. } => gpus.as_deref(),
            _ => None,
        }
    }

    /// The `docker run --shm-size` value this cell sets, if any (docker only).
    pub fn shm_size(&self) -> Option<&str> {
        match self {
            RuntimeFlags::EndToEndDockerVllmTorch { shm_size, .. }
            | RuntimeFlags::EvalDockerVllmTorch { shm_size, .. }
            | RuntimeFlags::MaxMemoryDockerVllmTorch { shm_size, .. }
            | RuntimeFlags::EndToEndDockerSglangTorch { shm_size, .. }
            | RuntimeFlags::EvalDockerSglangTorch { shm_size, .. }
            | RuntimeFlags::MaxMemoryDockerSglangTorch { shm_size, .. } => shm_size.as_deref(),
            _ => None,
        }
    }

    /// The `docker run --ipc` value this cell sets, if any (docker only).
    pub fn ipc(&self) -> Option<&str> {
        match self {
            RuntimeFlags::EndToEndDockerVllmTorch { ipc, .. }
            | RuntimeFlags::EvalDockerVllmTorch { ipc, .. }
            | RuntimeFlags::MaxMemoryDockerVllmTorch { ipc, .. }
            | RuntimeFlags::EndToEndDockerSglangTorch { ipc, .. }
            | RuntimeFlags::EvalDockerSglangTorch { ipc, .. }
            | RuntimeFlags::MaxMemoryDockerSglangTorch { ipc, .. } => ipc.as_deref(),
            _ => None,
        }
    }

    /// Env forwards (`K=V`, or bare `K` to inherit) this cell sets — docker and
    /// uv cells carry them; empty for llama.cpp cells.
    pub fn envs(&self) -> &[String] {
        match self {
            RuntimeFlags::EndToEndDockerVllmTorch { envs, .. }
            | RuntimeFlags::EvalDockerVllmTorch { envs, .. }
            | RuntimeFlags::MaxMemoryDockerVllmTorch { envs, .. }
            | RuntimeFlags::EndToEndUvVllmTorch { envs, .. }
            | RuntimeFlags::EvalUvVllmTorch { envs, .. }
            | RuntimeFlags::MaxMemoryUvVllmTorch { envs, .. }
            | RuntimeFlags::EndToEndDockerSglangTorch { envs, .. }
            | RuntimeFlags::EvalDockerSglangTorch { envs, .. }
            | RuntimeFlags::MaxMemoryDockerSglangTorch { envs, .. }
            | RuntimeFlags::EndToEndUvSglangTorch { envs, .. }
            | RuntimeFlags::EvalUvSglangTorch { envs, .. }
            | RuntimeFlags::MaxMemoryUvSglangTorch { envs, .. } => envs,
            _ => &[],
        }
    }

    /// [`Self::envs`] for in-place edits — see [`Self::without_env_values`].
    fn envs_mut(&mut self) -> &mut [String] {
        match self {
            RuntimeFlags::EndToEndDockerVllmTorch { envs, .. }
            | RuntimeFlags::EvalDockerVllmTorch { envs, .. }
            | RuntimeFlags::MaxMemoryDockerVllmTorch { envs, .. }
            | RuntimeFlags::EndToEndUvVllmTorch { envs, .. }
            | RuntimeFlags::EvalUvVllmTorch { envs, .. }
            | RuntimeFlags::MaxMemoryUvVllmTorch { envs, .. }
            | RuntimeFlags::EndToEndDockerSglangTorch { envs, .. }
            | RuntimeFlags::EvalDockerSglangTorch { envs, .. }
            | RuntimeFlags::MaxMemoryDockerSglangTorch { envs, .. }
            | RuntimeFlags::EndToEndUvSglangTorch { envs, .. }
            | RuntimeFlags::EvalUvSglangTorch { envs, .. }
            | RuntimeFlags::MaxMemoryUvSglangTorch { envs, .. } => envs,
            _ => &mut [],
        }
    }

    /// The `raw` escape-hatch entries authored on this cell.
    fn raw(&self) -> &[String] {
        match self {
            RuntimeFlags::PrefillLlamacppCliStockToolsGgufText { raw, .. }
            | RuntimeFlags::DecodeLlamacppCliStockToolsGgufText { raw, .. }
            | RuntimeFlags::MaxMemoryLlamacppCliStockToolsGgufText { raw, .. }
            | RuntimeFlags::EndToEndLlamacppCliStockToolsGgufText { raw, .. }
            | RuntimeFlags::EvalLlamacppCliStockToolsGgufText { raw, .. }
            | RuntimeFlags::VlLlamacppCliStockToolsGgufVision { raw, .. }
            | RuntimeFlags::EndToEndDockerVllmTorch { raw, .. }
            | RuntimeFlags::EvalDockerVllmTorch { raw, .. }
            | RuntimeFlags::MaxMemoryDockerVllmTorch { raw, .. }
            | RuntimeFlags::EndToEndUvVllmTorch { raw, .. }
            | RuntimeFlags::EvalUvVllmTorch { raw, .. }
            | RuntimeFlags::MaxMemoryUvVllmTorch { raw, .. }
            | RuntimeFlags::EndToEndDockerSglangTorch { raw, .. }
            | RuntimeFlags::EvalDockerSglangTorch { raw, .. }
            | RuntimeFlags::MaxMemoryDockerSglangTorch { raw, .. }
            | RuntimeFlags::EndToEndUvSglangTorch { raw, .. }
            | RuntimeFlags::EvalUvSglangTorch { raw, .. }
            | RuntimeFlags::MaxMemoryUvSglangTorch { raw, .. } => raw,
            // OpenVINO takes typed pipeline properties, not a command line, so
            // there is no argv escape hatch either.
            RuntimeFlags::PrefillUvOpenvinoOpenvino { .. }
            | RuntimeFlags::DecodeUvOpenvinoOpenvino { .. }
            | RuntimeFlags::EndToEndUvOpenvinoOpenvino { .. }
            | RuntimeFlags::MaxMemoryUvOpenvinoOpenvino { .. } => &[],
            // iOS engines are in-process — no argv escape hatch.
            RuntimeFlags::PrefillLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::DecodeLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::MaxMemoryLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::EndToEndLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::EvalLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::VlLlamacppIosPipetteGgufVision { .. }
            | RuntimeFlags::PrefillMlxIosPipetteMlx { .. }
            | RuntimeFlags::DecodeMlxIosPipetteMlx { .. }
            | RuntimeFlags::MaxMemoryMlxIosPipetteMlx { .. }
            | RuntimeFlags::EndToEndMlxIosPipetteMlx { .. }
            | RuntimeFlags::EvalMlxIosPipetteMlx { .. } => &[],
        }
    }

    /// Flag names a `raw` entry on this cell may not carry: the tool's
    /// typed-knob aliases plus the benchmark/tool's reserved flags.
    pub fn raw_denylist(&self) -> &'static [&'static str] {
        match self {
            RuntimeFlags::PrefillLlamacppCliStockToolsGgufText { .. } => DENY_LLAMA_PREFILL,
            RuntimeFlags::DecodeLlamacppCliStockToolsGgufText { .. } => DENY_LLAMA_DECODE,
            RuntimeFlags::MaxMemoryLlamacppCliStockToolsGgufText { .. } => DENY_LLAMA_MAXMEM,
            RuntimeFlags::EndToEndLlamacppCliStockToolsGgufText { .. }
            | RuntimeFlags::EvalLlamacppCliStockToolsGgufText { .. }
            | RuntimeFlags::VlLlamacppCliStockToolsGgufVision { .. } => DENY_LLAMA_SERVER,
            RuntimeFlags::EndToEndDockerVllmTorch { .. }
            | RuntimeFlags::EvalDockerVllmTorch { .. }
            | RuntimeFlags::MaxMemoryDockerVllmTorch { .. }
            | RuntimeFlags::EndToEndUvVllmTorch { .. }
            | RuntimeFlags::EvalUvVllmTorch { .. }
            | RuntimeFlags::MaxMemoryUvVllmTorch { .. } => DENY_VLLM,
            RuntimeFlags::EndToEndDockerSglangTorch { .. }
            | RuntimeFlags::EvalDockerSglangTorch { .. }
            | RuntimeFlags::MaxMemoryDockerSglangTorch { .. }
            | RuntimeFlags::EndToEndUvSglangTorch { .. }
            | RuntimeFlags::EvalUvSglangTorch { .. }
            | RuntimeFlags::MaxMemoryUvSglangTorch { .. } => DENY_SGLANG,
            RuntimeFlags::PrefillLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::DecodeLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::MaxMemoryLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::EndToEndLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::EvalLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::VlLlamacppIosPipetteGgufVision { .. }
            | RuntimeFlags::PrefillMlxIosPipetteMlx { .. }
            | RuntimeFlags::DecodeMlxIosPipetteMlx { .. }
            | RuntimeFlags::MaxMemoryMlxIosPipetteMlx { .. }
            | RuntimeFlags::EndToEndMlxIosPipetteMlx { .. }
            | RuntimeFlags::EvalMlxIosPipetteMlx { .. } => &[],
            RuntimeFlags::PrefillUvOpenvinoOpenvino { .. }
            | RuntimeFlags::DecodeUvOpenvinoOpenvino { .. }
            | RuntimeFlags::EndToEndUvOpenvinoOpenvino { .. }
            | RuntimeFlags::MaxMemoryUvOpenvinoOpenvino { .. } => &[],
        }
    }

    /// The flags the benchmark/tool fixes for this cell — the reserved subset of
    /// [`Self::raw_denylist`]. The remaining denied names are typed-knob
    /// aliases; the split drives which `raw` rejection reason is reported.
    fn tool_reserved(&self) -> &'static [&'static str] {
        use reserved_flags::llamacpp_cli_stock_tools as llama;
        match self {
            RuntimeFlags::PrefillLlamacppCliStockToolsGgufText { .. } => llama::PREFILL,
            RuntimeFlags::DecodeLlamacppCliStockToolsGgufText { .. } => llama::DECODE,
            RuntimeFlags::MaxMemoryLlamacppCliStockToolsGgufText { .. } => llama::MAX_MEMORY,
            RuntimeFlags::EndToEndLlamacppCliStockToolsGgufText { .. }
            | RuntimeFlags::EvalLlamacppCliStockToolsGgufText { .. }
            | RuntimeFlags::VlLlamacppCliStockToolsGgufVision { .. } => llama::SERVER,
            RuntimeFlags::EndToEndDockerVllmTorch { .. }
            | RuntimeFlags::EvalDockerVllmTorch { .. }
            | RuntimeFlags::MaxMemoryDockerVllmTorch { .. }
            | RuntimeFlags::EndToEndUvVllmTorch { .. }
            | RuntimeFlags::EvalUvVllmTorch { .. }
            | RuntimeFlags::MaxMemoryUvVllmTorch { .. } => VLLM_RESERVED,
            RuntimeFlags::EndToEndDockerSglangTorch { .. }
            | RuntimeFlags::EvalDockerSglangTorch { .. }
            | RuntimeFlags::MaxMemoryDockerSglangTorch { .. }
            | RuntimeFlags::EndToEndUvSglangTorch { .. }
            | RuntimeFlags::EvalUvSglangTorch { .. }
            | RuntimeFlags::MaxMemoryUvSglangTorch { .. } => SGLANG_RESERVED,
            RuntimeFlags::PrefillLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::DecodeLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::MaxMemoryLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::EndToEndLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::EvalLlamacppIosPipetteGgufText { .. }
            | RuntimeFlags::VlLlamacppIosPipetteGgufVision { .. }
            | RuntimeFlags::PrefillMlxIosPipetteMlx { .. }
            | RuntimeFlags::DecodeMlxIosPipetteMlx { .. }
            | RuntimeFlags::MaxMemoryMlxIosPipetteMlx { .. }
            | RuntimeFlags::EndToEndMlxIosPipetteMlx { .. }
            | RuntimeFlags::EvalMlxIosPipetteMlx { .. } => &[],
            RuntimeFlags::PrefillUvOpenvinoOpenvino { .. }
            | RuntimeFlags::DecodeUvOpenvinoOpenvino { .. }
            | RuntimeFlags::EndToEndUvOpenvinoOpenvino { .. }
            | RuntimeFlags::MaxMemoryUvOpenvinoOpenvino { .. } => &[],
        }
    }

    /// Reject any `raw` entry that names a flag this cell owns: a typed-knob
    /// alias (author should set the field) or a benchmark/tool-reserved flag
    /// (fixed by the harness). See [`deny_match`] for the matched forms.
    fn validate_raw(&self) -> Result<(), RuntimeFlagError> {
        let deny = self.raw_denylist();
        let reserved = self.tool_reserved();
        self.raw().iter().try_for_each(|token| {
            let Some(denied) = deny_match(deny, token) else {
                return Ok(());
            };
            Err(if reserved.contains(&denied) {
                RuntimeFlagError::RawReservedFlag {
                    flag: token.clone(),
                }
            } else {
                RuntimeFlagError::RawAliasesTypedKnob {
                    flag: token.clone(),
                }
            })
        })
    }
}

/// The denylisted flag `token` matches, if any: an exact name, a `--flag=value`
/// (keyed on the pre-`=` name), or a getopt glued short form where a single-dash
/// flag is immediately followed by its value (`-t8`, `-ngl99`, `-c4096`).
/// Returns the canonical denied flag so the caller can map it to a typed knob.
fn deny_match<'a>(deny: &[&'a str], token: &str) -> Option<&'a str> {
    let name = token.split('=').next().unwrap_or(token);
    if let Some(d) = deny.iter().copied().find(|&d| d == name) {
        return Some(d);
    }
    deny.iter().copied().find(|&d| {
        d.starts_with('-')
            && !d.starts_with("--")
            && token.len() > d.len()
            && token.starts_with(d)
            && token.as_bytes()[d.len()].is_ascii_digit()
    })
}

// ---------------------------------------------------------------------------
// Wire form
// ---------------------------------------------------------------------------

/// The composite-key fields of [`RuntimeFlagRef`] — everything that identifies
/// the cell rather than configuring it. [`RuntimeFlags::submission_value`] drops
/// them; a plan entry must carry them.
const AXIS_KEYS: [&str; 3] = ["runtime_type", "model_type", "benchmark_type"];

/// Flat wire form of [`RuntimeFlags`]: the three axis references (the composite
/// key) plus every typed knob as an optional field. Exists only at the serde
/// boundary — the [`TryFrom`] routes the set knobs into the one variant the
/// triple names and rejects the rest, so the stored type is never a bag.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeFlagRef {
    pub runtime_type: RuntimeType,
    pub model_type: ModelType,
    pub benchmark_type: BenchmarkType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub number_gpu_layers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmap: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flash_attention: Option<LlamacppFlashAttention>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx_size: Option<u32>,
    /// llama.cpp / iOS ubatch (prefill chunk size on MLX iOS).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub n_ubatch: Option<u32>,
    /// llama.cpp sliding-window cache policy (`swa_full`): `true` allocates KV
    /// for the full context on the windowed layers, `false` for the window
    /// alone. iOS cells only — the stock CLI tools set it themselves.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swa_full: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub no_cache: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_parallel_size: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dtype: Option<NonEmptyString>,
    /// vLLM context bound (`--max-model-len`), covering prompt + output — the
    /// same semantics as llama.cpp's `ctx_size`. The client derives it from the
    /// benchmark when the cell leaves it unset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<u32>,
    /// The OpenVINO compute device the cell runs on. Substituted at the call,
    /// not at the install — one wheel serves all three — so it configures a
    /// cell rather than identifying a runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<OpenvinoDevice>,
    /// OpenVINO NPU static-shape prompt bound. Left unset the runtime keeps
    /// GenAI's 1024 default, which the standard 512-token suite fits inside.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_prompt_len: Option<u32>,
    /// OpenVINO NPU output reservation. Unset, the runtime derives it from the
    /// cell — GenAI's 128 default truncates a 256-token generation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_response_len: Option<u32>,
    /// OpenVINO `GENERATE_HINT` (e.g. `BEST_PERF`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate_hint: Option<OpenvinoGenerateHint>,
    /// Server-side prefix caching. The benchmarks measure cold prefill+decode,
    /// so the client sets this `false`; a cell asking for `true` is refused at
    /// execution, where the benchmark's policy lives.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_caching: Option<bool>,
    /// `docker run` GPU allocation (`--gpus`), e.g. `"all"`. Docker cells only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<String>,
    /// `docker run --shm-size`, e.g. `"16g"`. Docker cells only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shm_size: Option<String>,
    /// `docker run --ipc`, e.g. `"host"`. Docker cells only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ipc: Option<String>,
    /// Env forwards for the server process (`K=V`, or bare `K` to inherit from
    /// the launching env). Docker and uv cells.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub envs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub raw: Vec<String>,
}

impl RuntimeFlagRef {
    /// An all-unset ref for one cell. The engines start here when a run carries
    /// no plan entry, set the fields their execution overlaid, and `try_into` a
    /// [`RuntimeFlags`] — so the flags they report back are routed and
    /// validated by the same code that admits an authored entry.
    pub fn new(
        benchmark_type: BenchmarkType,
        runtime_type: RuntimeType,
        model_type: ModelType,
    ) -> Self {
        Self {
            runtime_type,
            model_type,
            benchmark_type,
            threads: None,
            number_gpu_layers: None,
            mmap: None,
            flash_attention: None,
            ctx_size: None,
            n_ubatch: None,
            swa_full: None,
            no_cache: None,
            tensor_parallel_size: None,
            dtype: None,
            max_model_len: None,
            device: None,
            max_prompt_len: None,
            min_response_len: None,
            generate_hint: None,
            prefix_caching: None,
            gpus: None,
            shm_size: None,
            ipc: None,
            envs: Vec::new(),
            raw: Vec::new(),
        }
    }

    /// Reject a knob that's set but not accepted by this cell.
    fn deny(&self, set: bool, knob: &'static str) -> Result<(), RuntimeFlagError> {
        if set {
            Err(RuntimeFlagError::KnobNotAllowed {
                knob,
                runtime: self.runtime_type,
                model: self.model_type,
                benchmark: self.benchmark_type,
            })
        } else {
            Ok(())
        }
    }

    fn deny_llama_knobs_except_ubatch(&self) -> Result<(), RuntimeFlagError> {
        self.deny(self.threads.is_some(), "threads")?;
        self.deny(self.number_gpu_layers.is_some(), "number_gpu_layers")?;
        self.deny(self.mmap.is_some(), "mmap")?;
        self.deny(self.flash_attention.is_some(), "flash_attention")
    }

    fn deny_llama_knobs(&self) -> Result<(), RuntimeFlagError> {
        self.deny_llama_knobs_except_ubatch()?;
        self.deny_n_ubatch()
    }

    fn deny_server_knobs(&self) -> Result<(), RuntimeFlagError> {
        self.deny(self.ctx_size.is_some(), "ctx_size")?;
        self.deny(self.no_cache.is_some(), "no_cache")
    }

    fn deny_n_ubatch(&self) -> Result<(), RuntimeFlagError> {
        self.deny(self.n_ubatch.is_some(), "n_ubatch")
    }

    fn deny_swa_full(&self) -> Result<(), RuntimeFlagError> {
        self.deny(self.swa_full.is_some(), "swa_full")
    }

    /// CLI-only llama knobs the iOS in-process engine does not expose.
    /// `threads` is not among these: the iOS engine takes it too (see the
    /// `*LlamacppIosPipette*` variants). These are the ones that only exist as argv.
    fn deny_cli_only_llama_knobs(&self) -> Result<(), RuntimeFlagError> {
        self.deny(self.mmap.is_some(), "mmap")?;
        self.deny(self.flash_attention.is_some(), "flash_attention")?;
        self.deny(self.no_cache.is_some(), "no_cache")
    }

    /// Shared rejects for every iOS cell (no docker/vllm/sglang/`raw` surface).
    fn deny_non_ios_families(&self) -> Result<(), RuntimeFlagError> {
        self.deny_torch_server_knobs()?;
        self.deny_docker_launch()?;
        self.deny_envs()?;
        self.deny(!self.raw.is_empty(), "raw")
    }

    /// Every knob only the torch server cells accept.
    fn deny_torch_server_knobs(&self) -> Result<(), RuntimeFlagError> {
        self.deny(self.tensor_parallel_size.is_some(), "tensor_parallel_size")?;
        self.deny(self.dtype.is_some(), "dtype")?;
        self.deny(self.max_model_len.is_some(), "max_model_len")?;
        self.deny(self.prefix_caching.is_some(), "prefix_caching")
    }

    /// The `docker run` launcher settings — only docker cells accept them (uv runs
    /// on the host; the llama.cpp tools aren't containerized).
    fn deny_docker_launch(&self) -> Result<(), RuntimeFlagError> {
        self.deny(self.gpus.is_some(), "gpus")?;
        self.deny(self.shm_size.is_some(), "shm_size")?;
        self.deny(self.ipc.is_some(), "ipc")
    }

    /// The OpenVINO pipeline properties — only `uv_openvino` cells accept them.
    /// Every other cell must reject them, or a knob that does nothing would be
    /// silently accepted and reported on the submission as if it had applied.
    fn deny_openvino_knobs(&self) -> Result<(), RuntimeFlagError> {
        self.deny(self.device.is_some(), "device")?;
        self.deny(self.max_prompt_len.is_some(), "max_prompt_len")?;
        self.deny(self.min_response_len.is_some(), "min_response_len")?;
        self.deny(self.generate_hint.is_some(), "generate_hint")
    }

    /// Env forwards — accepted by the docker/uv server cells, rejected by the
    /// llama.cpp cells (the stock CLI tools take env from the parent process).
    fn deny_envs(&self) -> Result<(), RuntimeFlagError> {
        self.deny(!self.envs.is_empty(), "envs")
    }
}

impl RuntimeFlags {
    /// The knobs alone, for a client that has already parsed its cell.
    ///
    /// A client resolves `--benchmark`, `--runtime` and `--model` before it reads any
    /// flags, so it knows the triple; repeating it here is a second source of truth that
    /// can only agree or be wrong. An authored plan still carries it — that is what picks
    /// one entry out of a variant's list — so the axes are stripped at the edge rather
    /// than dropped from the type.
    ///
    /// Not [`Self::submission_value`], which strips the same axes: that one also redacts
    /// env values, because a submitted record must not carry a secret. This is the
    /// opposite direction — the client is about to *run* these flags and needs the values.
    pub fn knobs_json(&self) -> serde_json::Value {
        let mut value = serde_json::to_value(self)
            // `RuntimeFlags` serializes infallibly (no maps, floats, or fallible
            // `Serialize` impls). Fall closed on an empty object: the caller then emits
            // `{}`, which the far side reads as "no knobs" rather than as this cell's.
            .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
        if let Some(fields) = value.as_object_mut() {
            AXIS_KEYS.iter().for_each(|key| {
                fields.remove(*key);
            });
        }
        value
    }

    /// Parse a client's `--runtime-flags` against the cell it resolved.
    ///
    /// One format: a JSON object of knobs. The cell comes from `--benchmark`, `--runtime`
    /// and `--model`, which a client parses first, so naming it again here would be a
    /// second source of truth — and the one-element array that used to wrap it had nothing
    /// left to select among once the axes were gone. Both older spellings are refused by
    /// name rather than quietly accepted, so an invocation that predates this reads as
    /// wrong instead of half-working. An empty object is `None` — a knob set with nothing
    /// in it and an absent one both mean "run on the engine's defaults".
    pub fn from_cell_json(
        json: &str,
        runtime_type: RuntimeType,
        model_type: ModelType,
        benchmark_type: BenchmarkType,
    ) -> Result<Option<Self>, RuntimeFlagCellError> {
        // An array, a string, a number: none of them is a knob set, and all of them are
        // the same mistake to the caller.
        let serde_json::Value::Object(mut fields) = serde_json::from_str(json)? else {
            return Err(RuntimeFlagCellError::NotAnObject);
        };
        if fields.is_empty() {
            return Ok(None);
        }
        [
            ("runtime_type", serde_json::to_value(runtime_type)?),
            ("model_type", serde_json::to_value(model_type)?),
            ("benchmark_type", serde_json::to_value(benchmark_type)?),
        ]
        .into_iter()
        .try_for_each(|(key, derived)| {
            if fields.contains_key(key) {
                return Err(RuntimeFlagCellError::AxisNotAccepted { key });
            }
            fields.insert(key.to_owned(), derived);
            Ok(())
        })?;
        let reference: RuntimeFlagRef = serde_json::from_value(serde_json::Value::Object(fields))?;
        Ok(Some(Self::try_from(reference)?))
    }
}

/// Why a client's `--runtime-flags` does not describe the cell it resolved.
#[derive(Debug, thiserror::Error)]
pub enum RuntimeFlagCellError {
    #[error("parsing runtime flags JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error(
        "runtime flags must be a JSON object of knobs, e.g. {{\"threads\":4}} \
         (the one-element array is no longer accepted)"
    )]
    NotAnObject,
    #[error(
        "runtime flags must not carry `{key}`: the cell comes from --benchmark, --runtime \
         and --model, which are parsed first"
    )]
    AxisNotAccepted { key: &'static str },
    #[error(transparent)]
    Flags(#[from] RuntimeFlagError),
}

impl TryFrom<RuntimeFlagRef> for RuntimeFlags {
    type Error = RuntimeFlagError;

    fn try_from(r: RuntimeFlagRef) -> Result<Self, RuntimeFlagError> {
        use BenchmarkType as B;
        use ModelType as M;
        use RuntimeType as R;

        // The OpenVINO knobs are rejected for every other runtime here rather
        // than in each arm's deny chain. There are twenty-odd arms and the
        // failure mode of forgetting one is silent — a knob accepted, reported
        // on the submission, and applied to nothing.
        if r.runtime_type != R::UvOpenvino {
            r.deny_openvino_knobs()?;
        }
        // `swa_full` is the in-process iOS engine's to set: the stock CLI tools
        // pin the same setting themselves, and no other runtime exposes it.
        // Rejected here rather than in each arm's deny chain, for the reason
        // above.
        if r.runtime_type != R::LlamacppIosPipette {
            r.deny_swa_full()?;
        }

        let flags = match (r.benchmark_type, r.runtime_type, r.model_type) {
            // llama-bench cells: reject server + vllm/sglang + launcher settings.
            (B::PrefillThroughput, R::LlamacppCliStockTools, M::GgufText) => {
                r.deny_server_knobs()?;
                r.deny_n_ubatch()?;
                r.deny_torch_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny_envs()?;
                RuntimeFlags::PrefillLlamacppCliStockToolsGgufText {
                    threads: r.threads,
                    number_gpu_layers: r.number_gpu_layers,
                    mmap: r.mmap,
                    flash_attention: r.flash_attention,
                    raw: r.raw,
                }
            }
            (B::DecodeThroughput, R::LlamacppCliStockTools, M::GgufText) => {
                r.deny_server_knobs()?;
                r.deny_n_ubatch()?;
                r.deny_torch_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny_envs()?;
                RuntimeFlags::DecodeLlamacppCliStockToolsGgufText {
                    threads: r.threads,
                    number_gpu_layers: r.number_gpu_layers,
                    mmap: r.mmap,
                    flash_attention: r.flash_attention,
                    raw: r.raw,
                }
            }
            (B::MaxMemoryUsage, R::LlamacppCliStockTools, M::GgufText) => {
                r.deny_server_knobs()?;
                r.deny_n_ubatch()?;
                r.deny_torch_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny_envs()?;
                RuntimeFlags::MaxMemoryLlamacppCliStockToolsGgufText {
                    threads: r.threads,
                    number_gpu_layers: r.number_gpu_layers,
                    mmap: r.mmap,
                    flash_attention: r.flash_attention,
                    raw: r.raw,
                }
            }
            // llama-server cells: reject vllm/sglang + launcher settings.
            (B::EndToEndLatency, R::LlamacppCliStockTools, M::GgufText) => {
                r.deny_n_ubatch()?;
                r.deny_torch_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny_envs()?;
                RuntimeFlags::EndToEndLlamacppCliStockToolsGgufText {
                    threads: r.threads,
                    number_gpu_layers: r.number_gpu_layers,
                    mmap: r.mmap,
                    flash_attention: r.flash_attention,
                    ctx_size: r.ctx_size,
                    no_cache: r.no_cache,
                    raw: r.raw,
                }
            }
            (B::Eval, R::LlamacppCliStockTools, M::GgufText) => {
                r.deny_n_ubatch()?;
                r.deny_torch_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny_envs()?;
                RuntimeFlags::EvalLlamacppCliStockToolsGgufText {
                    threads: r.threads,
                    number_gpu_layers: r.number_gpu_layers,
                    mmap: r.mmap,
                    flash_attention: r.flash_attention,
                    ctx_size: r.ctx_size,
                    no_cache: r.no_cache,
                    raw: r.raw,
                }
            }
            (B::VlThroughput, R::LlamacppCliStockTools, M::GgufVision) => {
                r.deny_n_ubatch()?;
                r.deny_torch_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny_envs()?;
                RuntimeFlags::VlLlamacppCliStockToolsGgufVision {
                    threads: r.threads,
                    number_gpu_layers: r.number_gpu_layers,
                    mmap: r.mmap,
                    flash_attention: r.flash_attention,
                    ctx_size: r.ctx_size,
                    no_cache: r.no_cache,
                    raw: r.raw,
                }
            }
            // Docker vLLM cells: reject llama + server flags; keep server tuning
            // (tensor_parallel_size + dtype) and the `docker run` launcher settings.
            (B::EndToEndLatency, R::DockerVllm, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                RuntimeFlags::EndToEndDockerVllmTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    dtype: r.dtype,
                    max_model_len: r.max_model_len,
                    prefix_caching: r.prefix_caching,
                    gpus: r.gpus,
                    shm_size: r.shm_size,
                    ipc: r.ipc,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            (B::Eval, R::DockerVllm, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                RuntimeFlags::EvalDockerVllmTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    dtype: r.dtype,
                    max_model_len: r.max_model_len,
                    prefix_caching: r.prefix_caching,
                    gpus: r.gpus,
                    shm_size: r.shm_size,
                    ipc: r.ipc,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            (B::MaxMemoryUsage, R::DockerVllm, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                RuntimeFlags::MaxMemoryDockerVllmTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    dtype: r.dtype,
                    max_model_len: r.max_model_len,
                    prefix_caching: r.prefix_caching,
                    gpus: r.gpus,
                    shm_size: r.shm_size,
                    ipc: r.ipc,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            // Uv vLLM cells: reject llama + server + launcher settings; keep server
            // tuning and env forwards (uv runs on the host — no `docker run`).
            (B::EndToEndLatency, R::UvVllm, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny_docker_launch()?;
                RuntimeFlags::EndToEndUvVllmTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    dtype: r.dtype,
                    max_model_len: r.max_model_len,
                    prefix_caching: r.prefix_caching,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            (B::Eval, R::UvVllm, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny_docker_launch()?;
                RuntimeFlags::EvalUvVllmTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    dtype: r.dtype,
                    max_model_len: r.max_model_len,
                    prefix_caching: r.prefix_caching,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            (B::MaxMemoryUsage, R::UvVllm, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny_docker_launch()?;
                RuntimeFlags::MaxMemoryUvVllmTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    dtype: r.dtype,
                    max_model_len: r.max_model_len,
                    prefix_caching: r.prefix_caching,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            // Docker SGLang cells: reject llama + server flags + dtype; keep
            // tensor_parallel_size and the `docker run` launcher settings.
            (B::EndToEndLatency, R::DockerSglang, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny(r.dtype.is_some(), "dtype")?;
                r.deny(r.max_model_len.is_some(), "max_model_len")?;
                RuntimeFlags::EndToEndDockerSglangTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    prefix_caching: r.prefix_caching,
                    gpus: r.gpus,
                    shm_size: r.shm_size,
                    ipc: r.ipc,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            (B::Eval, R::DockerSglang, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny(r.dtype.is_some(), "dtype")?;
                r.deny(r.max_model_len.is_some(), "max_model_len")?;
                RuntimeFlags::EvalDockerSglangTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    prefix_caching: r.prefix_caching,
                    gpus: r.gpus,
                    shm_size: r.shm_size,
                    ipc: r.ipc,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            (B::MaxMemoryUsage, R::DockerSglang, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny(r.dtype.is_some(), "dtype")?;
                r.deny(r.max_model_len.is_some(), "max_model_len")?;
                RuntimeFlags::MaxMemoryDockerSglangTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    prefix_caching: r.prefix_caching,
                    gpus: r.gpus,
                    shm_size: r.shm_size,
                    ipc: r.ipc,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            // Uv SGLang cells: reject llama + server + launcher settings + dtype;
            // keep tensor_parallel_size and env forwards.
            (B::EndToEndLatency, R::UvSglang, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny(r.dtype.is_some(), "dtype")?;
                r.deny(r.max_model_len.is_some(), "max_model_len")?;
                RuntimeFlags::EndToEndUvSglangTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    prefix_caching: r.prefix_caching,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            (B::Eval, R::UvSglang, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny(r.dtype.is_some(), "dtype")?;
                r.deny(r.max_model_len.is_some(), "max_model_len")?;
                RuntimeFlags::EvalUvSglangTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    prefix_caching: r.prefix_caching,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            (B::MaxMemoryUsage, R::UvSglang, M::Torch) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny(r.dtype.is_some(), "dtype")?;
                r.deny(r.max_model_len.is_some(), "max_model_len")?;
                RuntimeFlags::MaxMemoryUvSglangTorch {
                    tensor_parallel_size: r.tensor_parallel_size,
                    prefix_caching: r.prefix_caching,
                    envs: r.envs,
                    raw: r.raw,
                }
            }
            // iOS llama.cpp in-process: ngl + ctx + ubatch; no raw argv.
            (B::PrefillThroughput, R::LlamacppIosPipette, M::GgufText) => {
                r.deny_cli_only_llama_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::PrefillLlamacppIosPipetteGgufText {
                    number_gpu_layers: r.number_gpu_layers,
                    ctx_size: r.ctx_size,
                    n_ubatch: r.n_ubatch,
                    threads: r.threads,
                    swa_full: r.swa_full,
                }
            }
            (B::DecodeThroughput, R::LlamacppIosPipette, M::GgufText) => {
                r.deny_cli_only_llama_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::DecodeLlamacppIosPipetteGgufText {
                    number_gpu_layers: r.number_gpu_layers,
                    ctx_size: r.ctx_size,
                    n_ubatch: r.n_ubatch,
                    threads: r.threads,
                    swa_full: r.swa_full,
                }
            }
            (B::MaxMemoryUsage, R::LlamacppIosPipette, M::GgufText) => {
                r.deny_cli_only_llama_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::MaxMemoryLlamacppIosPipetteGgufText {
                    number_gpu_layers: r.number_gpu_layers,
                    ctx_size: r.ctx_size,
                    n_ubatch: r.n_ubatch,
                    threads: r.threads,
                    swa_full: r.swa_full,
                }
            }
            (B::EndToEndLatency, R::LlamacppIosPipette, M::GgufText) => {
                r.deny_cli_only_llama_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::EndToEndLlamacppIosPipetteGgufText {
                    number_gpu_layers: r.number_gpu_layers,
                    ctx_size: r.ctx_size,
                    n_ubatch: r.n_ubatch,
                    threads: r.threads,
                    swa_full: r.swa_full,
                }
            }
            (B::Eval, R::LlamacppIosPipette, M::GgufText) => {
                r.deny_cli_only_llama_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::EvalLlamacppIosPipetteGgufText {
                    number_gpu_layers: r.number_gpu_layers,
                    ctx_size: r.ctx_size,
                    n_ubatch: r.n_ubatch,
                    threads: r.threads,
                    swa_full: r.swa_full,
                }
            }
            (B::VlThroughput, R::LlamacppIosPipette, M::GgufVision) => {
                r.deny_cli_only_llama_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::VlLlamacppIosPipetteGgufVision {
                    number_gpu_layers: r.number_gpu_layers,
                    ctx_size: r.ctx_size,
                    n_ubatch: r.n_ubatch,
                    threads: r.threads,
                    swa_full: r.swa_full,
                }
            }
            // iOS MLX: only n_ubatch (prefill chunk).
            (B::PrefillThroughput, R::MlxIosPipette, M::Mlx) => {
                r.deny_llama_knobs_except_ubatch()?;
                r.deny_server_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::PrefillMlxIosPipetteMlx {
                    n_ubatch: r.n_ubatch,
                }
            }
            (B::DecodeThroughput, R::MlxIosPipette, M::Mlx) => {
                r.deny_llama_knobs_except_ubatch()?;
                r.deny_server_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::DecodeMlxIosPipetteMlx {
                    n_ubatch: r.n_ubatch,
                }
            }
            (B::MaxMemoryUsage, R::MlxIosPipette, M::Mlx) => {
                r.deny_llama_knobs_except_ubatch()?;
                r.deny_server_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::MaxMemoryMlxIosPipetteMlx {
                    n_ubatch: r.n_ubatch,
                }
            }
            (B::EndToEndLatency, R::MlxIosPipette, M::Mlx) => {
                r.deny_llama_knobs_except_ubatch()?;
                r.deny_server_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::EndToEndMlxIosPipetteMlx {
                    n_ubatch: r.n_ubatch,
                }
            }
            (B::Eval, R::MlxIosPipette, M::Mlx) => {
                r.deny_llama_knobs_except_ubatch()?;
                r.deny_server_knobs()?;
                r.deny_non_ios_families()?;
                RuntimeFlags::EvalMlxIosPipetteMlx {
                    n_ubatch: r.n_ubatch,
                }
            }
            // OpenVINO x IR. No `raw`, no env forwards, no launcher settings:
            // the driver is spawned directly with typed pipeline properties.
            (B::PrefillThroughput, R::UvOpenvino, M::Openvino)
            | (B::DecodeThroughput, R::UvOpenvino, M::Openvino)
            | (B::EndToEndLatency, R::UvOpenvino, M::Openvino)
            | (B::MaxMemoryUsage, R::UvOpenvino, M::Openvino) => {
                r.deny_llama_knobs()?;
                r.deny_server_knobs()?;
                r.deny_n_ubatch()?;
                r.deny_torch_server_knobs()?;
                r.deny_docker_launch()?;
                r.deny_envs()?;
                r.deny(!r.raw.is_empty(), "raw")?;
                let device = r.device;
                let max_prompt_len = r.max_prompt_len;
                let min_response_len = r.min_response_len;
                let generate_hint = r.generate_hint;
                match r.benchmark_type {
                    B::PrefillThroughput => RuntimeFlags::PrefillUvOpenvinoOpenvino {
                        device,
                        max_prompt_len,
                        min_response_len,
                        generate_hint,
                    },
                    B::DecodeThroughput => RuntimeFlags::DecodeUvOpenvinoOpenvino {
                        device,
                        max_prompt_len,
                        min_response_len,
                        generate_hint,
                    },
                    B::EndToEndLatency => RuntimeFlags::EndToEndUvOpenvinoOpenvino {
                        device,
                        max_prompt_len,
                        min_response_len,
                        generate_hint,
                    },
                    B::MaxMemoryUsage => RuntimeFlags::MaxMemoryUvOpenvinoOpenvino {
                        device,
                        max_prompt_len,
                        min_response_len,
                        generate_hint,
                    },
                    // Spelled out rather than caught by `_`: the tuple above
                    // admits only four benchmark types today, but adding a
                    // fifth there would otherwise route it silently to the
                    // memory variant instead of failing to compile.
                    B::Eval | B::VlThroughput => {
                        return Err(RuntimeFlagError::NoSuchCombination {
                            benchmark: r.benchmark_type,
                            runtime: r.runtime_type,
                            model: r.model_type,
                        })
                    }
                }
            }
            _ => {
                return Err(RuntimeFlagError::NoSuchCombination {
                    benchmark: r.benchmark_type,
                    runtime: r.runtime_type,
                    model: r.model_type,
                })
            }
        };
        flags.validate_raw()?;
        Ok(flags)
    }
}

impl From<RuntimeFlags> for RuntimeFlagRef {
    fn from(f: RuntimeFlags) -> Self {
        let (benchmark_type, runtime_type, model_type) = f.axes();
        let mut out = RuntimeFlagRef::new(benchmark_type, runtime_type, model_type);
        match f {
            RuntimeFlags::PrefillLlamacppCliStockToolsGgufText {
                threads,
                number_gpu_layers,
                mmap,
                flash_attention,
                raw,
            }
            | RuntimeFlags::DecodeLlamacppCliStockToolsGgufText {
                threads,
                number_gpu_layers,
                mmap,
                flash_attention,
                raw,
            }
            | RuntimeFlags::MaxMemoryLlamacppCliStockToolsGgufText {
                threads,
                number_gpu_layers,
                mmap,
                flash_attention,
                raw,
            } => {
                out.threads = threads;
                out.number_gpu_layers = number_gpu_layers;
                out.mmap = mmap;
                out.flash_attention = flash_attention;
                out.raw = raw;
            }
            RuntimeFlags::EndToEndLlamacppCliStockToolsGgufText {
                threads,
                number_gpu_layers,
                mmap,
                flash_attention,
                ctx_size,
                no_cache,
                raw,
            }
            | RuntimeFlags::EvalLlamacppCliStockToolsGgufText {
                threads,
                number_gpu_layers,
                mmap,
                flash_attention,
                ctx_size,
                no_cache,
                raw,
            }
            | RuntimeFlags::VlLlamacppCliStockToolsGgufVision {
                threads,
                number_gpu_layers,
                mmap,
                flash_attention,
                ctx_size,
                no_cache,
                raw,
            } => {
                out.threads = threads;
                out.number_gpu_layers = number_gpu_layers;
                out.mmap = mmap;
                out.flash_attention = flash_attention;
                out.ctx_size = ctx_size;
                out.no_cache = no_cache;
                out.raw = raw;
            }
            RuntimeFlags::EndToEndDockerVllmTorch {
                tensor_parallel_size,
                dtype,
                max_model_len,
                prefix_caching,
                gpus,
                shm_size,
                ipc,
                envs,
                raw,
            }
            | RuntimeFlags::EvalDockerVllmTorch {
                tensor_parallel_size,
                dtype,
                max_model_len,
                prefix_caching,
                gpus,
                shm_size,
                ipc,
                envs,
                raw,
            }
            | RuntimeFlags::MaxMemoryDockerVllmTorch {
                tensor_parallel_size,
                dtype,
                max_model_len,
                prefix_caching,
                gpus,
                shm_size,
                ipc,
                envs,
                raw,
            } => {
                out.tensor_parallel_size = tensor_parallel_size;
                out.dtype = dtype;
                out.max_model_len = max_model_len;
                out.prefix_caching = prefix_caching;
                out.gpus = gpus;
                out.shm_size = shm_size;
                out.ipc = ipc;
                out.envs = envs;
                out.raw = raw;
            }
            RuntimeFlags::EndToEndUvVllmTorch {
                tensor_parallel_size,
                dtype,
                max_model_len,
                prefix_caching,
                envs,
                raw,
            }
            | RuntimeFlags::EvalUvVllmTorch {
                tensor_parallel_size,
                dtype,
                max_model_len,
                prefix_caching,
                envs,
                raw,
            }
            | RuntimeFlags::MaxMemoryUvVllmTorch {
                tensor_parallel_size,
                dtype,
                max_model_len,
                prefix_caching,
                envs,
                raw,
            } => {
                out.tensor_parallel_size = tensor_parallel_size;
                out.dtype = dtype;
                out.max_model_len = max_model_len;
                out.prefix_caching = prefix_caching;
                out.envs = envs;
                out.raw = raw;
            }
            RuntimeFlags::EndToEndDockerSglangTorch {
                tensor_parallel_size,
                prefix_caching,
                gpus,
                shm_size,
                ipc,
                envs,
                raw,
            }
            | RuntimeFlags::EvalDockerSglangTorch {
                tensor_parallel_size,
                prefix_caching,
                gpus,
                shm_size,
                ipc,
                envs,
                raw,
            }
            | RuntimeFlags::MaxMemoryDockerSglangTorch {
                tensor_parallel_size,
                prefix_caching,
                gpus,
                shm_size,
                ipc,
                envs,
                raw,
            } => {
                out.tensor_parallel_size = tensor_parallel_size;
                out.prefix_caching = prefix_caching;
                out.gpus = gpus;
                out.shm_size = shm_size;
                out.ipc = ipc;
                out.envs = envs;
                out.raw = raw;
            }
            RuntimeFlags::EndToEndUvSglangTorch {
                tensor_parallel_size,
                prefix_caching,
                envs,
                raw,
            }
            | RuntimeFlags::EvalUvSglangTorch {
                tensor_parallel_size,
                prefix_caching,
                envs,
                raw,
            }
            | RuntimeFlags::MaxMemoryUvSglangTorch {
                tensor_parallel_size,
                prefix_caching,
                envs,
                raw,
            } => {
                out.tensor_parallel_size = tensor_parallel_size;
                out.prefix_caching = prefix_caching;
                out.envs = envs;
                out.raw = raw;
            }
            RuntimeFlags::PrefillLlamacppIosPipetteGgufText {
                number_gpu_layers,
                ctx_size,
                n_ubatch,
                threads,
                swa_full,
            }
            | RuntimeFlags::DecodeLlamacppIosPipetteGgufText {
                number_gpu_layers,
                ctx_size,
                n_ubatch,
                threads,
                swa_full,
            }
            | RuntimeFlags::MaxMemoryLlamacppIosPipetteGgufText {
                number_gpu_layers,
                ctx_size,
                n_ubatch,
                threads,
                swa_full,
            }
            | RuntimeFlags::EndToEndLlamacppIosPipetteGgufText {
                number_gpu_layers,
                ctx_size,
                n_ubatch,
                threads,
                swa_full,
            }
            | RuntimeFlags::EvalLlamacppIosPipetteGgufText {
                number_gpu_layers,
                ctx_size,
                n_ubatch,
                threads,
                swa_full,
            }
            | RuntimeFlags::VlLlamacppIosPipetteGgufVision {
                number_gpu_layers,
                ctx_size,
                n_ubatch,
                threads,
                swa_full,
            } => {
                out.number_gpu_layers = number_gpu_layers;
                out.ctx_size = ctx_size;
                out.n_ubatch = n_ubatch;
                out.threads = threads;
                out.swa_full = swa_full;
            }
            RuntimeFlags::PrefillMlxIosPipetteMlx { n_ubatch }
            | RuntimeFlags::DecodeMlxIosPipetteMlx { n_ubatch }
            | RuntimeFlags::MaxMemoryMlxIosPipetteMlx { n_ubatch }
            | RuntimeFlags::EndToEndMlxIosPipetteMlx { n_ubatch }
            | RuntimeFlags::EvalMlxIosPipetteMlx { n_ubatch } => {
                out.n_ubatch = n_ubatch;
            }
            RuntimeFlags::PrefillUvOpenvinoOpenvino {
                device,
                max_prompt_len,
                min_response_len,
                generate_hint,
            }
            | RuntimeFlags::DecodeUvOpenvinoOpenvino {
                device,
                max_prompt_len,
                min_response_len,
                generate_hint,
            }
            | RuntimeFlags::EndToEndUvOpenvinoOpenvino {
                device,
                max_prompt_len,
                min_response_len,
                generate_hint,
            }
            | RuntimeFlags::MaxMemoryUvOpenvinoOpenvino {
                device,
                max_prompt_len,
                min_response_len,
                generate_hint,
            } => {
                out.device = device;
                out.max_prompt_len = max_prompt_len;
                out.min_response_len = min_response_len;
                out.generate_hint = generate_hint;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use strum::EnumCount as _;

    use super::*;

    fn llamacpp_cli() -> anyhow::Result<Runtime> {
        Ok(toml::from_str(
            "type = \"llamacpp_cli_stock_tools\"\nsource = \"github_release\"\nversion = \"b5000\"\nflavor = \"macos-arm64\"",
        )?)
    }

    fn gguf_text() -> anyhow::Result<Model> {
        Ok(toml::from_str(
            "type = \"gguf_text\"\nsource = \"huggingface\"\norg = \"org\"\nrepo_name = \"repo\"\npath = \"m.gguf\"",
        )?)
    }

    fn torch() -> anyhow::Result<Model> {
        Ok(toml::from_str(
            "type = \"torch\"\nsource = \"huggingface\"\norg = \"x\"\nrepo_name = \"y\"",
        )?)
    }

    fn parse(wire: &str) -> Result<RuntimeFlags, toml::de::Error> {
        toml::from_str(wire)
    }

    #[test]
    fn prefill_cli_maps_to_bench_variant() -> anyhow::Result<()> {
        let f = parse(
            "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"prefill_throughput\"\nthreads = 4\nnumber_gpu_layers = 99",
        )?;
        assert!(matches!(
            f,
            RuntimeFlags::PrefillLlamacppCliStockToolsGgufText { .. }
        ));
        assert!(f.matches(
            BenchmarkType::PrefillThroughput,
            &llamacpp_cli()?,
            &gguf_text()?
        ));
        assert!(!f.matches(BenchmarkType::Eval, &llamacpp_cli()?, &gguf_text()?));
        assert!(!f.matches(
            BenchmarkType::PrefillThroughput,
            &llamacpp_cli()?,
            &torch()?
        ));
        Ok(())
    }

    #[test]
    fn eval_cli_maps_to_server_variant() -> anyhow::Result<()> {
        let f = parse(
            "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"eval\"\nctx_size = 8192\nno_cache = true",
        )?;
        assert!(matches!(
            f,
            RuntimeFlags::EvalLlamacppCliStockToolsGgufText { .. }
        ));
        Ok(())
    }

    #[rstest]
    // ctx_size is a llama-server knob; prefill is a bench cell.
    #[case::server_knob_on_bench_cell(
        "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"prefill_throughput\"\nctx_size = 8192"
    )]
    // apk / AFM have no typed-flag cells — any entry is NoSuchCombination.
    #[case::in_process_apk_has_no_variant(
        "runtime_type = \"llamacpp_apk_pipette\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"eval\""
    )]
    #[case::apple_foundation_has_no_variant(
        "runtime_type = \"apple_foundation\"\nmodel_type = \"apple_foundation_text\"\nbenchmark_type = \"eval\""
    )]
    // torch never pairs with llama.cpp.
    #[case::torch_never_pairs_with_llamacpp(
        "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"torch\"\nbenchmark_type = \"eval\""
    )]
    // dtype is a vLLM knob; sglang doesn't take it.
    #[case::sglang_rejects_dtype(
        "runtime_type = \"docker_sglang\"\nmodel_type = \"torch\"\nbenchmark_type = \"eval\"\ndtype = \"bfloat16\""
    )]
    // `-t` is the typed-`threads` alias — must go through the knob.
    #[case::raw_aliasing_a_typed_knob(
        "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"prefill_throughput\"\nraw = [\"-t\", \"8\"]"
    )]
    // llama-bench fixes --repetitions (one rep per invocation, looped
    // externally), so a raw override is reserved on every bench cell.
    #[case::raw_repetitions_reserved_on_bench(
        "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"prefill_throughput\"\nraw = [\"--repetitions\", \"5\"]"
    )]
    // The prompt depth sizes llama-bench's context (`n_ctx = n_prompt + n_gen +
    // n_depth`), and a prefill cell picks its row on the prompt and gen counts
    // alone — so an override here would be measured against a deep KV cache and
    // recorded as an ordinary prefill.
    #[case::raw_depth_reserved_on_prefill(
        "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"prefill_throughput\"\nraw = [\"-d\", \"32768\"]"
    )]
    // Same flag, same context arithmetic, and this cell reports the memory that
    // arithmetic decides.
    #[case::raw_depth_reserved_on_max_memory(
        "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"max_memory_usage\"\nraw = [\"--n-depth\", \"32768\"]"
    )]
    // getopt glued short form `-t8` still aliases the typed `threads` knob.
    #[case::raw_glued_short_form(
        "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"prefill_throughput\"\nraw = [\"-t8\"]"
    )]
    // gpus/shm_size/ipc are `docker run` options — uv runs on the host, so a uv
    // cell has no field for them.
    #[case::uv_rejects_docker_gpus(
        "runtime_type = \"uv_vllm\"\nmodel_type = \"torch\"\nbenchmark_type = \"eval\"\ngpus = \"all\""
    )]
    // env forwards aren't a llama.cpp-cell flag (the stock tools inherit env).
    #[case::llama_rejects_envs(
        "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"eval\"\nenvs = [\"HF_TOKEN=x\"]"
    )]
    fn rejected_wire_forms(#[case] wire: &str) {
        assert!(parse(wire).is_err());
    }

    #[rstest]
    // A typed-knob alias points the author at the field.
    #[case("prefill_throughput", "-t", "aliases a typed knob")]
    // A benchmark/tool-reserved flag can't be overridden at all.
    #[case(
        "prefill_throughput",
        "--repetitions",
        "reserved by the benchmark/tool"
    )]
    fn raw_rejection_distinguishes_alias_from_reserved(
        #[case] benchmark: &str,
        #[case] flag: &str,
        #[case] expected: &str,
    ) {
        let wire = format!(
            "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"{benchmark}\"\nraw = [\"{flag}\"]"
        );
        let result = parse(&wire);
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(format!("{e}").contains(expected), "got: {e}");
        }
    }

    // Every valid triple, exhaustively. The variant's identity is hand-encoded
    // three times (its name, its `axes()` arm, its `TryFrom` routing pattern)
    // with nothing forcing them to agree — this locks all three, plus the
    // `From`/serde round-trip, for each cell.
    #[rstest]
    #[case(
        "llamacpp_cli_stock_tools",
        "gguf_text",
        "prefill_throughput",
        RuntimeType::LlamacppCliStockTools,
        ModelType::GgufText,
        BenchmarkType::PrefillThroughput
    )]
    #[case(
        "llamacpp_cli_stock_tools",
        "gguf_text",
        "decode_throughput",
        RuntimeType::LlamacppCliStockTools,
        ModelType::GgufText,
        BenchmarkType::DecodeThroughput
    )]
    #[case(
        "llamacpp_cli_stock_tools",
        "gguf_text",
        "max_memory_usage",
        RuntimeType::LlamacppCliStockTools,
        ModelType::GgufText,
        BenchmarkType::MaxMemoryUsage
    )]
    #[case(
        "llamacpp_cli_stock_tools",
        "gguf_text",
        "end_to_end_latency",
        RuntimeType::LlamacppCliStockTools,
        ModelType::GgufText,
        BenchmarkType::EndToEndLatency
    )]
    #[case(
        "llamacpp_cli_stock_tools",
        "gguf_text",
        "eval",
        RuntimeType::LlamacppCliStockTools,
        ModelType::GgufText,
        BenchmarkType::Eval
    )]
    #[case(
        "llamacpp_cli_stock_tools",
        "gguf_vision",
        "vl_throughput",
        RuntimeType::LlamacppCliStockTools,
        ModelType::GgufVision,
        BenchmarkType::VlThroughput
    )]
    #[case(
        "docker_vllm",
        "torch",
        "end_to_end_latency",
        RuntimeType::DockerVllm,
        ModelType::Torch,
        BenchmarkType::EndToEndLatency
    )]
    #[case(
        "docker_vllm",
        "torch",
        "eval",
        RuntimeType::DockerVllm,
        ModelType::Torch,
        BenchmarkType::Eval
    )]
    #[case(
        "docker_vllm",
        "torch",
        "max_memory_usage",
        RuntimeType::DockerVllm,
        ModelType::Torch,
        BenchmarkType::MaxMemoryUsage
    )]
    #[case(
        "uv_vllm",
        "torch",
        "end_to_end_latency",
        RuntimeType::UvVllm,
        ModelType::Torch,
        BenchmarkType::EndToEndLatency
    )]
    #[case(
        "uv_vllm",
        "torch",
        "eval",
        RuntimeType::UvVllm,
        ModelType::Torch,
        BenchmarkType::Eval
    )]
    #[case(
        "uv_vllm",
        "torch",
        "max_memory_usage",
        RuntimeType::UvVllm,
        ModelType::Torch,
        BenchmarkType::MaxMemoryUsage
    )]
    #[case(
        "docker_sglang",
        "torch",
        "end_to_end_latency",
        RuntimeType::DockerSglang,
        ModelType::Torch,
        BenchmarkType::EndToEndLatency
    )]
    #[case(
        "docker_sglang",
        "torch",
        "eval",
        RuntimeType::DockerSglang,
        ModelType::Torch,
        BenchmarkType::Eval
    )]
    #[case(
        "docker_sglang",
        "torch",
        "max_memory_usage",
        RuntimeType::DockerSglang,
        ModelType::Torch,
        BenchmarkType::MaxMemoryUsage
    )]
    #[case(
        "uv_sglang",
        "torch",
        "end_to_end_latency",
        RuntimeType::UvSglang,
        ModelType::Torch,
        BenchmarkType::EndToEndLatency
    )]
    #[case(
        "uv_sglang",
        "torch",
        "eval",
        RuntimeType::UvSglang,
        ModelType::Torch,
        BenchmarkType::Eval
    )]
    #[case(
        "uv_sglang",
        "torch",
        "max_memory_usage",
        RuntimeType::UvSglang,
        ModelType::Torch,
        BenchmarkType::MaxMemoryUsage
    )]
    #[case(
        "llamacpp_ios_pipette",
        "gguf_text",
        "prefill_throughput",
        RuntimeType::LlamacppIosPipette,
        ModelType::GgufText,
        BenchmarkType::PrefillThroughput
    )]
    #[case(
        "llamacpp_ios_pipette",
        "gguf_text",
        "decode_throughput",
        RuntimeType::LlamacppIosPipette,
        ModelType::GgufText,
        BenchmarkType::DecodeThroughput
    )]
    #[case(
        "llamacpp_ios_pipette",
        "gguf_text",
        "max_memory_usage",
        RuntimeType::LlamacppIosPipette,
        ModelType::GgufText,
        BenchmarkType::MaxMemoryUsage
    )]
    #[case(
        "llamacpp_ios_pipette",
        "gguf_text",
        "end_to_end_latency",
        RuntimeType::LlamacppIosPipette,
        ModelType::GgufText,
        BenchmarkType::EndToEndLatency
    )]
    #[case(
        "llamacpp_ios_pipette",
        "gguf_text",
        "eval",
        RuntimeType::LlamacppIosPipette,
        ModelType::GgufText,
        BenchmarkType::Eval
    )]
    #[case(
        "llamacpp_ios_pipette",
        "gguf_vision",
        "vl_throughput",
        RuntimeType::LlamacppIosPipette,
        ModelType::GgufVision,
        BenchmarkType::VlThroughput
    )]
    #[case(
        "mlx_ios_pipette",
        "mlx",
        "prefill_throughput",
        RuntimeType::MlxIosPipette,
        ModelType::Mlx,
        BenchmarkType::PrefillThroughput
    )]
    #[case(
        "mlx_ios_pipette",
        "mlx",
        "decode_throughput",
        RuntimeType::MlxIosPipette,
        ModelType::Mlx,
        BenchmarkType::DecodeThroughput
    )]
    #[case(
        "mlx_ios_pipette",
        "mlx",
        "max_memory_usage",
        RuntimeType::MlxIosPipette,
        ModelType::Mlx,
        BenchmarkType::MaxMemoryUsage
    )]
    #[case(
        "mlx_ios_pipette",
        "mlx",
        "end_to_end_latency",
        RuntimeType::MlxIosPipette,
        ModelType::Mlx,
        BenchmarkType::EndToEndLatency
    )]
    #[case(
        "mlx_ios_pipette",
        "mlx",
        "eval",
        RuntimeType::MlxIosPipette,
        ModelType::Mlx,
        BenchmarkType::Eval
    )]
    fn variant_triple_is_consistent(
        #[case] runtime: &str,
        #[case] model: &str,
        #[case] benchmark: &str,
        #[case] rt: RuntimeType,
        #[case] mt: ModelType,
        #[case] bt: BenchmarkType,
    ) -> anyhow::Result<()> {
        let wire = format!(
            "runtime_type = \"{runtime}\"\nmodel_type = \"{model}\"\nbenchmark_type = \"{benchmark}\""
        );
        let flags = parse(&wire)?;
        // TryFrom routed the triple to a variant whose axes() reports it back.
        assert_eq!(flags.axes(), (bt, rt, mt));
        // From + serde round-trip preserves the variant (no dropped field group).
        assert_eq!(parse(&toml::to_string(&flags)?)?, flags);
        Ok(())
    }

    #[test]
    fn ios_llama_knobs_round_trip() -> anyhow::Result<()> {
        let f = parse(
            "runtime_type = \"llamacpp_ios_pipette\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"eval\"\nnumber_gpu_layers = 99\nctx_size = 8192\nn_ubatch = 512\nthreads = 6\nswa_full = false",
        )?;
        assert!(matches!(
            f,
            RuntimeFlags::EvalLlamacppIosPipetteGgufText {
                number_gpu_layers: Some(99),
                ctx_size: Some(8192),
                n_ubatch: Some(512),
                threads: Some(6),
                swa_full: Some(false),
            }
        ));
        assert_eq!(parse(&toml::to_string(&f)?)?, f);
        Ok(())
    }

    /// The in-process engine takes `threads` (llama's `n_threads`), unlike the argv-only
    /// knobs beside it — a cell that offloads fewer layers runs most of the model on the
    /// CPU, and sampling is there even at full offload.
    #[test]
    fn ios_llama_takes_threads_but_not_the_argv_knobs() -> anyhow::Result<()> {
        let head = "runtime_type = \"llamacpp_ios_pipette\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"eval\"\n";
        assert!(parse(&format!("{head}threads = 4")).is_ok());
        for argv_only in ["mmap = true", "flash_attention = \"on\"", "no_cache = true"] {
            assert!(
                parse(&format!("{head}{argv_only}")).is_err(),
                "{argv_only} has no in-process counterpart"
            );
        }
        Ok(())
    }

    #[test]
    fn ios_mlx_rejects_ngl() {
        assert!(parse(
            "runtime_type = \"mlx_ios_pipette\"\nmodel_type = \"mlx\"\nbenchmark_type = \"prefill_throughput\"\nnumber_gpu_layers = 1",
        )
        .is_err());
    }

    #[test]
    fn ios_rejects_raw() {
        // In-process iOS engines have no argv surface for a raw escape hatch.
        assert!(parse(
            "runtime_type = \"llamacpp_ios_pipette\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"eval\"\nraw = [\"--foo\"]",
        )
        .is_err());
        assert!(parse(
            "runtime_type = \"mlx_ios_pipette\"\nmodel_type = \"mlx\"\nbenchmark_type = \"prefill_throughput\"\nraw = [\"x\"]",
        )
        .is_err());
    }

    #[test]
    fn raw_ctx_size_allowed_on_prefill_bench() -> anyhow::Result<()> {
        // llama-bench prefill does not fix --ctx-size (unlike max_memory,
        // which reserves it), so it's a legal raw flag on this cell.
        // Parsing succeeds (it isn't rejected as a reserved/aliased flag),
        // and the raw token is retained on the cell.
        let f = parse(
            "runtime_type = \"llamacpp_cli_stock_tools\"\nmodel_type = \"gguf_text\"\nbenchmark_type = \"prefill_throughput\"\nraw = [\"--ctx-size\", \"4096\"]",
        )?;
        assert!(matches!(
            f,
            RuntimeFlags::PrefillLlamacppCliStockToolsGgufText { .. }
        ));
        Ok(())
    }

    #[test]
    fn vllm_knobs_and_raw_round_trip() -> anyhow::Result<()> {
        let f = parse(
            "runtime_type = \"docker_vllm\"\nmodel_type = \"torch\"\nbenchmark_type = \"eval\"\ntensor_parallel_size = 2\nmax_model_len = 4096\nraw = [\"--swap-space\", \"8\"]",
        )?;
        assert!(matches!(f, RuntimeFlags::EvalDockerVllmTorch { .. }));
        let wire = toml::to_string(&f)?;
        assert!(wire.contains("runtime_type = \"docker_vllm\""));
        assert_eq!(parse(&wire)?, f);
        Ok(())
    }

    /// `max_model_len` and `prefix_caching` are typed, so the argv spellings are
    /// closed to `raw` — an author pinning the context has one place to do it,
    /// and the client's own derived value can't collide with a raw token.
    #[rstest]
    #[case::max_model_len(r#"raw = ["--max-model-len", "4096"]"#)]
    #[case::max_model_len_glued(r#"raw = ["--max-model-len=4096"]"#)]
    #[case::prefix_cache_on(r#"raw = ["--enable-prefix-caching"]"#)]
    #[case::prefix_cache_off(r#"raw = ["--no-enable-prefix-caching"]"#)]
    fn vllm_raw_cannot_shadow_the_typed_server_knobs(#[case] raw: &str) {
        let result = parse(&format!(
            "runtime_type = \"docker_vllm\"\nmodel_type = \"torch\"\nbenchmark_type = \"eval\"\n{raw}"
        ));
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(format!("{e}").contains("aliases a typed knob"), "got: {e}");
        }
    }

    /// sglang's context length stays an operator concern (`--context-length`),
    /// so the cell has no `max_model_len` to set.
    #[test]
    fn sglang_rejects_max_model_len() {
        let result = parse(
            "runtime_type = \"docker_sglang\"\nmodel_type = \"torch\"\nbenchmark_type = \"eval\"\nmax_model_len = 4096",
        );
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(format!("{e}").contains("max_model_len"), "got: {e}");
        }
    }

    /// The submission shape: the cell's own settings, with env forwards
    /// reduced to names so a token can't ride along.
    #[test]
    fn submission_value_reports_the_cells_settings() -> anyhow::Result<()> {
        let f = parse(
            "runtime_type = \"docker_vllm\"\nmodel_type = \"torch\"\nbenchmark_type = \"eval\"\ntensor_parallel_size = 2\nmax_model_len = 8448\nprefix_caching = false\ngpus = \"all\"\nenvs = [\"HF_TOKEN=hunter2\", \"NCCL_DEBUG\"]\nraw = [\"--swap-space\", \"8\"]",
        )?;
        assert_eq!(
            f.submission_value(),
            serde_json::json!({
                "tensor_parallel_size": 2,
                "max_model_len": 8448,
                "prefix_caching": false,
                "gpus": "all",
                "envs": ["HF_TOKEN", "NCCL_DEBUG"],
                "raw": ["--swap-space", "8"],
            })
        );
        Ok(())
    }

    /// A cell that configured nothing submits an empty object: the axis keys
    /// are the payload's job, not this record's.
    #[test]
    fn submission_value_is_empty_when_the_cell_configures_nothing() -> anyhow::Result<()> {
        let f = parse(
            "runtime_type = \"docker_vllm\"\nmodel_type = \"torch\"\nbenchmark_type = \"eval\"",
        )?;
        assert_eq!(f.submission_value(), serde_json::json!({}));
        Ok(())
    }

    /// A docker cell carries the `docker run` launcher settings; the accessors read
    /// them back and the whole set survives the wire round-trip.
    #[test]
    fn docker_launch_settings_round_trip() -> anyhow::Result<()> {
        let f = parse(
            "runtime_type = \"docker_vllm\"\nmodel_type = \"torch\"\nbenchmark_type = \"end_to_end_latency\"\ngpus = \"all\"\nshm_size = \"16g\"\nipc = \"host\"\nenvs = [\"HF_TOKEN=x\"]",
        )?;
        assert!(matches!(f, RuntimeFlags::EndToEndDockerVllmTorch { .. }));
        assert_eq!(f.gpus(), Some("all"));
        assert_eq!(f.shm_size(), Some("16g"));
        assert_eq!(f.ipc(), Some("host"));
        assert_eq!(f.envs(), ["HF_TOKEN=x".to_string()]);
        assert_eq!(parse(&toml::to_string(&f)?)?, f);
        Ok(())
    }

    /// A uv cell carries `envs` but never the docker-only launcher settings — the
    /// accessors return `None` for them regardless of the variant.
    #[test]
    fn uv_carries_envs_but_not_docker_settings() -> anyhow::Result<()> {
        let f = parse(
            "runtime_type = \"uv_vllm\"\nmodel_type = \"torch\"\nbenchmark_type = \"eval\"\nenvs = [\"VLLM_USE_V1=1\"]",
        )?;
        assert!(matches!(f, RuntimeFlags::EvalUvVllmTorch { .. }));
        assert_eq!(f.envs(), ["VLLM_USE_V1=1".to_string()]);
        assert_eq!(f.gpus(), None);
        assert_eq!(f.shm_size(), None);
        assert_eq!(f.ipc(), None);
        assert_eq!(parse(&toml::to_string(&f)?)?, f);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Exhaustive field-acceptance sweep
    // -----------------------------------------------------------------------

    /// Every optional field on [`RuntimeFlagRef`], with a wire value that is
    /// valid for its type. `raw` uses a flag no denylist names, so a rejection
    /// can only mean the cell refuses `raw` outright rather than that this
    /// particular flag is reserved.
    const PROBES: &[(&str, &str)] = &[
        ("threads", "4"),
        ("number_gpu_layers", "99"),
        ("mmap", "true"),
        ("flash_attention", "\"auto\""),
        ("ctx_size", "8192"),
        ("n_ubatch", "512"),
        ("swa_full", "true"),
        ("no_cache", "true"),
        ("tensor_parallel_size", "2"),
        ("dtype", "\"bfloat16\""),
        ("gpus", "\"all\""),
        ("shm_size", "\"16g\""),
        ("ipc", "\"host\""),
        ("envs", "[\"HF_TOKEN=x\"]"),
        ("raw", "[\"--pipette-field-probe\"]"),
    ];

    /// Every cell that has a variant, as `(benchmark, runtime, model, accepted
    /// fields)` — transcribed from the variant field lists, not derived from
    /// them, so a field added to or dropped from a variant without a matching
    /// edit here fails the sweep.
    #[rustfmt::skip]
    const CELLS: &[(&str, &str, &str, &[&str])] = &[
        // llama.cpp CLI bench cells (llama-bench): no server or ubatch fields.
        ("prefill_throughput", "llamacpp_cli_stock_tools", "gguf_text",
         &["threads", "number_gpu_layers", "mmap", "flash_attention", "raw"]),
        ("decode_throughput", "llamacpp_cli_stock_tools", "gguf_text",
         &["threads", "number_gpu_layers", "mmap", "flash_attention", "raw"]),
        ("max_memory_usage", "llamacpp_cli_stock_tools", "gguf_text",
         &["threads", "number_gpu_layers", "mmap", "flash_attention", "raw"]),
        // llama.cpp CLI server cells (llama-server): + ctx_size / no_cache.
        ("end_to_end_latency", "llamacpp_cli_stock_tools", "gguf_text",
         &["threads", "number_gpu_layers", "mmap", "flash_attention", "ctx_size", "no_cache", "raw"]),
        ("eval", "llamacpp_cli_stock_tools", "gguf_text",
         &["threads", "number_gpu_layers", "mmap", "flash_attention", "ctx_size", "no_cache", "raw"]),
        ("vl_throughput", "llamacpp_cli_stock_tools", "gguf_vision",
         &["threads", "number_gpu_layers", "mmap", "flash_attention", "ctx_size", "no_cache", "raw"]),
        // Docker vLLM: server fields + `docker run` launcher settings + envs.
        ("end_to_end_latency", "docker_vllm", "torch",
         &["tensor_parallel_size", "dtype", "gpus", "shm_size", "ipc", "envs", "raw"]),
        ("eval", "docker_vllm", "torch",
         &["tensor_parallel_size", "dtype", "gpus", "shm_size", "ipc", "envs", "raw"]),
        ("max_memory_usage", "docker_vllm", "torch",
         &["tensor_parallel_size", "dtype", "gpus", "shm_size", "ipc", "envs", "raw"]),
        // Uv vLLM: same server fields, no launcher settings (runs on the host).
        ("end_to_end_latency", "uv_vllm", "torch",
         &["tensor_parallel_size", "dtype", "envs", "raw"]),
        ("eval", "uv_vllm", "torch",
         &["tensor_parallel_size", "dtype", "envs", "raw"]),
        ("max_memory_usage", "uv_vllm", "torch",
         &["tensor_parallel_size", "dtype", "envs", "raw"]),
        // Docker SGLang: as docker vLLM but no `dtype`.
        ("end_to_end_latency", "docker_sglang", "torch",
         &["tensor_parallel_size", "gpus", "shm_size", "ipc", "envs", "raw"]),
        ("eval", "docker_sglang", "torch",
         &["tensor_parallel_size", "gpus", "shm_size", "ipc", "envs", "raw"]),
        ("max_memory_usage", "docker_sglang", "torch",
         &["tensor_parallel_size", "gpus", "shm_size", "ipc", "envs", "raw"]),
        // Uv SGLang: tensor_parallel_size + envs only.
        ("end_to_end_latency", "uv_sglang", "torch", &["tensor_parallel_size", "envs", "raw"]),
        ("eval", "uv_sglang", "torch", &["tensor_parallel_size", "envs", "raw"]),
        ("max_memory_usage", "uv_sglang", "torch", &["tensor_parallel_size", "envs", "raw"]),
        // iOS llama.cpp (in-process): app load-path fields, no CLI argv surface.
        // `threads` is a load-path field here (llama's `n_threads`), unlike `mmap` /
        // `flash_attention` / `no_cache`, which only exist as argv.
        ("prefill_throughput", "llamacpp_ios_pipette", "gguf_text",
         &["number_gpu_layers", "ctx_size", "n_ubatch", "threads", "swa_full"]),
        ("decode_throughput", "llamacpp_ios_pipette", "gguf_text",
         &["number_gpu_layers", "ctx_size", "n_ubatch", "threads", "swa_full"]),
        ("max_memory_usage", "llamacpp_ios_pipette", "gguf_text",
         &["number_gpu_layers", "ctx_size", "n_ubatch", "threads", "swa_full"]),
        ("end_to_end_latency", "llamacpp_ios_pipette", "gguf_text",
         &["number_gpu_layers", "ctx_size", "n_ubatch", "threads", "swa_full"]),
        ("eval", "llamacpp_ios_pipette", "gguf_text",
         &["number_gpu_layers", "ctx_size", "n_ubatch", "threads", "swa_full"]),
        ("vl_throughput", "llamacpp_ios_pipette", "gguf_vision",
         &["number_gpu_layers", "ctx_size", "n_ubatch", "threads", "swa_full"]),
        // iOS MLX: prefill-chunk size only.
        ("prefill_throughput", "mlx_ios_pipette", "mlx", &["n_ubatch"]),
        ("decode_throughput", "mlx_ios_pipette", "mlx", &["n_ubatch"]),
        ("max_memory_usage", "mlx_ios_pipette", "mlx", &["n_ubatch"]),
        ("end_to_end_latency", "mlx_ios_pipette", "mlx", &["n_ubatch"]),
        ("eval", "mlx_ios_pipette", "mlx", &["n_ubatch"]),
        // OpenVINO: typed pipeline properties, no raw and no env forwards.
        ("prefill_throughput", "uv_openvino", "openvino",
         &["max_prompt_len", "min_response_len", "generate_hint"]),
        ("decode_throughput", "uv_openvino", "openvino",
         &["max_prompt_len", "min_response_len", "generate_hint"]),
        ("end_to_end_latency", "uv_openvino", "openvino",
         &["max_prompt_len", "min_response_len", "generate_hint"]),
        ("max_memory_usage", "uv_openvino", "openvino",
         &["max_prompt_len", "min_response_len", "generate_hint"]),
    ];

    fn axes_wire(benchmark: &str, runtime: &str, model: &str) -> String {
        format!(
            "runtime_type = \"{runtime}\"\nmodel_type = \"{model}\"\nbenchmark_type = \"{benchmark}\"\n"
        )
    }

    /// The [`CELLS`] table covers every variant exactly once. Without this the
    /// sweep below could silently skip a cell — a new variant would add no
    /// coverage and nothing would say so.
    #[test]
    fn cell_table_covers_every_variant() -> anyhow::Result<()> {
        let mut seen = std::collections::HashSet::new();
        for (benchmark, runtime, model, _) in CELLS {
            let flags = parse(&axes_wire(benchmark, runtime, model)).map_err(|e| {
                anyhow::anyhow!("{benchmark}/{runtime}/{model} has no variant: {e}")
            })?;
            anyhow::ensure!(
                seen.insert(std::mem::discriminant(&flags)),
                "{benchmark}/{runtime}/{model} duplicates an earlier row's variant"
            );
        }
        anyhow::ensure!(
            seen.len() == RuntimeFlags::COUNT,
            "table covers {} variants, enum has {}",
            seen.len(),
            RuntimeFlags::COUNT
        );
        Ok(())
    }

    /// Every variant accepts exactly the fields it declares, rejects every
    /// other one, and carries the values it accepted.
    ///
    /// The `TryFrom<RuntimeFlagRef>` routing promises this cell by cell in
    /// hand-written arms: nothing structural forces a variant's arm to deny the
    /// fields that variant has no home for, nor to copy the ones it does. A
    /// missing `deny` silently drops an author's flag; a stray one rejects a
    /// legitimate cell; an arm that accepts a field and forgets to read it off
    /// the ref drops the value while still parsing. Sweeping [`CELLS`] ×
    /// [`PROBES`] pins all three.
    ///
    /// The value check compares against the **authored wire**. The
    /// `parse(to_string(f)) == f` idiom used elsewhere in this module cannot
    /// stand in for it: a dropped field is absent from both sides of that
    /// comparison, so it agrees precisely when the value was lost.
    #[test]
    fn every_cell_accepts_exactly_its_declared_fields_and_keeps_their_values() -> anyhow::Result<()>
    {
        for (benchmark, runtime, model, accepted) in CELLS {
            for (field, value) in PROBES {
                let wire = format!(
                    "{}{field} = {value}\n",
                    axes_wire(benchmark, runtime, model)
                );
                let expected = accepted.contains(field);
                let parsed = parse(&wire);
                anyhow::ensure!(
                    parsed.is_ok() == expected,
                    "{benchmark}/{runtime}/{model}: `{field}` should be {} but was {}",
                    if expected { "accepted" } else { "rejected" },
                    if expected { "rejected" } else { "accepted" },
                );
                let Ok(flags) = parsed else { continue };
                let emitted = toml::Value::try_from(&flags)?;
                let authored = toml::from_str::<toml::Value>(&format!("{field} = {value}"))?;
                anyhow::ensure!(
                    emitted.get(field) == authored.get(field),
                    "{benchmark}/{runtime}/{model}: `{field}` was accepted but came back as {:?}, authored {:?}",
                    emitted.get(field),
                    authored.get(field),
                );
            }
        }
        Ok(())
    }
    /// The knobs only exist for OpenVINO cells. Accepting one elsewhere would
    /// report it on the submission as though it had applied to something.
    #[test]
    fn openvino_knobs_are_rejected_on_other_runtimes() -> anyhow::Result<()> {
        let mut r = RuntimeFlagRef::new(
            BenchmarkType::DecodeThroughput,
            RuntimeType::LlamacppCliStockTools,
            ModelType::GgufText,
        );
        r.min_response_len = Some(256);
        let Err(err) = RuntimeFlags::try_from(r) else {
            anyhow::bail!("expected min_response_len to be refused on a llama.cpp cell");
        };
        assert!(err.to_string().contains("min_response_len"), "got {err}");
        Ok(())
    }

    /// A device outside the named set is the operator's to spell — an indexed
    /// GPU or a virtual device is legitimate — so it has to survive the trip a
    /// plan takes: authored text, typed flags, and back.
    #[test]
    fn an_unnamed_openvino_device_survives_the_flags() -> anyhow::Result<()> {
        let mut r = RuntimeFlagRef::new(
            BenchmarkType::EndToEndLatency,
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        r.device = Some(OpenvinoDevice::Custom("GPU.1".to_owned()));

        let flags = RuntimeFlags::try_from(r)?;
        let json = serde_json::to_string(&flags)?;
        assert!(json.contains("\"device\":\"GPU.1\""), "got {json}");

        let back: RuntimeFlagRef = serde_json::from_str::<RuntimeFlags>(&json)?.into();
        assert_eq!(
            back.device,
            Some(OpenvinoDevice::Custom("GPU.1".to_owned()))
        );
        Ok(())
    }

    /// Round-trip through the flat wire form, which is how a plan authors them.
    #[test]
    fn openvino_knobs_round_trip_through_the_ref() -> anyhow::Result<()> {
        let mut r = RuntimeFlagRef::new(
            BenchmarkType::EndToEndLatency,
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        r.max_prompt_len = Some(1024);
        r.min_response_len = Some(256);
        r.generate_hint = Some(OpenvinoGenerateHint::BestPerf);

        let flags = RuntimeFlags::try_from(r)?;
        assert_eq!(
            flags.axes(),
            (
                BenchmarkType::EndToEndLatency,
                RuntimeType::UvOpenvino,
                ModelType::Openvino
            )
        );
        let back: RuntimeFlagRef = flags.into();
        assert_eq!(back.max_prompt_len, Some(1024));
        assert_eq!(back.min_response_len, Some(256));
        assert_eq!(back.generate_hint, Some(OpenvinoGenerateHint::BestPerf));
        Ok(())
    }

    /// OpenVINO takes typed properties, not a command line, so there is no
    /// escape hatch and a `raw` entry has to be refused rather than ignored.
    #[test]
    fn openvino_cells_take_no_raw_entries() -> anyhow::Result<()> {
        let mut r = RuntimeFlagRef::new(
            BenchmarkType::DecodeThroughput,
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        r.raw = vec!["--some-flag".to_owned()];
        let Err(err) = RuntimeFlags::try_from(r) else {
            anyhow::bail!("expected raw to be refused on an openvino cell");
        };
        assert!(err.to_string().contains("raw"), "got {err}");
        Ok(())
    }
    /// The property is a closed, case-sensitive vocabulary — OpenVINO rejects
    /// `best_perf` as readily as nonsense — so the wire form is validated at
    /// parse rather than at device compile.
    #[test]
    fn generate_hint_is_a_closed_vocabulary() -> anyhow::Result<()> {
        assert_eq!(OpenvinoGenerateHint::BestPerf.as_property(), "BEST_PERF");
        assert_eq!(
            OpenvinoGenerateHint::FastCompile.as_property(),
            "FAST_COMPILE"
        );
        // Authored kebab-case, rendered SCREAMING_SNAKE for the plugin.
        let parsed: OpenvinoGenerateHint = serde_json::from_str("\"best-perf\"")?;
        assert_eq!(parsed, OpenvinoGenerateHint::BestPerf);
        assert!(serde_json::from_str::<OpenvinoGenerateHint>("\"BEST_PERF\"").is_err());
        assert!(serde_json::from_str::<OpenvinoGenerateHint>("\"turbo\"").is_err());
        Ok(())
    }
    /// Overriding the derived pipeline properties is allowed, so the override
    /// has to be *visible*: a cell that raised the static shape or changed the
    /// compile hint measures something different from one that did not, under
    /// the same benchmark id. `submission_value` is what the warehouse sees, so
    /// an authored knob must appear there — and a derived default must not,
    /// or every row would claim a setting it never made.
    #[test]
    fn an_authored_override_is_visible_on_the_submission() -> anyhow::Result<()> {
        let mut r = RuntimeFlagRef::new(
            BenchmarkType::DecodeThroughput,
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        r.max_prompt_len = Some(2048);
        r.generate_hint = Some(OpenvinoGenerateHint::FastCompile);
        let submitted = RuntimeFlags::try_from(r)?.submission_value();

        assert_eq!(
            submitted.get("max_prompt_len"),
            Some(&serde_json::json!(2048))
        );
        assert_eq!(
            submitted.get("generate_hint"),
            Some(&serde_json::json!("fast-compile"))
        );
        // Not authored, so not claimed.
        assert!(submitted.get("min_response_len").is_none());

        // A cell that overrode nothing carries no properties at all, rather
        // than reporting the values the runtime derived for it.
        let bare = RuntimeFlagRef::new(
            BenchmarkType::DecodeThroughput,
            RuntimeType::UvOpenvino,
            ModelType::Openvino,
        );
        let bare = RuntimeFlags::try_from(bare)?.submission_value();
        ["max_prompt_len", "min_response_len", "generate_hint"]
            .into_iter()
            .for_each(|key| assert!(bare.get(key).is_none(), "{key} should be absent"));
        Ok(())
    }
}
