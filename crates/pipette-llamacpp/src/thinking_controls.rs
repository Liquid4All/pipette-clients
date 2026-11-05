//! What could decide how llama-server's model thinks, or swap the model it
//! serves, outside a cell's typed knobs: a raw flag, an environment variable,
//! or a key in one of llama.cpp's own config files.
//! [`crate::server::ServerCommandBuilder`] refuses them on a vlm_eval cell,
//! whose recipe owns both. An ASR cell, whose recipe owns its template and
//! thinking, gets the config files' check alone; every other cell launches
//! without these checks. The names are the vendored llama.cpp's
//! (`common/arg.cpp`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use pipette_plan_types::EnvForward;

use crate::runtimes::canonical_flag_name;

/// The flag that renders the chat template without Jinja, which alone reads
/// `enable_thinking`: under it every thinking switch is invisible to
/// [`crate::server::thinking_switch`] and out of a vlm_eval cell's reach.
pub const NO_JINJA: &str = "--no-jinja";

/// llama.cpp's flags for whether and how a model thinks, which on a vlm_eval
/// cell the model flags' `enable_thinking` decides ([`crate::server::thinking`]).
/// Plan-types already closes `--reasoning`, `--reasoning-budget` and
/// `--reasoning-format` to `raw` on every server cell.
pub const REASONING_FLAGS: &[&str] = &[
    "-rea",
    "--reasoning",
    "--reasoning-budget",
    "--reasoning-budget-message",
    "--reasoning-effort",
    "--reasoning-format",
    "--reasoning-preserve",
    "--no-reasoning-preserve",
    "--chat-template-kwargs",
];

/// llama.cpp's flags that replace the chat template (which alone renders
/// `enable_thinking`), prefill the reply from a trailing assistant message, or
/// skip parsing the reply (leaving the reasoning in the content). Set out of
/// sight, in a variable or a config key, they change the prompt, the thinking
/// or the row for a reason no recorded command shows; on argv they stay
/// visible, and a vlm_eval's recipe owns them ([`RECIPE_FLAGS`]).
pub const TEMPLATE_FLAGS: &[&str] = &[
    "--chat-template",
    "--chat-template-file",
    "--prefill-assistant",
    "--no-prefill-assistant",
    "--skip-chat-parsing",
    "--no-skip-chat-parsing",
];

/// llama.cpp's flags that take the model or its projector from somewhere
/// other than the `--model` and `--mmproj` paths pipette launches: a download
/// (`-mu`, `-dr`, `-hf`, `-mmu`) replaces one, a raw projector after pipette's
/// overrides it, and a Hugging Face repo whose root holds a `preset.ini` turns
/// the server into a router running that file's options, which no recorded
/// command shows. The router's own sources go with them, and so do the
/// built-in presets, which set `--hf-repo` themselves. Refused on a vlm_eval
/// cell. `--no-mmproj` and the offload flags are not here: they change where a
/// projector runs or whether `-hf` fetches one, not which one serves.
pub const MODEL_SOURCE_FLAGS: &[&str] = &[
    "-hf",
    "-hfr",
    "--hf-repo",
    "-hff",
    "--hf-file",
    "-dr",
    "--docker-repo",
    "-mu",
    "--model-url",
    "-mm",
    "--mmproj",
    "-mmu",
    "--mmproj-url",
    "--models-dir",
    "--models-preset",
    "--embd-gemma-default",
    "--fim-qwen-1.5b-default",
    "--fim-qwen-3b-default",
    "--fim-qwen-7b-default",
    "--fim-qwen-7b-spec",
    "--fim-qwen-14b-spec",
    "--fim-qwen-30b-default",
    "--gpt-oss-20b-default",
    "--gpt-oss-120b-default",
    "--vision-gemma-4b-default",
    "--vision-gemma-12b-default",
];

/// The environment variables of `--jinja`, [`REASONING_FLAGS`],
/// [`TEMPLATE_FLAGS`] and [`MODEL_SOURCE_FLAGS`], with `LLAMA_ARG_NO_<NAME>`
/// for an option with a negative flag (`common_arg::get_value_from_env`).
/// llama-server reads them before its argv, and the recorded command is argv
/// alone, so a server that saw one would think, or serve another model, for a
/// reason no row shows.
pub const HIDDEN_ENVS: &[&str] = &[
    "LLAMA_ARG_JINJA",
    "LLAMA_ARG_NO_JINJA",
    "LLAMA_ARG_REASONING",
    "LLAMA_ARG_THINK_BUDGET",
    "LLAMA_ARG_THINK_BUDGET_MESSAGE",
    "LLAMA_ARG_REASONING_EFFORT",
    "LLAMA_ARG_THINK",
    "LLAMA_ARG_REASONING_PRESERVE",
    "LLAMA_ARG_NO_REASONING_PRESERVE",
    "LLAMA_ARG_CHAT_TEMPLATE_KWARGS",
    "LLAMA_ARG_CHAT_TEMPLATE",
    "LLAMA_ARG_CHAT_TEMPLATE_FILE",
    "LLAMA_ARG_PREFILL_ASSISTANT",
    "LLAMA_ARG_NO_PREFILL_ASSISTANT",
    "LLAMA_ARG_SKIP_CHAT_PARSING",
    "LLAMA_ARG_NO_SKIP_CHAT_PARSING",
    "LLAMA_ARG_HF_REPO",
    "LLAMA_ARG_HF_FILE",
    "LLAMA_ARG_DOCKER_REPO",
    "LLAMA_ARG_MODEL_URL",
    "LLAMA_ARG_MMPROJ",
    "LLAMA_ARG_MMPROJ_URL",
    "LLAMA_ARG_MODELS_DIR",
    "LLAMA_ARG_MODELS_PRESET",
];

/// What a vlm_eval's recipe (its catalog entry) owns besides thinking, Jinja
/// and the model's source. Plan-types refuses them in a vlm_eval's `raw`; here
/// they are the keys a llama.cpp config file may not set under a vlm_eval,
/// since it applies them to every launch.
pub const RECIPE_FLAGS: &[&str] = &[
    "--samplers",
    "--sampler-seq",
    "--sampling-seq",
    "-s",
    "--seed",
    "--ignore-eos",
    "--temp",
    "--temperature",
    "--top-k",
    "--top-p",
    "--min-p",
    "--top-nsigma",
    "--top-n-sigma",
    "--xtc-probability",
    "--xtc-threshold",
    "--typical",
    "--typical-p",
    "--repeat-last-n",
    "--repeat-penalty",
    "--presence-penalty",
    "--frequency-penalty",
    "--dry-multiplier",
    "--dry-base",
    "--dry-allowed-length",
    "--dry-penalty-last-n",
    "--dry-sequence-breaker",
    "--adaptive-target",
    "--adaptive-decay",
    "--dynatemp-range",
    "--dynatemp-exp",
    "--mirostat",
    "--mirostat-lr",
    "--mirostat-ent",
    "-l",
    "--logit-bias",
    "--grammar",
    "--grammar-file",
    "-j",
    "--json-schema",
    "-jf",
    "--json-schema-file",
    "-bs",
    "--backend-sampling",
    "-n",
    "--predict",
    "--n-predict",
    "-r",
    "--reverse-prompt",
    "-sp",
    "--special",
    "--chat-template",
    "--chat-template-file",
    "--prefill-assistant",
    "--no-prefill-assistant",
    "--skip-chat-parsing",
    "--no-skip-chat-parsing",
    "-sys",
    "--system-prompt",
    "-sysf",
    "--system-prompt-file",
    "--override-kv",
    "--image-min-tokens",
    "--image-max-tokens",
    "--video-fps",
    "--video-timestamp-interval",
    "--no-mmproj",
    "--no-mmproj-auto",
    "--lora",
    "--lora-scaled",
    "--lora-init-without-apply",
    "--control-vector",
    "--control-vector-scaled",
    "--control-vector-layer-range",
    "--dflash",
    "--rope-scaling",
    "--rope-scale",
    "--rope-freq-base",
    "--rope-freq-scale",
    "--yarn-orig-ctx",
    "--yarn-ext-factor",
    "--yarn-attn-factor",
    "--yarn-beta-slow",
    "--yarn-beta-fast",
    "--context-shift",
    "--no-context-shift",
    "--keep",
];

const LLAMA_ARG_PREFIX: &str = "LLAMA_ARG_";

/// Whether `name` is a llama.cpp option variable, in any case.
pub fn is_llama_arg(name: &str) -> bool {
    name.get(..LLAMA_ARG_PREFIX.len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case(LLAMA_ARG_PREFIX))
}

/// The first llama.cpp option variable `envs` set: none may reach a vlm_eval's
/// llama-server or its calibration.
pub fn llama_arg_env<'a>(envs: impl IntoIterator<Item = &'a EnvForward>) -> Option<String> {
    envs.into_iter()
        .find(|env| is_llama_arg(env.name()))
        .map(|env| env.name().to_owned())
}

/// The variable llama.cpp builds before b9360 read `--chat-template-kwargs`
/// from, which an asr_eval's prompt profile owns.
pub const TEMPLATE_KWARGS_ENV: &str = "LLAMA_CHAT_TEMPLATE_KWARGS";

/// Whether `envs` set [`TEMPLATE_KWARGS_ENV`], in any case.
pub fn template_kwargs_env<'a>(envs: impl IntoIterator<Item = &'a EnvForward>) -> Option<String> {
    envs.into_iter()
        .find(|env| env.name().eq_ignore_ascii_case(TEMPLATE_KWARGS_ENV))
        .map(|env| env.name().to_owned())
}

/// Drop from `command`'s environment every llama.cpp option variable pipette
/// itself inherited, so a vlm_eval's tools see only their recorded argv.
pub fn remove_inherited_llama_args(command: &mut std::process::Command) {
    for (name, _) in std::env::vars_os() {
        if is_llama_arg(&name.to_string_lossy()) {
            command.env_remove(name);
        }
    }
}

/// What an asr_eval's prompt profile and entry own beyond a vlm_eval's
/// [`RECIPE_FLAGS`]: the beam search, and the slot split and per-slot limit
/// the context is sized for (the chat template, the metadata overrides that
/// carry MiniCPM-o's end-of-turn token, the sampler chain every request spells
/// and the output cap, which is the entry's `parameter_max_tokens`, are a
/// recipe's on both). Refused in that cell's `raw`, which plan-types also
/// closes to the recipe's flags, and as a llama.cpp config-file key. The names
/// are those of pipette's ASR build of llama.cpp (`common/arg.cpp`), whose
/// `--beam-search` and `--kv-unified-per-slot` stock llama.cpp lacks.
pub const ASR_OWNED_FLAGS: &[&str] = &[
    "--chat-template",
    "--chat-template-file",
    "--override-kv",
    "--beam-search",
    "--samplers",
    "--sampler-seq",
    "--sampling-seq",
    "-np",
    "--parallel",
    "--kv-unified-per-slot",
    "-n",
    "--predict",
    "--n-predict",
];

/// Whose catalog recipe a launch runs under, which decides what its llama.cpp
/// config files may not set ([`config_control`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipeKeys {
    /// No recipe: only thinking, Jinja and the model's source are guarded.
    None,
    /// A vlm_eval's: also [`RECIPE_FLAGS`], a drafter and every option
    /// through its variable name.
    VlmEval,
    /// An asr_eval's: a vlm_eval's, and [`ASR_OWNED_FLAGS`].
    AsrEval,
}

/// The first of `names` in `raw`, spelled any way llama.cpp parses it.
pub fn raw_flag(raw: &[String], names: &[&'static str]) -> Option<&'static str> {
    raw.iter().find_map(|token| {
        let token = canonical_flag_name(token);
        names.iter().copied().find(|name| *name == token)
    })
}

/// The first of [`HIDDEN_ENVS`] that `envs` set, in any case: Windows reads a
/// variable's name in any.
pub fn hidden_env<'a>(envs: impl IntoIterator<Item = &'a EnvForward>) -> Option<&'static str> {
    envs.into_iter().find_map(|env| {
        HIDDEN_ENVS
            .iter()
            .copied()
            .find(|name| name.eq_ignore_ascii_case(env.name()))
    })
}

/// The config files llama.cpp applies to every launch before its environment
/// and argv (`common_params_apply_system_config`), system-wide then the
/// user's, under the launch's environment `var`. There is no flag to turn
/// them off. Without a `HOME` llama.cpp takes the account's home directory,
/// [`account_home`] in production.
pub fn config_files(
    var: impl Fn(&str) -> Option<OsString>,
    account_home: impl FnOnce() -> Option<PathBuf>,
) -> Vec<PathBuf> {
    let dir = |name: &str| {
        var(name)
            .filter(|value| !value.is_empty())
            .map(PathBuf::from)
    };
    let dirs = if cfg!(windows) {
        [dir("PROGRAMDATA"), dir("APPDATA")]
    } else {
        [
            Some(PathBuf::from("/etc")),
            dir("XDG_CONFIG_HOME").or_else(|| {
                dir("HOME")
                    .or_else(account_home)
                    .map(|home| home.join(".config"))
            }),
        ]
    };
    dirs.into_iter()
        .flatten()
        .map(|dir| dir.join("llama.cpp").join("config.ini"))
        .collect()
}

/// The account's home directory, where llama.cpp looks when a launch has no
/// `HOME`: `getpwuid(getuid())->pw_dir`, not pipette's own `HOME`.
#[cfg(unix)]
pub fn account_home() -> Option<PathBuf> {
    // SAFETY: takes nothing and cannot fail.
    let uid = unsafe { libc::getuid() };
    // SAFETY: takes a constant and touches no memory.
    let size_max = unsafe { libc::sysconf(libc::_SC_GETPW_R_SIZE_MAX) };
    let start = usize::try_from(size_max)
        .ok()
        .filter(|size| *size > 0)
        .unwrap_or(PASSWD_BUFFER_FALLBACK);
    with_growing_buffer(start, |strings| home_of(uid, strings))
}

/// The first `getpwuid_r` buffer where `sysconf` names no size.
#[cfg(unix)]
const PASSWD_BUFFER_FALLBACK: usize = 1024;

/// The largest `getpwuid_r` buffer tried: past it an entry is not a home
/// directory worth looking for.
#[cfg(unix)]
const PASSWD_BUFFER_CAP: usize = 1 << 20;

#[cfg(unix)]
#[derive(Debug)]
enum Passwd {
    Home(PathBuf),
    /// No entry, no home directory, or a failure other than a short buffer.
    None,
    /// `ERANGE`: the entry's strings did not fit.
    TooSmall,
}

#[cfg(unix)]
fn with_growing_buffer(
    start: usize,
    mut lookup: impl FnMut(&mut [libc::c_char]) -> Passwd,
) -> Option<PathBuf> {
    let mut size = start;
    loop {
        match lookup(&mut vec![0; size]) {
            Passwd::Home(home) => return Some(home),
            Passwd::None => return None,
            Passwd::TooSmall => {
                size = size
                    .checked_mul(2)
                    .filter(|size| *size <= PASSWD_BUFFER_CAP)?;
            }
        }
    }
}

#[cfg(unix)]
fn home_of(uid: libc::uid_t, strings: &mut [libc::c_char]) -> Passwd {
    use std::ffi::{CStr, OsStr};
    use std::os::unix::ffi::OsStrExt;

    let mut entry = std::mem::MaybeUninit::<libc::passwd>::uninit();
    let mut found = std::ptr::null_mut();
    // SAFETY: each pointer is to a live local, `strings` as long as passed.
    let status = unsafe {
        libc::getpwuid_r(
            uid,
            entry.as_mut_ptr(),
            strings.as_mut_ptr(),
            strings.len(),
            &raw mut found,
        )
    };
    if status == libc::ERANGE {
        return Passwd::TooSmall;
    }
    if status != 0 || found.is_null() {
        return Passwd::None;
    }
    // SAFETY: on success `found` points at `entry`, filled in, and its
    // `pw_dir` at a NUL-terminated string in `strings`, both still alive.
    let dir = unsafe { (*found).pw_dir };
    if dir.is_null() {
        return Passwd::None;
    }
    // SAFETY: as above.
    let dir = unsafe { CStr::from_ptr(dir) }.to_bytes();
    if dir.is_empty() {
        return Passwd::None;
    }
    Passwd::Home(PathBuf::from(OsStr::from_bytes(dir)))
}

/// Windows takes no account directory: without `%APPDATA%` llama.cpp skips the
/// user's file.
#[cfg(not(unix))]
pub fn account_home() -> Option<PathBuf> {
    None
}

/// The first key of the config file at `path` that sets a control
/// [`hidden_env`] or a raw flag check guards, `None` when there is no such
/// file.
///
/// Read wider than llama.cpp's preset loader, which matches a key exactly
/// against option names without their dashes and env names, and applies only
/// the unheaded, `[default]` and `[*]` sections at startup: here dashes are
/// stripped, `_` reads as `-`, an env name matches in any case and every
/// section counts, so no spelling or section a later loader might honour
/// slips by. Lines end where llama.cpp's own parser ends them: `\r`, `\n` or
/// `\r\n`.
pub fn config_control(path: &Path, recipe: RecipeKeys) -> std::io::Result<Option<String>> {
    // Lossy: llama.cpp reads the file as bytes, so a non-UTF-8 comment must not
    // make it unreadable here.
    let text = match std::fs::read(path) {
        Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err),
    };
    Ok(text
        .split(['\r', '\n'])
        .filter_map(|line| {
            let line = line.trim();
            let (key, _) = line.split_once('=')?;
            (!line.starts_with(['[', ';', '#'])).then(|| key.trim())
        })
        .find(|key| guarded_key(key, recipe))
        .map(str::to_owned))
}

/// Whether a config file's `key` sets a guarded control. `jinja` is one there,
/// though `--jinja` is harmless on argv: `jinja = false` turns Jinja off. On a
/// vlm_eval every option variable and every [`RECIPE_FLAGS`] option is one too;
/// on an asr_eval every [`ASR_OWNED_FLAGS`] option as well.
fn guarded_key(key: &str, recipe: RecipeKeys) -> bool {
    // A recipe's cell, whichever: every option variable is closed.
    let closed = recipe != RecipeKeys::None;
    if is_llama_arg(key) {
        return closed
            || HIDDEN_ENVS
                .iter()
                .any(|name| name.eq_ignore_ascii_case(key));
    }
    let option = key.trim_start_matches('-').replace('_', "-");
    let (recipe, asr): (&[&str], &[&str]) = match recipe {
        RecipeKeys::None => (&[], &[]),
        RecipeKeys::VlmEval => (RECIPE_FLAGS, &[]),
        RecipeKeys::AsrEval => (RECIPE_FLAGS, ASR_OWNED_FLAGS),
    };
    // A drafter changes how a sampled decode draws its tokens. A slot count
    // from the file would divide a context sized for the slots argv names.
    if closed
        && (option.starts_with("spec-")
            || option.contains("draft")
            || matches!(option.as_str(), "parallel" | "np"))
    {
        return true;
    }
    option == "jinja"
        || REASONING_FLAGS
            .iter()
            .chain(TEMPLATE_FLAGS)
            .chain(MODEL_SOURCE_FLAGS)
            .chain([&NO_JINJA])
            .chain(recipe)
            .chain(asr)
            .any(|flag| flag.trim_start_matches('-') == option)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    /// llama.cpp reads `_` as `-` in a long flag.
    #[rstest]
    #[case::canonical(&["--reasoning-budget", "0"], Some("--reasoning-budget"))]
    #[case::underscores(&["--reasoning_budget", "0"], Some("--reasoning-budget"))]
    #[case::no_jinja_underscore(&["--no_jinja"], Some("--no-jinja"))]
    #[case::kwargs_underscores(&["--chat_template_kwargs", "{}"], Some("--chat-template-kwargs"))]
    #[case::effort_underscore(&["--reasoning_effort", "low"], Some("--reasoning-effort"))]
    #[case::short(&["-rea", "on"], Some("-rea"))]
    #[case::hf_repo(&["-hf", "org/model"], Some("-hf"))]
    #[case::hf_repo_underscored(&["--hf_repo", "org/model"], Some("--hf-repo"))]
    #[case::hf_file(&["-hff", "model.gguf"], Some("-hff"))]
    #[case::models_preset(&["--models_preset", "p.ini"], Some("--models-preset"))]
    #[case::builtin_preset(&["--gpt-oss-20b-default"], Some("--gpt-oss-20b-default"))]
    #[case::fim_preset(&["--fim_qwen_7b_spec"], Some("--fim-qwen-7b-spec"))]
    #[case::projector(&["-mm", "other.gguf"], Some("-mm"))]
    #[case::projector_url(&["--mmproj_url", "https://x/p.gguf"], Some("--mmproj-url"))]
    #[case::projector_placement(&["--no-mmproj", "--mmproj-offload"], None)]
    // A short flag is looked up as it is.
    #[case::short_is_not_canonicalised(&["-r_ea", "on"], None)]
    #[case::other_flags(&["--jinja", "--verbose"], None)]
    fn raw_flags_match_every_spelling_llama_cpp_parses(
        #[case] raw: &[&str],
        #[case] expected: Option<&str>,
    ) {
        let raw: Vec<String> = raw.iter().map(|s| (*s).to_owned()).collect();
        let names: Vec<&'static str> = REASONING_FLAGS
            .iter()
            .chain(MODEL_SOURCE_FLAGS)
            .copied()
            .chain([NO_JINJA])
            .collect();
        assert_eq!(raw_flag(&raw, &names), expected);
    }

    #[cfg(unix)]
    #[rstest]
    #[case::xdg(&[("XDG_CONFIG_HOME", "/x"), ("HOME", "/h")], "/x/llama.cpp/config.ini")]
    #[case::home(&[("HOME", "/h")], "/h/.config/llama.cpp/config.ini")]
    #[case::empty_xdg(&[("XDG_CONFIG_HOME", ""), ("HOME", "/h")], "/h/.config/llama.cpp/config.ini")]
    #[case::empty_home(&[("HOME", "")], "/account/.config/llama.cpp/config.ini")]
    #[case::no_home(&[], "/account/.config/llama.cpp/config.ini")]
    fn the_config_files_are_llama_cpps(#[case] vars: &[(&str, &str)], #[case] user: &str) {
        let var = |name: &str| {
            vars.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| OsString::from(value))
        };
        assert_eq!(
            config_files(var, || Some(PathBuf::from("/account"))),
            [
                PathBuf::from("/etc/llama.cpp/config.ini"),
                PathBuf::from(user)
            ]
        );
    }

    #[cfg(windows)]
    #[test]
    fn the_config_files_are_llama_cpps_on_windows() {
        let var = |name: &str| match name {
            "PROGRAMDATA" => Some(OsString::from(r"C:\ProgramData")),
            "APPDATA" => Some(OsString::from(r"C:\Users\u\AppData\Roaming")),
            _ => None,
        };
        assert_eq!(
            config_files(var, || Some(PathBuf::from(r"C:\account"))),
            [
                PathBuf::from(r"C:\ProgramData\llama.cpp\config.ini"),
                PathBuf::from(r"C:\Users\u\AppData\Roaming\llama.cpp\config.ini"),
            ]
        );
    }

    #[rstest]
    #[case::option("reasoning = on", Some("reasoning"))]
    #[case::dashed_option("--reasoning-budget = 0", Some("--reasoning-budget"))]
    #[case::underscored_option("chat_template_kwargs = {}", Some("chat_template_kwargs"))]
    #[case::short_option("rea = on", Some("rea"))]
    #[case::jinja_off("jinja = false", Some("jinja"))]
    #[case::no_jinja("no-jinja = true", Some("no-jinja"))]
    #[case::env_name("LLAMA_ARG_THINK_BUDGET = 0", Some("LLAMA_ARG_THINK_BUDGET"))]
    #[case::lowercase_env_name("llama_arg_reasoning = on", Some("llama_arg_reasoning"))]
    #[case::in_a_section(
        "[*]\nthreads = 4\n[model]\nreasoning-effort = high",
        Some("reasoning-effort")
    )]
    #[case::lone_cr("[default]\rthreads = 4\rreasoning = on\r", Some("reasoning"))]
    #[case::hf_repo("hf-repo = org/presets", Some("hf-repo"))]
    #[case::hf_repo_env_name("LLAMA_ARG_HF_REPO = org/presets", Some("LLAMA_ARG_HF_REPO"))]
    #[case::models_preset("models_preset = p.ini", Some("models_preset"))]
    #[case::builtin_preset("vision-gemma-4b-default = true", Some("vision-gemma-4b-default"))]
    #[case::projector_url("mmproj-url = https://x/p.gguf", Some("mmproj-url"))]
    #[case::projector_env_name("LLAMA_ARG_MMPROJ = p.gguf", Some("LLAMA_ARG_MMPROJ"))]
    #[case::template("chat-template = chatml", Some("chat-template"))]
    #[case::template_file("chat-template-file = t.jinja", Some("chat-template-file"))]
    #[case::skip_chat_parsing("skip_chat_parsing = true", Some("skip_chat_parsing"))]
    #[case::prefill_assistant("no-prefill-assistant = true", Some("no-prefill-assistant"))]
    #[case::template_env_name(
        "LLAMA_ARG_CHAT_TEMPLATE_FILE = t.jinja",
        Some("LLAMA_ARG_CHAT_TEMPLATE_FILE")
    )]
    #[case::other_keys("threads = 4\nLLAMA_ARG_CTX_SIZE = 4096\n; reasoning = on", None)]
    fn a_config_key_that_sets_a_control_is_found(
        #[case] text: &str,
        #[case] expected: Option<&str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("config.ini");
        std::fs::write(&path, text)?;
        assert_eq!(
            config_control(&path, RecipeKeys::None)?.as_deref(),
            expected
        );
        assert_eq!(
            config_control(&dir.path().join("absent.ini"), RecipeKeys::None)?,
            None
        );
        Ok(())
    }

    #[test]
    fn a_config_that_is_not_utf8_is_still_read() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("config.ini");
        std::fs::write(&path, b"; caf\xe9\nreasoning = on\n")?;
        assert_eq!(
            config_control(&path, RecipeKeys::None)?.as_deref(),
            Some("reasoning")
        );
        Ok(())
    }

    #[rstest]
    #[case::sampler("top_k = 40", Some("top_k"))]
    #[case::sampler_dashed("--temp = 0.1", Some("--temp"))]
    #[case::image_budget("image-max-tokens = 32", Some("image-max-tokens"))]
    #[case::any_option_variable("LLAMA_ARG_CTX_SIZE = 4096", Some("LLAMA_ARG_CTX_SIZE"))]
    #[case::adapter("lora = a.gguf", Some("lora"))]
    #[case::positions("rope_freq_base = 1000000", Some("rope_freq_base"))]
    #[case::drafter("spec-type = ngram", Some("spec-type"))]
    #[case::draft_model("model-draft = d.gguf", Some("model-draft"))]
    #[case::slots("parallel = 4", Some("parallel"))]
    #[case::short_slots("np = 4", Some("np"))]
    #[case::performance("threads = 4\nubatch-size = 512", None)]
    fn a_vlm_evals_config_may_not_set_its_recipe(
        #[case] text: &str,
        #[case] vlm_eval: Option<&str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("config.ini");
        std::fs::write(&path, text)?;
        assert_eq!(
            config_control(&path, RecipeKeys::VlmEval)?.as_deref(),
            vlm_eval
        );
        assert_eq!(
            config_control(&path, RecipeKeys::None)?,
            None,
            "{text} elsewhere"
        );
        Ok(())
    }

    /// An asr_eval's config may set nothing a vlm_eval's may not, nor its
    /// beam search or per-slot cache, which a vlm_eval's config may.
    #[rstest]
    #[case::sampler("top_k = 40", Some("top_k"), Some("top_k"))]
    #[case::any_option_variable(
        "LLAMA_ARG_N_PARALLEL = 4",
        Some("LLAMA_ARG_N_PARALLEL"),
        Some("LLAMA_ARG_N_PARALLEL")
    )]
    #[case::drafter("spec-type = ngram", Some("spec-type"), Some("spec-type"))]
    #[case::beams("beam_search = 3", Some("beam_search"), None)]
    #[case::slots("np = 4", Some("np"), Some("np"))]
    #[case::per_slot("kv-unified-per-slot = 1024", Some("kv-unified-per-slot"), None)]
    #[case::performance("threads = 4\nubatch-size = 512", None, None)]
    fn an_asr_evals_config_may_not_set_its_recipe(
        #[case] text: &str,
        #[case] asr_eval: Option<&str>,
        #[case] vlm_eval: Option<&str>,
    ) -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("config.ini");
        std::fs::write(&path, text)?;
        assert_eq!(
            config_control(&path, RecipeKeys::AsrEval)?.as_deref(),
            asr_eval
        );
        assert_eq!(
            config_control(&path, RecipeKeys::VlmEval)?.as_deref(),
            vlm_eval
        );
        assert_eq!(
            config_control(&path, RecipeKeys::None)?,
            None,
            "{text} elsewhere"
        );
        Ok(())
    }

    #[test]
    fn a_config_whose_value_is_not_utf8_refuses_nothing() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("config.ini");
        std::fs::write(&path, b"threads = 4 ; \xff")?;
        assert_eq!(config_control(&path, RecipeKeys::None)?, None);
        Ok(())
    }

    #[rstest]
    #[case::upper("LLAMA_ARG_TOP_K", true)]
    #[case::lower("llama_arg_image_max_tokens", true)]
    #[case::too_short("LLAMA_ARG", false)]
    #[case::backend("GGML_CUDA_GRAPHS", false)]
    fn a_llama_option_variable_is_named_so_in_any_case(#[case] name: &str, #[case] is: bool) {
        assert_eq!(is_llama_arg(name), is);
    }

    #[cfg(unix)]
    #[test]
    fn the_account_home_is_absolute() {
        assert!(account_home().is_none_or(|home| home.is_absolute()));
    }

    #[cfg(unix)]
    #[rstest]
    #[case::fits_at_once(Some(1000), &[1024])]
    #[case::grows_until_it_fits(Some(4000), &[1024, 2048, 4096])]
    #[case::gives_up_past_the_cap(None, &[1024, 2048, 4096, 8192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576])]
    fn the_passwd_buffer_grows_on_erange(#[case] needs: Option<usize>, #[case] tried: &[usize]) {
        let mut sizes = Vec::new();
        let got = with_growing_buffer(1024, |strings| {
            sizes.push(strings.len());
            if needs.is_some_and(|needs| strings.len() >= needs) {
                Passwd::Home(PathBuf::from("/home/u"))
            } else {
                Passwd::TooSmall
            }
        });
        assert_eq!(got.is_some(), needs.is_some());
        assert_eq!(sizes, tried);
    }

    #[cfg(unix)]
    #[test]
    fn a_missing_entry_is_not_retried() {
        let mut calls = 0;
        assert_eq!(
            with_growing_buffer(1024, |_| {
                calls += 1;
                Passwd::None
            }),
            None
        );
        assert_eq!(calls, 1);
    }
}
