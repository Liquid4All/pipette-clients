//! `pipette-ctl commands`: print the pipette client invocations the runner
//! would issue for a plan's cells, without running them.

use std::{collections::HashSet, path::Path};

use pipette_plan_types::{Plan, RunnableCell};

use crate::{
    runner::{CellFilter, Shard, run::order_cells, state_io::load_state_index},
    shell::quote_argv,
};

/// Includes the `--sync` the runner appends but not the ssh/adb/local
/// wrapper. Shards by position in the full ordered matrix, as `run` does, so
/// `commands --shard i/N` previews `run --shard i/N`.
pub fn print_commands(
    plans_dir: &Path,
    plan: &Plan,
    filter: CellFilter,
    shard: Option<Shard>,
) -> anyhow::Result<()> {
    let ordered = order_cells(plan.runnable_cells()?.into_iter().collect());
    let state = load_state_index(plans_dir, &plan.plan_id)?;
    let selected: Vec<RunnableCell> = ordered
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| shard.is_none_or(|s| s.contains(*idx)))
        .filter(|(_, cell)| filter.matches(state.state_for(cell)))
        .map(|(_, cell)| cell)
        .collect();
    let rendered = render_commands(plan, &selected)?;
    if !rendered.is_empty() {
        println!("{rendered}");
    }
    Ok(())
}

/// One line per distinct invocation, in `cells`' order.
fn render_commands(plan: &Plan, cells: &[RunnableCell]) -> anyhow::Result<String> {
    let lines: Vec<String> = cells
        .iter()
        .flat_map(|cell| plan.transports.iter().map(move |cfg| (cell, cfg)))
        .filter(|(cell, cfg)| {
            cell.allowed_clients.is_empty()
                || cell.allowed_clients.iter().any(|c| c == cfg.client_id())
        })
        .map(|(cell, cfg)| {
            let mut argv = cell.build_argv(cfg)?;
            // Mirror the runner: iOS carries `submit=1` in its headless args and
            // rejects a bare `--sync`, so a preview that always appends it shows
            // a command the runner would never issue.
            if cfg.appends_sync_flag() {
                argv.push("--sync".to_string());
            }
            Ok(quote_argv(cfg.shell(), &argv))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut seen = HashSet::new();
    Ok(lines
        .into_iter()
        .filter(|line| seen.insert(line.clone()))
        .collect::<Vec<_>>()
        .join("\n"))
}

#[cfg(test)]
mod tests {
    use pipette_plan_types::Plan;

    use super::*;

    fn demo_plan() -> anyhow::Result<Plan> {
        let toml = r#"
plan_id    = "demo"
benchmarks = ["eval_smoke"]

[[transports]]
client_id   = "linux1"
type        = "ssh"
host        = "edge-ci-linux1"
binary_path = "/home/pipette/bin/pipette-torch-oai"
work_dir    = "/home/pipette/edge-evals"
shell       = "posix"

[[transports]]
client_id   = "mac1"
type        = "local"
binary_path = "/Users/pipette/bin/pipette-llamacpp"
work_dir    = "/Users/pipette/edge-evals"
shell       = "posix"

[[variants]]
clients  = ["linux1"]
models   = [{ type = "torch", source = "huggingface", org = "Qwen", repo_name = "Qwen2.5-0.5B-Instruct" }]
runtimes = [{ type = "docker_vllm", image_name = "vllm/vllm-openai", image_tag = "v0.21.0", flavor = "nvidia_gpu" }]
runtime_flags = [{ benchmark_type = "eval", runtime_type = "docker_vllm", model_type = "torch", max_model_len = 4096 }]

[[variants]]
clients  = ["mac1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "unsloth", repo_name = "gemma-4-E2B-it-GGUF", path = "g.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b9050", flavor = "macos-arm64" }]
runtime_flags = [{ benchmark_type = "eval", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", ctx_size = 8192 }]
"#;
        Ok(Plan::parse(toml)?)
    }

    #[test]
    fn render_emits_flat_pipette_cli_lines_without_transport_headers() -> anyhow::Result<()> {
        let plan = demo_plan()?;
        let cells = order_cells(plan.runnable_cells()?.into_iter().collect());
        let out = render_commands(&plan, &cells)?;

        assert!(!out.contains('#'));
        out.lines()
            .for_each(|line| assert!(line.starts_with('/'), "unexpected line: {line}"));

        assert!(out.contains(
            "/home/pipette/bin/pipette-torch-oai --work-dir /home/pipette/edge-evals benchmarks run \
             --benchmark eval_smoke --model "
        ));
        assert!(out.contains(r#""type":"torch""#), "got: {out}");
        assert!(out.contains(r#""type":"docker_vllm""#), "got: {out}");
        assert!(out.contains("--runtime-flags "), "got: {out}");
        assert!(out.contains(r#""max_model_len":4096"#), "got: {out}");
        // `--runtime-flags` carries only the knobs: the cell's axes come from
        // `--benchmark`/`--runtime`/`--model`.
        assert!(!out.contains(r#""runtime_type""#), "got: {out}");
        assert!(out.contains(
            "/Users/pipette/bin/pipette-llamacpp --work-dir /Users/pipette/edge-evals benchmarks run \
             --benchmark eval_smoke --model "
        ));
        assert!(out.contains(r#""type":"gguf_text""#), "got: {out}");
        assert!(
            out.contains(r#""type":"llamacpp_cli_stock_tools""#),
            "got: {out}"
        );
        assert!(out.contains(r#""ctx_size":8192"#), "got: {out}");
        Ok(())
    }

    #[test]
    fn render_dedupes_identical_commands_across_transports() -> anyhow::Result<()> {
        let toml = r#"
plan_id    = "demo"
benchmarks = ["eval_smoke"]

[[transports]]
client_id   = "a"
type        = "ssh"
host        = "host-a"
binary_path = "/bin/pipette-llamacpp"
work_dir    = "/wd"
shell       = "posix"

[[transports]]
client_id   = "b"
type        = "ssh"
host        = "host-b"
binary_path = "/bin/pipette-llamacpp"
work_dir    = "/wd"
shell       = "posix"

[[variants]]
clients  = ["a", "b"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "g.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b1", flavor = "macos-arm64" }]
"#;
        let plan = Plan::parse(toml)?;
        let cells = order_cells(plan.runnable_cells()?.into_iter().collect());
        let out = render_commands(&plan, &cells)?;
        assert_eq!(
            out.lines().count(),
            1,
            "identical commands must collapse: {out}"
        );
        Ok(())
    }

    #[test]
    fn render_empty_when_no_cells() -> anyhow::Result<()> {
        let plan = demo_plan()?;
        assert_eq!(render_commands(&plan, &[])?, "");
        Ok(())
    }

    #[test]
    fn rendered_runtime_flags_json_survives_shell_split() -> anyhow::Result<()> {
        use anyhow::Context;
        let plan = demo_plan()?;
        let cells = order_cells(plan.runnable_cells()?.into_iter().collect());
        let out = render_commands(&plan, &cells)?;
        out.lines().try_for_each(|line| -> anyhow::Result<()> {
            let argv = shlex::split(line)
                .ok_or_else(|| anyhow::anyhow!("not shell-splittable: {line}"))?;
            if let Some(i) = argv.iter().position(|a| a == "--runtime-flags") {
                let payload = argv
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--runtime-flags without a value"))?;
                serde_json::from_str::<serde_json::Value>(payload)
                    .with_context(|| format!("payload not valid JSON after split: {payload}"))?;
            }
            Ok(())
        })
    }
}
