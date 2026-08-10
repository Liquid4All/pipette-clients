//! `pipette-plan commands` — print the exact pipette client
//! invocations the runner would issue for the cells in a plan, without
//! running them.

use std::{collections::HashSet, path::Path};

use pipette_plan_types::{Plan, RunnableCell};

use crate::{
    runner::{run::order_cells, state_io::load_state_index, ListState, Shard},
    shell::quote_argv,
};

/// Print the pipette client invocations for every cell matching
/// `filter` (and `shard`, if set), one per line, without running them.
///
/// The trailing `--sync` the runner appends is included; the
/// ssh/adb/local transport wrapper is not — the output is just the
/// final pipette CLI. Sharding mirrors `run` exactly: it selects by
/// position in the full ordered matrix, so `commands --shard i/N`
/// previews precisely what `run --shard i/N` would execute.
pub fn print_commands(
    plans_dir: &Path,
    plan: &Plan,
    filter: ListState,
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

/// Render the deduplicated pipette CLI invocations for `cells` against
/// `plan`'s transports, preserving `cells`' order. Pure (no I/O) so it
/// can be unit-tested directly.
fn render_commands(plan: &Plan, cells: &[RunnableCell]) -> anyhow::Result<String> {
    let lines: Vec<String> = cells
        .iter()
        .flat_map(|cell| plan.transports.iter().map(move |cfg| (cell, cfg)))
        .filter(|(cell, cfg)| {
            cell.allowed_clients.is_empty()
                || cell
                    .allowed_clients
                    .iter()
                    .any(|c| c.as_ref() == cfg.client_id())
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
    // Dedup, preserving first-seen order.
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
binary_path = "/home/yuri/bin/pipette-torch-oai"
work_dir    = "/home/yuri/edge-evals"
shell       = "posix"

[[transports]]
client_id   = "mac1"
type        = "local"
binary_path = "/Users/yuri/bin/pipette-llamacpp"
work_dir    = "/Users/yuri/edge-evals"
shell       = "posix"

[[variants]]
clients  = ["linux1"]
models   = [{ type = "torch", source = "huggingface", org = "Qwen", repo_name = "Qwen2.5-0.5B-Instruct" }]
runtimes = [{ type = "docker_vllm", image_name = "vllm/vllm-openai", image_tag = "v0.21.0", flavor = "nvidia_gpu" }]
runtime_flags = [{ benchmark_type = "eval", runtime_type = "docker_vllm", model_type = "torch", max_model_len = 4096 }]

[[variants]]
clients  = ["mac1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "unsloth", repo_name = "gemma-4-E2B-it-GGUF", path = "g.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b9050", flavor = "macos-arm64" }]
runtime_flags = [{ benchmark_type = "eval", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", ctx_size = 8192 }]
"#;
        Plan::parse(toml)
    }

    #[test]
    fn render_emits_flat_pipette_cli_lines_without_transport_headers() -> anyhow::Result<()> {
        let plan = demo_plan()?;
        let cells = order_cells(plan.runnable_cells()?.into_iter().collect());
        let out = render_commands(&plan, &cells)?;

        // No `#` header lines (the previous grouped format used them);
        // every line is a bare pipette CLI invocation.
        assert!(!out.contains('#'));
        out.lines()
            .for_each(|line| assert!(line.starts_with('/'), "unexpected line: {line}"));

        // `--model` and `--runtime` now both ship as their canonical JSON
        // (projector inline for VL — no `--mmproj`), and `--runtime-flags`
        // as the structured payload the client renders. Assert the stable
        // command prefix plus each JSON blob's content rather than a
        // pre-rendered flag string.
        assert!(out.contains(
            "/home/yuri/bin/pipette-torch-oai --work-dir /home/yuri/edge-evals benchmarks run \
             --benchmark eval_smoke --model "
        ));
        assert!(out.contains(r#""type":"torch""#), "got: {out}");
        assert!(out.contains(r#""type":"docker_vllm""#), "got: {out}");
        assert!(out.contains("--runtime-flags "), "got: {out}");
        assert!(out.contains(r#""max_model_len":4096"#), "got: {out}");
        // The knobs alone — the cell's axes come from `--benchmark`/`--runtime`/`--model`,
        // which the client parses before it reads any flags.
        assert!(!out.contains(r#""runtime_type""#), "got: {out}");
        assert!(out.contains(
            "/Users/yuri/bin/pipette-llamacpp --work-dir /Users/yuri/edge-evals benchmarks run \
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
        // Two transports with the same binary/work_dir, and a cell
        // allowed on both, must collapse to a single line.
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
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", version = "b1", flavor = "macos-arm64" }]
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
        // The `--runtime-flags` JSON blob must survive a posix-shell round-trip:
        // shell-split each rendered line and confirm the payload parses back as
        // JSON (i.e. `quote_argv` quoted the blob so nothing is lost/re-split).
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
