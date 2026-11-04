//! The shipped local-dispatch examples (`examples/plans/*.toml`) parse and
//! expand, and the MLX VL examples carry their catalog row. Here rather than in
//! `pipette-plan-types`, which is the management server's crate and does not
//! carry them.

use anyhow::Context;

use pipette_artifacts_catalog::runtime::mlx;
use pipette_plan_types::{Plan, UvRuntimeSource};

fn example_plan_paths() -> anyhow::Result<Vec<std::path::PathBuf>> {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/plans");
    let paths = std::fs::read_dir(dir)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("toml"))
        .collect::<Vec<_>>();
    assert!(!paths.is_empty(), "no example plans found in {dir}");
    Ok(paths)
}

/// Parsing runs full validation (`#[serde(try_from = "PlanIntake")]`), so an
/// authored flag with a bad key or shape fails here.
#[test]
fn shipped_example_plans_parse() -> anyhow::Result<()> {
    example_plan_paths()?.iter().try_for_each(|path| {
        let toml = std::fs::read_to_string(path)?;
        Plan::parse(&toml).map_err(|e| anyhow::anyhow!("{}: {e:#}", path.display()))?;
        anyhow::Ok(())
    })
}

/// Materializing resolves flags onto their cells, so a `*_flags` entry that
/// names no real cell fails here.
#[test]
fn shipped_example_plans_materialize_runnable_cells() -> anyhow::Result<()> {
    example_plan_paths()?.iter().try_for_each(|path| {
        let toml = std::fs::read_to_string(path)?;
        let plan = Plan::parse(&toml).map_err(|e| anyhow::anyhow!("{}: {e:#}", path.display()))?;
        let cells = plan
            .runnable_cells()
            .map_err(|e| anyhow::anyhow!("{}: {e:#}", path.display()))?;
        anyhow::ensure!(
            !cells.is_empty(),
            "{}: produced no runnable cells",
            path.display()
        );
        anyhow::Ok(())
    })
}

#[test]
fn mlx_vl_examples_carry_their_catalog_row_verbatim() -> anyhow::Result<()> {
    [
        "mlx-macos-vl-spec-decode.toml",
        "scheduler/mlx-vl-spec-decode.toml",
    ]
    .into_iter()
    .try_for_each(|file| {
        let path = format!("{}/../../examples/plans/{file}", env!("CARGO_MANIFEST_DIR"));
        let plan: toml::Value = toml::from_str(&std::fs::read_to_string(&path)?)?;
        let runtimes = plan
            .get("variants")
            .and_then(toml::Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|variant| variant.get("runtimes").and_then(toml::Value::as_array))
            .flatten()
            .collect::<Vec<_>>();
        anyhow::ensure!(!runtimes.is_empty(), "{file}: no runtimes");
        runtimes.into_iter().try_for_each(|runtime| {
            let version = runtime
                .get("version")
                .and_then(toml::Value::as_str)
                .with_context(|| format!("{file}: a runtime has no version"))?;
            let UvRuntimeSource::PipRequirementsText { contents, .. } =
                mlx::declared_from_catalog(version)?.source
            else {
                anyhow::bail!("{version}: the catalog row is not a requirements body");
            };
            let body = runtime
                .get("source")
                .and_then(|source| source.get("contents"))
                .and_then(toml::Value::as_str);
            anyhow::ensure!(
                body == Some(contents.as_ref()),
                "{file}: the {version} body is not that catalog row"
            );
            anyhow::Ok(())
        })
    })
}
