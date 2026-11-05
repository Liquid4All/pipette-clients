//! The plan ladder: cells drawn as a tree descending
//! `benchmark → runtime → runtime flags → model flags → benchmark flags`,
//! ending in the models that share that whole path, so a flag entry shared by
//! dozens of models is printed once.
//!
//! Every object is drawn as its own serialization, unedited: the ladder is
//! read to check what a plan will send, so it shows what the client is handed.

use std::collections::BTreeMap;

use serde::Serialize;

use pipette_plan_types::RunnableCell;

/// A flag object the variant did not author. Printed rather than omitted:
/// "runs on the engine's defaults" is an answer, a missing line a question.
const UNSET: &str = "—";

struct Node {
    label: String,
    children: Vec<Node>,
}

impl Node {
    fn leaf(label: impl Into<String>) -> Self {
        Self {
            label: label.into(),
            children: Vec::new(),
        }
    }

    fn with(label: impl Into<String>, children: Vec<Node>) -> Self {
        Self {
            label: label.into(),
            children,
        }
    }

    /// Append this subtree to `lines`, `tree(1)`-style.
    fn draw(&self, prefix: &str, connector: &str, carry: &str, lines: &mut Vec<String>) {
        lines.push(format!("{prefix}{connector}{}", self.label));
        let child_prefix = format!("{prefix}{carry}");
        let Some(last) = self.children.len().checked_sub(1) else {
            return;
        };
        self.children.iter().enumerate().for_each(|(index, child)| {
            let (connector, carry) = if index == last {
                ("└── ", "    ")
            } else {
                ("├── ", "│   ")
            };
            child.draw(&child_prefix, connector, carry, lines);
        });
    }
}

/// One cell, flattened to the path the ladder groups it by.
struct Row {
    benchmark: String,
    runtime: String,
    runtime_flags: Option<String>,
    model_flags: Option<String>,
    benchmark_flags: Option<String>,
    model: String,
}

impl Row {
    fn of(cell: &RunnableCell) -> Self {
        Self {
            benchmark: cell.benchmark.as_ref().to_string(),
            runtime: json(&cell.runtime),
            runtime_flags: cell.runtime_flags.as_ref().map(json),
            model_flags: cell.model_flags.as_ref().map(json),
            benchmark_flags: cell.benchmark_flags.as_ref().map(json),
            model: json(&cell.model),
        }
    }
}

/// Which flag levels this drawing shows. A level nobody authored is left
/// out rather than drawn as a column of dashes; decided across the whole
/// drawing so sibling branches stay the same depth.
#[derive(Clone, Copy)]
struct Levels {
    runtime_flags: bool,
    model_flags: bool,
    benchmark_flags: bool,
}

impl Levels {
    fn of(rows: &[Row]) -> Self {
        Self {
            runtime_flags: rows.iter().any(|row| row.runtime_flags.is_some()),
            model_flags: rows.iter().any(|row| row.model_flags.is_some()),
            benchmark_flags: rows.iter().any(|row| row.benchmark_flags.is_some()),
        }
    }
}

/// An object as the plan serializes it, axis keys included.
fn json<T: Serialize>(value: &T) -> String {
    // These types serialize infallibly (no maps, floats, or fallible
    // `Serialize` impls). Say so on the impossible branch rather than
    // printing nothing, which would draw the object as absent.
    serde_json::to_string(value).unwrap_or_else(|err| format!("<unserializable: {err}>"))
}

pub fn cell_ladder(cells: &[RunnableCell]) -> String {
    let rows: Vec<Row> = cells.iter().map(Row::of).collect();
    let levels = Levels::of(&rows);
    let rows: Vec<&Row> = rows.iter().collect();

    benchmark_nodes(&rows, levels)
        .iter()
        .flat_map(|benchmark| {
            let mut lines = Vec::new();
            benchmark.draw("", "", "", &mut lines);
            lines
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Group `rows` by `key`, in key order, and turn each group into a node.
fn grouped<'a, K, Key, Label, Children>(
    rows: &[&'a Row],
    key: Key,
    label: Label,
    children: Children,
) -> Vec<Node>
where
    K: Ord,
    Key: Fn(&Row) -> K,
    Label: Fn(&K, &[&'a Row]) -> String,
    Children: Fn(&[&'a Row]) -> Vec<Node>,
{
    rows.iter()
        .copied()
        .fold(BTreeMap::<K, Vec<&Row>>::new(), |mut groups, row| {
            groups.entry(key(row)).or_default().push(row);
            groups
        })
        .into_iter()
        .map(|(key, rows)| Node::with(label(&key, &rows), children(&rows)))
        .collect()
}

fn benchmark_nodes(rows: &[&Row], levels: Levels) -> Vec<Node> {
    grouped(
        rows,
        |row| row.benchmark.clone(),
        |benchmark, rows| format!("{benchmark}  ({})", rows.len()),
        |rows| runtime_nodes(rows, levels),
    )
}

fn runtime_nodes(rows: &[&Row], levels: Levels) -> Vec<Node> {
    grouped(
        rows,
        |row| row.runtime.clone(),
        |runtime, _| runtime.clone(),
        |rows| runtime_flag_nodes(rows, levels),
    )
}

fn runtime_flag_nodes(rows: &[&Row], levels: Levels) -> Vec<Node> {
    if !levels.runtime_flags {
        return model_flag_nodes(rows, levels);
    }
    grouped(
        rows,
        |row| row.runtime_flags.clone(),
        |flags, _| flag_label("runtime_flags", flags),
        |rows| model_flag_nodes(rows, levels),
    )
}

fn model_flag_nodes(rows: &[&Row], levels: Levels) -> Vec<Node> {
    if !levels.model_flags {
        return benchmark_flag_nodes(rows, levels);
    }
    grouped(
        rows,
        |row| row.model_flags.clone(),
        |flags, _| flag_label("model_flags", flags),
        |rows| benchmark_flag_nodes(rows, levels),
    )
}

fn benchmark_flag_nodes(rows: &[&Row], levels: Levels) -> Vec<Node> {
    if !levels.benchmark_flags {
        return model_nodes(rows);
    }
    grouped(
        rows,
        |row| row.benchmark_flags.clone(),
        |flags, _| flag_label("benchmark_flags", flags),
        model_nodes,
    )
}

fn model_nodes(rows: &[&Row]) -> Vec<Node> {
    let mut models: Vec<&str> = rows.iter().map(|row| row.model.as_str()).collect();
    models.sort_unstable();
    models.into_iter().map(Node::leaf).collect()
}

fn flag_label(name: &str, flags: &Option<String>) -> String {
    match flags {
        Some(json) => format!("{name}  {json}"),
        None => format!("{name} {UNSET}"),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use pipette_plan_types::Plan;

    use super::*;

    /// Runtime flags on the prefill cell only, so a populated branch sits
    /// next to an empty one.
    const PLAN: &str = r#"
plan_id          = "ladder"
benchmarks       = ["prefill_throughput_256", "decode_throughput_256_100"]

[[transports]]
client_id = "t1"
type = "local"
binary_path = "/bin/pipette"
work_dir = "/tmp"
shell = "posix"

[[variants]]
clients  = ["t1"]
models   = [{ type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "a.gguf" }]
runtimes = [{ type = "llamacpp_cli_stock_tools", source = "github_release", repository_url = "github.com/ggml-org/llama.cpp", version = "b9050", flavor = "macos-arm64" }]
runtime_flags = [{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", threads = 4 }]
"#;

    fn cells(plan: &str) -> anyhow::Result<Vec<RunnableCell>> {
        Ok(Plan::parse(plan)?.runnable_cells()?.into_iter().collect())
    }

    fn ladder() -> anyhow::Result<String> {
        Ok(cell_ladder(&cells(PLAN)?))
    }

    /// `PLAN` with a `benchmark_flags` entry carrying a `readiness` block.
    fn with_a_block_knob() -> String {
        PLAN.replace(
            "runtime_flags = [",
            r#"benchmark_flags = [{ benchmark_type = "prefill_throughput", runtime_type = "llamacpp_cli_stock_tools", model_type = "gguf_text", readiness = { max_wait_secs = 300, skip_thermal = true } }]
runtime_flags = ["#,
        )
    }

    /// `PLAN` with a second model, so the first sits under a branch that has
    /// a sibling still to come.
    fn with_two_models() -> String {
        PLAN.replace(
            r#"path = "a.gguf" }]"#,
            r#"path = "a.gguf" }, { type = "gguf_text", source = "huggingface", org = "o", repo_name = "r", path = "b.gguf" }]"#,
        )
    }

    #[rstest]
    #[case::a_benchmark_roots_its_runtime(
        PLAN.to_string(),
        "prefill_throughput_256  (1)\n└── {\"type\":\"llamacpp_cli_stock_tools\","
    )]
    #[case::the_runtime_flags_are_a_branch(
        PLAN.to_string(),
        "    └── runtime_flags  {\"runtime_type\":\"llamacpp_cli_stock_tools\""
    )]
    #[case::a_knob_is_not_reformatted(PLAN.to_string(), r#""threads":4}"#)]
    #[case::cells_with_no_flags_group_under_the_glyph(PLAN.to_string(), "    └── runtime_flags —")]
    #[case::a_model_is_a_leaf(
        PLAN.to_string(),
        r#"        └── {"type":"gguf_text","source":"huggingface","org":"o","repo_name":"r","path":"a.gguf"}"#
    )]
    #[case::a_block_knob_stays_nested(
        with_a_block_knob(),
        r#""readiness":{"max_wait_secs":300,"skip_thermal":true}"#
    )]
    #[case::a_continuing_branch_carries_its_rule(with_two_models(), "        ├── {")]
    #[case::a_last_branch_closes(with_two_models(), "        └── {")]
    fn the_ladder_draws(#[case] plan: String, #[case] expected: &str) -> anyhow::Result<()> {
        let ladder = cell_ladder(&cells(&plan)?);

        assert!(ladder.contains(expected), "missing {expected:?}:\n{ladder}");
        Ok(())
    }

    #[test]
    fn each_benchmark_roots_its_own_branch() -> anyhow::Result<()> {
        let ladder = ladder()?;

        assert_eq!(
            ladder
                .lines()
                .filter(|line| !line.starts_with([' ', '│', '├', '└']))
                .count(),
            2,
            "one unindented root per benchmark:\n{ladder}"
        );
        Ok(())
    }

    /// Once anything authors a level it appears on every branch.
    #[test]
    fn an_authored_level_appears() -> anyhow::Result<()> {
        let ladder = cell_ladder(&cells(&with_a_block_knob())?);

        assert!(
            ladder.contains("benchmark_flags  {"),
            "the authored object is a branch:\n{ladder}"
        );
        assert!(
            ladder.contains(&format!("benchmark_flags {UNSET}")),
            "and the cells without it group under the glyph:\n{ladder}"
        );
        assert!(
            !ladder.contains("model_flags"),
            "the level still nobody authored stays out:\n{ladder}"
        );
        Ok(())
    }
}
