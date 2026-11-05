#!/usr/bin/env python3
"""Keep the runtime crates pipette-cli's alone, each one's surface to `run`, and
its `execute/` to its benchmarks.

Every workspace member is classified here, as a runtime crate or not, so a new
crate fails this check until someone decides which it is. The two halves of
the rule then apply to the runtime crates:

- only pipette-cli depends on one. That is cargo-deny's `[bans]` in deny.toml,
  and this check requires its entries to name exactly `RUNTIME_CRATES`;
- the CLI drives one through its `run` and nothing else, so `lib.rs` declares
  every module private and re-exports `execute::run`. A `pub mod` or a second
  `pub use` compiles, and hands other crates the runtime's insides.

One crate has an exception, listed in `EXTRA_EXPORTS`. pipette-coreai keeps its
Swift sidecar in a user cache, not in the runtime store, so the CLI pulls,
lists and removes Core AI through the sidecar functions it names.

`lib.rs` is the only file read for `pub` items: an item in a private module is
out of reach unless `lib.rs` re-exports it. A `#[macro_export]` macro lands at
the crate root from any file, so every source file is read for that.

Inside a runtime crate, `execute/` is the dispatch (`mod.rs`) and one file per
benchmark type. A benchmark file reaches neither the dispatch nor another
benchmark (`super::`, `crate::execute`); a module nested in it, such as its
tests, may still `use super::*;` for the file's own items. What benchmarks
share lives in a module at the crate root.
"""

import re
import subprocess
import sys
from pathlib import Path

RUNTIME_CRATES = (
    "pipette-llamacpp",
    "pipette-mlx",
    "pipette-coreai",
    "pipette-onnx-genai",
    "pipette-openvino",
    "pipette-ovms",
    "pipette-torch-oai",
    "pipette-trt-edgellm",
)
OTHER_CRATES = (
    "pipette-android",
    "pipette-artifacts",
    "pipette-artifacts-catalog",
    "pipette-cli",
    "pipette-completions",
    "pipette-contract",
    "pipette-device",
    "pipette-http",
    "pipette-ios-proxy",
    "pipette-memprobe",
    "pipette-mgmt-client",
    "pipette-ops",
    "pipette-ctl",
    "pipette-programs",
    "pipette-readiness",
    "pipette-ssh",
    "pipette-subprocess",
    "pipette-venv",
    "pipette-vl-spec-decode-corpus",
    "pipette-wait",
    "pipette-workspace",
)
RUN = "pub use execute::run;"
# The items a runtime crate may export beyond `run`, by crate.
EXTRA_EXPORTS = {
    "pipette-coreai": {
        "cached_sidecar",
        "clear_sidecar_cache",
        "require_coreai_sidecar",
        "sidecar_obtainable",
    },
}
# A `pub use` of one item or a braced list, read as one statement.
USE_NAMES = re.compile(r"pub use [\w:]+::(?:\{([^}]*)\}|(\w+));")
# pipette-plan-types' `BenchmarkType`, as the files in `execute/` are named.
BENCHMARKS = (
    "asr_eval",
    "asr_max_memory",
    "asr_throughput",
    "decode_throughput",
    "end_to_end_latency",
    "eval",
    "max_memory_usage",
    "prefill_throughput",
    "vl_max_memory",
    "vl_spec_decode",
    "vl_throughput",
    "vlm_eval",
)

MEMBERS = re.compile(r"^members\s*=\s*\[(.*?)\]", re.MULTILINE | re.DOTALL)
PACKAGE_NAME = re.compile(r'^\[package\]\s*\nname\s*=\s*"([^"]+)"', re.MULTILINE)
BAN = re.compile(
    r'^\s*\{\s*crate\s*=\s*"([^"]+)",\s*wrappers\s*=\s*\["pipette-cli"\]',
    re.MULTILINE,
)
# `pub` opening a line, but not `pub(crate)` and the like.
PUB = re.compile(r"\s*pub\b(?!\s*\()")
MACRO_EXPORT = re.compile(r"#\s*\[\s*macro_export\b")
# A path into the dispatch or another benchmark.
REACH = re.compile(r"\b(?:super|crate::execute)::")
# Indented: a module nested in the benchmark file, taking the file's own items.
OWN_ITEMS = re.compile(r"\s+use super::\*;")


def members() -> tuple[dict[str, Path], list[str]]:
    """Workspace members by package name, and what could not be read."""
    text = re.sub(r"(?m)#.*$", "", Path("Cargo.toml").read_text(encoding="utf-8"))
    block = MEMBERS.search(text)
    if block is None:
        return {}, ["Cargo.toml: no `members = [...]`"]
    found = {}
    problems = []
    for member in re.findall(r'"([^"]+)"', block.group(1)):
        if "*" in member:
            problems.append(f"Cargo.toml: `{member}`: list each member, not a glob")
            continue
        manifest = Path(member) / "Cargo.toml"
        name = PACKAGE_NAME.search(manifest.read_text(encoding="utf-8"))
        if name is None:
            problems.append(f"{manifest}: no `name` right under `[package]`")
            continue
        found[name.group(1)] = Path(member)
    return found, problems


def classification(workspace: dict[str, Path]) -> list[str]:
    """Members in neither list or both, and listed crates that are not members."""
    runtime = set(RUNTIME_CRATES)
    other = set(OTHER_CRATES)
    found = [
        f"{name} ({workspace[name]}): not classified; add it to RUNTIME_CRATES "
        "or OTHER_CRATES"
        for name in sorted(workspace.keys() - runtime - other)
    ]
    found += [
        f"{name}: in both RUNTIME_CRATES and OTHER_CRATES"
        for name in sorted(runtime & other)
    ]
    found += [
        f"{name}: listed, but not a workspace member"
        for name in sorted((runtime | other) - workspace.keys())
    ]
    return found


def bans() -> list[str]:
    """Runtime crates deny.toml does not ban, and bans on crates that are not."""
    banned = set(BAN.findall(Path("deny.toml").read_text(encoding="utf-8")))
    runtime = set(RUNTIME_CRATES)
    found = [
        f'deny.toml: no `{{ crate = "{name}", wrappers = ["pipette-cli"] }}`'
        for name in sorted(runtime - banned)
    ]
    found += [
        f"deny.toml: bans {name}, which is not in RUNTIME_CRATES"
        for name in sorted(banned - runtime)
    ]
    return found


def sources(crate: Path) -> list[Path]:
    """Tracked `.rs` files under the crate's `src/`."""
    result = subprocess.run(
        ["git", "ls-files", "--", str(crate / "src")],
        capture_output=True,
        text=True,
        check=True,
    )
    return [Path(line) for line in result.stdout.splitlines() if line.endswith(".rs")]


def exported_names(statement: str) -> set[str]:
    """The names a `pub use` statement exports, or `{statement}` itself for any
    other `pub` item, so it never matches an allowed name."""
    match = USE_NAMES.fullmatch(statement.replace("{ ", "{").replace(" }", "}"))
    if match is None:
        return {statement}
    listed = match.group(1)
    if listed is None:
        return {match.group(2)}
    return {name.strip() for name in listed.split(",") if name.strip()}


def exports(crate: Path) -> list[str]:
    """What `crate` exports beyond `run`, and a missing `run`."""
    lib = crate / "src" / "lib.rs"
    found = []
    exports_run = False
    allowed = EXTRA_EXPORTS.get(crate.name, set())
    lines = lib.read_text(encoding="utf-8").splitlines()
    for lineno, line in enumerate(lines, 1):
        if not PUB.match(line):
            continue
        # rustfmt wraps a long `pub use` list, so read on to its `;`.
        statement = line.strip()
        following = iter(lines[lineno:])
        while not statement.endswith(";"):
            rest = next(following, None)
            if rest is None:
                break
            statement = f"{statement} {rest.strip()}"
        if statement == RUN:
            exports_run = True
        elif not exported_names(statement) <= allowed:
            found.append(f"{lib}:{lineno}: `{statement}`")
    if not exports_run:
        found.append(f"{lib}: no `{RUN}`")
    for path in sources(crate):
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, 1):
            if MACRO_EXPORT.search(line):
                found.append(f"{path}:{lineno}: `#[macro_export]`")
    return found


def benchmarks(crate: Path) -> list[str]:
    """Files in `execute/` besides `mod.rs` and the benchmarks, and benchmark
    lines that reach into the dispatch or another benchmark."""
    execute = crate / "src" / "execute"
    found = []
    for path in sources(crate):
        if not path.is_relative_to(execute) or path == execute / "mod.rs":
            continue
        if path.parent != execute or path.stem not in BENCHMARKS:
            found.append(f"{path}: not `mod.rs` or a benchmark type")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for lineno, line in enumerate(lines, 1):
            if line.lstrip().startswith("//") or OWN_ITEMS.fullmatch(line):
                continue
            if REACH.search(line):
                found.append(f"{path}:{lineno}: `{line.strip()}`")
    return found


def report(headline: str, found: list[str]) -> None:
    print(f"error: {headline}", file=sys.stderr)
    for problem in found:
        print(f"  {problem}", file=sys.stderr)
    print(file=sys.stderr)


def main() -> int:
    workspace, unreadable = members()
    classified = unreadable + classification(workspace)
    if classified:
        report(
            "every workspace crate is classified in this check as a runtime crate "
            "(only pipette-cli may depend on it, and it exports `run` alone) or not.",
            classified,
        )
    banned = bans()
    if banned:
        report(
            "deny.toml's `[bans]` must name exactly the runtime crates, each with "
            '`wrappers = ["pipette-cli"]`.',
            banned,
        )
    exported = [
        problem
        for name in RUNTIME_CRATES
        if name in workspace
        for problem in exports(workspace[name])
    ]
    if exported:
        report(
            f"a runtime crate exports more than `run`. Its `lib.rs` declares every "
            f"module private and exports `{RUN}` alone. A test that needs an "
            "internal goes inside the crate; code another crate needs goes in a "
            "shared crate.",
            exported,
        )
    misplaced = [
        problem
        for name in RUNTIME_CRATES
        if name in workspace
        for problem in benchmarks(workspace[name])
    ]
    if misplaced:
        report(
            "a runtime crate's `execute/` holds the dispatch (`mod.rs`) and one file "
            "per benchmark type, and a benchmark file reaches neither the dispatch "
            "nor another benchmark. What benchmarks share goes in a module at the "
            "crate root, such as `server.rs` or `response.rs`; a new benchmark type "
            "goes in BENCHMARKS.",
            misplaced,
        )
    if classified or banned or exported or misplaced:
        return 1
    print(
        f"runtime-crates OK ({len(workspace)} crates classified, "
        f"{len(RUNTIME_CRATES)} runtime crates export `run` and their "
        "EXTRA_EXPORTS alone and keep `execute/` to their benchmarks)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
