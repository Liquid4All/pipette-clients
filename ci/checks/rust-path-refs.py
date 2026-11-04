#!/usr/bin/env python3
"""Verify that Rust paths named outside the Rust tree still resolve.

Kotlin, Swift and Markdown cite Rust code by module path
(`pipette_cli::client::worker::profile`) or file path
(`crates/pipette-ops/src/types.rs`). Nothing compiles those mentions, and the
mobile jobs' path filters skip Rust-only changes, so a rename would leave them
dangling.

Module segments are resolved against the crate's `src/` tree; the first item
after them must be declared or `pub use`-d in that module (a text search, not a
parse).
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Both `pipette_device::` and `pipette-device::` appear in prose.
# `pipette-plan-types` is absent: it lives in pipette-mgmt.
REFERENCE = re.compile(
    r"\bpipette[_-](ops|cli|artifacts[_-]catalog|artifacts|device)((?:::[A-Za-z_][A-Za-z0-9_]*)+)"
)

CRATE_SRC = {
    "ops": Path("crates/pipette-ops/src"),
    "cli": Path("crates/pipette-cli/src"),
    "artifacts": Path("crates/pipette-artifacts/src"),
    "artifacts_catalog": Path("crates/pipette-artifacts-catalog/src"),
    "device": Path("crates/pipette-device/src"),
}

# `crates/pipette-plan-types/` is skipped: it lives in pipette-mgmt.
FILE_REFERENCE = re.compile(
    r"\bcrates/pipette-(?!plan-types/)[a-z-]+/src/[A-Za-z0-9_/]+\.rs\b"
)

SEARCH_GLOBS = ["android/*", "ios/*", "docs/*", "*.md"]


def tracked_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", *SEARCH_GLOBS],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [Path(f) for f in out]


def module_source(node: Path) -> Path | None:
    """The file backing module `node`.

    `lib.rs` exists only at a crate root, where it lets a crate-root re-export
    (`pipette_ops::RunRequest`) resolve.
    """
    for candidate in (node.with_suffix(".rs"), node / "mod.rs", node / "lib.rs"):
        if candidate.exists():
            return candidate
    return None


def declares(source: Path, item: str) -> bool:
    """True when `source` declares `item` or re-exports it.

    Searches the whole file, since `pub use` lists wrap across lines.
    """
    text = source.read_text(encoding="utf-8")
    declared = re.search(
        rf"\b(fn|struct|enum|trait|type|const|static|mod)\s+{re.escape(item)}\b", text
    )
    if declared:
        return True
    return any(
        re.search(rf"\b{re.escape(item)}\b", statement)
        for statement in re.findall(r"pub use [^;]+;", text)
    )


def resolves(src: Path, segments: list[str]) -> bool:
    """True when the path names a real module chain and a real item in it."""
    node = src
    for segment in segments:
        if module_source(node / segment) is not None:
            node = node / segment
            continue
        # Not a module, so this names an item. Anything after it (a field or
        # variant) would need a real parse, so only this one is checked.
        source = module_source(node)
        return source is not None and declares(source, segment)
    return True


def self_test() -> None:
    """Regression fixtures for the resolver. Raises AssertionError on failure."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        (src / "client").mkdir(parents=True)
        (src / "lib.rs").write_text(
            "pub mod client;\npub mod device;\npub use run_request::RunRequest;\n"
        )
        (src / "device.rs").write_text("pub struct ThermalTelemetry {}\n")
        (src / "client" / "mod.rs").write_text("pub use inner::clear_remote_state;\n")

        # (name, path segments, expected)
        cases = [
            ("module only", ["device"], True),
            ("item in module", ["device", "ThermalTelemetry"], True),
            ("nested module", ["client"], True),
            ("crate-root re-export", ["RunRequest"], True),
            ("item via pub use", ["client", "clear_remote_state"], True),
            ("unknown module", ["nope"], False),
            ("unknown item", ["device", "no_such_item"], False),
            ("unknown item under a module", ["client", "no_such_item"], False),
        ]
        for name, segments, expected in cases:
            got = resolves(src, segments)
            if got is not expected:
                path = "::".join(segments)
                raise AssertionError(
                    f"rust-path-refs self-test failed: {name} ({path}) "
                    f"resolved to {got}, expected {expected}"
                )


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    only_self_test = argv == ["--self-test"]
    if argv and not only_self_test:
        print(f"usage: {sys.argv[0]} [--self-test]", file=sys.stderr)
        return 2

    # Fixtures always run first so CI catches resolver regressions.
    try:
        self_test()
    except AssertionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if only_self_test:
        print("rust-path-refs self-test OK")
        return 0

    violations: list[str] = []
    for path in tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            for match in REFERENCE.finditer(line):
                crate = match.group(1).replace("-", "_")
                segments = match.group(2).lstrip(":").split("::")
                if not resolves(CRATE_SRC[crate], segments):
                    violations.append(f"{path}:{line_no}: {match.group(0)}")
            for match in FILE_REFERENCE.finditer(line):
                if not Path(match.group(0)).is_file():
                    violations.append(f"{path}:{line_no}: {match.group(0)}")

    if violations:
        print("error: a Rust path named outside the Rust tree no longer resolves.")
        print("Update the reference to the module's current path, or drop it.")
        print()
        for v in violations:
            print(f"  {v}")
        return 1

    print("cross-language Rust path references OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
