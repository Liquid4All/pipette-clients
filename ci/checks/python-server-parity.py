#!/usr/bin/env python3
"""Hold the ONNX GenAI and OpenVINO Python servers and their suites to one copy
of their shared code.

Each region is compared top-level statement by statement, with the comments
above each, minus the statements each engine owns. An exclusion that names
nothing in either span, or a statement identical in both, is stale and fails.

Fix a failure by making the same edit in both files.
"""

import ast
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ONNX = Path("crates/pipette-onnx-genai")
OPENVINO = Path("crates/pipette-openvino")
HUNK = re.compile(r"^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@")


@dataclass(frozen=True)
class Region:
    """Top-level statements `first` through `last` of both files, minus
    `engine_specific`."""

    name: str
    files: tuple[Path, Path]
    first: str
    last: str
    engine_specific: frozenset[str]


REGIONS = [
    Region(
        name="server",
        files=(
            ONNX / "src/python/pipette_onnx_genai_server.py",
            OPENVINO / "src/python/pipette_openvino_server.py",
        ),
        first="MODEL_ID",
        last="make_server",
        engine_specific=frozenset(
            {
                "log",  # the prefix names the engine
                "Engine",  # each engine's own runtime
                "preload_cuda_libraries",  # ONNX GenAI's Engine helper
                "PARTIAL",  # OpenVINO's Engine helpers
                "Detokenizer",
            }
        ),
    ),
    Region(
        name="test",
        files=(
            ONNX / "tests/pipette_onnx_genai_server_test.py",
            OPENVINO / "tests/pipette_openvino_server_test.py",
        ),
        first="serve",
        last="ParentWatch",
        engine_specific=frozenset(
            {
                "OverHttp",  # the real engine over HTTP
                "Main",  # each engine's command line
            }
        ),
    ),
]


class SpanError(Exception):
    """A span's anchor statements are missing or out of order in a file."""


@dataclass(frozen=True)
class Statement:
    name: str
    start: int
    lines: tuple[str, ...]


def name_of(node: ast.stmt) -> str:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return node.target.id
    return f"<{type(node).__name__}>"


def statements(path: Path) -> list[Statement]:
    """Each top-level statement with the comment lines between it and the one
    before."""
    source = (ROOT / path).read_text(encoding="utf-8")
    lines = source.splitlines()
    found = []
    previous_end = 0
    for node in ast.parse(source, filename=str(path)).body:
        decorators = getattr(node, "decorator_list", [])
        first = min([node.lineno, *(d.lineno for d in decorators)])
        start = previous_end + 1
        while start < first and not lines[start - 1].strip():
            start += 1
        end = node.end_lineno or node.lineno
        found.append(Statement(name_of(node), start, tuple(lines[start - 1 : end])))
        previous_end = end
    return found


def span(path: Path, region: Region) -> list[Statement]:
    found = statements(path)
    names = [s.name for s in found]
    for anchor in (region.first, region.last):
        if anchor not in names:
            raise SpanError(
                f"{path}: no top-level `{anchor}` to anchor the {region.name} span"
            )
    first, last = names.index(region.first), names.index(region.last)
    if last < first:
        raise SpanError(f"{path}: `{region.last}` comes before `{region.first}`")
    return found[first : last + 1]


def diff(a: Statement, a_path: Path, b: Statement, b_path: Path) -> list[str]:
    """A unified diff of two statements, numbered as lines of their files."""
    out = []
    for line in difflib.unified_diff(
        list(a.lines), list(b.lines), str(a_path), str(b_path), lineterm=""
    ):
        hunk = HUNK.match(line)
        if hunk:
            a_line = int(hunk[1]) + a.start - 1
            b_line = int(hunk[3]) + b.start - 1
            line = f"@@ -{a_line}{hunk[2] or ''} +{b_line}{hunk[4] or ''} @@"
        out.append(line)
    return out


def problems(region: Region) -> list[str]:
    a_path, b_path = region.files
    try:
        a_span, b_span = span(a_path, region), span(b_path, region)
    except SpanError as exc:
        return [str(exc)]
    a_lines = {s.name: s.lines for s in a_span}
    b_lines = {s.name: s.lines for s in b_span}
    absent = sorted(region.engine_specific - a_lines.keys() - b_lines.keys())
    shared = sorted(
        name
        for name in region.engine_specific & a_lines.keys() & b_lines.keys()
        if a_lines[name] == b_lines[name]
    )
    stale = []
    if absent:
        stale.append(f"{region.name} span: stale exclusions, in neither file: {absent}")
    if shared:
        stale.append(
            f"{region.name} span: stale exclusions, identical in both files: {shared}"
        )
    if stale:
        return stale
    a_kept = [s for s in a_span if s.name not in region.engine_specific]
    b_kept = [s for s in b_span if s.name not in region.engine_specific]
    a_names, b_names = [s.name for s in a_kept], [s.name for s in b_kept]
    if a_names != b_names:
        found = [f"{region.name} span: the shared statements differ in name or order"]
        for path, kept, other in ((a_path, a_kept, b_names), (b_path, b_kept, a_names)):
            found.extend(
                f"{path}:{s.start}: `{s.name}` is in this file only"
                for s in kept
                if s.name not in other
            )
        found.append(f"{a_path} order: {', '.join(a_names)}")
        found.append(f"{b_path} order: {', '.join(b_names)}")
        return found
    found = []
    for a, b in zip(a_kept, b_kept, strict=True):
        if a.lines != b.lines:
            found.append(f"{region.name} span: `{a.name}` differs")
            found.extend(diff(a, a_path, b, b_path))
    return found


def main() -> int:
    found = [p for region in REGIONS for p in problems(region)]
    if found:
        print(
            "error: the code the ONNX GenAI and OpenVINO Python servers share "
            "differs between them, or an exclusion is stale; make the same edit "
            "in both files, or drop the stale exclusion from "
            "ci/checks/python-server-parity.py.",
            file=sys.stderr,
        )
        print("\n".join(found), file=sys.stderr)
        return 1
    print(f"python server parity OK ({len(REGIONS)} spans checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
