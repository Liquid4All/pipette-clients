#!/usr/bin/env python3
"""Keep the iOS llama.cpp pins tagged, and the variants listed the same way everywhere.

`vendor/llama.cpp.tag` and each `vendor/llamacpp-<v>.tag` record `<tag> <sha>`,
and the sha is the commit the submodule is pinned at. ios/build-llama.sh reads
the tag from that file (CI checks out shallow, with no tags), so a stale file
ships an untagged build.

A variant is a `vendor/llamacpp-<v>` submodule (docs/pipette-ios/build.md,
"llama.cpp variants"). The internal app carries it only where build.yml's
`ios-archive` job names it, so each one needs:

- its `vendor/llamacpp-<v>.tag`, as above;
- its path in the checkout step's `submodule update`;
- a build step whose env sets `PIPETTE_LLAMA_VARIANT: <v>`;
- `llamacpp-<v>` in the archive step's `pins` array, beside main's `llamacpp`,
  which that step passes to xcodebuild as `PIPETTE_LLAMA_PINS`;
- when its .gitmodules entry says `update = none` (a private repository), its
  repository in the token step's `repositories`, which names no other, and an
  `insteadOf` in the checkout step that reads its
  `submodule.vendor/llamacpp-<v>.url`.

And nothing in ios-archive names a variant that is not a submodule.
"""

import re
import subprocess
import sys
from pathlib import Path

WORKFLOW = Path(".github/workflows/build.yml")
PREFIX = "vendor/llamacpp-"
# Up to the next job's key.
JOB = re.compile(r"^  ios-archive:\n(.*?)(?=^  \S|\Z)", re.M | re.S)
STEP_VARIANT = re.compile(r'^ +PIPETTE_LLAMA_VARIANT: "?([^"\s]+)"?$', re.M)
PINS = re.compile(r"^ +pins=\(\n(.*?)^ +\)$", re.M | re.S)
PINS_SETTING = '"PIPETTE_LLAMA_PINS=${pins[*]}"'
REPOSITORIES = re.compile(r"^ +repositories: (.*)$", re.M)
# Up to the next step.
CHECKOUT = re.compile(
    r"^( +)- name: Check out the vendored sources\n(.*?)(?=^\1- |\Z)", re.M | re.S
)
# A path argument, not the `submodule.<path>.url` key an insteadOf reads.
CHECKOUT_PATH = re.compile(
    r"(?<!submodule\.)\bvendor/llamacpp-([a-z0-9][a-z0-9.-]*?)(?=[\s\\]|$)", re.M
)


def submodules() -> dict[str, dict[str, str]]:
    """Each `vendor/llamacpp-<v>` submodule's .gitmodules entry, by `<v>`."""
    out = subprocess.run(
        ["git", "config", "-f", ".gitmodules", "--list"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    entries: dict[str, dict[str, str]] = {}
    for line in out.splitlines():
        key, _, value = line.partition("=")
        name, _, field = key.removeprefix("submodule.").rpartition(".")
        if name.startswith(PREFIX):
            entries.setdefault(name.removeprefix(PREFIX), {})[field] = value
    return entries


def gitlink(path: str) -> str:
    """The commit the index pins submodule `path` at, or "" for no submodule."""
    stage = subprocess.run(
        ["git", "ls-files", "--stage", "--", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return stage[1] if stage[:1] == ["160000"] else ""


def tag_errors(path: str) -> list[str]:
    """Why `<path>.tag` does not name the commit `path` is pinned at, if it does not."""
    tag = Path(f"{path}.tag")
    if not tag.is_file():
        return [f"{path} has no {tag}"]
    recorded = tag.read_text().split()
    pinned = gitlink(path)
    if recorded[1:2] == [pinned]:
        return []
    return [
        f"{tag} records {' '.join(recorded[:2]) or 'nothing'}, but {path} is "
        f"pinned at {pinned or 'no commit'}; write that commit's tag and sha to it"
    ]


def main() -> int:
    match = JOB.search(WORKFLOW.read_text())
    job = match.group(1) if match else ""
    checkout_match = CHECKOUT.search(job)
    checkout = checkout_match.group(2) if checkout_match else ""
    errors = []

    entries = submodules()
    errors += [
        error
        for path in ["vendor/llama.cpp", *(f"{PREFIX}{v}" for v in sorted(entries))]
        for error in tag_errors(path)
    ]

    pins = PINS.search(job)
    pinned = set(pins.group(1).split()) if pins else set()
    if "llamacpp" not in pinned:
        errors.append(
            f"{WORKFLOW}: ios-archive's pins array has no llamacpp (vendor/llama.cpp)"
        )
    if PINS_SETTING not in job:
        errors.append(
            f"{WORKFLOW}: ios-archive does not pass {PINS_SETTING} to xcodebuild"
        )

    lists = {
        "vendor/llamacpp-<v> submodule": set(entries),
        "path in ios-archive's checkout step (vendor/llamacpp-<v>)": set(
            CHECKOUT_PATH.findall(checkout)
        ),
        "ios-archive build step (PIPETTE_LLAMA_VARIANT: <v>)": set(
            STEP_VARIANT.findall(job)
        ),
        "entry in ios-archive's pins array (llamacpp-<v>)": {
            pin.removeprefix("llamacpp-") for pin in pinned - {"llamacpp"}
        },
    }
    every = set().union(*lists.values())
    errors += [
        f"llama.cpp variant {v!r} is missing its {where}"
        for v in sorted(every)
        for where, names in lists.items()
        if v not in names
    ]

    private = {
        v: entry["url"].removesuffix(".git").rsplit("/", 1)[-1]
        for v, entry in entries.items()
        if entry.get("update") == "none"
    }
    token = REPOSITORIES.search(job)
    readable = {r.strip() for r in token.group(1).split(",")} if token else set()
    errors += [
        f"{WORKFLOW}: ios-archive's token does not read {repo}, which "
        f"{PREFIX}{v} (update = none) is from"
        for v, repo in sorted(private.items())
        if repo not in readable
    ]
    errors += [
        f"{WORKFLOW}: ios-archive's checkout step has no insteadOf for "
        f"{PREFIX}{v} (update = none); read submodule.{PREFIX}{v}.url there"
        for v in sorted(private)
        if f"submodule.{PREFIX}{v}.url" not in checkout
    ]
    errors += [
        f"{WORKFLOW}: ios-archive's token reads {repo}, which no "
        f"`update = none` {PREFIX}<v> submodule is from"
        for repo in sorted(readable - set(private.values()))
    ]

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
