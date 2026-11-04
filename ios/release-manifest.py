#!/usr/bin/env python3
"""Write the manifest a release publishes beside its iOS app archive.

    release-manifest.py <client_version> --frameworks DIR [--private-thermal]
        [--private-memory] [--root DIR]

Prints `{"client_version": ..., "runtimes": [...]}`: the runtime descriptors
`headlessrun runtimes list` prints for this build. pipette-ios-proxy reads it
to choose a build that carries a job's runtime before pulling one
(tools/pipette-ios-proxy/DESIGN.md, section 9).

The app builds its descriptors at runtime (`Runtime.thisBuild`), and the
archive cannot run on the runner, so this restates that encoding from the
sources the archive compiled:

- which runtimes: the list in `RuntimesCommands.runtimeRows`;
- the llama.cpp builds: one per `llamacpp*.framework` in `--frameworks`, from
  the identity ios/build-llama.sh writes into its Info.plist;
- the other versions: the `*BuildInfo.json` files the generators write beside
  each `*BuildInfo.swift`, from the same values;
- `private_thermal`, `private_memory`: the caller's probes of the archived
  binary.

A runtime type this script cannot encode fails it. The proxy compares the
manifest with the phone's own listing after each install, so a drift here
withdraws the build rather than sending it jobs it cannot run.

Stdlib only and 3.9-compatible: macOS runners may run the system python3.
"""

from __future__ import annotations

import argparse
import json
import plistlib
import re
import sys
import xml.parsers.expat
from pathlib import Path

APP = Path("ios/Pipette/Pipette")
RUNTIMES_COMMANDS = APP / "Commands/RuntimesCommands.swift"
MLX_INFO = APP / "Generated/MLX/MLXBuildInfo.json"
ONNX_INFO = APP / "Generated/OnnxGenai/OnnxGenaiBuildInfo.json"

# `PIPETTE_LLAMA_MAIN_PIN` in ios/native/pipette_llama.h: vendor/llama.cpp's
# build, which every app carries.
MAIN_PIN = "llamacpp"

# Every on-device runtime's flavor is this one case (`*IosPipetteFlavor`).
FLAVOR = "ios-arm64"

# `RuntimesCommands.runtimeRows` lists a row per llama.cpp build the app
# carries and one per case of its `others` array literal, e.g.
# `let others = [RuntimeType.mlxIosPipette, .onnxGenaiIosPipette, ...]`.
LISTED_LLAMA = "LlamaPins.probeAll()"
LISTED = re.compile(r"let others = \[(RuntimeType\.\w+(?:,\s*\.\w+)*)\]")


class ManifestError(Exception):
    pass


def sources(path: Path, names: list[str]) -> dict[str, dict[str, str]]:
    """Each named pin as a descriptor's `SourceRepository`: its repository and
    its `submissionVersion`, the tag when it has one, else the commit."""
    pins = json.loads(path.read_text())
    missing = [name for name in names if name not in pins]
    if missing:
        raise ManifestError(f"{path} has no {', '.join(missing)}")
    return {
        name: {
            "repository_url": pins[name]["repository_url"],
            "repository_version": pins[name]["tag"] or pins[name]["commit"],
        }
        for name in names
    }


def llama_builds(frameworks: Path) -> list[dict]:
    """A descriptor per `llamacpp*.framework` in `frameworks`, in the order
    `LlamaPins.carried` lists them: main first, then by name."""
    paths = sorted(
        frameworks.glob(f"{MAIN_PIN}*.framework"),
        key=lambda path: (path.stem != MAIN_PIN, path.stem),
    )
    if not paths or paths[0].stem != MAIN_PIN:
        raise ManifestError(f"{frameworks} has no {MAIN_PIN}.framework")
    names = {}
    for path in paths:
        try:
            with open(path / "Info.plist", "rb") as file:
                info = plistlib.load(file)
        except xml.parsers.expat.ExpatError as error:
            raise ManifestError(f"{path}/Info.plist: {error}") from error
        url = info.get("PipetteLlamaRepositoryUrl")
        commit = info.get("PipetteLlamaCommit")
        if not url or not commit:
            raise ManifestError(f"{path}/Info.plist does not name its llama.cpp build")
        # `LlamaPin.submissionVersion`: the tag when the build has one, else the commit.
        version = info.get("PipetteLlamaTag") or commit
        # A job picks its build by this identity, so a second one could never run.
        if (url, version) in names:
            raise ManifestError(
                f"{names[url, version]} and {path.name} are both {url}@{version}"
            )
        names[url, version] = path.name
    return [
        {
            "type": "llamacpp_ios_pipette",
            "repository_url": url,
            "repository_version": version,
            "flavor": FLAVOR,
        }
        for url, version in names
    ]


def listed(root: Path) -> list[str]:
    """The `RuntimeType` cases `RuntimesCommands.runtimeRows` lists beside the
    llama.cpp builds, in order."""
    path = root / RUNTIMES_COMMANDS
    source = path.read_text()
    if LISTED_LLAMA not in source:
        raise ManifestError(
            f"{path} does not list the llama.cpp builds from {LISTED_LLAMA}"
        )
    match = LISTED.search(source)
    if match is None:
        raise ManifestError(f"{path} has no `let others = [RuntimeType.…]` list")
    return re.findall(r"\.(\w+)", match.group(1))


def descriptors(
    root: Path, frameworks: Path, private_thermal: bool, private_memory: bool
) -> list[dict]:
    encoders = {
        "mlxIosPipette": lambda: {
            "type": "mlx_ios_pipette",
            "packages": sources(
                root / MLX_INFO, ["mlx_swift", "mlx_swift_lm", "swift_transformers"]
            ),
            "flavor": FLAVOR,
        },
        "onnxGenaiIosPipette": lambda: {
            "type": "onnx_genai_ios_pipette",
            "packages": sources(root / ONNX_INFO, ["onnxruntime", "onnxruntime_genai"]),
            "flavor": FLAVOR,
        },
        "appleFoundation": lambda: {"type": "apple_foundation"},
    }
    out = llama_builds(frameworks)
    for case in listed(root):
        if case not in encoders:
            raise ManifestError(
                f"{root / RUNTIMES_COMMANDS} lists RuntimeType.{case}, which this "
                "script cannot encode; teach it the case's descriptor"
            )
        out.append(encoders[case]())
    for descriptor in out:
        # Omitted when false, as the app's encoder omits it.
        if private_thermal:
            descriptor["private_thermal"] = True
        # Defaults to `private_thermal`, so written only when the two differ.
        if private_memory != private_thermal:
            descriptor["private_memory"] = private_memory
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("client_version", help="the version the archive is named")
    parser.add_argument(
        "--private-thermal",
        action="store_true",
        help="the archive compiled in the private SoC temperature read",
    )
    parser.add_argument(
        "--private-memory",
        action="store_true",
        help="the archive compiled in the private memory flush and footprint peak",
    )
    parser.add_argument(
        "--frameworks",
        type=Path,
        required=True,
        help="the folder holding the llamacpp*.framework builds the archive carries",
    )
    parser.add_argument(
        "--root", type=Path, default=Path("."), help="the repository checkout"
    )
    args = parser.parse_args()
    try:
        manifest = {
            "client_version": args.client_version,
            "runtimes": descriptors(
                args.root, args.frameworks, args.private_thermal, args.private_memory
            ),
        }
    except (ManifestError, OSError, ValueError, KeyError) as error:
        print(f"release-manifest.py: {error}", file=sys.stderr)
        return 1
    json.dump(manifest, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
