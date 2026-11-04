#!/usr/bin/env python3
"""Generate ThirdPartyLicenses.json for the iOS app.

Attribution is split by how a dependency enters the app:

  - Swift packages (everything in Package.resolved, transitive ones included)
    are handled at build time by the LicenseList plugin, not here.
  - Vendored source trees compiled into the app (llama.cpp in its frameworks,
    ONNX Runtime, ONNX Runtime GenAI) are listed here, since LicenseList cannot
    see sources outside the package graph.

Re-run after adding or bumping a vendored submodule and commit the result.
"""

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "ios/Pipette/Pipette/ThirdPartyLicenses.json"

# (display name, license, path to the license text). Add a row when another
# vendored source tree is compiled in.
COMPONENTS = [
    ("llama.cpp", "MIT License", REPO / "vendor/llama.cpp/LICENSE"),
    ("ONNX Runtime", "MIT License", REPO / "vendor/onnxruntime/LICENSE"),
    ("ONNX Runtime GenAI", "MIT License", REPO / "vendor/onnxruntime-genai/LICENSE"),
]


def main() -> None:
    entries = []
    for name, license_id, license_path in COMPONENTS:
        if not license_path.exists():
            sys.exit(
                f"missing license file: {license_path} (is the submodule checked out?)"
            )
        entries.append(
            {
                "name": name,
                "license": license_id,
                "text": license_path.read_text().strip() + "\n",
            }
        )

    OUT.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {OUT.relative_to(REPO)} ({len(entries)} entries)")


if __name__ == "__main__":
    main()
