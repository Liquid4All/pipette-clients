#!/usr/bin/env python3
"""Generate ThirdPartyLicenses.json for the iOS app.

Attribution is split by how a dependency enters the binary:

  - Swift packages (SwiftPM) — every dependency in Package.resolved, including
    the MLX stack (mlx-swift, mlx-swift-lm) and all transitive packages — are
    handled automatically at build time by the LicenseList plugin. They are NOT
    listed here.
  - Non-SwiftPM components compiled into the binary — currently just the
    vendored llama.cpp (the app is native Swift and links no Rust) — are listed
    here, since LicenseList can't see sources outside the package graph.

So this file produces exactly the non-SwiftPM entries. Re-run after bumping the
vendored llama.cpp submodule and commit the result.
"""

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
OUT = REPO / "ios/Pipette/Pipette/ThirdPartyLicenses.json"

# Non-SwiftPM components compiled into the app: (display name, SPDX license,
# path to the license text). Add a row here when another vendored source tree
# is compiled in.
COMPONENTS = [
    ("llama.cpp", "MIT License", REPO / "vendor/llama.cpp/LICENSE"),
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
