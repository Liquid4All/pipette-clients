#!/usr/bin/env bash
#
# Ad-hoc sign the app in an unsigned Pipette.xcarchive with the app's entitlements.
#
# An unsigned binary carries none, and `xcodebuild -exportArchive`, the fleet's
# signing path, keeps only what the binary carries, so without this its builds
# lack increased-memory-limit. Xcode refuses an ad-hoc identity for an iOS target
# with entitlements, hence codesign after the archive.
#
# Fleet builds are only ever development-signed, so this also grants
# increased-debugging-memory-limit, which raises a 12 GB iPhone's per-app limit from
# 6144 to 6656 MiB. Pipette.entitlements leaves it out for the App Store.
#
#   ios/adhoc-sign-archive.sh <path/to/Pipette.xcarchive> [<derived data path>]
#
# The archive's -derivedDataPath lets the build-settings read reuse its resolved
# packages. A variable in Pipette.entitlements that cannot be expanded fails the run.
set -euo pipefail

ARCHIVE="${1:?usage: $0 <path/to/Pipette.xcarchive> [<derived data path>]}"
DERIVED_DATA=()
if [[ $# -ge 2 ]]; then DERIVED_DATA=(-derivedDataPath "$2"); fi
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT="$REPO_ROOT/ios/Pipette/Pipette.xcodeproj"
APP="$ARCHIVE/Products/Applications/Pipette.app"
[[ -d "$APP" ]] || { echo "error: no app at $APP" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

xcodebuild -showBuildSettings -json -project "$PROJECT" -scheme Pipette \
    -configuration Release -destination 'generic/platform=iOS' \
    -skipPackagePluginValidation -skipMacroValidation \
    ${DERIVED_DATA[@]+"${DERIVED_DATA[@]}"} > "$work/settings.json"

python3 - "$work/settings.json" "$work/entitlements.plist" <<'EOF'
import json, plistlib, re, sys

settings_path, out_path = sys.argv[1], sys.argv[2]
settings = next(t["buildSettings"] for t in json.load(open(settings_path))
                if t["target"] == "Pipette")
source = settings["SRCROOT"] + "/" + settings["CODE_SIGN_ENTITLEMENTS"]
var = re.compile(r"\$[({](\w+)[)}]")

def expand(value):
    if isinstance(value, str):
        out = var.sub(lambda m: settings.get(m.group(1), m.group(0)), value)
        # Any leftover macro, including forms like $(VAR:rfc1034identifier).
        if re.search(r"\$[({]", out):
            sys.exit(f"error: {source}: cannot expand {out!r}")
        return out
    if isinstance(value, list):
        return [expand(v) for v in value]
    if isinstance(value, dict):
        return {k: expand(v) for k, v in value.items()}
    return value

with open(source, "rb") as f:
    entitlements = expand(plistlib.load(f))
entitlements["com.apple.developer.kernel.increased-debugging-memory-limit"] = True
with open(out_path, "wb") as f:
    plistlib.dump(entitlements, f)
EOF

# Nested code first, since the app's signature seals it.
for framework in "$APP"/Frameworks/*.framework; do
    [[ -e "$framework" ]] || continue
    codesign --force --sign - --timestamp=none "$framework"
done
codesign --force --sign - --timestamp=none --generate-entitlement-der \
    --entitlements "$work/entitlements.plist" "$APP"

codesign --verify --deep --strict "$APP"
# --deep passes an unsigned nested framework, so each is verified on its own.
for framework in "$APP"/Frameworks/*.framework; do
    [[ -e "$framework" ]] || continue
    codesign --verify --strict "$framework"
done
codesign -d --entitlements - --xml "$APP" > "$work/signed.plist"
for key in increased-memory-limit increased-debugging-memory-limit; do
    value="$(/usr/libexec/PlistBuddy -c \
        "Print :com.apple.developer.kernel.$key" "$work/signed.plist" || true)"
    if [[ "$value" != true ]]; then
        echo "error: $APP is not signed with $key set to true" >&2
        exit 1
    fi
done
echo "==> Ad-hoc signed $APP with its entitlements"
