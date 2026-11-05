#!/usr/bin/env bash
#
# Install a pinned CI tool, verifying its SHA-256 before anything is unpacked or
# executed.
#
#   ci/install-tool.sh <cargo-deny|cargo-machete|gitleaks>
#
# Pins live here rather than at call sites so they cannot drift between
# workflows. This file is meant to be identical in each Pipette repository, so a
# drift reads as a diff.
#
# Dependabot cannot see these pins: bump by hand, with the digest
# `shasum -a 256` reports for the new asset.
#
# Linux runners only: it wants `sha256sum`, and installs into /usr/local/bin.
set -euo pipefail

name="${1-}"
case "${name}" in
  cargo-deny)
    url="https://github.com/EmbarkStudios/cargo-deny/releases/download/0.20.2/cargo-deny-0.20.2-x86_64-unknown-linux-musl.tar.gz"
    sha="9f12ed4c49936e09b48bf862b595cde2fe64fcbd9d74dfacac6131ca824c8d5f"
    ;;
  cargo-machete)
    url="https://github.com/bnjbvr/cargo-machete/releases/download/v0.9.2/cargo-machete-v0.9.2-x86_64-unknown-linux-musl.tar.gz"
    sha="48200087f54c55aabcd4db4af1e25742b49846c02a1b1bfa134711945b35b2e9"
    ;;
  gitleaks)
    url="https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz"
    sha="551f6fc83ea457d62a0d98237cbad105af8d557003051f41f3e7ca7b3f2470eb"
    ;;
  *)
    echo "usage: $0 <cargo-deny|cargo-machete|gitleaks>" >&2
    exit 2
    ;;
esac

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp}"' EXIT

curl -fsSLo "${tmp}/pkg.tar.gz" "${url}"
# Verify before unpacking so a substituted archive never reaches tar.
echo "${sha}  ${tmp}/pkg.tar.gz" | sha256sum -c -
tar -xzf "${tmp}/pkg.tar.gz" -C "${tmp}"

# `-print -quit`, not `| head`: under pipefail a closed pipe exits 141 before
# the check below can report anything.
bin="$(find "${tmp}" -type f -name "${name}" -print -quit)"
if [[ -z "${bin}" ]]; then
  echo "$0: no file named '${name}' inside ${url}" >&2
  exit 1
fi

sudo=""
if [[ "$(id -u)" -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
  sudo="sudo"
fi
${sudo} install -m 0755 "${bin}" "/usr/local/bin/${name}"
# The URL carries the version, so the run log records exactly what ran.
echo "installed ${name} -> /usr/local/bin/${name} (${url})"
