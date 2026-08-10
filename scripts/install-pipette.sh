#!/usr/bin/env bash
# Install the `pipette` CLI on a fleet box.
#
# Two steps, each idempotent, so re-running is how you upgrade:
#   1. download the `pipette-cli-<triple>` release asset and install it at
#      <work-dir>/pipette — the path every plan addresses;
#   2. `pipette --work-dir <work-dir> init`, creating <work-dir>/.pipette.
#
# `init` leaves an existing workspace alone, so upgrading in place is safe.
#
# Usage:
#   ./scripts/install-pipette.sh -t macos   -H boston-mbp-m5-1 -u liquid
#   ./scripts/install-pipette.sh -t windows -H boston-gmktec   -u liquid
#   ./scripts/install-pipette.sh -t android -S R3GL30CRBGM
#   ./scripts/install-pipette.sh -t android -S R3GL30CRBGM \
#       -H boston-linux-belink -u liquid    # adb runs on the controller
#   ./scripts/install-pipette.sh -t linux   -H boston-pi5 -u liquid -a aarch64
#   ./scripts/install-pipette.sh -t macos   -H boston-macstudio-m3-1 -u liquid \
#       -b pipette-plan                     # the driver, not a client
#
# Flags:
#   -t  target: macos | windows | android | linux   (required)
#   -b  binary: pipette (client, default) | pipette-plan (driver)
#   -H  ssh host                                    (ssh targets)
#   -u  ssh user
#   -p  ssh port                                    (default 22)
#   -S  adb serial                                  (android)
#   -H  with -t android: the box whose adb server owns the device; adb runs
#       there over ssh, so this machine needs no adb and no tunnel
#   -a  arch: x86_64 | aarch64                      (windows/linux, default x86_64)
#   -d  work dir                                    (default per target, below)
#   -e  shell command run before adb on the intermediate host, joined with &&
#       (non-interactive ssh skips the login profile that puts adb on PATH)
#   -V  verify only: report state, install nothing

set -euo pipefail

readonly REPO="Liquid4All/edge-evals-llama.cpp"

# Fleet defaults, from devices-topology-map.md.
readonly DEFAULT_DIR_MACOS="/Users/liquid/workplace/pipette"
readonly DEFAULT_DIR_WINDOWS='C:\pipette'
readonly DEFAULT_DIR_ANDROID="/data/local/tmp/pipette"
readonly DEFAULT_DIR_LINUX="/home/liquid/edge-evals"

die() { echo "Error: $*" >&2; exit 1; }

# The Usage/Flags block above, located rather than hardcoded so editing the
# header cannot silently truncate the help.
usage() { sed -n '/^# Usage:/,/^$/p' "$0" | sed '$d' >&2; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

target="" host="" ssh_user="" port="22" serial="" arch="x86_64" work_dir=""
verify_only=0 tool="pipette" pre_exec=""
while getopts "t:H:u:p:S:a:d:b:e:V" opt; do
  case "$opt" in
    t) target="$OPTARG" ;;
    b) tool="$OPTARG" ;;
    e) pre_exec="$OPTARG" ;;
    H) host="$OPTARG" ;;
    u) ssh_user="$OPTARG" ;;
    p) port="$OPTARG" ;;
    S) serial="$OPTARG" ;;
    a) arch="$OPTARG" ;;
    d) work_dir="$OPTARG" ;;
    V) verify_only=1 ;;
    *) usage; exit 1 ;;
  esac
done

# `pipette` is the client, released as `pipette-cli-<triple>`; `pipette-plan` is
# the driver, from the same tag.
case "$tool" in
  pipette)      asset_prefix="pipette-cli" ;;
  pipette-plan) asset_prefix="pipette-plan" ;;
  *) die "-b must be one of: pipette pipette-plan" ;;
esac

case "$target" in
  macos)
    asset="${asset_prefix}-aarch64-apple-darwin.tar.gz"
    : "${work_dir:=$DEFAULT_DIR_MACOS}"
    ;;
  windows)
    case "$arch" in
      x86_64)  asset="${asset_prefix}-x86_64-pc-windows-msvc.zip" ;;
      aarch64) asset="${asset_prefix}-aarch64-pc-windows-msvc.zip" ;;
      *) die "unsupported arch for windows: $arch" ;;
    esac
    : "${work_dir:=$DEFAULT_DIR_WINDOWS}"
    ;;
  android)
    asset="${asset_prefix}-aarch64-linux-android.tar.gz"
    : "${work_dir:=$DEFAULT_DIR_ANDROID}"
    ;;
  linux)
    case "$arch" in
      x86_64)  asset="${asset_prefix}-x86_64-unknown-linux-gnu.tar.gz" ;;
      aarch64) asset="${asset_prefix}-aarch64-unknown-linux-gnu.tar.gz" ;;
      *) die "unsupported arch for linux: $arch" ;;
    esac
    : "${work_dir:=$DEFAULT_DIR_LINUX}"
    ;;
  *) die "-t must be one of: macos windows android linux" ;;
esac

if [[ "$tool" == "pipette-plan" && "$target" == "android" ]]; then
  die "pipette-plan has no android build — the driver runs on a host, not a device"
fi

if [[ -n "$pre_exec" && ( "$target" != "android" || -z "$host" ) ]]; then
  die "-e applies only to '-t android -H <controller>', where adb runs over ssh"
fi

# Single-quote for a remote posix shell, escaping embedded quotes — the
# android-through-a-host path nests the device command inside the controller's
# shell, exactly as the plan's `adb_over_ssh` transport does.
sq() {
  # Each embedded quote becomes '\'' — close, escaped quote, reopen.
  local repl="'\\''"
  printf "'%s'" "${1//\'/$repl}"
}

if [[ "$target" == "android" ]]; then
  [[ -n "$serial" ]] || die "-S <serial> is required for android"
  if [[ -n "$host" ]]; then
    # The handsets are paired to one controller (see devices-topology-map.md),
    # so adb runs there and this box needs only ssh.
    require_cmd ssh
    require_cmd scp
    ssh_dest="${ssh_user:+${ssh_user}@}${host}"
    # `pre_exec` is interpolated raw — it is shell, the way slurm's `pre_exec` is
    # — while the serial is data and gets quoted.
    remote_adb() { printf '%s%s' "${pre_exec:+$pre_exec && }" "adb -s $(sq "$serial")"; }
    run() {
      ssh -o BatchMode=yes -p "$port" "$ssh_dest" \
        "$(remote_adb) shell $(sq "$1")"
    }
    push() {
      local staged="/tmp/pipette-${serial}.$$"
      scp -q -P "$port" "$1" "${ssh_dest}:${staged}"
      # Clean up whatever the push did, without reporting its failure as success.
      ssh -o BatchMode=yes -p "$port" "$ssh_dest" \
        "$(remote_adb) push $(sq "$staged") $(sq "$2") >/dev/null; \
         rc=\$?; rm -f $(sq "$staged"); exit \$rc"
    }
    label="adb-ssh:${ssh_dest}:${serial}"
  else
    require_cmd adb
    run() { adb -s "$serial" shell "$1"; }
    push() { adb -s "$serial" push "$1" "$2" >/dev/null; }
    label="$serial"
  fi
else
  [[ -n "$host" ]] || die "-H <host> is required for $target"
  require_cmd ssh
  require_cmd scp
  ssh_dest="${ssh_user:+${ssh_user}@}${host}"
  run() { ssh -o BatchMode=yes -p "$port" "$ssh_dest" "$@"; }
  push() { scp -q -P "$port" "$1" "${ssh_dest}:$2"; }
  label="$ssh_dest"
fi

# ---------------------------------------------------------------------------
# Remote snippets. Each is a single command string so it works over both
# `ssh <cmd>` and `adb shell <cmd>`; Windows goes through powershell.
# ---------------------------------------------------------------------------

posix_report() {
  local d="$1"
  cat <<EOF
d=$d
printf 'binary:   '; [ -x "\$d/$tool" ] && "\$d/$tool" --version 2>&1 | head -1 || echo '(absent)'
EOF
}

ps_report() {
  local d="$1"
  cat <<EOF
\$d = '$d'
if (Test-Path "\$d\\${tool}.exe") {
  Write-Host -NoNewline 'binary:   '
  & "\$d\\${tool}.exe" --version
} else {
  Write-Host 'binary:   (absent)'
}
EOF
}

# Multi-line PowerShell needs `-EncodedCommand`: `-Command` would go through
# cmd.exe quoting, and `-Command -` parses stdin a line at a time, which breaks
# any if/foreach block spanning lines.
# `$ProgressPreference` off because a redirected stderr gets the progress stream
# as CLIXML, which buries the real output whenever the caller merges streams.
pwsh() {
  local encoded
  encoded="$(printf '$ProgressPreference = "SilentlyContinue"\n%s' "$1" \
    | iconv -f UTF-8 -t UTF-16LE | base64 | tr -d '\n')"
  run "powershell -NoProfile -NonInteractive -OutputFormat Text -EncodedCommand $encoded"
}

echo "== ${label} (${target}, work dir ${work_dir})"
echo "-- before"
case "$target" in
  windows) pwsh "$(ps_report "$work_dir")" ;;
  *)       run "$(posix_report "$work_dir")" ;;
esac

if (( verify_only )); then exit 0; fi

require_cmd gh
gh auth status >/dev/null 2>&1 || die "'gh' is not authenticated. Run 'gh auth login'."
tag="$(gh release view --repo "$REPO" --json tagName --jq '.tagName')"

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
echo "-- downloading ${asset} from ${tag}"
gh release download "$tag" --repo "$REPO" --pattern "$asset" --dir "$tmpdir"
case "$asset" in
  *.tar.gz) require_cmd tar;   tar xzf "${tmpdir}/${asset}" -C "$tmpdir" ;;
  *.zip)    require_cmd unzip; unzip -q "${tmpdir}/${asset}" -d "$tmpdir" ;;
esac

echo "-- installing"
case "$target" in
  windows)
    pwsh "Stop-Process -Name ${tool} -Force -ErrorAction SilentlyContinue; New-Item -ItemType Directory -Force -Path '${work_dir}' | Out-Null"
    push "${tmpdir}/${tool}.exe" "${work_dir}\\${tool}.exe"
    run "${work_dir}\\${tool}.exe --work-dir ${work_dir} init"
    ;;
  android)
    run "mkdir -p ${work_dir}"
    run "pkill -x ${tool} || true"
    push "${tmpdir}/${tool}" "${work_dir}/${tool}"
    run "chmod +x ${work_dir}/${tool}"
    run "${work_dir}/${tool} --work-dir ${work_dir} init"
    ;;
  macos|linux)
    run "mkdir -p ${work_dir}"
    run "pkill -x ${tool} || true"
    push "${tmpdir}/${tool}" "${work_dir}/${tool}"
    # The release binary is unsigned; macOS refuses to exec it without an
    # ad-hoc signature.
    if [[ "$target" == "macos" ]]; then
      run "codesign --force --sign - ${work_dir}/${tool} && codesign --verify --strict ${work_dir}/${tool}"
    else
      run "chmod +x ${work_dir}/${tool}"
    fi
    run "${work_dir}/${tool} --work-dir ${work_dir} init"
    ;;
esac

echo "-- after"
case "$target" in
  windows) pwsh "$(ps_report "$work_dir")" ;;
  *)       run "$(posix_report "$work_dir")" ;;
esac
echo "installed ${tool} ${tag} on ${label}"
