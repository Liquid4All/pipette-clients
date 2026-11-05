#!/usr/bin/env bash
#
# android-engine-ab.sh: alternate rustEngine and kotlinEngine benchmark passes on a device.
#
# Usage:
#   ./scripts/android-engine-ab.sh [options]
#
# Options:
#   -s SERIAL        ADB device serial (default: $ANDROID_SERIAL or lone device)
#   -n PASSES        Number of passes per arm (default: 5)
#   -b BENCHMARKS    Comma-separated benchmark matrix
#                    (default: prefill_throughput_512,decode_throughput_512_128,end_to_end_latency_512_128,max_memory_usage_512)
#   -m MATCH         Model name substring filter (optional)
#   -w MAX_WAIT      Readiness max wait in seconds (default: 300)
#   --skip-thermal   Skip thermal readiness wait
#   -r RUST_APK      Path to rustEngine release APK
#   -k KOTLIN_APK    Path to kotlinEngine release APK
#   -o OUTPUT_DIR    Results output directory (default: target/engine-ab-results)
#   --null-ab        Run null A/B test (rustEngine vs rustEngine) to measure noise floor
#   -h, --help       Show this help message

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SERIAL="${ANDROID_SERIAL:-}"
PASSES=5
BENCHMARKS="prefill_throughput_512,decode_throughput_512_128,end_to_end_latency_512_128,max_memory_usage_512"
MATCH=""
MAX_WAIT=300
SKIP_THERMAL=false
RUST_APK="$REPO_ROOT/android/Pipette/app/build/outputs/apk/rustEngine/release/app-rustEngine-release.apk"
KOTLIN_APK="$REPO_ROOT/android/Pipette/app/build/outputs/apk/kotlinEngine/release/app-kotlinEngine-release.apk"
OUTPUT_DIR="$REPO_ROOT/target/engine-ab-results"
NULL_AB=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    -s|--serial)
      SERIAL="$2"
      shift 2
      ;;
    -n|--passes)
      PASSES="$2"
      shift 2
      ;;
    -b|--benchmarks)
      BENCHMARKS="$2"
      shift 2
      ;;
    -m|--match)
      MATCH="$2"
      shift 2
      ;;
    -w|--max-wait)
      MAX_WAIT="$2"
      shift 2
      ;;
    --skip-thermal)
      SKIP_THERMAL=true
      shift
      ;;
    -r|--rust-apk)
      RUST_APK="$2"
      shift 2
      ;;
    -k|--kotlin-apk)
      KOTLIN_APK="$2"
      shift 2
      ;;
    -o|--output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --null-ab)
      NULL_AB=true
      shift
      ;;
    -h|--help)
      sed -n '3,20p' "$0" | sed 's/^# *//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

ADB_BIN="adb"
if ! command -v adb >/dev/null 2>&1; then
  for candidate in \
    "${ANDROID_HOME:-}/platform-tools/adb" \
    "${ANDROID_SDK_ROOT:-}/platform-tools/adb" \
    "$HOME/Library/Android/sdk/platform-tools/adb"; do
    if [[ -x "$candidate" ]]; then
      ADB_BIN="$candidate"
      break
    fi
  done
fi

ADB_CMD=("$ADB_BIN")
if [[ -n "$SERIAL" ]]; then
  ADB_CMD=("$ADB_BIN" -s "$SERIAL")
fi

if ! "${ADB_CMD[@]}" get-state >/dev/null 2>&1; then
  echo "Error: ADB device not available or not responding." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
echo "Starting Android Engine A/B Harness"
echo "Device Serial: ${SERIAL:-default}"
echo "Passes Per Arm: $PASSES"
echo "Benchmarks: $BENCHMARKS"
echo "Null A/B Mode: $NULL_AB"
echo "Output Directory: $OUTPUT_DIR"
echo "----------------------------------------"

run_pass() {
  local pass="$1"
  local arm_label="$2"
  local target_flavor="$3"
  local apk_path="$4"
  local pass_dir="$OUTPUT_DIR/pass_${pass}_${arm_label}"
  mkdir -p "$pass_dir"

  echo "==> [Pass $pass | Arm $arm_label] Installing $apk_path"
  if [[ ! -f "$apk_path" ]]; then
    echo "Error: APK not found at $apk_path" >&2
    exit 1
  fi

  # -r keeps app data, so downloaded models under getExternalFilesDir survive the swap.
  "${ADB_CMD[@]}" install -r -d "$apk_path"
  "${ADB_CMD[@]}" shell pm grant ai.liquid.pipette android.permission.WRITE_SECURE_SETTINGS >/dev/null 2>&1 || true

  "${ADB_CMD[@]}" logcat -c
  "${ADB_CMD[@]}" shell am start -W -n ai.liquid.pipette/.headless.HeadlessRunActivity -e verb version >/dev/null 2>&1
  local version_line
  version_line="$("${ADB_CMD[@]}" logcat -d -s pipette-headless:I | grep '\[HEADLESS\] version' | tail -n 1 || true)"
  echo "Reported: $version_line"

  if ! echo "$version_line" | grep -q "flavor=${target_flavor}"; then
    echo "Error: Installed APK flavor does not match target flavor ${target_flavor}: ${version_line}" >&2
    exit 1
  fi

  echo "==> [Pass $pass | Arm $arm_label] Launching benchmark run"
  "${ADB_CMD[@]}" logcat -c
  local run_args=(-e verb run -e submit 0 -e benchmarks "$BENCHMARKS" -e readiness_max_wait_secs "$MAX_WAIT")
  if [[ -n "$MATCH" ]]; then
    run_args+=(-e match "$MATCH")
  fi
  if [[ "$SKIP_THERMAL" == "true" ]]; then
    run_args+=(-e readiness_skip_thermal 1)
  fi

  "${ADB_CMD[@]}" shell am start -W -n ai.liquid.pipette/.headless.HeadlessRunActivity "${run_args[@]}" >/dev/null 2>&1

  local started_line
  started_line="$("${ADB_CMD[@]}" logcat -d -s pipette-headless:I | grep '\[HEADLESS\] STARTED' | tail -n 1 || true)"
  local job_id
  job_id="$(echo "$started_line" | sed -n 's/.*job=\([a-zA-Z0-9_-]*\).*/\1/p')"

  if [[ -z "$job_id" ]]; then
    echo "Error: Failed to launch benchmark run; no STARTED job emitted in logcat" >&2
    exit 1
  fi
  echo "Job ID: $job_id"

  local elapsed=0
  local timeout_secs=3600
  echo -n "==> [Pass $pass | Arm $arm_label] Polling job execution status: "
  while [[ $elapsed -lt $timeout_secs ]]; do
    sleep 5
    elapsed=$((elapsed + 5))
    "${ADB_CMD[@]}" logcat -c
    "${ADB_CMD[@]}" shell am start -W -n ai.liquid.pipette/.headless.HeadlessRunActivity -e verb status >/dev/null 2>&1
    local status_line
    status_line="$("${ADB_CMD[@]}" logcat -d -s pipette-headless:I | grep '\[HEADLESS\] status' | tail -n 1 || true)"
    if echo "$status_line" | grep -q 'running=false'; then
      echo " completed in ${elapsed}s"
      break
    fi
    echo -n "."
  done

  if [[ $elapsed -ge $timeout_secs ]]; then
    echo ""
    echo "Error: Benchmark run timed out after ${timeout_secs}s" >&2
    exit 1
  fi

  echo "==> [Pass $pass | Arm $arm_label] Extracting results"
  "${ADB_CMD[@]}" logcat -c
  "${ADB_CMD[@]}" shell am start -W -n ai.liquid.pipette/.headless.HeadlessRunActivity -e verb results >/dev/null 2>&1
  "${ADB_CMD[@]}" logcat -d -s pipette-headless:I | grep "\\[HEADLESS\\] result .*job=${job_id}" > "$pass_dir/results.txt" || true

  "${ADB_CMD[@]}" logcat -c
  "${ADB_CMD[@]}" shell am start -W -n ai.liquid.pipette/.headless.HeadlessRunActivity -e verb job-export -e id "$job_id" >/dev/null 2>&1
  local export_line
  export_line="$("${ADB_CMD[@]}" logcat -d -s pipette-headless:I | grep '\[HEADLESS\] EXPORTED' | tail -n 1 || true)"
  echo "$export_line"
  echo "$export_line" > "$pass_dir/export.txt"

  local remote_path
  remote_path="$(echo "$export_line" | sed -n 's/.*path=\([^ ]*\).*/\1/p')"
  if [[ -n "$remote_path" ]]; then
    "${ADB_CMD[@]}" pull "$remote_path" "$pass_dir/" >/dev/null 2>&1 || true
  fi
  mkdir -p "$pass_dir/payloads"
  "${ADB_CMD[@]}" pull "/storage/emulated/0/Android/data/ai.liquid.pipette/files/exports/${job_id}/." "$pass_dir/payloads/" >/dev/null 2>&1 || true

  echo "Results saved to $pass_dir"
}

for ((i = 1; i <= PASSES; i++)); do
  echo ""
  echo "========================================"
  echo "Iteration $i of $PASSES"
  echo "========================================"

  run_pass "$i" "arm_a_rust" "rust" "$RUST_APK"

  if [[ "$NULL_AB" == "true" ]]; then
    run_pass "$i" "arm_b_rust" "rust" "$RUST_APK"
  else
    run_pass "$i" "arm_b_kotlin" "kotlin" "$KOTLIN_APK"
  fi
done

echo ""
echo "A/B benchmark comparison completed successfully."
echo "All outputs recorded under: $OUTPUT_DIR"
echo ""

if [[ -f "$SCRIPT_DIR/analyze-engine-ab.py" ]]; then
  python3 "$SCRIPT_DIR/analyze-engine-ab.py" "$OUTPUT_DIR" | tee "$OUTPUT_DIR/summary.md"
fi

