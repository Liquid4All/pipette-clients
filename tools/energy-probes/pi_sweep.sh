#!/bin/bash
# Token sweep on a Raspberry Pi 5: decode n_gen and prefill n_prompt, each at
# -r 2 and -r 6, so (E6-E2)/4 is one repetition with model load and warm-up
# differenced out. Energy is the PMIC rail sum from pi_power_sampler.sh, mean
# over the run times its elapsed time. Output lines feed fit.py. Takes 60 s of
# idle before the first run and 30 s after the last, so the same CSV holds the
# baseline. Stop any benchmark worker on the Pi before running this.
LB=${LB:-$HOME/workplace/pipette/.pipette/runtimes/llama-cpp__ggml-org_llama.cpp__b10516__linux-arm64-cpu/blobs/llama-b10516/llama-bench}
M=${M:-$HOME/workplace/pipette/.pipette/models/unsloth__Llama-3.2-1B-I_c9a31313/blobs/Llama-3.2-1B-Instruct-Q4_K_M.gguf}
CSV=${CSV:-/tmp/pi_sweep_samples.csv}
LD_LIBRARY_PATH=$(dirname "$LB")
export LD_LIBRARY_PATH
"$(dirname "$0")/pi_power_sampler.sh" "$CSV" &
sleep 60
window() { # t0 t1 -> "n mean_w joules" over rows in [t0,t1]
  awk -F, -v a="$1" -v b="$2" 'NR > 1 && $1 >= a && $1 <= b { s += $2; n++ }
    END { if (n) printf "%d %.3f %.2f\n", n, s / n, s / n * (b - a); else print "0 0 0" }' "$CSV"
}
run() { # phase n r
  local args t0 t1 out ts n mean j
  if [ "$1" = decode ]; then args=(--n-prompt 0 --n-gen "$2" --n-depth 512); else args=(--n-prompt "$2" --n-gen 0); fi
  t0=$(date +%s.%N | cut -c1-14)
  out=$("$LB" --model "$M" --mmap 0 -t 4 -fa on -r "$3" "${args[@]}" --output json 2>/dev/null)
  t1=$(date +%s.%N | cut -c1-14)
  ts=$(echo "$out" | python3 -c "import sys,json; s=sys.stdin.read(); j=s[:s.rfind(']')+1]; d=json.loads(j[j.find('['):]); print('%.3f'%(sum(d[0]['samples_ns'])/1e9))" 2>/dev/null)
  read -r n mean j < <(window "$t0" "$t1")
  echo "$1 n=$2 r=$3 bench_measured_s=$ts elapsed_s=$(awk -v a="$t0" -v b="$t1" "BEGIN { printf \"%.3f\", b - a }") pmic_mean_w=$mean pmic_n=$n pmic_j=$j"
}
t_idle_end=$(date +%s.%N | cut -c1-14)
for n in 50 100 200 400; do run decode "$n" 2; run decode "$n" 6; done
for n in 256 512 1024 2048; do run prefill "$n" 2; run prefill "$n" 6; done
t_tail=$(date +%s.%N | cut -c1-14)
sleep 30
read -r n mean _ < <(window 0 "$t_idle_end"); echo "idle_before n=$n mean_w=$mean"
read -r n mean _ < <(window "$t_tail" 9999999999); echo "idle_after n=$n mean_w=$mean"
touch "$CSV.stop"; wait
