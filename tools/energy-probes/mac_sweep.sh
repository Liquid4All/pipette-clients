#!/bin/bash
# Token sweep on macOS: decode n_gen and prefill n_prompt, each at -r 2 and -r 6,
# so (E6-E2)/4 is one repetition with model load and warm-up differenced out.
cd /tmp/ep && clang -O1 -o ewrap ewrap.c -framework CoreFoundation -framework IOKit -lIOReport 2>&1 | grep -v warning | head -3
LB=/tmp/llama-b10516/llama-b10516/llama-bench
M=/Users/pipette/edge-evals/data/models/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/LFM2.5-1.2B-Instruct-Q4_0.gguf
export DYLD_LIBRARY_PATH=/tmp/llama-b10516/llama-b10516
run() { # phase n r
  if [ "$1" = decode ]; then args="--n-prompt 0 --n-gen $2 --n-depth 512"; else args="--n-prompt $2 --n-gen 0"; fi
  out=$(./ewrap $LB --model $M --mmap 0 -r $3 $args --output json 2>/dev/null)
  ts=$(echo "$out" | python3 -c "import sys,json; s=sys.stdin.read(); j=s[:s.rfind(']')+1]; d=json.loads(j[j.find('['):]); print('%.3f'%(sum(d[0]['samples_ns'])/1e9))" 2>/dev/null)
  echo "$1 n=$2 r=$3 bench_measured_s=$ts $(echo "$out" | grep EWRAP)"
}
sleep 5
for n in 50 100 200 400 800; do run decode $n 2; run decode $n 6; done
for n in 256 512 1024 2048 4096; do run prefill $n 2; run prefill $n 6; done
