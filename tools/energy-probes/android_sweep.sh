#!/bin/sh
# Token sweep on an Android handset: each length at -r 2 and -r 6, so
# (E6-E2)/4 is one repetition with model load and warm-up differenced out.
# Run with the phone's mediator unit stopped. android_sampler.sh, pushed as
# /data/local/tmp/energy_sampler.sh, collects alongside.
cd /data/local/tmp/pipette || exit 1
D=/data/local/tmp/pipette/.pipette/runtimes/llama-cpp__ggml-org_llama.cpp__b10516__android-arm64-v8a/blobs/llama-b10516; export LD_LIBRARY_PATH=$D; LB=$D/llama-bench
M=.pipette/models/LiquidAI__LFM2.5-1.2B-I_1eb107bb/blobs/LFM2.5-1.2B-Instruct-Q4_0.gguf
rm -f /data/local/tmp/energy_stop; nohup sh /data/local/tmp/energy_sampler.sh > /dev/null 2>&1 &
sleep 15
run() {
  if [ "$1" = decode ]; then args="--n-prompt 0 --n-gen $2 --n-depth 512"; else args="--n-prompt $2 --n-gen 0"; fi
  t0=$(date +%s.%N | cut -c1-14)
  ts=$($LB --model $M --mmap 0 -r $3 $args --output json 2>/dev/null | tr -d '\n' | sed 's/.*"samples_ns": *\[\([^]]*\)\].*/\1/' | tr ',' '\n' | awk '{s+=$1} END {printf "%.3f", s/1e9}')
  t1=$(date +%s.%N | cut -c1-14)
  echo "$1 n=$2 r=$3 bench_measured_s=$ts t0=$t0 t1=$t1"
  sleep 8   # let the charger refill so runs start from the same state
}
run decode 50 2 > /dev/null
for n in 50 100 200 400 800; do run decode $n 2; run decode $n 6; done
for n in 256 512 1024 2048 4096; do run prefill $n 2; run prefill $n 6; done
sleep 10; touch /data/local/tmp/energy_stop; sleep 2
echo SAMPLES_BEGIN; cat /data/local/tmp/energy_samples.txt; echo SAMPLES_END
