#!/bin/bash
# Stepped compute ramp (0 -> 28 threads -> 0): prints SoC counters and PSTR per step.
# Needs `e` (ioreport_energy.c) and `smc` (smc_power.c) built in /tmp/ep.
cd /tmp/ep || exit 1
soc() { ./e 4000 | awk '/CPU Energy|GPU Energy|^DRAM|^AMCC|^DCS|^ANE|^FAB/ && !/SRAM|DTL/ {s+=$(NF-1); if ($0 ~ /CPU Energy/) c+=$(NF-1); if ($0 ~ /GPU Energy/) g+=$(NF-1)} END {printf "%.2f %.2f %.2f", c, g, s}'; }
pstr() { ./smc 4 PSTR | awk -F'[= ]' '{s+=$2} END {printf "%.1f", s/NR}'; }
step() {
  n=$1; pids=""
  if [ "$n" -gt 0 ]; then for _ in $(seq 1 "$n"); do awk 'BEGIN{while(1){x=x*1.000001+1}}' & pids="$pids $!"; done; fi
  sleep 2
  p=$(pstr); read c g s < <(soc)
  printf "threads=%-2s  SoC: CPU=%6s GPU=%5s all=%6s W   PSTR system=%6s W\n" "$n" "$c" "$g" "$s" "$p"
  for p in $pids; do kill $p 2>/dev/null; done
}
for n in 0 1 2 4 8 16 28 0; do step $n; done
