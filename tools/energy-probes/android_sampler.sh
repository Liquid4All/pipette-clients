#!/bin/sh
# 2 Hz battery sampler -> /data/local/tmp/energy_samples.txt ; stops when /data/local/tmp/energy_stop exists
rm -f /data/local/tmp/energy_stop
cur() { service call batteryproperties 1 i32 2 | sed -n 's/.*00000001 \([0-9a-f]*\).*/\1/p' | head -1; }
hex2s() { printf '%d' $((0x$1 > 0x7fffffff ? 0x$1 - 0x100000000 : 0x$1)); }
: > /data/local/tmp/energy_samples.txt
while [ ! -e /data/local/tmp/energy_stop ]; do
  c=$(hex2s "$(cur)"); b=$(dumpsys battery | awk '/^  voltage|^  temperature|^  level|^  AC powered/ {printf "%s=%s ", $1$2, $NF}')
  echo "$(date +%s.%N | cut -c1-14) current_uA=$c $b" >> /data/local/tmp/energy_samples.txt
  sleep 0.4
done
