#!/bin/bash
# Samples an iPhone's AppleSmartBattery entry from the host Mac over the
# CoreDevice network tunnel, with no root and no USB. pymobiledevice3's native
# tunnel mode rides the pairing remotepairingd already holds, and the
# diagnostics relay answers through it. One row per poll; the entry itself
# publishes every 20 s, so a poll every 1.5 s misses nothing.
#
# usage: ios_power_sampler.sh <udid> <out.csv>      (udid from `pymobiledevice3 remote browse`)
# stop:  touch <out.csv>.stop
# Needs uvx on the host; nothing is installed.
U=$1; OUT=$2; STOP=$OUT.stop
[ -n "$U" ] && [ -n "$OUT" ] || { echo "usage: $0 <udid> <out.csv>" >&2; exit 2; }
rm -f "$STOP"
PARSE=$(dirname "$0")/ios_power_parse.py
T=$(mktemp)
uvx --quiet pymobiledevice3 remote start-tunnel --udid "$U" --script-mode > "$T" 2>&1 &
TP=$!
for _ in $(seq 1 60); do grep -qE "^[0-9a-f:.]+ [0-9]+$" "$T" 2>/dev/null && break; sleep 0.5; done
read -r HOST PORT < <(grep -E "^[0-9a-f:.]+ [0-9]+$" "$T" | head -1)
[ -n "$PORT" ] || { echo "no tunnel" >&2; kill $TP 2>/dev/null; exit 1; }
echo "t_host,UpdateTime,cnt,acc_PIn_mWs,PIn_mW,SysLoad_mW,BattPow_mW,acc_Batt,acc_BattDis,InstAmp_mA,Volt_mV,Temp,IsCharging,Level,AdapterW,AdapterV" > "$OUT"
while [ ! -e "$STOP" ]; do
  t=$(date +%s.%N | cut -c1-14)
  uvx --quiet pymobiledevice3 diagnostics ioregistry --ioclass AppleSmartBattery --rsd "$HOST" "$PORT" 2>/dev/null \
    | python3 "$PARSE" "$t" >> "$OUT"
  sleep 1.5
done
kill $TP 2>/dev/null; wait $TP 2>/dev/null; rm -f "$T"
