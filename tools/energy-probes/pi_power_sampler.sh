#!/bin/bash
# Samples the Raspberry Pi 5's PMIC rail ADCs at about 4 Hz, on the Pi itself,
# with no root. `vcgencmd pmic_read_adc` returns one current and one voltage
# per regulator rail as an instantaneous snapshot (no accumulator, no
# averaging, ~25 ms per call through the firmware mailbox), so single rows are
# noisy and a window is the mean of its rows. `rails_w` is the sum of V*I over
# every rail; it covers the SoC, DRAM, I/O and the WiFi radio, and omits the
# regulator losses, the fan, the USB ports and anything on a HAT. The 5 V
# input has a voltage channel only.
#
# usage: pi_power_sampler.sh <out.csv>
# stop:  touch <out.csv>.stop
OUT=$1; STOP=$OUT.stop
[ -n "$OUT" ] || { echo "usage: $0 <out.csv>" >&2; exit 2; }
rm -f "$STOP"
echo "t,rails_w,core_w,ddr_w,wifi_w,ext5v_v,throttled" > "$OUT"
while [ ! -e "$STOP" ]; do
  t=$(date +%s.%N | cut -c1-14)
  th=$(vcgencmd get_throttled | cut -d= -f2)
  vcgencmd pmic_read_adc | awk -v t="$t" -v th="$th" '
    /current/ { split($2, a, "="); i[$1] = a[2] + 0 }
    /volt/    { split($2, a, "="); v[$1] = a[2] + 0 }
    END {
      p = 0
      for (k in i) { vk = k; sub(/_A$/, "_V", vk); w[k] = i[k] * v[vk]; p += w[k] }
      printf "%s,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n", t, p, w["VDD_CORE_A"],
        w["DDR_VDD2_A"] + w["DDR_VDDQ_A"], w["3V7_WL_SW_A"], v["EXT5V_V"], th
    }' >> "$OUT"
  sleep 0.2
done
