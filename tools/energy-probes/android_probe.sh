#!/bin/sh
# read-only Android power-source survey, run as adb shell
echo "--- dumpsys battery"; dumpsys battery | grep -E "AC powered|USB powered|Wireless|level|voltage|temperature|status|Charge counter|current"
echo "--- power_supply sysfs"
for d in /sys/class/power_supply/*; do
  for f in current_now voltage_now charge_counter power_now energy_now current_avg; do
    [ -e $d/$f ] && echo "$d/$f = $(cat $d/$f 2>&1 | head -1)"
  done
done
echo "--- powercap"; ls /sys/class/powercap 2>&1 | head
echo "--- IPowerStats (ODPM)"; dumpsys android.hardware.power.stats.IPowerStats/default 2>&1 | head -40
echo "--- powerstats service"; dumpsys powerstats 2>&1 | grep -iE "channel|rail|energy|meter" | head -30
echo "--- battery props via cmd"; cmd battery 2>&1 | head -3
