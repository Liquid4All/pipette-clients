#!/bin/sh
ls -l /sys/class/power_supply/ 2>&1 | head -5
cat /sys/class/power_supply/battery/current_now 2>&1
# IBatteryPropertiesRegistrar.getProperty(id): 1=CHARGE_COUNTER 2=CURRENT_NOW 3=CURRENT_AVERAGE 4=CAPACITY 5=ENERGY_COUNTER
for id in 1 2 3 5; do echo "prop $id: $(service call batteryproperties 1 i32 $id 2>&1 | tr -d '\n' | head -c 300)"; done
echo "--- CURRENT_NOW 10 samples @0.2s"
i=0; while [ $i -lt 10 ]; do service call batteryproperties 1 i32 2 | tr -d '\n'; echo; sleep 0.2; i=$((i+1)); done
