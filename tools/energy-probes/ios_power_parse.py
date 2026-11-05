"""One CSV row from a pymobiledevice3 AppleSmartBattery ioregistry dump.

Reads the JSON on stdin; the first argument is the host timestamp. Power fields
are in mW, the accumulators in mW*s, current in mA, voltage in mV, temperature
in centi-degrees C. Those units are inferred from the accumulator delta
matching the instantaneous reading, not documented by Apple.
"""

import json
import sys

t = sys.argv[1]
s = sys.stdin.read()
s = s[s.find("{") : s.rfind("}") + 1]
try:
    d = json.loads(s)
except ValueError:
    sys.exit(0)
d = d[0] if isinstance(d, list) else d
p = d["PowerTelemetryData"]
a = d.get("AdapterDetails", {})
row = [
    t,
    d["UpdateTime"],
    p["SystemPowerInAccumulatorCount"],
    p["AccumulatedSystemPowerIn"],
    p["SystemPowerIn"],
    p["SystemLoad"],
    p["BatteryPower"],
    p["AccumulatedBatteryPower"],
    p["AccumulatedBatteryDischarge"],
    d["InstantAmperage"],
    d["Voltage"],
    d["Temperature"],
    int(d["IsCharging"]),
    d["CurrentCapacity"],
    a.get("Watts"),
    a.get("AdapterVoltage"),
]
print(",".join(str(x) for x in row))
