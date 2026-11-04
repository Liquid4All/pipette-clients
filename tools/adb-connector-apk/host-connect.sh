#!/usr/bin/env bash
#
# Finds and connects wireless-debugging devices whose port you do not know: adbd picks a fresh
# random port each time it starts listening.
#
# Three phases per device, in order: the fixed port 5555, the mDNS candidates, then an nmap sweep
# of the ephemeral range. The phase that connected is reported on stderr. mDNS is only a fast path:
# the list holds stale records from earlier boots and misses a live device about a third of the
# time (measured in "Finding the port from the host" in README.md), so the sweep stays required.
#
# Usage: ./host-connect.sh 192.0.2.10 [192.0.2.11 ...]
set -uo pipefail

# Linux's ephemeral range, where every adbd wireless port observed on this fleet has landed.
# A full 65k sweep over wifi costs minutes per device for no extra coverage.
readonly EPHEMERAL_RANGE="32768-60999"
# `adb tcpip 5555` pins this one, so when it is up it saves a scan.
readonly FIXED_PORT=5555

# The adb server's mDNS cache misses a live device in gaps of about 1s to 8s. Five asks 0.5s
# apart cover the short gaps and give up on the long ones, because a wait long enough for the
# longest gap costs more than the sweep it avoids.
readonly MDNS_ATTEMPTS=5
readonly MDNS_RETRY_DELAY_SECS=0.5

# Preflighted because both missing tools read as a device fault: without `timeout` the 5555 fast
# path never fires, and without `nmap` a healthy device the mDNS phase misses is reported as
# needing a USB trip. Neither ships on a stock macOS.
missing=()
for tool in adb nmap timeout; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "missing required tool(s): ${missing[*]}" >&2
  echo "on macOS: brew install ${missing[*]/timeout/coreutils}" >&2
  exit 1
fi

# Ports this host has heard announced for a device, one per line. An empty answer is
# inconclusive, not proof the device is silent.
#
# Only `_adb-tls-connect` names a port `adb connect` can use (the pairing service wants a code).
# An IPv6 `IP:PORT` splits into more than two fields and is dropped: every phase uses IPv4.
mdns_ports() {
  local host=$1 ports attempt
  for ((attempt = 1; attempt <= MDNS_ATTEMPTS; attempt++)); do
    ports=$(adb mdns services 2>/dev/null |
      awk -v host="$host" '/_adb-tls-connect\._tcp/ {
        if (split($NF, a, ":") == 2 && a[1] == host && !seen[a[2]]++) print a[2]
      }')
    if [[ -n $ports ]]; then
      printf '%s\n' "$ports"
      return 0
    fi
    [[ $attempt -lt $MDNS_ATTEMPTS ]] && sleep "$MDNS_RETRY_DELAY_SECS"
  done
  return 1
}

port_open() {
  timeout 4 bash -c "echo > /dev/tcp/$1/$2" 2>/dev/null
}

connect_to() {
  local host=$1 port=$2 out state
  out=$(adb connect "$host:$port" 2>&1)
  # "already connected to H:P" is a success too, and the common case when re-running.
  if ! grep -qE "^(already )?connected to" <<<"$out"; then
    echo "  connect failed on $host:$port ($out)" >&2
    return 1
  fi
  # `adb connect` succeeds as soon as TCP is up, before any adb handshake, so any open port passes
  # and the device would sit `offline`. get-state is the cheapest proof that adbd answered.
  state=$(adb -s "$host:$port" get-state 2>/dev/null | tr -d '\r\n')
  if [[ $state != "device" ]]; then
    echo "  $host:$port is open but not adbd (state=${state:-none}); ignoring" >&2
    adb disconnect "$host:$port" >/dev/null 2>&1
    return 1
  fi
  echo "  connected $host:$port"
  return 0
}

for host in "$@"; do
  echo "$host"

  if ! ping -c 2 -W 2 "$host" >/dev/null 2>&1; then
    # A different fault from a dead listener, with a different fix (wifi/DHCP, not re-arming adb).
    echo "  unreachable: no ping response. This is a network fault, not an adb one" >&2
    continue
  fi

  if port_open "$host" "$FIXED_PORT" && connect_to "$host" "$FIXED_PORT"; then
    echo "  phase: fixed port" >&2
    continue
  fi

  # Every candidate is tried: a refused `adb connect` costs milliseconds against the ~6s sweep.
  mdns_hit=""
  for port in $(mdns_ports "$host"); do
    connect_to "$host" "$port" && mdns_hit=1 && break
  done
  if [[ -n $mdns_hit ]]; then
    echo "  phase: mDNS" >&2
    continue
  fi

  echo "  scanning $EPHEMERAL_RANGE for an adbd listener..."
  # -Pn: ping already succeeded, so skip host discovery.
  found=$(nmap -Pn --open -T4 -p "$EPHEMERAL_RANGE" "$host" 2>/dev/null |
    awk -F/ '/^[0-9]+\/tcp/ {print $1}')

  if [[ -z $found ]]; then
    # Wireless debugging is off. If the app is installed, suspect a force-stop (FLAG_STOPPED blocks
    # the boot receiver) or a reinstall that dropped the WRITE_SECURE_SETTINGS grant.
    echo "  no adb listener found. Device needs re-arming (see README)" >&2
    continue
  fi

  for port in $found; do
    if connect_to "$host" "$port"; then
      echo "  phase: nmap sweep" >&2
      break
    fi
  done
done

echo
adb devices
