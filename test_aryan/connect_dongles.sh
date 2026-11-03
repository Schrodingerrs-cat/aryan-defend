#!/usr/bin/env bash
# Join each named drone's Wi-Fi on ITS OWN USB dongle (from the repo's dongles.json), and pin the NetworkManager
# profile to that dongle so it can never join through the wrong one. Your laptop's own Wi-Fi is not touched.
#
#   ./connect_dongles.sh --list                # which dongle flies which drone, and whether it is plugged in
#   ./connect_dongles.sh TELLO-1 TELLO-3       # join those two
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
REG=../drone_deployment_infra/config/dongles.json
lookup() { python3 - "$REG" "$1" <<'PY'
import json, sys
reg, name = sys.argv[1], sys.argv[2].upper()
for d in json.load(open(reg))["dongles"]:
    if (d.get("drone") or "").upper() == name:
        print(d["iface"], d["ssid"]); break
else:
    sys.exit(f"{name} is not in {reg}")
PY
}
if [[ "${1:-}" == "--list" || $# -eq 0 ]]; then
  python3 - "$REG" <<'PY'
import json, os, sys
for d in json.load(open(sys.argv[1]))["dongles"]:
    here = os.path.isdir(f"/sys/class/net/{d['iface']}")
    print(f"dongle {d['number']}: {d['iface']}  ->  {d['drone']}  ({'plugged in' if here else 'not plugged in'})")
PY
  exit 0
fi
for name in "$@"; do
  read -r iface ssid < <(lookup "$name")
  [[ -d /sys/class/net/$iface ]] || { echo "$name: dongle $iface is not plugged in" >&2; exit 1; }
  nmcli -t -f NAME connection show | grep -qx "$ssid" || nmcli connection add type wifi ifname "$iface" con-name "$ssid" ssid "$ssid" >/dev/null
  nmcli connection modify "$ssid" connection.interface-name "$iface" ipv4.never-default yes
  nmcli connection up "$ssid" ifname "$iface"
  echo "$name joined on $iface ($(ip -4 -o addr show dev "$iface" | awk '{print $4}'))"
done
