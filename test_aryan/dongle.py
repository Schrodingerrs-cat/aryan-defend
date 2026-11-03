"""Which USB Wi-Fi dongle flies which Tello, and sockets pinned to that dongle.

Every Tello is 192.168.10.1, so with two drones on one PC each drone's sockets must be pinned to its own dongle's
interface (SO_BINDTODEVICE, Linux). The dongle -> drone map is the team's registry
drone_deployment_infra/config/dongles.json (written by core/tello_dongle_setup.py).

    entry = resolve("TELLO-1")          # {'iface': 'wlx...', 'ssid': 'TELLO-1', 'video_port': 17001, ...}
    problems = check(entry)             # [] if the dongle is present and joined to that drone's Wi-Fi
    sock = dev_socket(entry["iface"], 9000)
"""
import fcntl
import json
import os
import socket
import struct
import subprocess
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent.parent / "drone_deployment_infra" / "config" / "dongles.json"


def registry():
    return json.loads(REGISTRY.read_text())["dongles"]


def resolve(name):
    """Registry entry for a drone name like 'TELLO-1' (also accepts 'tello_1' or '1')."""
    key = str(name).upper().replace("_", "-")
    if key.isdigit():
        key = f"TELLO-{key}"
    for d in registry():
        if (d.get("drone") or "").upper() == key:
            if not d.get("iface"):
                raise SystemExit(f"{key} has no dongle interface recorded in {REGISTRY}")
            return d
    raise SystemExit(f"{key} is not in {REGISTRY}; known: {[d.get('drone') for d in registry()]}. "
                     "Map it with core/tello_dongle_setup.py")


def iface_ip(iface):
    """IPv4 address of an interface, or None."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack("256s", iface[:15].encode()))[20:24])
    except OSError:
        return None
    finally:
        s.close()


def _active_wifi(iface):
    """(ssid, bssid) the dongle is associated with, via nmcli; (None, None) if unknown."""
    try:
        out = subprocess.run(["nmcli", "-t", "-f", "ACTIVE,SSID,BSSID", "dev", "wifi", "list", "ifname", iface],
                             text=True, capture_output=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    for line in out.splitlines():
        if line.startswith("yes:"):
            rest = line[4:].replace("\\:", "\x00")
            ssid, _, bssid = rest.partition(":")
            return ssid.replace("\x00", ":"), bssid.replace("\x00", ":").upper()
    return None, None


def check(entry):
    """Problems that mean this dongle is not talking to the drone the registry says (empty list = good)."""
    iface, bad = entry["iface"], []
    if not os.path.isdir(f"/sys/class/net/{iface}"):
        return [f"dongle {iface} is not plugged in"]
    ip = iface_ip(iface)
    if not ip or not ip.startswith("192.168.10."):
        bad.append(f"{iface} has no Tello address (got {ip}); join the {entry.get('ssid')} Wi-Fi on it")
    ssid, bssid = _active_wifi(iface)
    if ssid is not None and ssid != entry.get("ssid"):
        bad.append(f"{iface} is joined to '{ssid}' but the registry says it flies '{entry.get('ssid')}'")
    elif bssid and entry.get("bssid") and bssid != entry["bssid"].upper():
        bad.append(f"{iface} is on access point {bssid}, registry expects {entry['bssid']} ({entry.get('ssid')})")
    return bad


def dev_socket(iface, port, timeout=None):
    """UDP socket bound to one interface, so two drones with the same IP do not share it."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BINDTODEVICE, iface.encode())
    s.bind(("0.0.0.0", port))
    if timeout is not None:
        s.settimeout(timeout)
    return s
