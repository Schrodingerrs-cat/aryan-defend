"""publish_pose_hook.py — publish this drone's live pose as drone_telemetry v1 UDP JSON.

Sits between LiveMapper.snapshot() and the network; called by tello_keys.py when
--publish-telemetry HOST:PORT is given.  No drone_gui needed for the 1v1 test.

Wire format matches drone_telemetry README exactly so that attack_feed.py and any
other consumer can read it without modification.

Frame conversion:
  LiveMapper outputs xyz in **map frame** (same as config/map.yaml):
    x right, y up, z out of the tag-10 wall (into the room)
  drone_telemetry wire format wants tag-frame xyz for the named tag:
    x right, y down, z into the wall  (pupil-apriltags convention)
  For tag 10 (origin, R=I, t=0 in the map):
    wire_xyz = [x_map, -y_map, -z_map]
  Yaw: LiveMapper heading_deg (0=facing wall, +=turned right) == drone_telemetry yaw
    for tag 10 (same origin, same sign convention).

Usage (called from tello_keys.py --publish-telemetry):
  hook = PosePublisher("127.0.0.1:9100", drone_name="TELLO-2", team="attack",
                       get_snapshot=lm.snapshot, get_state=rec.state)
  hook.start()
  # ... fly loop ...
  hook.stop()

Standalone smoke-test (prints datagrams to stdout):
  python3 publish_pose_hook.py --drone TELLO-2 --team attack --dry-run
"""
from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path
from typing import Callable

# Fake pose age when live_mapper has no fix
_NO_FIX_AGE = 9999.0
# Publish rate (Hz) — 10 Hz matches drone_gui default
_HZ = 10.0
# Sequence counter (module-level, shared across instances if multiple hooks run)
_seq = 0


def _parse_dest(dest: str) -> tuple[str, int]:
    host, _, port = dest.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"bad --publish-telemetry dest {dest!r}: want host:port")
    return host, int(port)


class PosePublisher:
    """Background thread that publishes one drone's pose to a UDP destination at ~10 Hz.

    Args:
        dest:          "host:port" string, e.g. "127.0.0.1:9100"
        drone_name:    registry name, e.g. "TELLO-2"
        team:          "attack" or "defense"
        get_snapshot:  callable -> LiveMapper.snapshot() dict
                       (keys: xyz, lost, used, heading, fix_age, ...)
        get_state:     callable -> tello state dict (keys: bat, h, tof, vgx, vgy, vgz)
        hz:            publish rate
    """

    def __init__(
        self,
        dest: str,
        drone_name: str,
        team: str,
        get_snapshot: Callable[[], dict],
        get_state: Callable[[], dict],
        hz: float = _HZ,
    ):
        self.dest = _parse_dest(dest)
        self.drone_name = drone_name
        self.team = team
        self.get_snapshot = get_snapshot
        self.get_state = get_state
        self.hz = hz
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name=f"pose-pub-{drone_name}")

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sock.close()

    # ------------------------------------------------------------------
    def _build_datagram(self) -> bytes:
        global _seq
        snap = self.get_snapshot() if self.get_snapshot else {}
        state = self.get_state() if self.get_state else {}

        xyz_map = snap.get("xyz")          # [x, y, z] in map frame (y up, z out)
        heading = snap.get("heading")      # degrees: 0=facing tag-10 wall, +=right
        lost = snap.get("lost", True)
        used_tags = snap.get("used", [])   # list of tag IDs that contributed to the fix
        fix_age = snap.get("fix_age", _NO_FIX_AGE)
        if fix_age is None or fix_age == float("inf"):
            fix_age = _NO_FIX_AGE

        # Convert map-frame xyz → tag-10 wire-format xyz (y down, z into wall)
        if xyz_map is not None and not lost:
            x_m, y_m, z_m = xyz_map
            wire_xyz = [round(x_m, 4), round(-y_m, 4), round(-z_m, 4)]
            wire_yaw = round(heading, 2) if heading is not None else 0.0
            # use the first tag actually seen; default to origin tag 10
            wire_tag = int(used_tags[0]) if used_tags else 10
            pose_dict = {
                "tag": wire_tag,
                "xyz": wire_xyz,
                "yaw": wire_yaw,
                "age": round(fix_age, 3),
            }
        else:
            pose_dict = None

        # Height (m above takeoff)
        h_raw = state.get("h")
        h_m = float(h_raw) / 100.0 if h_raw is not None else None   # Tello h is in cm
        tof_raw = state.get("tof")
        tof_m = float(tof_raw) / 100.0 if tof_raw is not None else None

        # Velocity (Tello vg axes, cm/s → m/s)
        vel = [
            round(float(state.get("vgx", 0)) / 100.0, 3),
            round(float(state.get("vgy", 0)) / 100.0, 3),
            round(float(state.get("vgz", 0)) / 100.0, 3),
        ]

        bat = state.get("bat")

        drone_entry = {
            "name": self.drone_name,
            "slot": 1,
            "team": self.team,
            "link": True,
            "fly": not lost,
            "mode": "manual" if not lost else "ground",
            "bat": int(bat) if bat is not None else None,
            "h": round(h_m, 3) if h_m is not None else 0.0,
            "tof": round(tof_m, 3) if tof_m is not None else None,
            "vel": vel,
            "pose": pose_dict,
        }

        _seq += 1
        msg = {
            "v": 1,
            "src": "real",
            "seq": _seq,
            "t": round(time.time(), 3),
            "hz": self.hz,
            "drones": [drone_entry],
        }
        return json.dumps(msg, separators=(",", ":")).encode()

    def _loop(self):
        period = 1.0 / max(0.5, self.hz)
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                payload = self._build_datagram()
                self._sock.sendto(payload, self.dest)
            except Exception as exc:
                print(f"[pose-pub {self.drone_name}] send error: {exc}", flush=True)
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.0, period - elapsed))


# -----------------------------------------------------------------------
# Smoke-test / standalone dry-run
# -----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pose publisher smoke-test")
    ap.add_argument("--dest", default="127.0.0.1:9100")
    ap.add_argument("--drone", default="TELLO-2")
    ap.add_argument("--team", default="attack", choices=["attack", "defense"])
    ap.add_argument("--dry-run", action="store_true", help="print JSON instead of sending")
    ap.add_argument("--ticks", type=int, default=5)
    args = ap.parse_args()

    import math

    tick = [0]

    def fake_snapshot():
        t = tick[0] * 0.1
        return {
            "xyz": [1.5 + 0.3 * math.sin(t), 1.2, 2.0 + 0.5 * math.cos(t)],
            "lost": False,
            "used": [10],
            "heading": 15.0 * math.sin(t),
            "fix_age": 0.05,
        }

    def fake_state():
        return {"bat": 85, "h": 120, "tof": 118, "vgx": 10, "vgy": 0, "vgz": 0}

    if args.dry_run:
        pub = PosePublisher(args.dest, args.drone, args.team, fake_snapshot, fake_state)
        for i in range(args.ticks):
            tick[0] = i
            data = pub._build_datagram()
            msg = json.loads(data)
            print(json.dumps(msg, indent=2))
            time.sleep(0.1)
    else:
        pub = PosePublisher(args.dest, args.drone, args.team, fake_snapshot, fake_state)
        pub.start()
        print(f"Publishing {args.team} pose for {args.drone} -> {args.dest}  (Ctrl-C to stop)")
        try:
            for i in range(args.ticks * 10):
                tick[0] = i
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            pub.stop()
        print("done.")
