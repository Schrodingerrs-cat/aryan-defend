"""Record Tello video + RGB frames + IMU/state data to disk. Never takes off.

  python3 tello_record.py                 # video.mp4 + frames.csv + imu.csv
  python3 tello_record.py --save-frames   # also dump every frame as RGB .npy
  python3 tello_record.py --seconds 60    # stop automatically

Output: data/<YYYYmmdd_HHMMSS>/
  video.mp4    camera video (960x720)
  frames.csv   frame_id, time (unix s), plus the latest IMU/state values at that frame
  imu.csv      every state packet (~10 Hz): time + all Tello state fields
  frames/      (--save-frames) frame_000123.npy, uint8 RGB, shape (H, W, 3)

Tello "IMU": attitude pitch/roll/yaw (deg), velocity vgx/vgy/vgz (dm/s),
acceleration agx/agy/agz (cm/s^2 -> 0.001g units), height h (cm), baro (m),
tof (cm; also tof_m in metres, blank if invalid), battery, temperature. There is no raw gyro stream.
Press q in the preview window to stop.
"""
import argparse
import csv
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tello_stream import TelloStream

STATE_FIELDS = ["pitch", "roll", "yaw", "vgx", "vgy", "vgz", "templ", "temph",
                "tof", "h", "bat", "baro", "time", "agx", "agy", "agz"]


# CSV column names: the Tello's own "time" field (motor-on seconds) must not collide with our unix "time".
STATE_HEADERS = ["motor_time" if k == "time" else k for k in STATE_FIELDS]


def tof_m(state):
    """Time-of-flight distance below the drone in metres, '' if missing/invalid.

    The state packet reports cm; the sensor returns ~6553 (and 0) when it has no valid reading.
    """
    v = state.get("tof")
    if v is None or v <= 0 or v >= 6000:
        return ""
    return f"{v / 100:.2f}"


class StateListener:
    """Receives the ~10 Hz state broadcast on UDP 8890."""

    def __init__(self, on_state, iface=None):
        if iface:       # pinned to one drone's dongle (several Tellos on one PC)
            from dongle import dev_socket
            self.sock = dev_socket(iface, 8890, timeout=1)
        else:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind(("", 8890))
            self.sock.settimeout(1)
        self.on_state = on_state
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def parse(text):
        # "pitch:0;roll:0;yaw:0;vgx:0;...;agz:-999.00;\r\n"
        out = {}
        for item in text.strip().split(";"):
            if ":" in item:
                k, v = item.split(":", 1)
                try:
                    out[k] = float(v)
                except ValueError:
                    pass
        return out

    def _loop(self):
        while self.running:
            try:
                data, _ = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            except OSError:
                return
            state = self.parse(data.decode("ascii", errors="ignore"))
            if state:
                self.on_state(time.time(), state)

    def stop(self):
        self.running = False
        self.sock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-frames", action="store_true", help="also save each frame as RGB .npy")
    ap.add_argument("--seconds", type=float, default=0, help="auto-stop after N seconds (0 = until q)")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "data"),
                    help="base output directory (default: data/ next to this script)")
    ap.add_argument("--fps", type=float, default=30, help="fps written into video.mp4 header")
    args = ap.parse_args()

    out_dir = Path(args.out) / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    if args.save_frames:
        (out_dir / "frames").mkdir()

    imu_f = open(out_dir / "imu.csv", "w", newline="")
    imu_w = csv.writer(imu_f)
    imu_w.writerow(["time"] + STATE_HEADERS + ["tof_m"])
    frames_f = open(out_dir / "frames.csv", "w", newline="")
    frames_w = csv.writer(frames_f)
    frames_w.writerow(["frame_id", "time", "state_time"] + STATE_HEADERS + ["tof_m"])

    latest = {"t": 0.0, "s": {}}

    def on_state(t, s):
        latest["t"], latest["s"] = t, s
        imu_w.writerow([f"{t:.4f}"] + [s.get(k, "") for k in STATE_FIELDS] + [tof_m(s)])

    listener = StateListener(on_state)
    tello = TelloStream()
    tello.start()
    print(f"Recording to {out_dir}  (q to stop)")

    writer = None
    last_id = 0
    n = 0
    start = time.time()
    try:
        while True:
            if args.seconds and time.time() - start > args.seconds:
                break
            item = tello.read_new(last_id)
            if item is None:
                if cv2.waitKey(2) & 0xFF == ord("q"):
                    break
                continue
            frame_bgr, last_id, t = item

            if writer is None:
                h, w = frame_bgr.shape[:2]
                writer = cv2.VideoWriter(str(out_dir / "video.mp4"),
                                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
            writer.write(frame_bgr)

            s = latest["s"]
            frames_w.writerow([last_id, f"{t:.4f}", f"{latest['t']:.4f}"]
                              + [s.get(k, "") for k in STATE_FIELDS] + [tof_m(s)])
            if args.save_frames:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                np.save(out_dir / "frames" / f"frame_{last_id:06d}.npy", rgb)
            n += 1

            cv2.putText(frame_bgr, f"REC {n}  bat {s.get('bat', '?')}%  tof {tof_m(s) or '--'} m", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Tello record", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        listener.stop()
        tello.stop()
        if writer is not None:
            writer.release()
        imu_f.close()
        frames_f.close()
        cv2.destroyAllWindows()
        dur = time.time() - start
        print(f"Saved {n} frames in {dur:.1f}s ({n / max(dur, 1e-6):.1f} fps) to {out_dir}")


if __name__ == "__main__":
    main()
