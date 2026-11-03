"""LIVE AprilTag localization: connect to the Tello, see tags, show where the drone is. Never takes off.

  python3 tello_localize_live.py                       # live drone (join the TELLO-XXXXXX Wi-Fi first)
  python3 tello_localize_live.py --record              # also record video/IMU/etc. to data/<ts>/ (reprocess later
                                                       #   with localization_pipeline.py)
  python3 tello_localize_live.py --map maps/my_map.yaml --drone tello_4
  python3 tello_localize_live.py --replay data/<session>/video.mp4   # test with no drone: replays a recording

Left: the camera with detected tags outlined (green = in the map, grey = seen but not mapped).
Right: top-down map (wall at top): tags, the drone, its heading + field of view, trail, and a line to the
tag it is using. Press q to quit, c to clear the trail.

Same pipeline as localization_pipeline.py (repo detector + map + Kalman filter), so what you see live is what
the offline run will give. The Tello only receives command/streamon/battery?/keepalive here, no flight commands.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from height_fusion import HeightCfg, HeightFilter, tof_to_height  # noqa: E402
from localization_pipeline import (DEFAULT_REPO, TopDown, draw_panel, heading_deg, load_repo,  # noqa: E402
                                   tag_shapes, view_bounds)

W_CAM, H, W_MAP = 720, 540, 560


class ReplaySource:
    """Plays a recorded video at real-time speed, like a camera."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.t0, self.idx, self.frame, self.done = time.time(), 0, None, False

    def read_new(self, last_id):
        target = int((time.time() - self.t0) * self.fps) + 1
        if target == last_id:
            return None
        while self.idx < target:
            ok, f = self.cap.read()
            if not ok:
                self.done = True
                return None
            self.idx, self.frame = self.idx + 1, f
        return self.frame.copy(), self.idx, time.time()

    def stop(self):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map", type=Path, default=HERE / "maps" / "latest_layout.yaml")
    ap.add_argument("--drone", default="tello_3", help="camera calibration (config/camera/<drone>.yaml)")
    ap.add_argument("--tags", type=Path, default=None)
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--floor-y", type=float, default=-1.22)
    ap.add_argument("--no-height-fusion", action="store_true", help="raw tag height, no ToF / smoothing")
    ap.add_argument("--record", action="store_true", help="also record the session (FlightRecorder, no pose)")
    ap.add_argument("--tello", default=None, metavar="TELLO-N", help="use this drone's USB dongle (see dongle.py)")
    ap.add_argument("--replay", type=Path, default=None, help="video file to use instead of the drone")
    ap.add_argument("--headless", action="store_true", help="no window (with --seconds / --save for testing)")
    ap.add_argument("--seconds", type=float, default=0, help="auto-stop after N s")
    ap.add_argument("--save", type=Path, default=None, help="write the composite view to this mp4")
    args = ap.parse_args()

    R = load_repo(args.repo)
    cfg = R["load_tag_config"](args.tags) if args.tags else R["load_tag_config"]()
    cam = R["CONFIG_DIR"] / "camera" / f"{args.drone}.yaml"
    K, _ = R["camera_from_dict"](R["load_yaml"](cam if cam.exists() else R["CONFIG_DIR"] / "camera" / "default.yaml"))
    hfov = float(np.degrees(2 * np.arctan(480.0 / K[0, 0])))
    from positional_kalman_filter import FilteredPositioner
    fp = FilteredPositioner(args.drone, map_path=args.map)
    tag_map = fp.positioner.tag_map
    mapped = set(tag_map.poses)
    shapes = tag_shapes(tag_map, cfg, args.floor_y)
    shape_pos = {s["id"]: s["pos"] for s in shapes}
    xr, zr = view_bounds(shapes, [[[0, 0, 4.0], [1.5, 0, 3.0], [-1.0, 0, 3.0]]])   # room-sized default view
    td = TopDown(xr, zr, W_MAP, H)
    bg = td.background(shapes, xr, zr, args.floor_y)

    det = fp.positioner.detector
    orig, hold = det.detect, {}

    def cached(gray):
        hold["obs"] = orig(gray)
        return hold["obs"]
    det.detect = cached

    rec = state_src = None
    if args.replay:
        src = ReplaySource(args.replay)
        print(f"replaying {args.replay} (no drone)")
    else:
        from tello_stream import TelloStream
        src = TelloStream(drone=args.tello)
        src.start()
        if args.record:
            from flight_recorder import FlightRecorder
            rec = FlightRecorder(src, HERE / "data", frames="jpg", pose=False,
                                 meta=dict(script="tello_localize_live.py", map=str(args.map)))
            rec.start()
        else:
            from tello_record import StateListener
            lastst = {}
            state_src = StateListener(lambda t, s: lastst.update(s), iface=src.iface)
    live_csv = None
    if rec:
        f = open(rec.out_dir / "localization_live.csv", "w", newline="")
        live_csv = (f, csv.writer(f))
        live_csv[1].writerow(["time", "frame_id", "tags_seen", "tags_used", "x", "y", "z", "heading_deg",
                              "accepted", "lost", "fix_age", "latency_ms"])

    hcfg = HeightCfg()
    hf = None if args.no_height_fusion else HeightFilter(hcfg)
    writer = None
    last_id, last_R, last_pos, trail = 0, None, None, []
    n, t_start, t_fps, fps = 0, time.time(), time.time(), 0.0
    print("running. q = quit, c = clear trail")
    try:
        while True:
            if args.seconds and time.time() - t_start > args.seconds:
                break
            item = src.read_new(last_id)
            if item is None:
                if getattr(src, "done", False):
                    break
                if not args.headless and (cv2.waitKey(2) & 0xFF) == ord("q"):
                    break
                continue
            frame, last_id, t_arr = item
            t0 = time.time()
            res = fp.locate(frame, time.monotonic())
            raw = res["raw"]
            if raw:
                last_R = np.asarray(raw["R"], float)
            st = rec.state() if rec else (dict(lastst) if state_src else {})
            xyz, hmode = res["xyz"], ""
            if hf is not None:      # stable height: ToF (when valid in flight) + outlier-gated tag height
                z_f, hmode = hf.step(time.monotonic(), raw["xyz"][1] - args.floor_y if raw else None,
                                     raw["range"] if raw else None, tof_to_height(st, hcfg))
                if xyz is not None and z_f is not None:
                    xyz = [xyz[0], z_f + args.floor_y, xyz[2]]
            q = dict(xyz=xyz, lost=res["lost"], used=list(raw["tags"]) if raw else [],
                     heading=heading_deg(last_R) if last_R is not None else None)
            for o in hold.get("obs", []):
                if o.accepted:
                    col = (0, 220, 0) if int(o.tag_id) in mapped else (170, 170, 170)
                    cv2.polylines(frame, [o.corners.round().astype(np.int32)], True, col, 3)
                    cv2.putText(frame, str(int(o.tag_id)), tuple(o.corners[0].round().astype(int) + [0, -8]),
                                0, 0.9, col, 2)
            if xyz:
                last_pos = xyz
                trail.append(xyz)
            m = draw_panel(td, bg, shape_pos, q, trail, last_pos, args.floor_y, hfov, H)
            n += 1
            if time.time() - t_fps > 1.0:
                fps, n, t_fps = n / (time.time() - t_fps), 0, time.time()
            latency = (time.time() - t_arr) * 1000
            info = f"{fps:4.1f} fps  proc {1000 * (time.time() - t0):3.0f} ms"
            if st:
                info += f"  bat {st.get('bat', '?'):.0f}%  tof {st.get('tof', 0) / 100:.2f} m"
            cv2.putText(m, info, (W_MAP - 330, H - 8), 0, 0.45, (200, 200, 200), 1)
            view = np.hstack([cv2.resize(frame, (W_CAM, H)), m])
            if live_csv:
                live_csv[1].writerow([f"{time.time():.4f}", last_id, " ".join(str(int(o.tag_id)) for o in hold.get("obs", []) if o.accepted),
                                      " ".join(map(str, q["used"])), *(xyz or ["", "", ""]),
                                      "" if q["heading"] is None else round(q["heading"], 1), int(res["accepted"]),
                                      int(res["lost"]), f"{res['fix_age']:.2f}" if res["fix_age"] != float("inf") else "",
                                      round(latency, 1)])
            if args.save:
                if writer is None:
                    writer = cv2.VideoWriter(str(args.save), cv2.VideoWriter_fourcc(*"mp4v"), 15, view.shape[1::-1])
                writer.write(view)
            if not args.headless:
                cv2.imshow("Tello localization (q quit, c clear trail)", view)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("c"):
                    trail.clear()
    finally:
        if writer:
            writer.release()
        if live_csv:
            live_csv[0].close()
        if rec:
            rec.stop()
        if state_src:
            state_src.stop()
        src.stop()
        cv2.destroyAllWindows()
        print(f"stopped; avg {fps:.1f} fps in the last second of running")


if __name__ == "__main__":
    main()
