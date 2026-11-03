"""Live tag mapping + localization + stable height, as a background worker for tello_keys.py.

While you fly, a worker thread takes the newest camera frame, detects AprilTags (repo detector), feeds every frame with
two or more tags into the repo's MapBuilder (tag 10 = origin, others chained outward), re-solves the map about once a
second, localizes the drone in the growing map (repo FilteredPositioner) and fuses height (height_fusion.HeightFilter:
ToF when valid, outlier-gated tag height otherwise). The main loop only draws:

    lm = LiveMapper(tello, state_fn, out_dir); lm.start()
    frame = lm.annotate(frame)            # tag outlines + map status on the camera view
    cv2.imshow("map", lm.panel())         # top-down map: tags found so far, drone, heading, trail
    lm.save()  /  lm.stop()               # stop() saves the final map

Outputs in out_dir: live_map.yaml (same layout as the repo's config/map.yaml; autosaved every 10 s),
localization_live.csv (per processed frame), and a copy maps/live_<name>.yaml on stop.
Never sends a drone command. Detection runs on its own thread and skips frames if it falls behind; the recorder
still saves every frame.
"""
import csv
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from height_fusion import HeightCfg, HeightFilter, tof_to_height
from localization_pipeline import (DEFAULT_REPO, TopDown, draw_panel, heading_deg, load_repo, tag_shapes,
                                   view_bounds)
from map_format import to_repo_format

HERE = Path(__file__).resolve().parent
W_MAP, H_MAP = 560, 540


class LiveMapper:
    def __init__(self, stream, state_fn, out_dir, drone="tello_3", tags=None, repo=DEFAULT_REPO, floor_y=-1.22,
                 fusion=True, clock=time.monotonic):
        self.stream, self.state_fn, self.floor_y, self.clock = stream, state_fn, floor_y, clock
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        R = load_repo(repo)
        self.R = R
        self.cfg = R["load_tag_config"](tags) if tags else R["load_tag_config"]()
        cam = R["CONFIG_DIR"] / "camera" / f"{drone}.yaml"
        K, _ = R["camera_from_dict"](R["load_yaml"](cam if cam.exists() else R["CONFIG_DIR"] / "camera" / "default.yaml"))
        self.hfov = float(np.degrees(2 * np.arctan(480.0 / K[0, 0])))
        self.designated = {t for t, s in self.cfg.sizes_mm.items() if s}
        self.builder = R["MapBuilder"](self.cfg.origin_tag)
        self.map_path = self.out_dir / "live_map.yaml"
        origin = self.cfg.origin_tag
        R["TagMap"](origin, {origin: np.eye(4)}, {}, {}).save(self.map_path)     # bootstrap: only tag 10
        from positional_kalman_filter import FilteredPositioner
        self.fp = FilteredPositioner(drone, map_path=self.map_path)
        self.tag_map = self.fp.positioner.tag_map
        self.hcfg = HeightCfg()
        self.hf = HeightFilter(self.hcfg) if fusion else None
        det = self.fp.positioner.detector
        orig, self._hold = det.detect, {}

        def cached(gray):                      # one detection per frame, shared by the map builder and the positioner
            self._hold["obs"] = orig(gray)
            return self._hold["obs"]
        det.detect = cached

        self._lock = threading.Lock()
        self._snap = dict(obs=[], q=None, trail=[], last_pos=None, n=0, fps=0.0, hmode="", map_version=0)
        self._running = False
        self._thread = None
        self._csv = None
        self._td = self._bg = self._shape_pos = None
        self._panel_key = None
        self._shown_version = -1
        self._panel_t = 0.0
        self.map_version = 0
        self.last_save = 0.0
        self._clear = False

    # ------------------------------------------------------------------ worker
    def start(self):
        f = open(self.out_dir / "localization_live.csv", "w", newline="")
        self._csv = (f, csv.writer(f))
        self._csv[1].writerow(["time", "frame_id", "tags_seen", "tags_used", "x", "y", "z", "height_above_floor",
                               "heading_deg", "accepted", "lost", "fix_age", "height_mode", "tof_trust", "mapped_tags"])
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        last_id, last_R, trail, last_pos = 0, None, [], None
        n, t_fps, fps, t_rebuild = 0, time.time(), 0.0, time.time()
        while self._running:
            item = self.stream.read_new(last_id)
            if item is None:
                time.sleep(0.005)
                continue
            frame, last_id, _ = item
            if self._clear:
                trail, last_pos, self._clear = [], None, False
            t = self.clock()
            try:
                res = self.fp.locate(frame, t)
                obs = [o for o in self._hold.get("obs", []) if o.accepted]
                self.builder.add_frame([o for o in obs if o.tag_id in self.designated])
                if time.time() - t_rebuild > 1.0:
                    self._rebuild_map()
                    t_rebuild = time.time()
                raw = res["raw"]
                if raw:
                    last_R = np.asarray(raw["R"], float)
                xyz, hmode, st = res["xyz"], "", (self.state_fn() or {})
                if self.hf is not None:
                    z_f, hmode = self.hf.step(t, raw["xyz"][1] - self.floor_y if raw else None,
                                              raw["range"] if raw else None, tof_to_height(st, self.hcfg))
                    if xyz is not None and z_f is not None:
                        xyz = [xyz[0], z_f + self.floor_y, xyz[2]]
                q = dict(xyz=xyz, lost=res["lost"], used=list(raw["tags"]) if raw else [],
                         heading=heading_deg(last_R) if last_R is not None else None)
                if xyz:
                    last_pos = xyz
                    trail.append(xyz)
                    del trail[:-3000]
                n += 1
                if time.time() - t_fps > 1.0:
                    fps, n, t_fps = n / (time.time() - t_fps), 0, time.time()
                mapped = set(self.tag_map.poses)
                with self._lock:
                    self._snap = dict(obs=[(int(o.tag_id), o.corners.copy(), int(o.tag_id) in mapped) for o in obs],
                                      q=q, trail=list(trail), last_pos=last_pos, fps=fps, hmode=hmode,
                                      map_version=self.map_version, tof_trust=getattr(self.hf, "tof_verdict", ""))
                self._csv[1].writerow([f"{time.time():.4f}", last_id, " ".join(str(int(o.tag_id)) for o in obs),
                                       " ".join(map(str, q["used"])), *(xyz or ["", "", ""]),
                                       "" if not xyz else round(xyz[1] - self.floor_y, 3),
                                       "" if q["heading"] is None else round(q["heading"], 1), int(res["accepted"]),
                                       int(res["lost"]), "" if res["fix_age"] == float("inf") else f"{res['fix_age']:.2f}",
                                       hmode, getattr(self.hf, "tof_verdict", ""), " ".join(map(str, sorted(mapped)))])
                if time.time() - self.last_save > 10:
                    self.save()
            except Exception as e:       # keep mapping alive through one bad frame
                print("live_mapper:", repr(e))

    def _rebuild_map(self):
        new = self.builder.build()
        self.fp.positioner.tag_map = self.tag_map = new
        self.map_version += 1

    # ------------------------------------------------------------------ main-thread helpers
    def clear_trail(self):
        self._clear = True

    def snapshot(self):
        with self._lock:
            return dict(self._snap)

    def status(self):
        s = self.snapshot()
        tags = sorted(self.tag_map.poses)
        return tags, s

    def annotate(self, frame):
        """Draw tag outlines (green = in the map, grey = seen but not linked yet) and the map status."""
        tags, s = self.status()
        for tid, corners, in_map in s["obs"]:
            col = (0, 220, 0) if in_map else (170, 170, 170)
            cv2.polylines(frame, [corners.round().astype(np.int32)], True, col, 3)
            cv2.putText(frame, str(tid), tuple(corners[0].round().astype(int) + [0, -8]), 0, 0.9, col, 2)
        samples = self.tag_map.samples
        cv2.putText(frame, "MAP tags: " + " ".join(f"{t}({samples.get(t, 0)})" for t in tags), (10, 100), 0, 0.55,
                    (255, 255, 0), 2)
        cv2.putText(frame, f"map {s['fps']:.0f} fps  height {s['hmode'] or '-'}  (grey tag = not linked: show it "
                    "together with a mapped tag)", (10, 122), 0, 0.45, (255, 255, 0), 1)
        return frame

    def panel(self):
        s = self.snapshot()
        now = time.time()
        if self._td is None or s["map_version"] != self._shown_version or now - self._panel_t > 2.0:
            shapes = tag_shapes(self.tag_map, self.cfg, self.floor_y)
            xr, zr = view_bounds(shapes, [s["trail"]] if s["trail"] else [[[0, 0, 4.0], [1.5, 0, 3.0], [-1.0, 0, 3.0]]])
            self._td = TopDown(xr, zr, W_MAP, H_MAP)
            self._bg = self._td.background(shapes, xr, zr, self.floor_y)
            self._shape_pos = {sh["id"]: sh["pos"] for sh in shapes}
            self._shown_version, self._panel_t = s["map_version"], now
        if s["q"] is None:
            q = dict(xyz=None, lost=True, used=[], heading=None)
        else:
            q = s["q"]
        m = draw_panel(self._td, self._bg, self._shape_pos, q, s["trail"], s["last_pos"], self.floor_y, self.hfov, H_MAP)
        cv2.putText(m, f"{len(self.tag_map.poses)} tags mapped: {sorted(self.tag_map.poses)}", (10, H_MAP - 100), 0, 0.5,
                    (255, 255, 0), 1)
        return m

    # ------------------------------------------------------------------ output
    def save(self, path=None):
        path = Path(path or self.map_path)
        self.tag_map.save(path)
        to_repo_format(path)
        self.last_save = time.time()
        return path

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._rebuild_map()
        path = self.save()
        if self._csv:
            self._csv[0].close()
        copy = HERE / "maps" / f"live_{self.out_dir.name}.yaml"
        copy.parent.mkdir(exist_ok=True)
        copy.write_text(path.read_text())
        mapped = sorted(self.tag_map.poses)
        unlinked = sorted(self.designated - set(mapped))
        print(f"live map: tags {mapped} (not linked: {unlinked}) -> {path} and {copy}")
        return copy
