"""Records everything during a flight session. Used by tello_keys.py.

All disk writing happens on one background thread, so the control loop never blocks on I/O.

Output: data/<YYYYmmdd_HHMMSS>/
  meta.json     run settings + final counts
  video.mp4     camera video (BGR->mp4v, lossy)
  frames/       frame_000123.jpg (q95) or .npy (lossless RGB uint8) or nothing; see `frames=`
  frames.csv    one row per camera frame: ids/times, latest IMU state, ToF, the rc command in effect
  imu.csv       every state packet (~10 Hz): attitude, velocity, acceleration, height, baro, ToF, battery
  commands.csv  every command event: each rc packet as sent (plus the pre-trim intent and key),
                takeoff / land / emergency / trim changes / hover
  pose.jsonl    (pose=True) one JSON line per YOLO inference: person boxes + 17 COCO keypoints
                (x, y, conf, pixel coords) tagged with frame_id so it joins frames.csv

All times are unix seconds from this laptop's clock (time.time()), so every file lines up.
"""
import csv
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tello_record import STATE_FIELDS, STATE_HEADERS, StateListener, tof_m

COCO_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13),
              (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]
POSE_WEIGHTS = Path(__file__).resolve().parent / "yolo11n-pose.pt"


def draw_pose(frame, persons):
    """Draw boxes + skeletons from FlightRecorder.latest_pose() onto frame (in place)."""
    for p in persons or []:
        x1, y1, x2, y2 = (int(v) for v in p["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        kp = p["kp"]
        for a, b in COCO_EDGES:
            if kp[a][2] > 0.5 and kp[b][2] > 0.5:
                cv2.line(frame, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])),
                         (255, 200, 0), 2)
        for x, y, c in kp:
            if c > 0.5:
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)


class FlightRecorder:
    def __init__(self, tello, out_base, frames="jpg", pose=True, fps=30.0, meta=None):
        assert frames in ("jpg", "npy", "none")
        self.tello = tello
        self.frames_mode = frames
        self.pose_enabled = pose
        self.fps = fps
        drone = getattr(tello, "drone", None)       # several drones can record in the same second
        self.out_dir = Path(out_base) / (datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{drone}" if drone else ""))
        self.out_dir.mkdir(parents=True)
        if frames != "none":
            (self.out_dir / "frames").mkdir()
        self.meta = dict(meta or {})
        if drone:
            self.meta.update(drone=drone, iface=tello.iface)
        self.meta.update(frames=frames, pose=pose, start_time=time.time(),
                         start_iso=datetime.now().isoformat(timespec="seconds"))

        self._q = queue.Queue(maxsize=400)
        self._running = False
        self._writer_done = False
        self._threads = []
        self._lock = threading.Lock()
        self._state = (0.0, {})                 # (time, state dict)
        self._ctl = dict(sent=(0, 0, 0, 0), flying=False)
        self._pose_in = None                    # (frame, id, t) newest frame for the pose thread
        self._pose_latest = []
        self.counts = dict(frames=0, dropped_frames=0, pose=0, cmds=0, imu=0)

    # ---------- public API ----------
    def start(self):
        if self.pose_enabled:  # load up front: importing torch inside a thread stalls the control loop
            print("Loading pose model ...")
            from ultralytics import YOLO
            self._model = YOLO(str(POSE_WEIGHTS))
            self._model(np.zeros((720, 960, 3), np.uint8), verbose=False, device="cpu")  # warm-up
        self._running = True
        self._listener = StateListener(self._on_state, iface=getattr(self.tello, "iface", None))
        for fn in (self._writer, self._capture) + ((self._pose_worker,) if self.pose_enabled else ()):
            t = threading.Thread(target=fn, daemon=True)
            t.start()
            self._threads.append(t)
        print(f"Recording to {self.out_dir}")

    def state(self, max_age=2.0):
        """Latest state packet dict, or {} if none newer than max_age seconds."""
        with self._lock:
            t, s = self._state
        return dict(s) if time.time() - t <= max_age else {}

    def set_control(self, sent, flying):
        """Tell the recorder the rc command currently in effect (tagged onto each frame)."""
        with self._lock:
            self._ctl = dict(sent=tuple(sent), flying=bool(flying))

    def log_rc(self, sent, intended, trim, key=""):
        self._put(("cmd", [time.time(), "rc", key, *sent, *intended, trim[0], trim[1], trim[3], 1, ""]))

    def log_event(self, event, trim=(0, 0, 0, 0), flying=False, key="", note=""):
        self._put(("cmd", [time.time(), event, key, "", "", "", "", "", "", "", "",
                           trim[0], trim[1], trim[3], int(flying), note]))

    def latest_pose(self):
        with self._lock:
            return self._pose_latest

    def stop(self):
        self._running = False
        self._listener.stop()
        for t in self._threads[1:]:   # capture + pose first, so the writer sees everything they queued
            t.join(timeout=10)
        self._writer_done = True
        self._threads[0].join(timeout=30)
        self.meta.update(end_time=time.time(), duration_s=round(time.time() - self.meta["start_time"], 2),
                         **self.counts)
        (self.out_dir / "meta.json").write_text(json.dumps(self.meta, indent=2))
        c = self.counts
        print(f"Saved {c['frames']} frames ({c['dropped_frames']} dropped), {c['imu']} IMU, "
              f"{c['cmds']} command rows, {c['pose']} pose inferences -> {self.out_dir}")

    # ---------- internals ----------
    def _put(self, item):
        try:
            self._q.put_nowait(item)
            return True
        except queue.Full:
            return False

    def _on_state(self, t, s):
        with self._lock:
            self._state = (t, s)
        self._put(("imu", [f"{t:.4f}"] + [s.get(k, "") for k in STATE_FIELDS] + [tof_m(s)]))

    def _capture(self):
        last_id = 0
        while self._running:
            item = self.tello.read_new(last_id)
            if item is None:
                time.sleep(0.002)
                continue
            frame, last_id, t = item
            with self._lock:
                st, state = self._state
                ctl = dict(self._ctl)
                self._pose_in = (frame, last_id, t)
            row = [last_id, f"{t:.4f}", f"{st:.4f}"] + [state.get(k, "") for k in STATE_FIELDS] \
                + [tof_m(state), *ctl["sent"], int(ctl["flying"])]
            if not self._put(("frame", (frame, last_id, row))):
                self.counts["dropped_frames"] += 1

    def _pose_worker(self):
        model = self._model
        last_id = 0
        while self._running:
            with self._lock:
                item = self._pose_in
            if item is None or item[1] == last_id:
                time.sleep(0.005)
                continue
            frame, last_id, t = item
            t0 = time.time()
            r = model(frame, verbose=False, device="cpu")[0]
            persons = []
            if r.boxes is not None and r.keypoints is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                kps = r.keypoints.data.cpu().numpy()
                for b, c, k in zip(boxes, confs, kps):
                    persons.append(dict(box=[round(float(v), 1) for v in b], conf=round(float(c), 3),
                                        kp=[[round(float(v), 1) for v in pt] for pt in k]))
            rec = dict(frame_id=last_id, frame_time=round(t, 4), time=round(time.time(), 4),
                       inference_ms=round((time.time() - t0) * 1000, 1), persons=persons)
            with self._lock:
                self._pose_latest = persons
            self._put(("pose", rec))

    def _writer(self):
        d = self.out_dir
        files = {}

        def open_csv(name, header):
            f = open(d / name, "w", newline="")
            w = csv.writer(f)
            w.writerow(header)
            files[name] = f
            return w

        imu_w = open_csv("imu.csv", ["time"] + STATE_HEADERS + ["tof_m"])
        cmd_w = open_csv("commands.csv", [
            "time", "event", "key", "lr", "fb", "ud", "yaw", "intent_lr", "intent_fb", "intent_ud",
            "intent_yaw", "trim_lr", "trim_fb", "trim_yaw", "flying", "note"])
        frm_w = open_csv("frames.csv", ["frame_id", "time", "state_time"] + STATE_HEADERS
                         + ["tof_m", "cmd_lr", "cmd_fb", "cmd_ud", "cmd_yaw", "flying"])
        pose_f = open(d / "pose.jsonl", "w") if self.pose_enabled else None
        video = None
        last_flush = time.time()

        def handle(kind, payload):
            nonlocal video
            if kind == "imu":
                imu_w.writerow(payload)
                self.counts["imu"] += 1
            elif kind == "cmd":
                cmd_w.writerow(payload)
                self.counts["cmds"] += 1
            elif kind == "pose":
                pose_f.write(json.dumps(payload) + "\n")
                self.counts["pose"] += 1
            elif kind == "frame":
                frame, fid, row = payload
                if video is None:
                    h, w = frame.shape[:2]
                    video = cv2.VideoWriter(str(d / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                            self.fps, (w, h))
                video.write(frame)
                if self.frames_mode == "jpg":
                    cv2.imwrite(str(d / "frames" / f"frame_{fid:06d}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                elif self.frames_mode == "npy":
                    np.save(d / "frames" / f"frame_{fid:06d}.npy", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                frm_w.writerow(row)
                self.counts["frames"] += 1

        try:
            while not (self._writer_done and self._q.empty()):
                try:
                    kind, payload = self._q.get(timeout=0.1)
                except queue.Empty:
                    continue
                handle(kind, payload)
                if time.time() - last_flush > 1.0:  # a crash loses at most ~1 s of rows
                    for f in files.values():
                        f.flush()
                    if pose_f:
                        pose_f.flush()
                    last_flush = time.time()
        finally:
            if video is not None:
                video.release()
            for f in files.values():
                f.close()
            if pose_f:
                pose_f.close()
