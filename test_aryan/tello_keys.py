"""Fly the Tello from the keyboard with live video, and record everything. Click the video window first!

  t        takeoff            l        land
  w / s    forward / back     a / d    left / right
  r / f    up / down          j / k    rotate left / right
  space    hover (stop)       x        EMERGENCY motor cut
  ESC      land and quit

Software trim (cancels steady drift while hovering; saved in trim.json, applied every rc packet):
  1 / 2    left-right trim  - / +     3 / 4    forward-back trim  - / +
  5 / 6    yaw trim         - / +     0        reset all trim
If it drifts right, press 1; if it drifts forward, press 3 (opposite of the drift).

Movement lasts only while a key is held (it stops ~0.4 s after release).

Recording (on by default) goes to data/<timestamp>/ - see flight_recorder.py for the files:
video, per-frame images, every rc command sent, IMU/state, ToF, and YOLO person pose.
  --no-record        fly without recording
  --no-pose          skip pose inference (saves CPU)
  --frames jpg|npy|none   per-frame images (default jpg; npy = lossless RGB but ~2 MB/frame)

Live mapping (--live-map): while you fly, detect AprilTags, build the tag map on the fly (tag 10 = origin; a new tag
is linked when it is in view together with a mapped one), localize the drone in it, and fuse height with the ToF.
A second window shows the top-down map; keys there too: m = save the map now, c = clear the trail. The map is saved
(repo config/map.yaml layout) to data/<ts>/live_map.yaml and maps/live_<ts>.yaml when you quit.
  --live-map         turn it on        --floor-y -1.22   floor level in the tag-10 frame (tag 10 centre is 1.22 m high)
  Tip: add --no-pose so YOLO does not compete with tag detection for the CPU.

Several drones on one PC (one USB dongle each, see dongle.py): run one copy per drone in its own terminal,
  python3 tello_keys.py --tello TELLO-1        python3 tello_keys.py --tello TELLO-2
Each copy pins its command, video and state sockets to that drone's dongle (dongles.json), checks the dongle really
is on that drone's Wi-Fi before sending anything, keeps its own trim file and data folder (data/<ts>_TELLO-N/).
Keys go to whichever window has focus, so click the window of the drone you want to fly.
"""
import argparse
import json
import time
from pathlib import Path

import cv2

from tello_stream import TelloStream

HERE = Path(__file__).resolve().parent
SPEED = 40          # rc range is -100..100; keep low indoors
HOLD = 0.4          # seconds a key press keeps the drone moving
MIN_TAKEOFF_BATT = 20
TRIM_STEP = 2
TRIM_FILE = HERE / "trim.json"
TRIM_KEYS = {  # key -> (channel index in rc tuple, direction)
    ord("1"): (0, -1), ord("2"): (0, 1),
    ord("3"): (1, -1), ord("4"): (1, 1),
    ord("5"): (3, -1), ord("6"): (3, 1),
}


def load_trim():
    try:
        return list(json.loads(TRIM_FILE.read_text()))[:4]
    except (OSError, ValueError, TypeError):
        return [0, 0, 0, 0]


def clamp(v, lim=100):
    return max(-lim, min(lim, int(v)))


# key -> (left/right, forward/back, up/down, yaw)
MOVES = {
    ord("w"): (0, SPEED, 0, 0),
    ord("s"): (0, -SPEED, 0, 0),
    ord("a"): (-SPEED, 0, 0, 0),
    ord("d"): (SPEED, 0, 0, 0),
    ord("r"): (0, 0, SPEED, 0),
    ord("f"): (0, 0, -SPEED, 0),
    ord("j"): (0, 0, 0, -SPEED),
    ord("k"): (0, 0, 0, SPEED),
}

ap = argparse.ArgumentParser()
ap.add_argument("--no-record", action="store_true")
ap.add_argument("--no-pose", action="store_true")
ap.add_argument("--frames", choices=["jpg", "npy", "none"], default="jpg")
ap.add_argument("--out", default=str(HERE / "data"))
ap.add_argument("--live-map", action="store_true", help="build the tag map + localize + fuse height while flying")
ap.add_argument("--floor-y", type=float, default=-1.22)
ap.add_argument("--map", default=None, metavar="PATH",
                help="fixed map file for --live-map localization; set to "
                     "drone_deployment_infra/config/map.yaml for the 1v1 match. "
                     "When given, LiveMapper localizes against this map instead of building a fresh one.")
ap.add_argument("--drone", default=None, help="camera calibration name for --live-map (default: from --tello, else tello_3)")
ap.add_argument("--tello", default=None, metavar="TELLO-N",
                help="fly this drone through ITS USB dongle (dongles.json registry); omit for the single-drone "
                     "setup (laptop Wi-Fi joined to one Tello)")
ap.add_argument("--publish-telemetry", default=None, metavar="HOST:PORT",
                help="publish this drone's pose as drone_telemetry v1 UDP JSON (needs --live-map); "
                     "e.g. 127.0.0.1:9100  -- team set by --team flag")
ap.add_argument("--team", default="attack", choices=["attack", "defense"],
                help="team label in the published telemetry (default: attack)")
args = ap.parse_args()

tello = TelloStream(drone=args.tello)
tello.start()
if args.tello:
    TRIM_FILE = HERE / f"trim_{tello.entry['drone']}.json"       # each drone has its own drift
    print(f"{tello.entry['drone']} via {tello.iface}; video -> localhost:{tello._fwd_port}")
cam_name = args.drone or (args.tello.lower().replace("-", "_") if args.tello else "tello_3")

trim = load_trim()  # [lr, fb, ud, yaw]
rec = None
if not args.no_record:
    from flight_recorder import FlightRecorder, draw_pose
    rec = FlightRecorder(tello, args.out, frames=args.frames, pose=not args.no_pose,
                         meta=dict(script="tello_keys.py", tello=args.tello, speed=SPEED, hold=HOLD, trim_step=TRIM_STEP,
                                   initial_trim=trim))
    rec.start()
    rec.log_event("start", trim=trim)

lm = state_src = None
if args.live_map:
    from live_mapper import LiveMapper
    if rec:
        state_fn, out_dir = rec.state, rec.out_dir
    else:       # no recorder: listen for the state broadcast ourselves (ToF for the height fusion)
        from datetime import datetime
        from tello_record import StateListener
        last_state = {}
        state_src = StateListener(lambda t, s: last_state.update(s), iface=tello.iface)
        state_fn, out_dir = (lambda: dict(last_state)), Path(args.out) / (datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{tello.drone}" if tello.drone else "") + "_livemap")
    lm = LiveMapper(tello, state_fn, out_dir, drone=cam_name, floor_y=args.floor_y)
    lm.start()
    # --map opt-in: use a fixed shared map for localization instead of the live-built one
    if args.map:
        from pathlib import Path as _Path
        _ext_map = _Path(args.map)
        if _ext_map.exists():
            import shutil as _sh
            _sh.copy2(_ext_map, lm.map_path)        # seed live_map.yaml with the venue map
            lm.fp = lm.fp.__class__(cam_name, map_path=lm.map_path)   # re-init positioner
            print(f"live mapping: using external map {_ext_map.name}")
        else:
            print(f"WARNING: --map {_ext_map} not found; building map from scratch")
    print("live mapping on; map -> %s" % lm.map_path)

# --publish-telemetry opt-in: wire live pose to the defender's AttackFeed (3 lines)
_pose_pub = None
if args.publish_telemetry:
    from publish_pose_hook import PosePublisher
    _pose_pub = PosePublisher(args.publish_telemetry, args.tello or "TELLO-X", args.team,
                              get_snapshot=(lm.snapshot if lm else lambda: {}),
                              get_state=(rec.state if rec else lambda: {}))
    _pose_pub.start()
    print(f"pose publisher: {args.team} {args.tello or 'TELLO-X'} -> {args.publish_telemetry}")

flying = False
rc = (0, 0, 0, 0)
key_char = ""
last_move = 0.0
last_rc_sent = 0.0

try:
    while True:
        state = rec.state() if rec else {}
        battery = state.get("bat", "?")
        frame = tello.read()
        if frame is not None:
            if rec:
                draw_pose(frame, rec.latest_pose())
            status = "FLYING" if flying else "LANDED"
            tof = state.get("tof")
            cv2.putText(frame, f"{status}  batt {battery}%  h {state.get('h', '?')}cm  tof {tof if tof is not None else '?'}cm",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"trim lr {trim[0]:+d} fb {trim[1]:+d} yaw {trim[3]:+d}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if rec:
                cv2.putText(frame, f"REC {rec.counts['frames']} fr", (10, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if lm:
                lm.annotate(frame)
            cv2.imshow("Tello keys", frame)
        if lm:
            cv2.imshow("Tello map (m save, c clear trail)", lm.panel())
        key = cv2.waitKey(10) & 0xFF
        now = time.monotonic()

        if key == 27:  # ESC
            break
        elif key == ord("x"):
            tello.send_nowait("emergency")
            flying = False
            rc = (0, 0, 0, 0)
            if rec:
                rec.log_event("emergency", trim, flying, "x")
            print("EMERGENCY motor cut")
        elif key == ord("t") and not flying:
            batt = state.get("bat")
            if batt is None:  # no state packet yet: ask the drone
                reply = tello.send("battery?")
                batt = float(reply) if reply and reply.isdigit() else None
            if batt is not None and batt >= MIN_TAKEOFF_BATT:
                tello.send_nowait("takeoff")
                flying = True
                if rec:
                    rec.log_event("takeoff", trim, flying, "t", f"battery={batt:.0f}")
                print("Takeoff")
            else:
                if rec:
                    rec.log_event("takeoff_refused", trim, flying, "t", f"battery={batt}")
                print(f"Battery too low or unknown ({batt}); not taking off")
        elif key == ord("l"):
            tello.send_nowait("land")
            flying = False
            rc = (0, 0, 0, 0)
            if rec:
                rec.log_event("land", trim, flying, "l")
            print("Land")
        elif key == ord("m") and lm:
            print("map saved:", lm.save())
        elif key == ord("c") and lm:
            lm.clear_trail()
        elif key == ord(" "):
            rc = (0, 0, 0, 0)
            if rec:
                rec.log_event("hover", trim, flying, "space")
        elif key in TRIM_KEYS:
            ch, d = TRIM_KEYS[key]
            trim[ch] = clamp(trim[ch] + d * TRIM_STEP, 30)
            TRIM_FILE.write_text(json.dumps(trim))
            if rec:
                rec.log_event("trim", trim, flying, chr(key))
            print("trim [lr, fb, ud, yaw] =", trim)
        elif key == ord("0"):
            trim = [0, 0, 0, 0]
            TRIM_FILE.write_text(json.dumps(trim))
            if rec:
                rec.log_event("trim_reset", trim, flying, "0")
            print("trim reset")
        elif key in MOVES:
            rc = MOVES[key]
            key_char = chr(key)
            last_move = now

        if now - last_move > HOLD:
            rc = (0, 0, 0, 0)
            key_char = ""

        # rc streams at ~20 Hz while flying
        if flying and now - last_rc_sent > 0.05:
            sent = tuple(clamp(c + t) for c, t in zip(rc, trim))
            tello.send_nowait("rc %d %d %d %d" % sent)
            last_rc_sent = now
            if rec:
                rec.set_control(sent, flying)
                rec.log_rc(sent, rc, trim, key_char)
        elif not flying and rec:
            rec.set_control((0, 0, 0, 0), False)
finally:
    if flying:
        tello.send_nowait("rc 0 0 0 0")
        tello.send_nowait("land")
        if rec:
            rec.log_event("land", trim, False, "", "auto-land on exit")
        print("Landing...")
        time.sleep(3)
    try:
        if _pose_pub:
            _pose_pub.stop()
        if lm:
            lm.stop()
        if state_src:
            state_src.stop()
        if rec:
            rec.log_event("stop", trim)
            rec.stop()
    finally:
        tello.stop()
        cv2.destroyAllWindows()
