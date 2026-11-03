# CLAUDE.md (test_aryan)

Our own Python 3 scripts for one Tello: fly it, record it, and localize it from AprilTags. Run everything from inside
this folder (scripts import each other by bare name). The laptop must be on the drone's `TELLO-XXXXXX` Wi-Fi for
anything that talks to the drone (that Wi-Fi has no internet). Part of `action-space-hackathon` (real-drone code it reuses lives in `../drone_deployment_infra`).

## Scripts

| Script | What it does | Sends flight commands? |
|---|---|---|
| `tello_video.py` | live camera; `q` quit, `s` snapshot | no |
| `tello_pose.py` | camera + YOLO person pose + arm-gesture labels | no |
| `tello_keys.py` | **keyboard flight + live video + full recording** to `data/<ts>/` (`t`/`l` takeoff/land, `wasd`, `r`/`f` up/down, `j`/`k` yaw, `1`-`6` software trim, `0` reset trim, space hover, `x` emergency, ESC land+quit). Flags: `--no-record`, `--no-pose`, `--frames jpg\|npy\|none` | **yes** |
| `live_mapper.py` | background worker used by `tello_keys.py --live-map`: detects tags, builds the tag map while you fly (`MapBuilder`), localizes in it, fuses ToF height, saves `live_map.yaml` in the repo's `map.yaml` layout | no |
| `dongle.py` | dongle <-> drone: reads the repo's `config/dongles.json`, checks the dongle is on the right `TELLO-N` Wi-Fi, makes sockets pinned to that dongle (`SO_BINDTODEVICE`). Used by `--tello TELLO-N` in `tello_keys.py` / `tello_localize_live.py` (command, state and video sockets all go through the dongle; own trim file + data folder per drone) | no |
| `map_format.py` | rewrites a map yaml in exactly the repo `config/map.yaml` layout (called automatically when we save maps) | no |
| `tello_record.py` | record video + IMU/state/ToF without flying | no |
| `tello_bench.py` | link benchmark: fps, command RTT, rc send rate, state rate; `--qr` adds video latency (point camera at the QR window) | no (only neutral `rc`) |
| `tello_localize_live.py` | **live** AprilTag localization: camera with tags outlined + top-down map. `--record`, `--replay <video.mp4>` (test with no drone), `--no-height-fusion` | no |
| `localization_pipeline.py` | **main offline entry**: sessions -> (optional new map) -> position per frame -> `localization/` outputs. `--build-map`, `--map`, `--no-video`, `--no-height-fusion` | no |
| `viewer_3d.py` | interactive 3D page (video synced to a rotatable scene) -> `localization/viewer_3d.html` | no |
| `localize_offline.py`, `build_map_offline.py`, `map_consistency.py` | earlier building blocks of the pipeline (single-session localize, map from recordings, map self-agreement check); still work | no |
| `tag_coverage.py` | aggregates `localization_summary.json` of all sessions into `data/tag_coverage.md/.csv` (which recordings show tags) | no |
| `test_tello.py` | raw-socket takeoff/hover/land smoke test | **yes** |

Shared modules: `tello_stream.py` (`TelloStream`: command socket + OpenCV/FFMPEG video thread + 5 s keepalive; the
keepalive swallows command replies, so flight code uses `send_nowait`), `flight_recorder.py` (`FlightRecorder`, single
writer thread, bounded queue), `height_fusion.py` (`HeightFilter`).

## Typical flows

```bash
python3 tello_keys.py                                   # fly + record -> data/<ts>/
python3 tello_keys.py --tello TELLO-1                   # 2+ drones on one PC: one terminal per drone, each via its own USB dongle
python3 tello_keys.py --live-map --no-pose              # same + build tag map / localize / fuse height live (m = save map)
python3 localization_pipeline.py data/<ts>              # localize with maps/latest_layout.yaml
python3 localization_pipeline.py data/<ts> --build-map  # tags moved? build a fresh map from this session first
python3 viewer_3d.py data/<ts>                          # 3D viewer (opens Chrome)
python3 tello_localize_live.py                          # live, on the drone's Wi-Fi
python3 tello_localize_live.py --replay data/<ts>/video.mp4   # same, no drone
```

## Data layout

`data/<YYYYmmdd_HHMMSS>/` (one per recording): `video.mp4` (mp4v - browsers can't play it; the viewer makes an H.264
copy), `frames/` (jpg or npy), `frames.csv` (per frame: times + latest IMU + ToF + rc in effect), `imu.csv` (state packets
~10 Hz), `commands.csv` (every rc packet sent with pre-trim intent and key, plus takeoff/land/trim events), `pose.jsonl`,
`meta.json`. `localization/` inside it holds the pipeline outputs: `localization.csv`, `map_view.mp4` (camera + top-down),
`map_static.png`, `summary.json`, and from the viewer `viewer_3d.html` + `viewer_video.mp4`.
Older sessions have two columns named `time` in `frames.csv`/`imu.csv` (unix + the Tello's motor time); the recorders now
name the second `motor_time`, and every reader here de-duplicates the header. All timestamps are laptop `time.time()`.
`maps/` holds tag maps (`latest_layout.yaml` = built from sessions 201826 + 203914) and `overview_*.png`.

## How localization works (and what it reuses)

Uses `../drone_deployment_infra` as-is: `TagDetector` (tag36h11, gates in `config/tags.yaml`),
`MapBuilder`/`TagMap` (chains tags outward from tag 10 from frames where two tags are visible together),
`Positioner`/`FilteredPositioner` (position in the tag-10 frame + Kalman filter), camera calibration from
`config/camera/<drone>.yaml` (default `tello_3`). Nothing in that repo is modified; its `config/map.yaml` is never
overwritten by our scripts (teammates fly from it). Map frame: origin = centre of tag 10, x right, y up, z out of the wall
into the room, metres. Heights shown = `y - floor_y` with `floor_y = -1.22` (tag 10's centre is 1.22 m above the floor).
Heading 0 = facing the wall, positive = turned right. `height_fusion.py` replaces the jumpy tag height by a 1-D Kalman
filter that adds the Tello's downward ToF when valid (tilt-corrected, 0.12-3.0 m) and outlier-gates the tag height.

## Gotchas and facts established

- **Tags get moved between sessions.** A map built from mixed layouts is garbage (tag 4 moved between 192028 and the
  later sessions). Build maps only from sessions that share one layout. The shared `config/map.yaml` (tags 4, 6, 10) does
  not match the current layout either.
- Sessions with tags at the venue: `185532`-`203914`. `200603`, `203515` show only floor/furniture; `232635`-`233346` were
  recorded in a different room (hardwood floor, radiator) and show no tags. `20261024_*` are new `tello_keys.py` flights
  (ToF valid in 16-76% of frames, `h` up to 70-130 cm); **not yet localized**.
- The Tello's ToF reads a constant 10 cm when idle/handheld/on the ground and `h`, `vgx/vgy/vgz` stay ~0 then: these
  sensors only mean something in flight. In all handheld sessions ToF was useless.
- `localization_pipeline.py --max-frames/--stride` **overwrites** that session's `localization/` outputs with a partial
  run; re-run the full pipeline afterwards (the viewer warns when `localization.csv` is partial).
- Position noise: tag poses flip/jump (gates in `tags.yaml` are mostly off); typical disagreement between tags is
  40-65 cm. Tag-derived height alone jumps >30 cm between frames 120-218 times per session; `height_fusion.py`
  cuts that to 3-4 (still tag-only in practice on the handheld sessions).
- **ToF is not "the floor".** It measures whatever is below the drone (hand, table). In handheld session 201826 it read
  ~0.6 m while the tags said ~1.2 m, and a naive fusion followed the ToF. `HeightFilter` therefore monitors
  `median(tag - ToF)` over 4 s and ignores the ToF while it is > 0.30 m off (`tof_trust` = ok/bad/unknown and
  `tof_tag_bias` in `localization.csv`); on a clear floor that median is also a check of `floor_y`.
- IMU pitch (positive = nose up) tracks the tag-derived camera pitch (corr 0.97); the camera sits about 11.6 degrees below
  the body axis. IMU roll did not match a simple tag-derived roll (corr -0.55), so the 3D view cone ignores roll.
- Unverified assumptions: `floor_y = -1.22` comes from the sim config (`tag10_height`), not a real measurement; small tag
  size in `tags.yaml` is 5.2 in (132.08 mm) but the printed sheets say 6.00 in (152.4 mm) (a rebuild with 6.00 in fit
  slightly worse); units of the state fields `vgx/vgy/vgz` (cm/s vs dm/s) are unchecked.
- Pose uses `ultralytics` YOLO on CPU (no working GPU driver); `mediapipe` is broken here. The pose model is loaded in
  `FlightRecorder.start()` on purpose: importing torch inside a thread stalls the control loop for seconds.
- Keyboard flight on Wayland has no key-release events, so movement is hold-timeout based (`HOLD` in `tello_keys.py`).
- Download any model weights before joining the drone's Wi-Fi (`yolo11n-pose.pt` is already here).

## Tested vs not

Tested on recordings: the whole localization pipeline, map building and consistency check, live script via `--replay`,
the 3D viewer (headless Chrome render), the recorder (with a fake drone). Tested only in simulation: `height_fusion.py`
(synthetic flight: RMS height error 30 cm raw tag -> 8.6 cm filtered -> 4.8 cm with ToF). **Never run against the real
drone in this session:** `--tello` dongle binding (only the registry lookup, the not-plugged-in checks and a loopback bind were tested), `tello_keys.py --live-map` / `live_mapper.py` (only imports and a short replay run were checked), `tello_localize_live.py` live mode, `tello_bench.py`, `tello_keys.py` trim keys.
