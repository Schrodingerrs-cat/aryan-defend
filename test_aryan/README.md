# test_aryan: fly, record and localize one Tello (or several)

Python 3 scripts for the DJI Tello: keyboard flight, full-data recording, AprilTag mapping and localization (offline and
live), a 3D viewer, and a link benchmark. They reuse this repo's tag detector, map builder and position filter from
`../drone_deployment_infra` and change nothing in it. Details and gotchas: [`CLAUDE.md`](CLAUDE.md).

## Setup

System Python 3.10 is enough (no repo venv): `opencv-python`, `numpy`, `pyyaml`, `matplotlib`, `plotly`,
`pupil-apriltags` (the repo detector's dependency), `ultralytics` for person pose. Run everything from this folder.
The laptop must be on the drone's `TELLO-XXXXXX` Wi-Fi (no internet there), so download model weights first
(`yolo11n-pose.pt` is included).

## Fly and record

```bash
python3 tello_keys.py                      # keys + live video + recording to data/<ts>/
python3 tello_keys.py --live-map --no-pose # also build the tag map, localize and fuse height while flying
python3 tello_keys.py --no-record          # just fly
python3 tello_keys.py --tello TELLO-1      # several drones on one PC: this drone via ITS USB dongle (dongles.json)
```

`t`/`l` takeoff/land, `w a s d` move, `r`/`f` up/down, `j`/`k` yaw, space hover, `x` emergency motor cut, ESC land and
quit, `1`-`6` software trim, `0` reset trim, `m` save map and `c` clear trail (with `--live-map`). Movement lasts while a
key is held. Takeoff needs at least 20% battery.

## Localize and view a recording

```bash
python3 localization_pipeline.py data/<ts>                 # position per frame with maps/latest_layout.yaml
python3 localization_pipeline.py data/<ts> --build-map     # tags moved? build a fresh map from this session first
python3 localization_pipeline.py data/<ts> --map maps/x.yaml --no-video
python3 viewer_3d.py data/<ts> --map maps/x.yaml           # interactive 3D page with the video in sync
python3 regularize_map.py maps/in.yaml --out maps/out.yaml --floor 1,2,4,7 --wall 5:+z,6:-z --same-plane 6,9
python3 tello_localize_live.py                             # live on the drone's Wi-Fi (--replay <video> to test offline)
python3 eval_ekf.py data/<ts>                              # check pose_ekf.py against the repo Kalman on a session
```

Outputs go to `data/<ts>/localization/` (`localization.csv`, `map_view.mp4`, `map_static.png`, `viewer_3d.html`).
The 3D viewer displays X right, Y out from the tag-10 wall, and Z up from the floor: takeoff is +Z and approaching
the origin is −Y. It converts the localisation/map coordinates below for display.
Maps are in the same layout as `drone_deployment_infra/config/map.yaml` (origin tag 10; x right, y up, z out of the
wall, metres), so the rest of the repo can read them. Tags get moved between sessions: build a map only from sessions
with one layout, and never overwrite the shared `config/map.yaml` without asking the team.

## What is here

| File | Purpose |
|---|---|
| `tello_keys.py` | keyboard flight + recording + optional live mapping |
| `flight_recorder.py`, `tello_record.py` | recorder (video, frames, IMU/state, ToF, rc commands, pose) |
| `tello_stream.py`, `dongle.py` | Tello link; per-dongle sockets for several drones |
| `live_mapper.py` | background tag mapping + localization + height fusion for `tello_keys.py --live-map` |
| `localization_pipeline.py` | offline entry: map building, position per frame, videos, overview |
| `height_fusion.py` | stable height from tag height + ToF, with a ToF-vs-tag consistency monitor |
| `pose_ekf.py`, `eval_ekf.py` | x/z filter that adds the Tello's own velocity (offline-validated only) |
| `viewer_3d.py`, `regularize_map.py`, `map_format.py` | 3D viewer; snap a map to the known layout; write repo map layout |
| `tello_localize_live.py`, `tello_bench.py`, `tello_video.py`, `tello_pose.py`, `test_tello.py` | live localization, link benchmark, camera, person pose, raw takeoff test |
| `localize_offline.py`, `build_map_offline.py`, `map_consistency.py`, `tag_coverage.py` | earlier building blocks |
| `maps/` | tag maps and overview images (`latest_layout.yaml` = 4, 5, 7, 8, 10; `layout_0944.yaml` = 1, 2, 4, 5, 6, 7, 9 measured + 8 copied) |

`data/` (recordings, 20+ GB) is git-ignored and stays on the machine that flew.

## Status

Tested on recordings: the localization pipeline, map building, the viewer, `height_fusion.py`, `pose_ekf.py` (jitter and
hidden-tag tests on one flight). **Not yet run on the real drone:** `--tello` dongle binding, `tello_bench.py`,
`tello_localize_live.py` live mode, trim keys. `pose_ekf.py` is not wired into the pipeline, and the trained policies
only ever saw the repo's own Kalman filter, so keep that one for anything the policy reads.
Unverified: floor level -1.22 m (from the sim config), small-tag size 5.2 vs 6.00 in (velocity data hints the tags may be
about 15% bigger than configured).
