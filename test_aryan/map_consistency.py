"""Does a tag map agree with itself on recorded video?

In any frame showing 2+ mapped tags, each tag gives its own estimate of the camera position.
If the map is right those estimates coincide; if tag positions are wrong (or the tags were moved
since mapping) they disagree. Prints the disagreement ("spread", cm) for each map.

  python3 map_consistency.py map_candidate.yaml            # compares config/map.yaml vs candidate
Note: a map built from the same recordings is partly in-sample, so treat it as a sanity check,
not proof; the real test is a tape-measured distance between two tags.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "drone_deployment_infra"
sys.path[:0] = [str(REPO / "cal_and_map"), str(REPO / "core")]
from tello_calibration.calib.io import camera_from_dict, load_yaml  # noqa: E402
from tello_calibration.tags.detect import TagDetector, load_tag_config  # noqa: E402
from tello_calibration.tags.mapping import TagMap  # noqa: E402
from tello_calibration.utils.config import CONFIG_DIR  # noqa: E402

import argparse  # noqa: E402
ap = argparse.ArgumentParser()
ap.add_argument("maps", nargs="*")
ap.add_argument("--tags", type=Path, default=None, help="tag size yaml used to detect (match how the map was built)")
ap.add_argument("--no-config", action="store_true", help="skip config/map.yaml")
ap.add_argument("--sessions", nargs="*", type=Path, default=None, help="session dirs (default: all with video)")
args = ap.parse_args()
maps = {} if args.no_config else {"config/map.yaml": TagMap.load(CONFIG_DIR / "map.yaml")}
for p in args.maps:
    maps[p] = TagMap.load(Path(p))
K, dist = camera_from_dict(load_yaml(CONFIG_DIR / "camera" / "tello_3.yaml"))
det = TagDetector(load_tag_config(args.tags) if args.tags else load_tag_config(), K, dist)

spreads = {k: [] for k in maps}
pairs = {k: {} for k in maps}
sessions = args.sessions or sorted(p for p in (HERE / "data").iterdir() if (p / "video.mp4").exists())
for sess in sessions:
    cap = cv2.VideoCapture(str(sess / "video.mp4"))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % 6 == 0:
            obs = [o for o in det.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if o.accepted]
            for name, m in maps.items():
                pts = {o.tag_id: m.camera_in_origin(o) for o in obs if o.tag_id in m.poses}
                pts = {k: v for k, v in pts.items() if v is not None}
                if len(pts) >= 2:
                    ids = sorted(pts)
                    for a in range(len(ids)):
                        for b in range(a + 1, len(ids)):
                            d = float(np.linalg.norm(pts[ids[a]] - pts[ids[b]]))
                            pairs[name].setdefault((ids[a], ids[b]), []).append(d)
                    arr = np.array(list(pts.values()))
                    spreads[name].append(float(np.linalg.norm(arr.max(0) - arr.min(0))))
        i += 1

for name in maps:
    s = np.array(spreads[name])
    print(f"\n=== {name}: tags {sorted(maps[name].poses)} ===")
    if len(s) == 0:
        print("  no frames with 2+ mapped tags"); continue
    print(f"  frames with 2+ mapped tags: {len(s)}   spread between tag estimates (cm): "
          f"median {np.median(s)*100:.0f}  p90 {np.percentile(s,90)*100:.0f}")
    for (a, b), v in sorted(pairs[name].items(), key=lambda kv: -len(kv[1]))[:6]:
        print(f"    tags {a}+{b}: {len(v):>4} frames, median disagreement {np.median(v)*100:.0f} cm")
