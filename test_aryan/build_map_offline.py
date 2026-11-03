"""Build a tag map (tags in the tag-10 frame) from recorded sessions instead of a live drone.

  python3 build_map_offline.py                                  # all sessions in data/ that have video
  python3 build_map_offline.py data/20261023_201826 data/20261023_203914 --stride 3
  python3 build_map_offline.py --out map_candidate.yaml

Same method as the repo's live tello_map.py: every frame where two or more tags are visible adds a
relative-pose edge; edges are averaged and chained outward from tag 10 (MapBuilder). The tag size
(tags.yaml) and the camera intrinsics (config/camera/<drone>.yaml) turn pixel size into metres.

Writes a CANDIDATE map (default test_aryan/map_candidate.yaml). It does NOT touch the shared
action-space-hackathon config/map.yaml; compare first with
  python3 localize_offline.py data/<session> --map map_candidate.yaml
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent / "drone_deployment_infra"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*", type=Path)
    ap.add_argument("--drone", default="tello_3")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", type=Path, default=HERE / "map_candidate.yaml")
    ap.add_argument("--tags", type=Path, default=None, help="tag size config (default: repo tags.yaml)")
    args = ap.parse_args()

    sys.path[:0] = [str(args.repo / "cal_and_map"), str(args.repo / "core")]
    from tello_calibration.calib.io import camera_from_dict, load_yaml
    from tello_calibration.tags.detect import TagDetector, load_tag_config
    from tello_calibration.tags.mapping import MapBuilder, TagMap
    from tello_calibration.utils.config import CONFIG_DIR

    cam = CONFIG_DIR / "camera" / f"{args.drone}.yaml"
    K, dist = camera_from_dict(load_yaml(cam if cam.exists() else CONFIG_DIR / "camera" / "default.yaml"))
    cfg = load_tag_config(args.tags) if args.tags else load_tag_config()
    detector = TagDetector(cfg, K, dist)
    designated = {t for t, s in cfg.sizes_mm.items() if s}      # tags with a known size
    print(f"camera fx={K[0, 0]:.1f}, origin tag {cfg.origin_tag}, tags with sizes: {sorted(designated)}")

    sessions = args.sessions or sorted(p for p in (HERE / "data").iterdir() if (p / "video.mp4").exists())
    builder = MapBuilder(cfg.origin_tag)
    for sess in sessions:
        cap = cv2.VideoCapture(str(sess / "video.mp4"))
        i = n = linked = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % args.stride == 0:
                obs = [o for o in detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                       if o.tag_id in designated]
                if sum(o.accepted for o in obs) >= 2:
                    linked += 1
                builder.add_frame(obs)
                n += 1
            i += 1
        print(f"  {sess.name}: {n} frames checked, {linked} with 2+ tags together", flush=True)

    tag_map = builder.build()
    tag_map.save(args.out)
    from map_format import to_repo_format; to_repo_format(args.out)
    print(f"\nmapped tags: {tag_map.mapped()}   (wanted {sorted(designated)})")
    missing = sorted(designated - set(tag_map.mapped()))
    if missing:
        print(f"NOT linked to tag 10 (never shared a view with the chain): {missing}")

    try:
        old = TagMap.load(CONFIG_DIR / "map.yaml")
    except Exception:
        old = None
    print("\n tag | samples |   x      y      z  (m, tag-10 frame) | vs current config/map.yaml")
    for tid in tag_map.mapped():
        p = tag_map.poses[tid][:3, 3]
        cmp = ""
        if old is not None and tid in old.poses:
            d = np.linalg.norm(p - old.poses[tid][:3, 3])
            cmp = f"differs by {d * 100:.0f} cm"
        elif old is not None:
            cmp = "NEW (not in current map)"
        print(f" {tid:>3} | {tag_map.samples[tid]:>7} | {p[0]:+6.2f} {p[1]:+6.2f} {p[2]:+6.2f} | {cmp}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
