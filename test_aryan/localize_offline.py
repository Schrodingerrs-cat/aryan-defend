"""Run the repo's AprilTag localization pipeline on a recorded session (no drone needed).

  python3 localize_offline.py data/20261023_203515
  python3 localize_offline.py data/20261023_203515 --drone tello_4 --no-video

Uses action-space-hackathon's own pipeline: core/positional.py (tag detection + map.yaml ->
drone position in the tag-10 frame) wrapped by core/positional_kalman_filter.py.

Reads <session>/video.mp4 + frames.csv (frame i of the video <-> row i of frames.csv).
Writes into <session>/:
  localization.csv    per frame: tags seen, raw position, filtered position + velocity, lost flag, IMU h/tof
  localization.mp4    video with tag outlines and the position overlaid   (skip with --no-video)
  localization.png    top-down (x,y) and height trace of the estimated path
and prints a summary: fix rate, which tags were seen, longest gap without a fix.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parent / "drone_deployment_infra"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--drone", default="tello_3", help="camera calibration to use (config/camera/<drone>.yaml)")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="drone_deployment_infra folder")
    ap.add_argument("--map", type=Path, default=None, help="tag map yaml (default: repo config/map.yaml)")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame (fast scan)")
    args = ap.parse_args()

    sys.path[:0] = [str(args.repo / "cal_and_map"), str(args.repo / "core")]
    from positional_kalman_filter import FilteredPositioner  # noqa: E402

    sess = args.session
    tag = f"_{args.map.stem}" if args.map else ""
    with open(sess / "frames.csv") as f:
        rd = csv.reader(f)
        header, seen_cols = [], set()
        for col in next(rd):  # older sessions have two "time" columns (unix + Tello motor time)
            header.append(col if col not in seen_cols else "motor_time")
            seen_cols.add(col)
        rows = [dict(zip(header, r)) for r in rd]
    if not (sess / "video.mp4").exists():
        sys.exit(f"{sess}: no video.mp4, skipping")
    cap = cv2.VideoCapture(str(sess / "video.mp4"))
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{sess.name}: {len(rows)} frames.csv rows, {n_video} video frames, drone calib = {args.drone}")
    if abs(len(rows) - n_video) > 2:
        print("WARNING: row/frame counts differ; frame<->row pairing may be off")

    fp = FilteredPositioner(args.drone, **({"map_path": args.map} if args.map else {}))
    detector = fp.positioner.detector
    mapped = set(fp.positioner.tag_map.poses)
    print("mapped tags:", sorted(mapped))

    writer = None
    out_rows = []
    tag_counts = {}
    t0 = float(rows[0]["time"])
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(rows) or (args.max_frames and i >= args.max_frames):
            break
        if i % args.stride:
            i += 1
            continue
        r = rows[i]
        t = float(r["time"]) - t0
        res = fp.locate(frame, t)
        raw = res["raw"]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        seen_all = [o for o in detector.detect(gray) if o.accepted]
        for o in seen_all:
            tag_counts[int(o.tag_id)] = tag_counts.get(int(o.tag_id), 0) + 1

        xyz = res["xyz"] or [None] * 3
        vel = res["vel"] or [None] * 3
        out_rows.append([r["frame_id"], r["time"], f"{t:.3f}",
                         " ".join(str(int(o.tag_id)) for o in seen_all),
                         " ".join(str(k) for k in (raw["tags"] if raw else [])),
                         *(raw["xyz"] if raw else ["", "", ""]), raw["range"] if raw else "",
                         *xyz, *vel, int(res["accepted"]), int(res["lost"]),
                         f"{res['fix_age']:.2f}" if res["fix_age"] != float("inf") else "",
                         r.get("h", ""), r.get("tof_m", ""), r.get("yaw", "")])

        if not args.no_video:
            for o in seen_all:
                col = (0, 220, 0) if int(o.tag_id) in mapped else (160, 160, 160)
                cv2.polylines(frame, [o.corners.round().astype(np.int32)], True, col, 2)
                cv2.putText(frame, str(int(o.tag_id)), tuple(o.corners[0].round().astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
            if res["xyz"] is not None:
                txt = "x %+.2f y %+.2f z %+.2f m%s" % (*res["xyz"], "  (LOST)" if res["lost"] else "")
            else:
                txt = "no position yet"
            cv2.putText(frame, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if res["lost"] else (0, 255, 0), 2)
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(str(sess / f"localization{tag}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         30, (w, h))
            writer.write(frame)
        i += 1
        if i % 200 == 0:
            print(f"  {i} frames ...", flush=True)
    if writer:
        writer.release()

    header = ["frame_id", "time", "t_rel", "tags_seen", "tags_used", "raw_x", "raw_y", "raw_z", "raw_range",
              "x", "y", "z", "vx", "vy", "vz", "accepted", "lost", "fix_age", "imu_h_cm", "tof_m", "imu_yaw"]
    with open(sess / f"localization{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(out_rows)

    # ---- summary ----
    n = len(out_rows)
    fixed = np.array([bool(r[4]) for r in out_rows])         # raw fix this frame
    lost = np.array([r[16] == 1 for r in out_rows])
    ts = np.array([float(r[2]) for r in out_rows])
    longest, run_start = 0.0, None
    for k in range(n):
        if not fixed[k]:
            run_start = ts[k] if run_start is None else run_start
            longest = max(longest, ts[k] - run_start)
        else:
            run_start = None
    print("\n=== localization summary ===")
    print(f"frames processed        {n} ({ts[-1]:.1f} s)" if n else "no frames")
    if n:
        print(f"frames with a map fix   {fixed.sum()} ({100 * fixed.mean():.1f}%)")
        print(f"frames flagged lost     {lost.sum()} ({100 * lost.mean():.1f}%)")
        print(f"longest gap w/o fix     {longest:.1f} s")
        print("tag sightings (id: frames)", dict(sorted(tag_counts.items())),
              "| mapped:", sorted(mapped))
        unmapped = sorted(set(tag_counts) - mapped)
        if unmapped:
            print(f"  seen but NOT in map.yaml: {unmapped} (can't be used for position)")
        xs = [(r[9], r[10], r[11]) for r in out_rows if r[9] is not None]
        if xs:
            a = np.array(xs, float)
            print(f"position range  x [{a[:, 0].min():+.2f}, {a[:, 0].max():+.2f}]  "
                  f"y [{a[:, 1].min():+.2f}, {a[:, 1].max():+.2f}]  z [{a[:, 2].min():+.2f}, {a[:, 2].max():+.2f}] m")
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
                sc = ax[0].scatter(a[:, 0], a[:, 2], c=np.arange(len(a)), s=6, cmap="viridis")
                ax[0].plot(0, 0, "r*", ms=14, label="tag 10 (origin)")
                for tid, p in fp.positioner.tag_map.poses.items():
                    t_ = np.asarray(p)[:3, 3]   # poses are 4x4 tag->origin matrices
                    ax[0].plot(t_[0], t_[2], "ks", ms=6)
                    ax[0].annotate(str(tid), (t_[0], t_[2]))
                ax[0].set_xlabel("x right of tag 10 (m)")
                ax[0].set_ylabel("z out of tag 10 face (m)")
                ax[0].set_title("Estimated path, top-down (color = time)")
                ax[0].axis("equal")
                ax[0].legend()
                fig.colorbar(sc, ax=ax[0], label="sample #")
                tt = [float(r[2]) for r in out_rows if r[9] is not None]
                ax[1].plot(tt, a[:, 1], label="y (up) from tag-10 frame")
                ax[1].set_xlabel("time (s)")
                ax[1].set_ylabel("y (m)")
                ax[1].set_title("Height-axis estimate")
                ax[1].legend()
                fig.tight_layout()
                fig.savefig(sess / f"localization{tag}.png", dpi=110)
            except Exception as e:  # plotting is a nicety
                print("plot skipped:", e)
    # ---- machine-readable summary (read by tag_coverage.py) ----
    windows, start, prev = [], None, None
    for r in out_rows:           # merge frames with any accepted tag into [start, end] windows (gap < 1 s)
        tr = float(r[2])
        if r[3]:
            if start is None:
                start = tr
            elif tr - prev > 1.0:
                windows.append([round(start, 1), round(prev, 1)])
                start = tr
            prev = tr
    if start is not None:
        windows.append([round(start, 1), round(prev, 1)])
    import json
    summary = dict(
        session=sess.name, drone_calib=args.drone, stride=args.stride, frames_in_csv=len(rows),
        frames_processed=n, duration_s=round(float(ts[-1]), 1) if n else 0,
        frames_with_any_tag=int(sum(1 for r in out_rows if r[3])),
        frames_with_map_fix=int(fixed.sum()) if n else 0,
        map_fix_pct=round(100 * float(fixed.mean()), 1) if n else 0,
        lost_pct=round(100 * float(lost.mean()), 1) if n else 0,
        longest_gap_without_fix_s=round(longest, 1),
        tag_frames={str(k): v for k, v in sorted(tag_counts.items())},
        mapped_tags=sorted(int(m) for m in mapped),
        seen_but_unmapped=sorted(set(tag_counts) - mapped),
        tag_windows_s=windows)
    (sess / f"localization_summary{tag}.json").write_text(json.dumps(summary, indent=2))

    print(f"\nwrote {sess}/localization.csv" + ("" if args.no_video else ", localization.mp4")
          + ", localization.png")


if __name__ == "__main__":
    main()
