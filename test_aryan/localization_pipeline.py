"""AprilTag localization pipeline: recorded sessions -> tag map -> drone position, with a map view.

  # tomorrow, with new recordings (build a fresh map from them, then localize them):
  python3 localization_pipeline.py data/<new1> data/<new2> --build-map

  # use an existing map (default maps/latest_layout.yaml) on any recordings:
  python3 localization_pipeline.py data/<session>
  python3 localization_pipeline.py data/<a> data/<b> --map maps/my_map.yaml

  # build the map from some sessions, localize others:
  python3 localization_pipeline.py data/<run> --build-map --map-from data/<a> data/<b>

Steps (action-space-hackathon's own pipeline: tello_calibration detector + MapBuilder + FilteredPositioner):
  1. (--build-map) detect tags in --map-from sessions, chain them outward from tag 10 -> maps/<label>.yaml
  2. per session: detect tags every frame, position in the tag-10 frame, Kalman-filtered
  3. outputs in data/<session>/localization/:
       map_view.mp4     camera view (tags outlined) next to a live top-down map: tags, drone, heading,
                        field of view, trail, and lines to the tags it is using
       map_static.png   whole run: top-down + side view (height) + stats
       localization.csv per-frame position, velocity, heading, height, tags, lost flag, IMU h / ToF
       summary.json
     and maps/overview_<label>.png with the tag layout and every session's path.

Map frame: origin = centre of tag 10, x right, y up, z out of the tag-10 wall into the room (metres).
Top-down view: the wall is at the top; the room extends downward. Heading 0 deg = facing the wall,
positive = turned right. Floor is y = --floor-y (tag 10's centre is 1.22 m above the floor).
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from height_fusion import HeightCfg, HeightFilter, tof_to_height  # noqa: E402
DEFAULT_REPO = HERE.parent / "drone_deployment_infra"
BG, GRID, WALL = (40, 40, 40), (70, 70, 70), (200, 200, 200)
C_WALLTAG, C_FLOORTAG, C_ORIGIN = (230, 160, 60), (60, 160, 255), (60, 60, 255)  # BGR
C_FIX, C_LOST = (80, 220, 80), (60, 60, 230)


# ----------------------------------------------------------------------------- repo + io
def load_repo(repo):
    sys.path[:0] = [str(repo / "cal_and_map"), str(repo / "core")]
    from tello_calibration.calib.io import camera_from_dict, load_yaml
    from tello_calibration.tags.detect import TagDetector, load_tag_config
    from tello_calibration.tags.mapping import MapBuilder, TagMap
    from tello_calibration.utils.config import CONFIG_DIR
    return dict(camera_from_dict=camera_from_dict, load_yaml=load_yaml, TagDetector=TagDetector,
                load_tag_config=load_tag_config, MapBuilder=MapBuilder, TagMap=TagMap, CONFIG_DIR=CONFIG_DIR)


def read_frames_csv(sess):
    with open(sess / "frames.csv") as f:
        rd = csv.reader(f)
        header, seen = [], set()
        for col in next(rd):          # older sessions have two "time" columns (unix + Tello motor time)
            header.append(col if col not in seen else "motor_time")
            seen.add(col)
        return [dict(zip(header, r)) for r in rd]


# ----------------------------------------------------------------------------- map building
def build_map(R, sessions, cfg, K, dist, stride, label):
    detector = R["TagDetector"](cfg, K, dist)
    designated = {t for t, s in cfg.sizes_mm.items() if s}
    builder = R["MapBuilder"](cfg.origin_tag)
    for sess in sessions:
        cap = cv2.VideoCapture(str(sess / "video.mp4"))
        i = n = linked = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % stride == 0:
                obs = [o for o in detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                       if o.tag_id in designated]
                linked += sum(o.accepted for o in obs) >= 2
                builder.add_frame(obs)
                n += 1
            i += 1
        print(f"  map-from {sess.name}: {n} frames checked, {linked} with 2+ tags together", flush=True)
    tag_map = builder.build()
    out = HERE / "maps" / f"{label}.yaml"
    out.parent.mkdir(exist_ok=True)
    tag_map.save(out)
    from map_format import to_repo_format; to_repo_format(out)   # same layout as the repo config/map.yaml
    (HERE / "maps" / f"{label}.json").write_text(json.dumps(dict(
        built=datetime.now().isoformat(timespec="seconds"), sessions=[s.name for s in sessions],
        stride=stride, tags=tag_map.mapped(), samples={str(k): v for k, v in tag_map.samples.items()},
        not_linked=sorted(designated - set(tag_map.mapped()))), indent=2))
    print(f"  map tags {tag_map.mapped()}  (not linked: {sorted(designated - set(tag_map.mapped()))}) -> {out}")
    return out


# ----------------------------------------------------------------------------- geometry
def tag_shapes(tag_map, cfg, floor_y):
    """Top-down footprint of each mapped tag: polygon in (x, z), normal arrow, and kind."""
    shapes = []
    for tid, M in sorted(tag_map.poses.items()):
        p, Rm = M[:3, 3], M[:3, :3]
        size = (cfg.sizes_mm.get(tid) or 150.0) / 1000.0
        a = size / 2
        ux, uy, n = Rm[:, 0], Rm[:, 1], Rm[:, 2]
        floor_tag = abs(n[1]) > 0.7 and tid != cfg.origin_tag
        if floor_tag:   # lying flat: draw the square footprint
            poly = [p[[0, 2]] + sx * a * ux[[0, 2]] + sy * a * uy[[0, 2]]
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        else:           # on a wall/lectern: a short segment along the tag's x-axis
            d = ux[[0, 2]] / (np.linalg.norm(ux[[0, 2]]) or 1.0)
            poly = [p[[0, 2]] - d * a, p[[0, 2]] + d * a]
        nh = n[[0, 2]] / (np.linalg.norm(n[[0, 2]]) or 1.0)
        shapes.append(dict(id=int(tid), kind="origin" if tid == cfg.origin_tag else
                           "floor" if floor_tag else "wall", poly=np.array(poly), pos=p,
                           normal=None if floor_tag else nh))
    return shapes


def heading_deg(Rm):
    """0 = facing the tag-10 wall, positive = turned right. None if the camera looks mostly up/down."""
    f = Rm[:, 2]
    if np.hypot(f[0], f[2]) < 0.3:
        return None
    return float(np.degrees(np.arctan2(f[0], -f[2])))


class TopDown:
    """World (x, z) -> pixel for a fixed-size top-down panel (wall at the top)."""

    def __init__(self, xr, zr, w, h, pad=34):
        self.w, self.h, self.pad = w, h, pad
        self.s = min((w - 2 * pad) / (xr[1] - xr[0]), (h - 2 * pad) / (zr[1] - zr[0]))
        self.x0, self.z0 = xr[0], zr[0]

    def px(self, x, z):
        return int(round(self.pad + (x - self.x0) * self.s)), int(round(self.pad + (z - self.z0) * self.s))

    def background(self, shapes, xr, zr, floor_y):
        img = np.full((self.h, self.w, 3), BG, np.uint8)
        for gx in range(int(np.floor(xr[0])), int(np.ceil(xr[1])) + 1):
            cv2.line(img, self.px(gx, zr[0]), self.px(gx, zr[1]), GRID, 1)
            cv2.putText(img, f"{gx} m", (self.px(gx, zr[1])[0] - 12, self.h - 8), 0, 0.4, (150, 150, 150), 1)
        for gz in range(int(np.floor(zr[0])), int(np.ceil(zr[1])) + 1):
            cv2.line(img, self.px(xr[0], gz), self.px(xr[1], gz), GRID, 1)
            cv2.putText(img, f"{gz} m", (4, self.px(xr[0], gz)[1] - 3), 0, 0.4, (150, 150, 150), 1)
        cv2.line(img, self.px(xr[0], 0), self.px(xr[1], 0), WALL, 3)
        cv2.putText(img, "wall (tag 10 face)", (self.pad, 16), 0, 0.45, WALL, 1)
        for s in shapes:
            col = {"origin": C_ORIGIN, "floor": C_FLOORTAG, "wall": C_WALLTAG}[s["kind"]]
            pts = np.array([self.px(*q) for q in s["poly"]], np.int32)
            if s["kind"] == "floor":
                cv2.fillPoly(img, [pts], col)
            else:
                cv2.polylines(img, [pts], False, col, 5)
                if s["normal"] is not None:
                    c = s["pos"][[0, 2]]
                    cv2.arrowedLine(img, self.px(*c), self.px(*(c + 0.35 * s["normal"])), col, 1, tipLength=0.3)
            lx, ly = self.px(*s["pos"][[0, 2]])
            lab = f"{s['id']}" + (" origin" if s["kind"] == "origin" else "") + \
                  (f" ({s['pos'][1] - floor_y:.2f} m high)" if s["kind"] in ("wall", "origin") else "")
            cv2.putText(img, lab, (lx + 7, ly - 7), 0, 0.5, (255, 255, 255), 1)
        return img


# ----------------------------------------------------------------------------- localization pass
def localize(R, sess, fp, cfg, args, floor_y):
    rows = read_frames_csv(sess)
    cap = cv2.VideoCapture(str(sess / "video.mp4"))
    det = fp.positioner.detector
    orig, hold = det.detect, {}

    def cached(gray):                       # one detection per frame, shared with FilteredPositioner
        hold["obs"] = orig(gray)
        return hold["obs"]
    try:
        det.detect = cached
        shared = True
    except AttributeError:
        shared = False
    mapped = set(fp.positioner.tag_map.poses)
    t0 = float(rows[0]["time"])
    res_all, last_R, i = [], None, 0
    hcfg = HeightCfg(tof_min=args.tof_min, tof_max=args.tof_max, tof_offset=args.tof_offset)
    hf = None if args.no_height_fusion else HeightFilter(hcfg)
    while True:
        ok, frame = cap.read()
        if not ok or i >= len(rows) or (args.max_frames and i >= args.max_frames):
            break
        if i % args.stride:
            i += 1
            continue
        r, t = rows[i], float(rows[i]["time"]) - t0
        res = fp.locate(frame, t)
        obs = hold.get("obs", []) if shared else orig(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        raw = res["raw"]
        if raw:
            last_R = np.asarray(raw["R"], float)
        xyz, z_tag, tof_h, hmode, tof_trust, tof_bias = res["xyz"], None, None, "", "", None
        if hf is not None:
            def _f(k):
                try:
                    return float(r.get(k, ""))
                except ValueError:
                    return None
            if _f("tof") is not None:
                tof_h = tof_to_height({"tof": _f("tof"), "pitch": _f("pitch") or 0.0, "roll": _f("roll") or 0.0}, hcfg)
            z_tag = raw["xyz"][1] - floor_y if raw else None
            z_fused, hmode = hf.step(t, z_tag, raw["range"] if raw else None, tof_h)
            tof_trust, tof_bias = hf.tof_verdict, hf.tof_bias
            if xyz is not None and z_fused is not None:
                xyz = [xyz[0], z_fused + floor_y, xyz[2]]        # stable height replaces the jumpy tag height
        res_all.append(dict(
            i=i, row=r, t=t, xyz=xyz, xyz_kf=res["xyz"], z_tag=z_tag, tof_h=tof_h, hmode=hmode, tof_trust=tof_trust, tof_bias=tof_bias, vel=res["vel"], lost=res["lost"], accepted=res["accepted"],
            fix_age=res["fix_age"], raw=raw, used=list(raw["tags"]) if raw else [],
            heading=heading_deg(last_R) if last_R is not None else None, R=last_R,
            seen=[(int(o.tag_id), o.corners.copy(), int(o.tag_id) in mapped) for o in obs if o.accepted]))
        i += 1
        if len(res_all) % 300 == 0:
            print(f"    {sess.name}: {len(res_all)} frames", flush=True)
    return res_all


def write_csv(path, res, floor_y):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_id", "time", "t_rel", "tags_seen", "tags_used", "x", "y", "z", "vx", "vy", "vz",
                    "raw_x", "raw_y", "raw_z", "raw_range", "height_above_floor", "heading_deg", "accepted",
                    "lost", "fix_age", "imu_h_cm", "tof_m", "imu_yaw", "height_tag_raw", "tof_height", "height_mode", "tof_trust", "tof_tag_bias"])
        for q in res:
            xyz, vel, raw = q["xyz"] or [""] * 3, q["vel"] or [""] * 3, q["raw"]
            w.writerow([q["row"]["frame_id"], q["row"]["time"], f"{q['t']:.3f}",
                        " ".join(str(s[0]) for s in q["seen"]), " ".join(map(str, q["used"])), *xyz, *vel,
                        *(raw["xyz"] if raw else ["", "", ""]), raw["range"] if raw else "",
                        round(xyz[1] - floor_y, 3) if q["xyz"] else "",
                        "" if q["heading"] is None else round(q["heading"], 1), int(q["accepted"]), int(q["lost"]),
                        "" if q["fix_age"] == float("inf") else f"{q['fix_age']:.2f}",
                        q["row"].get("h", ""), q["row"].get("tof_m", ""), q["row"].get("yaw", ""),
                        "" if q.get("z_tag") is None else round(q["z_tag"], 3),
                        "" if q.get("tof_h") is None else round(q["tof_h"], 3), q.get("hmode", ""),
                        q.get("tof_trust", ""), "" if q.get("tof_bias") is None else round(q["tof_bias"], 3)])


# ----------------------------------------------------------------------------- rendering
def view_bounds(shapes, paths):
    xs = [s["pos"][0] for s in shapes]
    zs = [s["pos"][2] for s in shapes]
    for p in paths:
        if len(p):
            a = np.asarray(p)
            xs += [np.percentile(a[:, 0], 2), np.percentile(a[:, 0], 98)]
            zs += [np.percentile(a[:, 2], 2), np.percentile(a[:, 2], 98)]
    xr = (min(xs + [-0.5]) - 0.5, max(xs + [0.5]) + 0.5)
    zr = (min(-0.4, min(zs) - 0.3), max(zs + [1.0]) + 0.5)
    return xr, zr


def draw_panel(td, bg, shape_pos, q, trail, last_pos, floor_y, hfov_deg, H):
    """Live map panel: drone (green = fix, red hollow = lost), heading, FOV, trail, lines to used tags."""
    m = bg.copy()
    if last_pos is not None:
        for a, b in zip(trail[-150:], trail[-149:]):
            cv2.line(m, td.px(a[0], a[2]), td.px(b[0], b[2]), (200, 200, 90), 2)
        dx, dy = td.px(last_pos[0], last_pos[2])
        col = C_LOST if q["lost"] else C_FIX
        if not q["lost"]:
            for tid in q["used"]:
                if tid in shape_pos:
                    cv2.line(m, (dx, dy), td.px(shape_pos[tid][0], shape_pos[tid][2]), (120, 200, 120), 1)
        if q["heading"] is not None:
            th = np.radians(q["heading"])
            fx, fz = np.sin(th), -np.cos(th)           # forward in (x, z)
            ov = m.copy()
            L = 1.6 * td.s
            a1, a2 = th - np.radians(hfov_deg / 2), th + np.radians(hfov_deg / 2)
            wedge = np.array([(dx, dy), (dx + L * np.sin(a1), dy - L * np.cos(a1)),
                              (dx + L * np.sin(a2), dy - L * np.cos(a2))], np.int32)
            cv2.fillPoly(ov, [wedge], col)
            m = cv2.addWeighted(ov, 0.25, m, 0.75, 0)
            tip = (int(dx + 16 * fx), int(dy + 16 * fz))
            cv2.arrowedLine(m, (dx, dy), tip, col, 3, tipLength=0.5)
        cv2.circle(m, (dx, dy), 7, col, -1 if not q["lost"] else 2)
        hgt = last_pos[1] - floor_y
        hd = "--" if q["heading"] is None else f"{q['heading']:+.0f}"
        info = [f"x {last_pos[0]:+.2f}  z {last_pos[2]:+.2f} m", f"height {hgt:.2f} m   heading {hd} deg",
                ("LOST (holding last position)" if q["lost"] else f"fix via tags {q['used']}")]
        for k, line in enumerate(info):
            cv2.putText(m, line, (10, H - 74 + 18 * k), 0, 0.5, col if k == 2 else (240, 240, 240), 1)
    else:
        cv2.putText(m, "no position yet (no mapped tag seen)", (10, H - 20), 0, 0.5, C_LOST, 1)
    return m


def render_video(R, sess, res, shapes, out, args, floor_y, hfov_deg):
    W_CAM, H, W_MAP = 720, 540, 560
    xyzs = [q["xyz"] for q in res if q["xyz"]]
    xr, zr = view_bounds(shapes, [xyzs])
    td = TopDown(xr, zr, W_MAP, H)
    bg = td.background(shapes, xr, zr, floor_y)
    for a, b in zip(xyzs, xyzs[1:]):                      # whole run, faint
        cv2.line(bg, td.px(a[0], a[2]), td.px(b[0], b[2]), (75, 75, 75), 1)
    cap = cv2.VideoCapture(str(sess / "video.mp4"))
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 30 / args.stride, (W_CAM + W_MAP, H))
    by_i = {q["i"]: q for q in res}
    shape_pos = {s["id"]: s["pos"] for s in shapes}
    trail, last_pos, i = [], None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        q = by_i.get(i)
        i += 1
        if q is None:
            continue
        for tid, corners, is_mapped in q["seen"]:
            col = (0, 220, 0) if is_mapped else (170, 170, 170)
            cv2.polylines(frame, [corners.round().astype(np.int32)], True, col, 3)
            cv2.putText(frame, str(tid), tuple(corners[0].round().astype(int) + [0, -8]), 0, 0.9, col, 2)
        cam = cv2.resize(frame, (W_CAM, H))
        if q["xyz"]:
            last_pos = q["xyz"]
            trail.append(q["xyz"])
        m = draw_panel(td, bg, shape_pos, q, trail, last_pos, floor_y, hfov_deg, H)
        cv2.putText(m, f"t {q['t']:.1f}s", (W_MAP - 90, 18), 0, 0.5, (200, 200, 200), 1)
        wr.write(np.hstack([cam, m]))
    wr.release()


def render_static(sess_name, res, shapes, out, floor_y, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    pts = np.array([[q["t"], *q["xyz"]] for q in res if q["xyz"] and not q["lost"]])
    lost_pts = np.array([[*q["xyz"]] for q in res if q["xyz"] and q["lost"]])
    col = {"origin": "red", "floor": "darkorange", "wall": "royalblue"}
    for s in shapes:
        p = s["poly"]
        ax[0].plot(*zip(*np.vstack([p, p[:1]])) if len(p) > 2 else zip(*p), color=col[s["kind"]], lw=5 if len(p) == 2 else 2)
        ax[0].annotate(str(s["id"]), s["pos"][[0, 2]], xytext=(5, 5), textcoords="offset points", fontsize=11, weight="bold")
        ax[1].plot(s["pos"][2], s["pos"][1], "s", color=col[s["kind"]], ms=9)
        ax[1].annotate(str(s["id"]), (s["pos"][2], s["pos"][1]), xytext=(5, 5), textcoords="offset points")
    if len(lost_pts):
        ax[0].plot(lost_pts[:, 0], lost_pts[:, 2], ".", color="lightgray", ms=2, label="held (lost)")
    if len(pts):
        sc = ax[0].scatter(pts[:, 1], pts[:, 3], c=pts[:, 0], s=5, cmap="viridis", label="position fix")
        fig.colorbar(sc, ax=ax[0], label="time (s)", shrink=0.8)
        ax[1].scatter(pts[:, 3], pts[:, 2], c=pts[:, 0], s=5, cmap="viridis")
    ax[0].axhline(0, color="black", lw=3)
    ax[0].invert_yaxis()
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x  (right of tag 10, m)")
    ax[0].set_ylabel("z  (out from the wall, m)  - wall at top")
    ax[0].set_title(f"{sess_name}: top-down")
    ax[0].legend(loc="lower right")
    ax[1].axhline(floor_y, color="saddlebrown", lw=2)
    ax[1].text(0, floor_y + 0.05, "floor", color="saddlebrown")
    ax[1].axvline(0, color="black", lw=3)
    ax[1].set_xlabel("z  (out from the wall, m)")
    ax[1].set_ylabel("y  (up, m; tag 10 centre = 0)")
    ax[1].set_title("side view (height)")
    ax[1].set_aspect("equal")
    fig.suptitle(f"map fix {summary['map_fix_pct']}% of frames | lost {summary['lost_pct']}% | "
                 f"longest gap {summary['longest_gap_s']} s | tags seen {summary['tag_frames']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def height_stats(res, floor_y):
    """How much the height jumps frame to frame: raw tag height vs the height we output."""
    raw_h, out_h = [], []
    for q in res:
        if q["raw"] and q["xyz"]:
            raw_h.append(q["raw"]["xyz"][1] - floor_y)
            out_h.append(q["xyz"][1] - floor_y)
    out = dict(tof_valid_pct=round(100 * float(np.mean([q.get("tof_h") is not None for q in res])), 1) if res else 0)
    tv = [q for q in res if q.get("tof_h") is not None]
    out["tof_flagged_not_floor_pct"] = round(100 * float(np.mean([q.get("tof_trust") == "bad" for q in tv])), 1) if tv else 0
    biases = [q["tof_bias"] for q in res if q.get("tof_bias") is not None]
    out["tof_tag_bias_median_m"] = round(float(np.median(biases)), 3) if biases else None
    if len(raw_h) > 2:
        dr, do = np.abs(np.diff(raw_h)), np.abs(np.diff(out_h))
        out.update(jumps_over_30cm_raw=int((dr > 0.3).sum()), jumps_over_30cm_output=int((do > 0.3).sum()),
                   p95_step_cm_raw=round(100 * float(np.percentile(dr, 95)), 1),
                   p95_step_cm_output=round(100 * float(np.percentile(do, 95)), 1))
    return out


def summarize(res, tag_map):
    n = len(res)
    fixed = np.array([bool(q["raw"]) for q in res]) if n else np.array([])
    lost = np.array([q["lost"] for q in res]) if n else np.array([])
    ts = np.array([q["t"] for q in res]) if n else np.array([])
    longest, start = 0.0, None
    for k in range(n):
        if not fixed[k]:
            start = ts[k] if start is None else start
            longest = max(longest, ts[k] - start)
        else:
            start = None
    counts = {}
    for q in res:
        for tid, _, _ in q["seen"]:
            counts[str(tid)] = counts.get(str(tid), 0) + 1
    return dict(frames=n, duration_s=round(float(ts[-1]), 1) if n else 0,
                map_fix_pct=round(100 * float(fixed.mean()), 1) if n else 0,
                lost_pct=round(100 * float(lost.mean()), 1) if n else 0, longest_gap_s=round(longest, 1),
                tag_frames=dict(sorted(counts.items())),
                seen_but_unmapped=sorted(int(k) for k in counts if int(k) not in tag_map.poses))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", type=Path, help="session folders to localize (data/<ts>)")
    ap.add_argument("--build-map", action="store_true", help="build a new map first (from --map-from)")
    ap.add_argument("--map-from", nargs="*", type=Path, default=None, help="sessions for --build-map (default: the ones given)")
    ap.add_argument("--map", type=Path, default=HERE / "maps" / "latest_layout.yaml")
    ap.add_argument("--map-stride", type=int, default=2)
    ap.add_argument("--drone", default="tello_3", help="camera calibration (config/camera/<drone>.yaml)")
    ap.add_argument("--tags", type=Path, default=None, help="tag size config (default: repo tags.yaml)")
    ap.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    ap.add_argument("--floor-y", type=float, default=-1.22, help="floor height in the tag-10 frame (m)")
    ap.add_argument("--no-height-fusion", action="store_true", help="use the raw tag height (no ToF / smoothing)")
    ap.add_argument("--tof-min", type=float, default=0.12, help="ToF below this (m) = no valid reading")
    ap.add_argument("--tof-max", type=float, default=3.0)
    ap.add_argument("--tof-offset", type=float, default=0.0, help="metres added to the ToF height")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    R = load_repo(args.repo)
    cfg = R["load_tag_config"](args.tags) if args.tags else R["load_tag_config"]()
    cam = R["CONFIG_DIR"] / "camera" / f"{args.drone}.yaml"
    K, dist = R["camera_from_dict"](R["load_yaml"](cam if cam.exists() else R["CONFIG_DIR"] / "camera" / "default.yaml"))
    hfov = float(np.degrees(2 * np.arctan(480.0 / K[0, 0])))
    sessions = [s for s in args.sessions if (s / "video.mp4").exists()]
    for s in set(args.sessions) - set(sessions):
        print(f"skip {s}: no video.mp4")

    label = "map_" + datetime.now().strftime("%Y%m%d_%H%M%S") if args.build_map else args.map.stem
    if args.build_map:
        print("== 1. building tag map")
        map_path = build_map(R, args.map_from or sessions, cfg, K, dist, args.map_stride, label)
    else:
        map_path = args.map
        print(f"== 1. using existing map {map_path}")
    sys.path.insert(0, str(args.repo / "core"))
    from positional_kalman_filter import FilteredPositioner
    tag_map = R["TagMap"].load(map_path)
    shapes = tag_shapes(tag_map, cfg, args.floor_y)

    all_paths = {}
    for sess in sessions:
        print(f"== 2. localizing {sess.name}")
        fp = FilteredPositioner(args.drone, map_path=map_path)
        res = localize(R, sess, fp, cfg, args, args.floor_y)
        out = sess / "localization"
        out.mkdir(exist_ok=True)
        write_csv(out / "localization.csv", res, args.floor_y)
        summ = summarize(res, tag_map)
        summ["height"] = height_stats(res, args.floor_y)
        summ.update(session=sess.name, map=str(map_path), drone_calib=args.drone, floor_y=args.floor_y)
        (out / "summary.json").write_text(json.dumps(summ, indent=2))
        render_static(sess.name, res, shapes, out / "map_static.png", args.floor_y, summ)
        if not args.no_video:
            print(f"== 3. rendering map_view.mp4 for {sess.name}")
            render_video(R, sess, res, shapes, out / "map_view.mp4", args, args.floor_y, hfov)
        all_paths[sess.name] = [q["xyz"] for q in res if q["xyz"] and not q["lost"]]
        print(f"   {sess.name}: fix {summ['map_fix_pct']}%  lost {summ['lost_pct']}%  longest gap "
              f"{summ['longest_gap_s']} s  tags {summ['tag_frames']}"
              + (f"  UNMAPPED seen: {summ['seen_but_unmapped']}" if summ["seen_but_unmapped"] else ""))
        hs = summ["height"]
        if "jumps_over_30cm_raw" in hs:
            print(f"   height: ToF valid {hs['tof_valid_pct']}% of frames (flagged 'not the floor': "
                  f"{hs['tof_flagged_not_floor_pct']}% of those; median tag-ToF bias {hs['tof_tag_bias_median_m']} m) | frame-to-frame jumps >30 cm: raw tag "
                  f"{hs['jumps_over_30cm_raw']} -> output {hs['jumps_over_30cm_output']} | p95 step "
                  f"{hs['p95_step_cm_raw']} -> {hs['p95_step_cm_output']} cm")
        print(f"   -> {out}")

    # layout overview: tags + every session's path
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 8))
    col = {"origin": "red", "floor": "darkorange", "wall": "royalblue"}
    for s in shapes:
        p = s["poly"]
        ax.plot(*zip(*np.vstack([p, p[:1]])) if len(p) > 2 else zip(*p), color=col[s["kind"]], lw=6 if len(p) == 2 else 3)
        ax.annotate(f"tag {s['id']}  ({s['pos'][1] - args.floor_y:.2f} m above floor)", s["pos"][[0, 2]], xytext=(6, 6),
                    textcoords="offset points", weight="bold")
    for name, pth in all_paths.items():
        if pth:
            a = np.array(pth)
            ax.plot(a[:, 0], a[:, 2], ".", ms=2, alpha=0.5, label=name)
    ax.axhline(0, color="black", lw=3)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xlabel("x right of tag 10 (m)")
    ax.set_ylabel("z out from the wall (m)")
    ax.set_title(f"Tag layout ({map_path.name}) with drone paths\nblue = wall tag, orange = floor tag, red = origin")
    ax.legend(loc="lower right", fontsize=8, markerscale=6)
    fig.tight_layout()
    ov = HERE / "maps" / f"overview_{label}.png"
    fig.savefig(ov, dpi=110)
    print(f"== overview: {ov}")


if __name__ == "__main__":
    main()
