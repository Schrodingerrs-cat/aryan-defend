"""Snap a built tag map to the known physical layout (removes the tilt/height error that chaining long-range tag links adds).

Tags lying on the floor get normal = up and centre height = floor level; tags on walls get a horizontal normal (the wall
direction you give) and stay upright; tags on the same wall share one distance to it. Positions along the floor/wall keep
the measured values. Prints how far each tag moved, and warns if a measured normal is > 15 deg from the one you said
(then the layout description is probably wrong for that tag).

  python3 regularize_map.py maps/live_20261024_094227.yaml --out maps/layout_0944.yaml \\
      --floor 1,2,4,7 --wall 5:+z,6:-z,9:-z,8:-x --same-plane 6,9 --add 8 --add-from maps/latest_layout.yaml.bak

Wall direction = the way the tag face points, in the tag-10 frame: +z = into the room (same as tag 10), -z = back
toward tag 10, +x / -x = right / left. Tags given with --add are copied unchanged from --add-from (use for tags the new
recording never linked; their pose is only as good as that other map).
"""
import argparse
from pathlib import Path

import numpy as np
import yaml

from map_format import to_repo_format

AXES = {"+z": (0, 0, 1), "-z": (0, 0, -1), "+x": (1, 0, 0), "-x": (-1, 0, 0)}


def load(path):
    d = yaml.safe_load(Path(path).read_text())
    return d, {int(k): (np.array(v["R"], float), np.array(v["t"], float), v) for k, v in d["tags"].items()}


def angle(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


def wall_R(normal):
    z = np.array(normal, float)
    y = np.array([0.0, 1.0, 0.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def floor_R(R):
    ux = R[:, 0].copy()
    ux[1] = 0.0
    ux /= np.linalg.norm(ux) or 1.0
    z = np.array([0.0, 1.0, 0.0])
    return np.column_stack([ux, np.cross(z, ux), z])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--floor", default="", help="comma list of tag ids lying on the floor")
    ap.add_argument("--wall", default="", help="id:dir,... e.g. 5:+z,6:-z,8:-x")
    ap.add_argument("--same-plane", default="", help="comma list of wall tags that share one wall (same coordinate)")
    ap.add_argument("--floor-y", type=float, default=-1.22)
    ap.add_argument("--add", default="", help="tags to copy from --add-from (not in the input map)")
    ap.add_argument("--add-from", type=Path, default=None)
    a = ap.parse_args()
    data, tags = load(a.map)
    origin = int(data["origin_tag"])
    floor = {int(t) for t in a.floor.split(",") if t}
    walls = {int(p.split(":")[0]): AXES[p.split(":")[1]] for p in a.wall.split(",") if p}
    out = {}
    for tid, (R, t, raw) in tags.items():
        R2, t2 = R.copy(), t.copy()
        if tid in floor:
            if angle(R[:, 2], (0, 1, 0)) > 15:
                print(f"  WARNING tag {tid}: measured normal {R[:, 2].round(2)} is not 'up'")
            R2, t2[1] = floor_R(R), a.floor_y
        elif tid in walls and tid != origin:
            n = walls[tid]
            if angle(R[:, 2], n) > 15:
                print(f"  WARNING tag {tid}: measured normal {R[:, 2].round(2)} is {angle(R[:, 2], n):.0f} deg from {n}")
            R2 = wall_R(n)
        out[tid] = [R2, t2, raw]
    same = [int(t) for t in a.same_plane.split(",") if t]
    if len(same) > 1:
        ax = int(np.argmax(np.abs(walls[same[0]])))          # the coordinate that is constant on that wall
        mean = float(np.mean([out[t][1][ax] for t in same]))
        for t in same:
            out[t][1][ax] = mean
    if a.add:
        _, src = load(a.add_from)
        for tid in (int(t) for t in a.add.split(",")):
            R, t, raw = src[tid]
            out[tid] = [R, t, dict(raw, seen=0, samples=0)]
            print(f"  tag {tid}: copied from {a.add_from} (NOT measured in this recording)")
    print("tag | moved (m) | height above floor (m)")
    for tid, (R2, t2, raw) in sorted(out.items()):
        moved = np.linalg.norm(t2 - tags[tid][1]) if tid in tags else float("nan")
        print(f"{tid:>3} | {moved:9.2f} | {t2[1] - a.floor_y:.2f}")
    doc = {"origin_tag": origin, "frame": data["frame"],
           "tags": {tid: {"R": R2.tolist(), "t": t2.tolist(), "seen": int(raw.get("seen", 0)),
                          "samples": int(raw.get("samples", 0))} for tid, (R2, t2, raw) in out.items()}}
    a.out.write_text(yaml.safe_dump(doc, default_flow_style=False, sort_keys=False))
    to_repo_format(a.out)
    a.out.with_suffix(".yaml.bak").unlink(missing_ok=True)
    print("->", a.out)


if __name__ == "__main__":
    main()
