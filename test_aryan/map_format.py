"""Write a tag map in exactly the layout of the repo's config/map.yaml (swarm_ops/map_tags.py output):
origin tag first with identity pose, block-style lists, key order origin_tag / frame / tags.

    python3 map_format.py maps/latest_layout.yaml          # rewrites in place (backup: .bak)
"""
import shutil
import sys
from pathlib import Path

import yaml

FRAME = "origin tag: x right, y up, z out of the tag face; meters"


def to_repo_format(path):
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    origin = int(data["origin_tag"])
    tags = {int(k): v for k, v in data["tags"].items()}
    tags.setdefault(origin, {"R": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "t": [0.0, 0.0, 0.0],
                             "seen": 0, "samples": 0})
    ordered = {origin: tags.pop(origin)}
    ordered.update(tags)
    out = {"origin_tag": origin, "frame": data.get("frame", FRAME),
           "tags": {k: {"R": [[float(x) for x in row] for row in v["R"]], "t": [float(x) for x in v["t"]],
                        "seen": int(v.get("seen", 0)), "samples": int(v.get("samples", 0))}
                    for k, v in ordered.items()}}
    shutil.copy(path, path.with_suffix(".yaml.bak"))
    path.write_text(yaml.safe_dump(out, default_flow_style=False, sort_keys=False))


if __name__ == "__main__":
    for p in sys.argv[1:]:
        to_repo_format(p)
