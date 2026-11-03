#!/bin/bash
# Render trajectory dumps (aryan_defend/ops/turing/traj_dump.sbatch -> outputs/traj/*.npz) to top-down mp4s. CPU only (matplotlib + ffmpeg).
# usage: aryan_defend/tools/render_videos.sh <dir with .npz> <out dir>   (run from the repo root)
# blue = defenders with trails, red = live attackers, grey x = downed attacker, shaded disc = protected site.
set -euo pipefail
IN=$1; OUT=$2; mkdir -p "$OUT"
for f in "$IN"/*.npz; do
  n=$(basename "$f" .npz)
  python3 simulation_infra/scripts/render_defend_traj.py "$f" --envs 2 --out "$OUT/$n.mp4" 2>&1 | grep -v Axes3D | tail -1
done
