# Videos (top-down, CPU-rendered from `evaluate.py --traj`; seed 5, 25 s, two episodes each)
Blue = defenders (with trails), red = live attackers, grey x = downed attacker, shaded disc = protected site.

| file | policy -> stage it is run on |
|---|---|
| `b3_stage7.mp4` `c3_stage7.mp4` `e3_stage7.mp4` `f3_stage7.mp4`, `all4_stage7.mp4` (side by side) | first ladders -> stage 7 (clean feed) |
| `d3_s7_on_s7.mp4` | long-run policy (93.1% stop rate) -> stage 7 |
| `g3_s9_on_s8.mp4`, `g3_s9_on_s10.mp4` | bridge-1 policy -> stage 8 (feed + 3 attackers) and stage 10 (4 attackers, losses) |
| `o3_s17_on_s17.mp4`, `o3_s17_on_s8.mp4` | bridge-2 policy -> stage 17 (full feed, 2 attackers) and stage 8 |
Regenerate: `aryan_defend/ops/turing/traj_dump.sbatch` then `aryan_defend/tools/render_videos.sh`.
