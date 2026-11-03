# Results

Stop rate = downed/ep / (downed/ep + breach/ep). Episodes end on the first breach, so breach/ep is ~ the share of episodes that were breached.
Fresh seed 23, ~1024 episodes per cell, `final_eval.sbatch`. Raw data: `results/stop_rates.csv`, `results/gates.csv`, `results/raw/`.

## Stop rates by checkpoint

| checkpoint | eval stage | downed/ep | breach/ep | stop rate |
|---|---|---|---|---|
| `e3_s7` | 7 | 5.413 | 0.623 | 89.7% |
| `f3_s7` | 7 | 5.394 | 0.597 | 90.0% |
| `b3_s7` | 7 | 4.099 | 0.742 | 84.7% |
| `c3_s7` | 7 | 5.667 | 0.577 | 90.8% |
| `e3_s7` | 8 | 0.198 | 1.0 | 16.6% |
| `f3_s7` | 8 | 0.161 | 1.0 | 13.8% |
| `b3_s7` | 8 | 0.155 | 1.0 | 13.4% |
| `c3_s7` | 8 | 0.182 | 1.0 | 15.4% |
| `g3_s8` | 8 | 1.01 | 0.992 | 50.4% |
| `g3_s8` | 9 | 0.9 | 0.997 | 47.4% |
| `g3_s8` | 10 | 0.737 | 1.0 | 42.4% |
| `d3_s7` | 7 | 6.352 | 0.47 | 93.1% |
| `d3_s7` | 8 | 0.252 | 1.0 | 20.1% |
| `g3_s9` | 8 | 1.098 | 0.988 | 52.6% |
| `g3_s9` | 9 | 1.023 | 0.996 | 50.7% |
| `g3_s9` | 10 | 0.8 | 0.999 | 44.5% |
| `n3_s17` | 17 | 1.095 | 0.99 | 52.5% |
| `m3_s17` | 17 | 1.08 | 0.993 | 52.1% |
| `n3_s17` | 8 | 0.802 | 0.994 | 44.6% |
| `m3_s17` | 8 | 0.838 | 0.991 | 45.8% |
| `n3_s17` | 10 | 0.601 | 0.999 | 37.6% |
| `m3_s17` | 10 | 0.617 | 0.998 | 38.2% |
| `o3_s17` | 17 | 1.191 | 0.986 | 54.7% |
| `o3_s17` | 8 | 0.781 | 0.992 | 44.1% |
| `o3_s17` | 10 | 0.676 | 0.996 | 40.4% |

Checkpoint tag = `<run>_s<stage trained through>`; policy folders are `policies/aryan_defend_rl_sparse_s<stage>_<run>`.

## Scripted baselines (same harness, from earlier measurements)

| stage 7 | stage 8 | stage 10 |
|---|---|---|
| 71% (assign-lead-safe) | 54% (assign-lead) | 56% (assign-lead) |

## Runs

| run | recipe | reached |
|---|---|---|
| b3 c3 e3 f3 | ladder 0..7, entropy 0.02/0.02/0.05/0.1 | stage 7 (84.7-90.8%), failed stage 8 (13-17%) |
| d3 | ladder, 32000 steps per stage, entropy 0.02 | stage 7 (**93.1%**), failed stage 8 |
| g3 | stage-7 ckpt of b3 -> bridge 11,12,13 -> 8 -> 9 | passed 8 (try 2), stage 9 failed the gate 3x but best checkpoint is the s9 policy: 52.6/50.7/44.5% on 8/9/10 |
| h3 i3 j3 | same bridge, other seeds | failed stage 12 |
| k3 l3 | resumed e3/f3 at stage 8 | failed stage 8 3x (0.4-0.55 vs 1.10) |
| m3 n3 o3 | stage-7 ckpt -> 14,15,16,17 -> 8 | passed 14-17, failed stage 8 3x (0.75-0.91 vs 1.10); s17 policies 52-55% on stage 17 |

Gate = beat the scripted baseline's downed/ep on the same stage (relaxed to 0.85-0.9x because downed/ep saturates near the wave-arrival ceiling on the easy stages).

## Videos

See `aryan_defend/videos/README.md` (stage-7 policies b3 c3 e3 f3 d3, and the realistic-stage policies g3 and o3).
