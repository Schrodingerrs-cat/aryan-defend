# Curriculum (`env.curriculum_set=aryan`, `curriculum_aryan.py`)

Select with `env.curriculum_set=aryan env.curriculum_stage=N` (training) or `--set "curriculum_set='aryan'" curriculum_stage=N` (evaluate.py).
The default table is untouched. The physics model follows the feed: velocity model (120 Hz) while `feed_model` is off, cascade firmware model (500 Hz)
once it is on. Policy rate is ~13.4 Hz in both, so checkpoints carry across stages.

| Stage | Attackers | Difficulty | Take-down | Feed | Notes |
|---|---|---|---|---|---|
| 0 | 1, stationary, 2.0 m | none | radius 0.8 | clean | first take-downs happen by chance |
| 1 | 1, stationary, 3.5 m | none | radius 0.7 | clean | |
| 2 | 1 | 0.5 | radius 0.6 | clean | non-linear + dodging starts |
| 3 | 1-2 | 0.6 | radius 0.5 | clean | |
| 4 | 2 | 0.7 | radius 0.4 | clean | coordination |
| 5 | 2 | 0.85 | radius 0.35 | clean | |
| 6 | 2 | 0.85 | **ram** 0.25 | clean | |
| 7 | 2 | **1.0** | ram 0.25 | clean | max aggression, perfect information |
| 8 | 2-3 | 1.0 | ram | **lag 0-0.3 s, noise 3 cm, 10% dropout** + own-state noise | realistic |
| 9 | 2-3 | 1.0 | ram | as 8 | + 20% defender loss |
| 10 | 2-4 | 1.0 | ram | as 8 | + 30% defender loss |
| 11 | 2 | 1.0 | ram | clean + own-state noise | bridge 7->8, attempt 1 |
| 12 | 2 | 1.0 | ram | lag 0-0.1 s, 1 cm, 3% dropout | collapsed here (5.9 -> 0.5 take-downs/ep) |
| 13 | 2 | 1.0 | ram | full feed (as 8) | |
| 14 | 2 | 0.7 | ram | lag 0-0.05 s, 0.5 cm, no dropout | bridge attempt 2: lag and aggression ramped together |
| 15 | 2 | 0.8 | ram | lag 0-0.1 s, 1 cm, 3% | |
| 16 | 2 | 0.9 | ram | lag 0-0.2 s, 2 cm, 6% | |
| 17 | 2 | 1.0 | ram | full feed | |

Run orders used: plain ladder `0..10`; bridge 1 `7 11 12 13 8 9 10`; bridge 2 `7 14 15 16 17 8 9 10`.
Attacker difficulty (see `attackers.py`): 1.0 = 0.8-1.2 m/s, 80% reactive evasion, mostly weave/detour/dash; 0.7 = 0.6-1.0 m/s, 32% evasion.

## Reward (`env.reward_mode=sparse`, knobs `sp_*`; defaults reproduce the original hardcoded values)
`sp_capture_team=10` (every defender, per take-down) | `sp_capture_own=20` (the defender that made it) | `sp_cleared=30` (wave cleared)
| `sp_breach=-20` | `sp_crash=-5` | `sp_time=-0.25` per second while any attacker is alive.
