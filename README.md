# counter-uas-rl-defender-policy

Counter-UAS reinforcement-learning defender policy: 3 defender drones vs 2+ attacker drones (sparse-reward curriculum MAPPO, Isaac Lab sim, DJI Tello).

Aryan's work on the Action Space anti-drone hackathon, task 3 (3 defender Tellos vs 2+ attackers): sparse-reward + curriculum MAPPO training on the
Turing cluster, the trained policies, results, videos, and the single-Tello localisation scripts (`test_aryan`).

This repo holds only **our** files. The simulator and deployment code live in the team's hackathon repo (`action-space-hackathon`); see "Using it" below.

| Folder | What |
|---|---|
| `aryan_defend/` | the main write-up: `README.md` (setup + results), `docs/` (CURRICULUM, RESULTS, LESSONS, RUNBOOK), `ops/turing/` (Slurm ladder / eval / trajectory scripts), `tools/`, `results/` (gates, stop rates, raw score logs), `videos/` |
| `policies/` | 7 trained defender policies in the hackathon repo's layout (checkpoint + `agent.yaml` + `env.yaml` + `policy.npz` + README) |
| `sim_patch/` | our changes to the simulator as a patch (`aryan-sim-changes.patch`: sparse reward knobs `sp_*`, NaN guards, `curriculum_set`) plus `curriculum_aryan.py` (the 18-stage table) |
| `test_aryan/` | single-Tello flight / recording / AprilTag localisation scripts, `data_small/` (small derived outputs), `videos/` (3 localisation / map videos), `DATA_MANIFEST.md` (what the 24 GB of raw recordings contained; they are not in git) |

## Headline result
Stop rate = downed / (downed + breach), fresh seed, ~1024 episodes. Target 90% on the hardest stages.

| Stage | Conditions | Best learned policy | Stop rate | Scripted best |
|---|---|---|---|---|
| 7 | clean feed, 2 attackers, max dodging | `aryan_defend_rl_sparse_s7_d3` | **93.1%** | 71% |
| 8 | delayed noisy feed, 3 attackers | `aryan_defend_rl_sparse_s9_g3` | 52.6% | 54% |
| 10 | 4 attackers, 30% defender losses | `aryan_defend_rl_sparse_s9_g3` | 44.5% | 56% |

Met on the clean-feed stage, not on the realistic stages. `aryan_defend/docs/LESSONS.md` says why and what to try next.

## Using it
```bash
git clone git@github.com:adityapat958/action-space-hackathon.git && cd action-space-hackathon
git checkout 8068f1c                                   # the main commit this was built on
git apply ../counter-uas-rl-defender-policy/sim_patch/aryan-sim-changes.patch   # adds sp_* knobs, NaN guards, curriculum_set=aryan (+ curriculum_aryan.py)
cp -r ../counter-uas-rl-defender-policy/aryan_defend ../counter-uas-rl-defender-policy/test_aryan .
cp -r ../counter-uas-rl-defender-policy/policies/aryan_* policies/
```
Then follow `aryan_defend/docs/RUNBOOK.md`. To deploy a policy: `policies/<name>/policy.npz` loads with `drone_deployment_infra/defend_live.py --policy rl`
(the `s7_*` policies assume a clean attacker feed; use `s9_g3` or `s17_o3` for the real tracker feed).

Reward used for every policy (not the repo defaults): +10 to all defenders per take-down, +20 extra to the one that made it, +30 wave cleared,
-20 breach, -5 own crash, -0.25 per second while any attacker is alive.
