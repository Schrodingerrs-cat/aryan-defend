# aryan_defend: sparse-reward + curriculum MAPPO defenders (3 defenders vs 2-4 attackers)

Everything Aryan built for the defend task (hackathon task 3): reward design, an 18-stage curriculum, the Turing (Slurm) launch/eval
scripts, trained policies, results and videos. The rest of this branch is `main` unchanged; only the items below were added.

| What | Where |
|---|---|
| Reward knobs (`sp_*`), NaN guards, `curriculum_set=aryan` | `simulation_infra/source/.../tasks/direct/tello_defend/` (`tello_defend_env.py`, `tello_defend_env_cfg.py`, new `curriculum_aryan.py`) |
| Trained policies (repo layout: ckpt + yamls + `policy.npz` + README) | `policies/aryan_defend_rl_sparse_*` (8 folders) |
| Turing scripts: ladder, stop-rate eval, trajectory dump, run checker | `aryan_defend/ops/turing/` |
| Packaging + video tools | `aryan_defend/tools/` |
| Results: every gate (101 rows), every stop-rate eval (25 rows), raw score logs | `aryan_defend/results/` |
| Videos (top-down animations of the stage-7 policies) | `aryan_defend/videos/` |
| Docs | `aryan_defend/docs/` : `CURRICULUM.md`, `RESULTS.md`, `LESSONS.md`, `RUNBOOK.md` |
| Aryan's single-Tello scripts and data (restored from before main dropped it) | `test_aryan/` (`DATA_MANIFEST.md` says what was too big for git) |

## The setup in one paragraph
Task `AS-Defend-Live-v0`, skrl MAPPO (one [256,128,64] ELU net per defender), **sparse reward that is not the repo default**:
+10 to every defender per take-down, +20 extra to the one that made it, +30 when a wave is cleared, -20 per breach, -5 per own crash,
-0.25 per second while any attacker is alive. Attackers are non-linear and dodging (weave, detour, loiter-then-dash, reactive sidestep) from stage 2 on.
Training climbs a curriculum ladder (restart each stage from the previous best checkpoint, advance when it beats the scripted `assign-lead` baseline).

## Result (honest summary)
Stop rate = downed / (downed + breach), fresh seed 23, ~1024 episodes. Target was 90% on the hard, realistic stages.

| Stage | Conditions | Best learned | Scripted best |
|---|---|---|---|
| 7 | clean feed, 2 attackers, max dodging | **93.1%** (`..._s7_d3`) | 71% |
| 8 | delayed noisy feed, 3 attackers | **52.6%** (`..._s9_g3`) | 54% |
| 10 | 4 attackers, 30% defender losses | **44.5%** (`..._s9_g3`) | 56% |

So the target is met on the clean-feed stage and **not** met on the realistic stages: the learned policies match the scripted baseline on stage 8 and are below it on stage 10.
`docs/RESULTS.md` has every number, `docs/LESSONS.md` says why and what to try next.
