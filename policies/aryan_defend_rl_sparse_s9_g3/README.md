# aryan_defend_rl_sparse_s9_g3: MAPPO sparse, trained through stage 9 via bridge 11-13: first policy that works on the realistic feed (~50%)

Made by Aryan on Turing. Reward is NOT the repo default: +10 all per take-down, +20 extra for the taker, +30 wave cleared,
-20 breach, -5 crash, -0.25/s while any attacker is alive. Stage table: `env.curriculum_set=aryan`
(`curriculum_aryan.py`, docs in `aryan_defend/docs/CURRICULUM.md`).

| | |
|---|---|
| task | `AS-Defend-Live-v0`, skrl MAPPO, one [256,128,64] ELU net per defender, obs 53, act 4, `clip_actions=False` |
| checkpoint | `best_agent.pt` (sha256 prefix `c551e7356917bbef`) |

## Fresh-seed eval (seed 23, ~1024 episodes; stop rate = downed / (downed + breach))
| stage | downed/ep | breach/ep | **stop rate** |
|---|---|---|---|
| 8 (feed + 3 attackers) | 1.10 | 0.99 | **52.6%** |
| 9 (+20% defender losses) | 1.02 | 1.00 | **50.7%** |
| 10 (4 attackers, 30% losses) | 0.80 | 1.00 | **44.5%** |

Re-export: `python3 drone_deployment_infra/export_policy.py best_agent.pt policy.npz --agent-yaml agent.yaml --env-yaml env.yaml --check`
