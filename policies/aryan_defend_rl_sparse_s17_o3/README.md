# aryan_defend_rl_sparse_s17_o3: MAPPO sparse, trained through the feed-lag/aggression ramp (stages 14-17): best on the full feed with 2 attackers

Made by Aryan on Turing. Reward is NOT the repo default: +10 all per take-down, +20 extra for the taker, +30 wave cleared,
-20 breach, -5 crash, -0.25/s while any attacker is alive. Stage table: `env.curriculum_set=aryan`
(`curriculum_aryan.py`, docs in `aryan_defend/docs/CURRICULUM.md`).

| | |
|---|---|
| task | `AS-Defend-Live-v0`, skrl MAPPO, one [256,128,64] ELU net per defender, obs 53, act 4, `clip_actions=False` |
| checkpoint | `best_agent.pt` (sha256 prefix `c6db5fb66d4d3fe2`) |

## Fresh-seed eval (seed 23, ~1024 episodes; stop rate = downed / (downed + breach))
| stage | downed/ep | breach/ep | **stop rate** |
|---|---|---|---|
| 17 (full feed, 2 attackers, difficulty 1.0) | 1.19 | 0.99 | **54.7%** |
| 8 (feed + 3 attackers) | 0.78 | 0.99 | 44.1% |
| 10 (4 attackers, 30% losses) | 0.68 | 1.00 | 40.4% |

Re-export: `python3 drone_deployment_infra/export_policy.py best_agent.pt policy.npz --agent-yaml agent.yaml --env-yaml env.yaml --check`
