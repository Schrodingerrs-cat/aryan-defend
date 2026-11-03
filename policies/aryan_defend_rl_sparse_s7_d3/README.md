# aryan_defend_rl_sparse_s7_d3: MAPPO sparse, stage 7, long run (32000 steps/stage): the best stage-7 policy, clean feed only

Made by Aryan on Turing. Reward is NOT the repo default: +10 all per take-down, +20 extra for the taker, +30 wave cleared,
-20 breach, -5 crash, -0.25/s while any attacker is alive. Stage table: `env.curriculum_set=aryan`
(`curriculum_aryan.py`, docs in `aryan_defend/docs/CURRICULUM.md`).

| | |
|---|---|
| task | `AS-Defend-Live-v0`, skrl MAPPO, one [256,128,64] ELU net per defender, obs 53, act 4, `clip_actions=False` |
| checkpoint | `agent_32000.pt` (sha256 prefix `44eb368ed0efed56`) |

## Fresh-seed eval (seed 23, ~1024 episodes; stop rate = downed / (downed + breach))
| stage | downed/ep | breach/ep | **stop rate** |
|---|---|---|---|
| 7 (clean feed) | 6.35 | 0.47 | **93.1%** |
| 8 (delayed noisy feed + 3 attackers) | 0.25 | 1.00 | 20.1% (collapses) |

Re-export: `python3 drone_deployment_infra/export_policy.py agent_32000.pt policy.npz --agent-yaml agent.yaml --env-yaml env.yaml --check`
