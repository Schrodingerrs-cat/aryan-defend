# aryan_defend_rl_sparse_s7_b3: MAPPO sparse, curriculum stage 7 (aryan run b3)

**Stage-7 policy (clean attacker feed), NOT yet a deploy policy for the realistic stages 8-10.** Made by Aryan on Turing.

| | |
|---|---|
| task | `AS-Defend-Live-v0` + `curriculum_stage=7`, `reward_mode=sparse` (2 attackers, difficulty 1.0, ram take-down 0.25 m, clean feed) |
| algorithm | skrl MAPPO, one policy per defender, [256,128,64] ELU, obs dim 53, act dim 4, `clip_actions=False`, lr 3e-4 |
| reward (NOT repo defaults) | +10 all per take-down, +20 extra for the taker, +30 wave cleared, -20 breach, -5 crash, -0.25/s while any attacker alive |
| training | curriculum ladder stages 0->7 (restart-from-checkpoint per stage, gate vs assign-lead), run dir `Action_Space_b3` on Turing |
| checkpoint | `best_agent.pt` (sha256 prefix `6bf1ccf158b02fc1`) |

## Fresh-seed eval (seed 23, ~1024 episodes, stop rate = downed/(downed+breach))
| stage | downed/ep | breach/ep | **stop rate** |
|---|---|---|---|
| 7 (clean feed) | 4.099 | 0.742 | **84.7%** |
| 8 (delayed noisy feed + 3 attackers) | ~0.2 | 1.0 | 13.4% (policy collapses: not usable) |

Re-export: `python3 drone_deployment_infra/export_policy.py best_agent.pt policy.npz --agent-yaml agent.yaml --env-yaml env.yaml --check`
