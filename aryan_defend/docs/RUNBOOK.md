# Runbook: train, evaluate, package (Turing / Slurm)

Assumes the Isaac Lab env under `~/isaac` and the repo checked out (editable install) in `$REPO` on Turing; account `fyuan3`.

1. **Train a ladder** (one run = one directory copy of the repo so locks and outputs do not collide):
   `ENT=0.05 ITERS=600 TRIES=3 GATE_FRAC=0.9 MODE=ladder START=0 END=10 REPO=$HOME/Action_Space_<run> sbatch -J <run> -p short -t 1-00:00:00 --gres=gpu:rtx_pro_6000_b:1 aryan_defend/ops/turing/ladder.sbatch`
   Bridge / resume: add `START=7 STAGE_LIST="7 14 15 16 17 8 9 10" CKPT=<.../best_agent.pt>`. Exploration variants: `ENT=0.1 EXTRA_AG="agent.models.policy.initial_log_std=1.0 agent.models.policy.min_log_std=-0.5"`.
   Partition `short` (up to 12 GPUs) started fastest; `quick` is 1 GPU/user. Logs go to node-local /tmp and are rsynced to `$REPO/simulation_infra/logs_synced`.
2. **Watch it**: `python aryan_defend/ops/turing/check_runs.py <run>...` (config check, NaN, trend) and `defend_sparse_scores.jsonl` in each run's `outputs/` (one `"gate":1` line per try).
3. **Score** (fresh seed, 1024 episodes): `CK=<ckpt> TAG=<name> STAGES="7 8 10" sbatch --export=ALL aryan_defend/ops/turing/final_eval.sbatch` -> `outputs/final_scores.jsonl`.
4. **Videos**: list `name checkpoint stage` lines in a file, `LIST=<file> sbatch --export=ALL aryan_defend/ops/turing/traj_dump.sbatch`, copy `outputs/traj/*.npz` here, `aryan_defend/tools/render_videos.sh <dir> aryan_defend/videos`.
5. **Package a policy** into `policies/`: `TURING=<user@host> aryan_defend/tools/package_policy.sh <name> <run dir> <ckpt file> "<README table rows>" "<description>"`.
6. **Deploy**: `policies/<name>/policy.npz` loads in `drone_deployment_infra/defend_live.py --policy rl` (see `drone_deployment_infra/README_RUN.md`). The `s7_*` policies were trained with a clean feed
   and are not suitable for the real (laggy, noisy) tracker feed; use `..._s9_g3` or `..._s17_o3`.

Known traps: Isaac sometimes segfaults in `simulation_app.close()` AFTER writing results (the scripts accept a result line as success); GPU nodes need `LD_LIBRARY_PATH=$HOME/glu_env/lib` (libGLU) or env creation hangs;
`evaluate.py --set` values are `ast.literal_eval`-ed (quote strings: `"curriculum_set='aryan'"`), and `None` cannot be passed through Hydra (use -1.0).
