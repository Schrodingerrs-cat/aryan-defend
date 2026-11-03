#!/bin/bash
# Package one Turing checkpoint into policies/<name>/ in the repo's usual layout
# (agent/env yaml + checkpoint + policy.npz export + README). Run from the repo root on a machine with torch.
#   aryan_defend/tools/package_policy.sh <name> <turing_run_dir> <checkpoint_file> "<README table rows>" "<one-line description>"
# <turing_run_dir> = .../logs_synced/skrl/tello_defend/<timestamp>_mappo_torch_sparse_sN_cascade_tK ; TURING = user@host.
set -euo pipefail
NAME=$1; RUN=$2; CKPT=$3; ROWS=$4; DESC=$5
D=policies/$NAME; mkdir -p "$D"
scp -q "$TURING:$RUN/checkpoints/$CKPT" "$D/$CKPT"
scp -q "$TURING:$RUN/params/agent.yaml" "$D/agent.yaml"
scp -q "$TURING:$RUN/params/env.yaml" "$D/env.yaml"
python3 drone_deployment_infra/export_policy.py "$D/$CKPT" "$D/policy.npz" --agent-yaml "$D/agent.yaml" --env-yaml "$D/env.yaml" --check
SHA=$(sha256sum "$D/$CKPT" | cut -c1-16)
cat > "$D/README.md" <<EOT
# $NAME: $DESC

Made by Aryan on Turing. Reward is NOT the repo default: +10 all per take-down, +20 extra for the taker, +30 wave cleared,
-20 breach, -5 crash, -0.25/s while any attacker is alive. Stage table: \`env.curriculum_set=aryan\`
(\`curriculum_aryan.py\`, docs in \`aryan_defend/docs/CURRICULUM.md\`).

| | |
|---|---|
| task | \`AS-Defend-Live-v0\`, skrl MAPPO, one [256,128,64] ELU net per defender, obs 53, act 4, \`clip_actions=False\` |
| checkpoint | \`$CKPT\` (sha256 prefix \`$SHA\`) |

## Fresh-seed eval (seed 23, ~1024 episodes; stop rate = downed / (downed + breach))
$ROWS

Re-export: \`python3 drone_deployment_infra/export_policy.py $CKPT policy.npz --agent-yaml agent.yaml --env-yaml env.yaml --check\`
EOT
echo "packaged $D"
