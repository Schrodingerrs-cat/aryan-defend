# ladder.sbatch -- Aryan's sparse-reward curriculum ladder for AS-Defend-Live-v0 on Turing (Slurm).
# Derived from upstream ops/defend_sparse_stage.sbatch. Differences: (1) our reward numbers are passed as env.sp_* overrides
# ($SP below, NOT the repo defaults), (2) the stage table is `env.curriculum_set=aryan` (18 stages, curriculum_aryan.py),
# (3) STAGE_LIST runs stages in any order (e.g. the bridge: "7 14 15 16 17 8 9 10"), (4) logs go to node-local /tmp and are rsynced
# to $REPO/simulation_infra/logs_synced (the networked home breaks TensorBoard writes), (5) a per-run lock dir stops duplicate copies,
# (6) skrl `clip_actions=False` (clip_actions=True + fixed LR gave NaN weights ~iteration 110-140 in every early run).
# Usage (see aryan_defend/docs/RUNBOOK.md):
#   ENT=0.05 ITERS=600 TRIES=3 GATE_FRAC=0.9 MODE=ladder START=0 END=10 REPO=$HOME/Action_Space_<run> sbatch -J <run> ladder.sbatch
#   resume / bridge: ... START=7 STAGE_LIST="7 14 15 16 17 8 9 10" CKPT=<best_agent.pt> EXTRA_AG="agent.models.policy.initial_log_std=1.0 ..."
#!/bin/bash
# ws-defend-sparse (session D, moth): AS-Defend-Live-v0, reward_mode=sparse, curriculum restart-from-checkpoint.
#   Throughput:  sbatch -J dsp_tput --export=ALL,MODE=tput,NE=2048,DYN=cascade ops/defend_sparse_stage.sbatch
#   Ladder:      sbatch -J dsp_ladder --export=ALL,MODE=ladder[,START=0,END=6,CKPT=...,ITERS=300] ops/defend_sparse_stage.sbatch
#                probes cascade@2048 (+1024) first, picks NE/DYN by rule (see pick()), then per stage: gate -> train -> eval,
#                advance on pass, one retry (ITERS more from the stage's own best) on fail, stop on 2nd fail.
#   One stage:   sbatch -J dsp_s0 --export=ALL,MODE=stage,S=0,ITERS=300[,CKPT=<prev best_agent.pt>][,NE=2048][,DYN=velocity] ops/defend_sparse_stage.sbatch
# Stage = gate baseline (assign-lead, same stage) -> train (fixed LR 3e-4, gamma 0.99, timeout bootstrap) -> eval best + last.
# Results: outputs/defend_sparse_scores.jsonl (one line per eval, tagged) + outputs/evaluate_results.jsonl (evaluate.py's own).
# Gate: policy downed/ep >= assign-lead downed/ep on the same stage. Evals always use cascade dynamics (deploy model).
#SBATCH -A fyuan3
#SBATCH -p short
#SBATCH --gres=gpu:rtx_pro_6000_b:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -o ladder_%x_%j.log   # relative to the submit dir; point at $REPO/simulation_infra/outputs/ if you prefer
set -u
export LD_LIBRARY_PATH=$HOME/glu_env/lib:${LD_LIBRARY_PATH:-}
export TELLO_USD_PATH=$HOME/isaac/assets/tello.usd
export ISAAC_ENV=$HOME/isaac/tools/isaac_env.sh
export PICK=${PICK:-c2048}
REPO=${REPO:-$HOME/Action_Space}
export ISAAC_V=5; source ${ISAAC_ENV:-$HOME/isaac/tools/isaac_env.sh}
unset CONDA_PREFIX CONDA_DEFAULT_ENV
export TELLO_USD_PATH=${TELLO_USD_PATH:-$HOME/isaac/assets/as_kit/tello.usd} OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
cd $REPO/simulation_infra && mkdir -p outputs
if [ "${MODE:-stage}" = ladder ]; then
  if ! mkdir outputs/dsp_ladder.lock 2>/dev/null; then echo "ladder already running: $(cat outputs/dsp_ladder.lock/job)"; exit 0; fi
  echo ${SLURM_JOB_ID:-$$} > outputs/dsp_ladder.lock/job
fi
LOCAL=/tmp/${USER}_${SLURM_JOB_ID:-x}; mkdir -p $LOCAL/logs $REPO/simulation_infra/logs_synced
[ -e logs ] && [ ! -L logs ] && mv logs logs_old_$(date +%s); rm -f logs; ln -s $LOCAL/logs logs
( while true; do sleep 60; rsync -a $LOCAL/logs/ $REPO/simulation_infra/logs_synced/ 2>/dev/null; done ) &
SYNCER=$!
trap 'kill $SYNCER; rsync -a $LOCAL/logs/ $REPO/simulation_infra/logs_synced/; rm -rf outputs/dsp_ladder.lock' EXIT
grep -q '"AS-Defend-Live-v0"' source/action_space_kit/action_space_kit/tasks/direct/tello_defend/__init__.py \
  || { echo "tello_defend lacks AS-Defend-Live-v0; sync it first"; exit 3; }
TASK=AS-Defend-Live-v0
MODE=${MODE:-stage}; NE=${NE:-2048}; DYN=${DYN:-cascade}; S=${S:-0}; ITERS=${ITERS:-300}; CKPT=${CKPT:-}
EVAL_EP=${EVAL_EP:-512}; EVAL_NE=${EVAL_NE:-256}; SEED=${SEED:-42}
OUT=outputs/defend_sparse_scores.jsonl
echo "node $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) MODE=$MODE S=$S NE=$NE DYN=$DYN ITERS=$ITERS CKPT=$CKPT start $(date +%T)"
run() { echo "== $1 $(date +%T)"; local t0=$(date +%s); shift; "$@"; local rc=$?; echo "== rc=$rc $(( $(date +%s)-t0 ))s $(date +%T)"; return $rc; }
AG="agent.models.policy.clip_actions=False agent.agent.learning_rate_scheduler=null agent.agent.learning_rate=3e-4 agent.agent.discount_factor=0.99 agent.agent.time_limit_bootstrap=True agent.agent.entropy_loss_scale=${ENT:-0.003} ${EXTRA_AG:-}"
SP="env.reward_mode=sparse env.sp_capture_team=10.0 env.sp_capture_own=20.0 env.sp_cleared=30.0 env.sp_breach=-20.0 env.sp_crash=-5.0 env.sp_time=-0.25"
SPEV="sp_capture_team=10.0 sp_capture_own=20.0 sp_cleared=30.0 sp_breach=-20.0 sp_crash=-5.0 sp_time=-0.25"

evretry() {  # run an eval; success = a NEW result line was written (Isaac often segfaults in simulation_app.close() afterwards)
  local n=$(wc -l < outputs/evaluate_results.jsonl 2>/dev/null || echo 0)
  for att in 1 2 3; do
    run "$@"
    local c=$(wc -l < outputs/evaluate_results.jsonl 2>/dev/null || echo 0)
    [ "$c" -gt "$n" ] && return 0
    echo "== no eval result (attempt $att), retrying"
  done
  return 1
}
ev() {  # ev TAG [evaluate.py args...]  -> runs evaluate, appends tagged summary line to $OUT
  local tag=$1; shift
  local before=$(wc -l < outputs/evaluate_results.jsonl 2>/dev/null || echo 0)
  evretry eval_$tag python -u scripts/evaluate.py --task $TASK --num_envs $EVAL_NE --episodes $EVAL_EP --seed 7 \
      --set "curriculum_set='aryan'" curriculum_stage=$S "reward_mode='sparse'" $SPEV "$@" || { echo "== STOP: eval $tag failed"; exit 4; }
  tail -n +$((before+1)) outputs/evaluate_results.jsonl 2>/dev/null | tail -1 | \
    python -c "import sys,json; [print(json.dumps(dict(ws='defend-sparse', stage=$S, tag='$tag', job='${SLURM_JOB_ID:-}', rec=json.loads(l)))) for l in sys.stdin]" >> $OUT
}

case "$MODE" in
tput)
  # 20 iterations x 16 rollouts = 320 policy steps per env; read the tqdm it/s near the end of the log
  run train_tput python -u scripts/skrl/train.py --task $TASK --algorithm MAPPO --headless --num_envs $NE --seed $SEED \
    --max_iterations 20 env.curriculum_set=aryan env.curriculum_stage=$S $SP $AG \
    agent.agent.experiment.experiment_name=sparse_tput_${DYN}_${NE} ;;
stage)
  [ "${SKIP_BASE:-0}" = 1 ] || ev assign-lead --baseline assign-lead
  RA=""; [ -n "$CKPT" ] && RA="--checkpoint $CKPT"
  NAME=sparse_s${S}_${DYN}
  run train python -u scripts/skrl/train.py --task $TASK --algorithm MAPPO --headless --num_envs $NE --seed $SEED \
    --max_iterations $ITERS $RA env.curriculum_set=aryan env.curriculum_stage=$S $SP $AG \
    agent.agent.experiment.experiment_name=$NAME
  RUN=$(ls -td logs/skrl/tello_defend/*_$NAME | head -1); echo "run dir $RUN"
  for ck in best_agent.pt $(ls $RUN/checkpoints | grep -v best | sort -V | tail -1); do
    ev "policy:$RUN/checkpoints/$ck" --algorithm MAPPO --checkpoint $RUN/checkpoints/$ck
  done ;;
ladder)
  # first ladder job to start wins; duplicates queued on other partitions exit (rm -r outputs/dsp_ladder.lock to reset)
  true
  START=${START:-0}; END=${END:-6}; MIN_IPS=${MIN_IPS:-2.7}   # 2.7 env-steps/s = 300 iters x 16 rollouts in <= 30 min
  ips() { python - "$1" <<'PY'
import re,sys
t=open(sys.argv[1],errors="ignore").read().replace("\r","\n")
v=[float(a) if u=="it/s" else 1/float(a) for a,u in re.findall(r"([0-9.]+)(it/s|s/it)",t)]
print(v[-1] if v else 0)
PY
  }
  probe() {  # probe NE DYN -> prints env-steps/s
    local lf=outputs/dsp_probe_${1}_${2}_${SLURM_JOB_ID:-x}.log
    python -u $REPO/ops/dsp_train.py --task AS-Defend-Live-S${START}-v0 --algorithm MAPPO --headless --num_envs $1 --seed $SEED --max_iterations 12 \
      env.reward_mode=sparse env.dynamics_model=$2 $AG agent.agent.experiment.experiment_name=sparse_tput_${2}_$1 > $lf 2>&1
    ips $lf
  }
  if [ -z "${PICK:-}" ]; then
    C2=$(probe 2048 cascade); echo "PROBE cascade 2048: $C2 steps/s ($(python -c "print(round(2048*$C2))") samples/s/agent)"
    [ "$C2" = 0 ] && { echo "== STOP: probe failed, see outputs/dsp_probe_2048_cascade_${SLURM_JOB_ID:-x}.log"; tail -30 outputs/dsp_probe_2048_cascade_${SLURM_JOB_ID:-x}.log; exit 3; }
    if python -c "exit(0 if $C2>=$MIN_IPS else 1)"; then PICK=c2048
    else C1=$(probe 1024 cascade); echo "PROBE cascade 1024: $C1 steps/s"
      # 1024 only helps if it is really faster; on A30 it isn't (1.95 vs 1.89 steps/s: cost = 37 substeps/step, not env count)
      if python -c "exit(0 if $C1>=$MIN_IPS else 1)"; then PICK=c1024; elif python -c "exit(0 if $C1>=1.3*$C2 else 1)"; then PICK=v012; else PICK=v012w; fi
    fi
  fi
  echo "PICK=$PICK (c2048: cascade 2048 all stages | c1024: cascade 1024 all | v012: velocity 2048 for S0-2, cascade 1024 S3+ | v012w: same but cascade 2048 S3+)"
  for S in ${STAGE_LIST:-$(seq $START $END)}; do
    case $PICK in c2048) NE=2048; DYN=cascade;; c4096) NE=4096; DYN=cascade;; c1024) NE=1024; DYN=cascade;;
                  v012) if [ $S -le 2 ]; then NE=2048; DYN=velocity; else NE=1024; DYN=cascade; fi;;
                  v012w) if [ $S -le 2 ]; then NE=2048; DYN=velocity; else NE=2048; DYN=cascade; fi;; esac
    # gate = max(assign-lead, assign-lead-safe): -safe avoids teammate collisions (jinx 2938b74), the stricter bar at S5+
    BASE=0; BL="assign-lead assign-lead-safe"
    [ $S = $START ] && [ -n "${BASE0:-}" ] && { BASE=$BASE0; BL=""; echo "gate baseline S$S reused: $BASE0"; }
    for b in $BL; do
      ev $b --baseline $b
      B=$(tail -1 $OUT | python -c "import sys,json; print(json.loads(sys.stdin.read())['rec']['downed_per_episode'])")
      BASE=$(python -c "print(max($BASE,$B))")
    done
    for try in $(seq 1 ${TRIES:-2}); do
      NAME=sparse_s${S}_${DYN}_t$try
      RA=""; [ -n "$CKPT" ] && RA="--checkpoint $CKPT"
      run train_s${S}_t$try python -u scripts/skrl/train.py --task $TASK --algorithm MAPPO --headless --num_envs $NE --seed $((SEED+try)) \
        --max_iterations $ITERS $RA env.curriculum_set=aryan env.curriculum_stage=$S $SP $AG \
        agent.agent.experiment.experiment_name=$NAME
      RUN=$(ls -td logs/skrl/tello_defend/*_$NAME | head -1); echo "run dir $RUN"
      BEST=""; BESTD=-1
      for ck in best_agent.pt $(ls $RUN/checkpoints | grep -v best | sort -V | tail -1); do
        ev "policy:$RUN/checkpoints/$ck" --algorithm MAPPO --checkpoint $RUN/checkpoints/$ck
        D=$(tail -1 $OUT | python -c "import sys,json; print(json.loads(sys.stdin.read())['rec']['downed_per_episode'])")
        if python -c "exit(0 if $D>$BESTD else 1)"; then BEST=$RUN/checkpoints/$ck; BESTD=$D; fi
      done
      PASS=$(python -c "print(int($BESTD>=${GATE_FRAC:-1.0}*$BASE))")
      echo "GATE S$S try$try: policy $BESTD vs assign-lead $BASE -> pass=$PASS ($BEST)"
      echo "{\"ws\":\"defend-sparse\",\"gate\":1,\"stage\":$S,\"try\":$try,\"policy\":$BESTD,\"assign_lead\":$BASE,\"pass\":$PASS,\"ckpt\":\"$PWD/$BEST\",\"ne\":$NE,\"dyn\":\"$DYN\",\"job\":\"${SLURM_JOB_ID:-}\"}" >> $OUT
      CKPT=$PWD/$BEST
      [ $PASS = 1 ] && break
    done
    [ $PASS = 1 ] || { echo "== STOP: S$S failed the gate twice; last ckpt $CKPT"; exit 42; }
  done ;;
*) echo "unknown MODE=$MODE"; exit 2 ;;
esac
echo "== ALLDONE $(date +%T)"
