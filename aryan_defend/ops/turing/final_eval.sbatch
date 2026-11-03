#!/bin/bash
# final_eval.sbatch -- stop rate of one checkpoint on chosen stages with a fresh seed (23), 1024 episodes each.
# stop rate = downed/ep / (downed/ep + breach/ep); appends one JSON line per stage to outputs/final_scores.jsonl
# usage: CK=<checkpoint.pt> TAG=<name> [STAGES="7 8 10"] [SEED=23] sbatch --export=ALL final_eval.sbatch
#SBATCH -A fyuan3
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -J ajs3_final
#SBATCH -o final_%x_%j.log
export LD_LIBRARY_PATH=$HOME/glu_env/lib:${LD_LIBRARY_PATH:-}
export TELLO_USD_PATH=$HOME/isaac/assets/tello.usd OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
source $HOME/isaac/tools/isaac_env.sh
cd ${REPO:-$HOME/Action_Space}/simulation_infra
SPEV="sp_capture_team=10.0 sp_capture_own=20.0 sp_cleared=30.0 sp_breach=-20.0 sp_crash=-5.0 sp_time=-0.25"
OUT=outputs/final_scores.jsonl
for S in ${STAGES:-7 8 10}; do
  for try in 1 2 3; do
    n0=$(wc -l < outputs/evaluate_results.jsonl 2>/dev/null || echo 0)
    python -u scripts/evaluate.py --task AS-Defend-Live-v0 --algorithm MAPPO --checkpoint $CK --num_envs 256 --episodes 1024 --seed ${SEED:-23} \
      --set "curriculum_set='aryan'" curriculum_stage=$S "reward_mode='sparse'" $SPEV > outputs/final_${TAG}_S${S}.log 2>&1
    n1=$(wc -l < outputs/evaluate_results.jsonl 2>/dev/null || echo 0)
    [ "$n1" -gt "$n0" ] && break
  done
  tail -1 outputs/evaluate_results.jsonl | python3 -c "
import sys,json
r=json.loads(sys.stdin.read()); d=r['downed_per_episode']; b=r['breach_rate']
print(json.dumps(dict(tag='$TAG', stage=$S, ckpt='$CK', downed_per_ep=d, breach_rate=b, stop_rate=d/(d+b) if d+b>0 else None, episodes=r['episodes'])))" | tee -a $OUT
done
