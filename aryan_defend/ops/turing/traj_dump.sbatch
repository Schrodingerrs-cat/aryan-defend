#!/bin/bash
# traj_dump.sbatch -- dump trajectories of a checkpoint for CPU rendering (aryan_defend/tools/render_videos.sh).
# usage: LIST=<file with lines "name checkpoint stage"> sbatch --export=ALL traj_dump.sbatch
#SBATCH -A fyuan3
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:30:00
#SBATCH -J traj_dump
#SBATCH -o traj_%x_%j.log
export LD_LIBRARY_PATH=$HOME/glu_env/lib:${LD_LIBRARY_PATH:-}
export TELLO_USD_PATH=$HOME/isaac/assets/tello.usd OMNI_KIT_ACCEPT_EULA=YES TERM=xterm
source $HOME/isaac/tools/isaac_env.sh
cd ${REPO:-$HOME/Action_Space}/simulation_infra
SPEV="sp_capture_team=10.0 sp_capture_own=20.0 sp_cleared=30.0 sp_breach=-20.0 sp_crash=-5.0 sp_time=-0.25"
mkdir -p outputs/traj
while read name ck st; do
  for try in 1 2 3; do
    python -u scripts/evaluate.py --task AS-Defend-Live-v0 --algorithm MAPPO --checkpoint $ck --num_envs 16 --episodes 16 --seed 5 \
      --traj outputs/traj/${name}.npz --traj_envs 4 --traj_seconds 25 \
      --set "curriculum_set='aryan'" curriculum_stage=$st "reward_mode='sparse'" $SPEV > outputs/traj/${name}.log 2>&1
    [ -f outputs/traj/${name}.npz ] && break   # Isaac segfaults in app.close() sometimes; the file is written before that
  done
done < "$LIST"
