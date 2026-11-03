"""check_runs.py -- audit the newest training run of each named run dir: reward numbers + clip_actions in the saved configs, NaN, downed/ep trend.

usage: python check_runs.py [run ...]     (RUNS_ROOT=<dir> if the Action_Space_<run> dirs are not in $HOME; needs the Isaac venv for tensorboard)
"""
import glob, math, os, re, sys, yaml, json
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
EXPECT_ENV = dict(reward_mode="sparse", sp_capture_team=10.0, sp_capture_own=20.0, sp_cleared=30.0, sp_breach=-20.0, sp_crash=-5.0, sp_time=-0.25)
bad = 0
ROOT = os.environ.get("RUNS_ROOT", "~")  # directory holding Action_Space_<run>/ (or <run>/ after reorganising)
for d in (sys.argv[1:] or ["b3", "c3", "e3", "f3", "d3"]):
    base = os.path.expanduser(f"{ROOT}/Action_Space_{d}/simulation_infra/")
    runs = sorted(glob.glob(base + "logs_synced/skrl/tello_defend/*_sparse_s*_cascade_t*"), key=os.path.getmtime)
    if not runs:
        print(f"{d}: no training run yet"); continue
    run = runs[-1]; name = os.path.basename(run)[-30:]
    out = [f"{d} {name}"]
    try:
        ag = yaml.unsafe_load(open(run + "/params/agent.yaml")); en = yaml.unsafe_load(open(run + "/params/env.yaml"))
        pol = ag["models"]["policy"]; a = ag["agent"]
        def g(o, *ks):
            for k in ks: o = o[k]
            return o
        out.append(f"clip_actions={pol['clip_actions']} lr={a['learning_rate']} sched={a['learning_rate_scheduler']} ent={a['entropy_loss_scale']} "
                   f"init_log_std={pol['initial_log_std']} min_log_std={pol['min_log_std']} iters_ts={ag['trainer']['timesteps']}")
        out.append(f"stage={en.get('curriculum_stage')} dyn={en.get('dynamics_model')} dec={en.get('decimation')} sim_dt={round(en['sim']['dt'],5)} "
                   f"spawn={en.get('spawn_radius')} att={en.get('active_attackers')} diff={en.get('attacker_difficulty')} mode={en.get('capture_mode')} feed={en.get('feed_model')}")
        wrong = {k: (en.get(k), v) for k, v in EXPECT_ENV.items() if en.get(k) != v}
        if pol["clip_actions"] is not False: wrong["clip_actions"] = pol["clip_actions"]
        if wrong: bad += 1; out.append(f"!!! CONFIG MISMATCH {wrong}")
        else: out.append("config OK (our rewards, clip_actions False)")
    except Exception as e:
        out.append(f"(params not readable yet: {type(e).__name__})")
    ev = glob.glob(run + "/events.out*")
    if ev:
        ea = EventAccumulator(ev[0], size_guidance={"scalars": 0}); ea.Reload()
        def ser(t):
            try: return [s.value for s in ea.Scalars(t)]
            except KeyError: return []
        std = ser("Policy / Standard deviation (defender_0)"); dn = [x for x in ser("Episode/downed") if not math.isnan(x)]
        nan = any(math.isnan(x) or math.isinf(x) for x in std + ser("Loss / Value loss (defender_0)"))
        if nan: bad += 1
        q = max(1, len(dn) // 5)
        out.append(f"iters={len(std)} NaN={'YES!!!' if nan else 'no'} std={std[-1] if std else float('nan'):.2f} downed/ep first={sum(dn[:q])/q if dn else 0:.2f} last={sum(dn[-q:])/q if dn else 0:.2f} max={max(dn) if dn else 0:.2f}")
    print(" | ".join(out))
    sc = base + "outputs/defend_sparse_scores.jsonl"
    if os.path.exists(sc):
        for l in open(sc):
            r = json.loads(l)
            if r.get("gate"): print(f"    GATE stage{r['stage']} try{r['try']}: policy {r['policy']:.2f} vs bar {r['assign_lead']:.2f} pass={r['pass']}")
print("PROBLEMS:", bad)
