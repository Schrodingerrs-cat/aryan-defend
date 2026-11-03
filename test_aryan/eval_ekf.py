"""Offline check of pose_ekf.PoseEKF on a recorded session: jitter and gap bridging vs the existing Kalman / freezing.
  python3 eval_ekf.py data/<session> [--vg-scale 0.1]
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np
from pose_ekf import EkfCfg, PoseEKF

def rd(p):
    rows = list(csv.reader(open(p))); h, s = [], set()
    for c in rows[0]:
        h.append(c if c not in s else "motor_time"); s.add(c)
    return [dict(zip(h, r)) for r in rows[1:]]

def fl(v):
    try: return float(v)
    except (TypeError, ValueError): return None

def load(sess):
    loc = rd(sess / "localization" / "localization.csv"); fr = {r["frame_id"]: r for r in rd(sess / "frames.csv")}
    out = []
    for r in loc:
        f = fr.get(r["frame_id"])
        if f: out.append((r, f))
    return out

def run(sess, cfg, mask=None):
    data = load(sess); ekf = PoseEKF(cfg); res = []
    for r, f in data:
        t = float(r["time"]); x, z = fl(r["raw_x"]), fl(r["raw_z"])
        fix = (x, z) if x is not None and z is not None else None
        if mask and any(a <= t < b for a, b in mask): fix = None
        h, tof = fl(f["h"]) or 0, fl(f["tof"]) or 0
        vg = (fl(f["vgx"]) or 0, fl(f["vgy"]) or 0)
        o = ekf.step(t, fix, fl(r["raw_range"]), fl(r["heading_deg"]), fl(f["yaw"]), vg, vg_valid=(h >= 20 or tof >= 15))
        res.append((t, fix, o, fl(r["x"]), fl(r["z"]), r["lost"] == "1"))
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("session", type=Path); ap.add_argument("--vg-scale", type=float, default=0.1)
    a = ap.parse_args(); cfg = EkfCfg(vg_scale=a.vg_scale); res = run(a.session, cfg)
    T = np.array([r[0] for r in res])
    # jitter: step between consecutive frames that both have output
    def steps(xy):
        out = []
        for p, q in zip(xy, xy[1:]):
            if p is not None and q is not None: out.append(np.hypot(q[0]-p[0], q[1]-p[1]))
        return np.array(out)
    raw = [r[1] for r in res]; kf = [(r[3], r[4]) if r[3] is not None else None for r in res]
    ek = [(r[2]["x"], r[2]["z"]) if r[2] else None for r in res]
    for name, xy in (("raw tag fix", raw), ("current Kalman", kf), ("new filter", ek)):
        s = steps(xy); print(f"{name:15s} frame steps >30 cm: {(s>0.3).sum():4d}   >10 cm: {(s>0.1).sum():4d}   p95 {np.percentile(s,95)*100:5.1f} cm   n={len(s)}")
    # gap bridging: gaps with no raw fix, flying, ended by a fix
    print("\ngap bridging (position error vs the first fix after the gap; fix itself is ~10 cm noisy):")
    gaps = []; i = 0
    while i < len(res):
        if res[i][1] is None and i > 0 and res[i-1][1] is not None:
            j = i
            while j < len(res) and res[j][1] is None: j += 1
            if j < len(res):
                k = [res[m][1] for m in range(j, min(j+5, len(res))) if res[m][1] is not None]
                if k and res[i-1][2] and res[j-1][2]:
                    end = np.median(np.array(k), 0); prev = np.array(res[i-1][1])
                    pred = np.array([res[j-1][2]["x"], res[j-1][2]["z"]])
                    gaps.append((res[j-1][0]-res[i-1][0], np.hypot(*(pred-end)), np.hypot(*(prev-end)), res[j-1][2]["sigma"]))
            i = j
        i += 1
    g = np.array(gaps)
    for lo, hi in ((0.1, 0.5), (0.5, 1.5), (1.5, 3), (3, 8), (8, 999)):
        m = (g[:,0] >= lo) & (g[:,0] < hi)
        if m.sum(): print(f"  gap {lo:>4}-{hi:<4} s  n={m.sum():3d}   freeze error median {np.median(g[m,2])*100:5.0f} cm   new filter median {np.median(g[m,1])*100:5.0f} cm   (its sigma {np.median(g[m,3])*100:.0f} cm)")

    # artificial gaps: hide the tags for L seconds in the middle of continuous tracking, compare at the end of the gap
    print("\nartificial gaps (tags hidden for L s during flight; error at the gap end vs the real fixes right after):")
    base = run(a.session, cfg); Tb = np.array([r[0] for r in base]); fixes = [r[1] for r in base]
    for L in (1, 2, 3, 5):
        e_new, e_frz, sig = [], [], []
        t = Tb[0] + 150
        while t < Tb[-1] - L - 2:
            idx = np.where((Tb >= t - 1) & (Tb < t))[0]; after = np.where((Tb >= t + L) & (Tb < t + L + 0.5))[0]
            pre = [fixes[i] for i in idx if fixes[i] is not None]; post = [fixes[i] for i in after if fixes[i] is not None]
            if len(pre) >= 5 and len(post) >= 3:
                res = run(a.session, cfg, mask=[(t, t + L)])
                k = np.searchsorted(Tb, t + L) - 1
                end = np.median(np.array(post), 0); frz = np.median(np.array(pre), 0)
                o = res[k][2]
                e_new.append(np.hypot(o["x"] - end[0], o["z"] - end[1])); e_frz.append(np.hypot(*(frz - end))); sig.append(o["sigma"])
            t += 3
        if e_new: print(f"  L={L} s  n={len(e_new):2d}   freeze: median {np.median(e_frz)*100:4.0f} cm p90 {np.percentile(e_frz,90)*100:4.0f}   new filter: median {np.median(e_new)*100:4.0f} cm p90 {np.percentile(e_new,90)*100:4.0f}   (sigma {np.median(sig)*100:.0f} cm)")
    # lag: how far the filtered output is from the raw fixes it is built on
    d = [np.hypot(r[2]["x"] - r[1][0], r[2]["z"] - r[1][1]) for r in base if r[1] is not None and r[2]]
    print(f"\nfiltered vs raw fix distance when a fix exists: median {np.median(d)*100:.0f} cm, p90 {np.percentile(d,90)*100:.0f} cm")
