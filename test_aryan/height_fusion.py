"""Stable drone height: fuse the jumpy tag-derived height with the Tello's downward ToF sensor.

The tag pose gives height too (map y + floor offset), but it jumps tens of cm to ~1 m when a tag pose flips or the
range is long. The Tello's downward ToF (state field `tof`, cm) is accurate to a few cm at low altitude, but only
works in flight: on the ground / when idle it sits at ~10 cm. This is a 1-D constant-velocity Kalman filter
[height, vertical speed] that
  * predicts between frames, so height keeps updating while no tag is visible,
  * takes a ToF height (tilt-corrected, validity-gated) as a precise measurement when the sensor is valid,
  * takes the tag height as a coarse measurement (noise grows with the range to the tag),
  * rejects outlier measurements (Mahalanobis gate) - this is what stops the tag-pose flips; after
    `max_rejects` rejections in a row it accepts the source again (e.g. the ToF moved from floor to a table).
With no valid ToF it still works as an outlier-gated smoother on the tag height alone.

ToF measures the distance to whatever is below the drone (floor, table, hand), not necessarily the floor. So a
consistency monitor compares ToF with the tag height over the last few seconds: if their median difference exceeds
`tof_tag_tol_bad` the ToF is ignored (`tof_verdict == "bad"`) until they agree again within `tof_tag_tol_ok`.
`tof_bias` is that median difference (tag - ToF): over a flight on a clear floor it is a data-driven check of the
assumed floor level / tag-10 height.

    hf = HeightFilter()
    z, mode = hf.step(t, z_tag=tag_height_or_None, tag_range=metres_or_None, tof_height=tof_to_height(state))
mode: "tof+tag", "tof", "tag", "predict" (what was accepted this step).

Heights are metres above the floor. ToF is measured from the drone's underside; set HeightCfg.tof_offset if the
camera-centre reference matters (default 0).
"""
from collections import deque
from dataclasses import dataclass
import math
import statistics


@dataclass
class HeightCfg:
    accel_sigma: float = 1.0        # [m/s^2] vertical acceleration noise (how fast height may change)
    tag_sigma0: float = 0.12        # [m] tag-height noise at zero range
    tag_sigma_per_m: float = 0.08   # [m] extra noise per metre of range to the tag
    tof_sigma0: float = 0.03        # [m]
    tof_sigma_per_m: float = 0.02   # [m] per metre of ToF reading
    tof_min: float = 0.12           # [m] at/below this the Tello is reporting its "no reading / on ground" 10 cm
    tof_max: float = 3.0            # [m] ToF is unreliable beyond this
    max_tilt_deg: float = 25.0      # ignore ToF when pitched/rolled more than this
    tof_offset: float = 0.0         # [m] added to the ToF height
    gate_chi2: float = 10.83        # reject if innovation^2 / S exceeds this (1 dof, 99.9%)
    max_rejects: int = 5            # this many rejections in a row from one source: accept it again
    max_dt: float = 1.0             # [s] bigger gaps just inflate the uncertainty
    tof_tag_tol_bad: float = 0.30   # [m] median |tag - ToF| above this: ToF is probably not looking at the floor
    tof_tag_tol_ok: float = 0.20    # [m] ...and trusted again below this (hysteresis)
    check_window: float = 4.0       # [s] window for that median
    check_min: int = 8              # samples needed before judging


def tof_to_height(state, cfg=None, max_age_ok=True):
    """Tilt-corrected ToF height [m] from a Tello state dict ({'tof': cm, 'pitch': deg, 'roll': deg}), or None."""
    cfg = cfg or HeightCfg()
    if not state or not max_age_ok:
        return None
    tof = state.get("tof")
    if tof is None:
        return None
    d = float(tof) / 100.0
    if not (cfg.tof_min < d < cfg.tof_max):
        return None
    pitch, roll = float(state.get("pitch", 0.0)), float(state.get("roll", 0.0))
    if abs(pitch) > cfg.max_tilt_deg or abs(roll) > cfg.max_tilt_deg:
        return None
    return d * math.cos(math.radians(pitch)) * math.cos(math.radians(roll)) + cfg.tof_offset


class HeightFilter:
    def __init__(self, cfg=None):
        self.cfg = cfg or HeightCfg()
        self.z = None
        self.v = 0.0
        self.P = [0.0, 0.0, 0.0]          # P00, P01, P11
        self.t = None
        self.rejects = {"tof": 0, "tag": 0}
        self.n_rejected = {"tof": 0, "tag": 0}
        self.pairs = deque()               # (t, z_tag - z_tof) while both are available
        self.tof_verdict = "unknown"       # "unknown" (not enough tag overlap yet: ToF is used), "ok", "bad"
        self.tof_bias = None               # median(z_tag - z_tof) over the window, metres

    def _predict(self, dt):
        c = self.cfg
        dt = max(0.0, dt)
        a2 = c.accel_sigma ** 2
        p00, p01, p11 = self.P
        self.z += self.v * dt
        self.P = [p00 + 2 * dt * p01 + dt * dt * p11 + a2 * dt ** 4 / 4,
                  p01 + dt * p11 + a2 * dt ** 3 / 2,
                  p11 + a2 * dt * dt]
        if dt > c.max_dt:                 # long gap: don't trust the old estimate
            self.P[0] += (dt - c.max_dt) * 0.5

    def _update(self, zm, sigma, key):
        p00, p01, p11 = self.P
        s = p00 + sigma ** 2
        nu = zm - self.z
        if nu * nu / s > self.cfg.gate_chi2:
            self.rejects[key] += 1
            self.n_rejected[key] += 1
            if self.rejects[key] < self.cfg.max_rejects:
                return False
            self.z, self.v, self.P = zm, 0.0, [sigma ** 2, 0.0, 0.25]   # persistent: believe it again
            self.rejects[key] = 0
            return True
        self.rejects[key] = 0
        k0, k1 = p00 / s, p01 / s
        self.z += k0 * nu
        self.v += k1 * nu
        self.P = [p00 - p00 * p00 / s, p01 - p00 * p01 / s, p11 - p01 * p01 / s]
        return True

    def _tof_allowed(self, t, z_tag, tof_height):
        c = self.cfg
        if z_tag is not None and tof_height is not None:
            self.pairs.append((t, z_tag - tof_height))
        while self.pairs and t - self.pairs[0][0] > c.check_window:
            self.pairs.popleft()
        if len(self.pairs) >= c.check_min:
            self.tof_bias = statistics.median(d for _, d in self.pairs)
            if abs(self.tof_bias) > c.tof_tag_tol_bad:
                self.tof_verdict = "bad"
            elif abs(self.tof_bias) < c.tof_tag_tol_ok:
                self.tof_verdict = "ok"
        return self.tof_verdict != "bad"

    def step(self, t, z_tag=None, tag_range=None, tof_height=None):
        c = self.cfg
        if not self._tof_allowed(t, z_tag, tof_height):
            tof_height = None                           # sensor is looking at something that isn't the floor
        tof_sigma = None if tof_height is None else c.tof_sigma0 + c.tof_sigma_per_m * tof_height
        tag_sigma = None if z_tag is None else c.tag_sigma0 + c.tag_sigma_per_m * (tag_range or 2.0)
        if self.z is None:                              # initialise from the best measurement
            if tof_height is not None:
                self.z, self.P = tof_height, [tof_sigma ** 2, 0.0, 0.25]
            elif z_tag is not None:
                self.z, self.P = z_tag, [tag_sigma ** 2, 0.0, 0.25]
            else:
                return None, "none"
            self.t = t
            return self.z, "tof" if tof_height is not None else "tag"
        self._predict(t - self.t)
        self.t = t
        used = []
        if tof_height is not None and self._update(tof_height, tof_sigma, "tof"):
            used.append("tof")
        if z_tag is not None and self._update(z_tag, tag_sigma, "tag"):
            used.append("tag")
        return self.z, "+".join(used) if used else "predict"
