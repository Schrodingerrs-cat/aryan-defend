"""Horizontal pose filter: AprilTag position fixes + the Tello's own velocity (vgx/vgy) + IMU yaw.

State [x, z, vx, vz] in the tag-10 frame (x right of tag 10, z out of the wall), constant-velocity model.
  * predict:   x += v dt, so the position keeps moving through gaps with no tag in view (instead of freezing)
  * tag fix:   position measurement with noise growing with range; Mahalanobis gate rejects pose flips
               (a source that keeps disagreeing is believed again after `max_rejects`)
  * velocity:  Tello vgx/vgy are in a FIXED frame set at takeoff (x = nose direction at takeoff, y = right of it),
               units dm/s (checked on 20261024_094227: fits tag motion with R2 0.58, body frame fits 0.19). Rotate into
               the tag frame by psi0, the heading the drone had at IMU yaw 0, learned online from
               (tag heading - IMU yaw) whenever a tag fix is available.
Output carries a position std; `lost` when it exceeds `lost_sigma` (long gap: velocity integration drifts).
Height stays in height_fusion.HeightFilter.

    ekf = PoseEKF()
    out = ekf.step(t, fix_xz=(x, z) or None, fix_range=m or None, tag_heading_deg=deg or None,
                   imu_yaw_deg=deg, vg=(vgx, vgy) or None, vg_valid=True)
    out -> dict(x, z, vx, vz, sigma, mode, lost)   mode: "fix", "fix+vel", "vel", "predict", "rejected"
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass
class EkfCfg:
    accel_sigma: float = 1.5        # [m/s^2] how fast velocity may change
    fix_sigma0: float = 0.06        # [m] tag fix noise at zero range
    fix_sigma_per_m: float = 0.08   # [m] extra per metre of range
    vel_sigma: float = 0.15         # [m/s] vg measurement noise (integer dm/s quantisation + scale error)
    vg_scale: float = 0.1           # m/s per vg unit (dm/s)
    gate_chi2: float = 9.21         # 2 dof, 99%
    max_rejects: int = 6
    lost_sigma: float = 0.6         # [m] declare lost above this position std
    max_dt: float = 1.0
    psi_alpha: float = 0.03         # smoothing of the (tag heading - IMU yaw) offset
    vel_tau: float = 4.0            # [s] velocity decays with this time constant when nothing measures it
    max_speed: float = 2.0          # [m/s] clamp


class PoseEKF:
    def __init__(self, cfg=None):
        self.cfg = cfg or EkfCfg()
        self.s = None                       # [x, z, vx, vz]
        self.P = None
        self.t = None
        self.rejects = 0
        self.psi0 = None                    # radians, heading of the takeoff x axis in the tag frame
        self._ps = self._pc = None

    # --- yaw offset ---------------------------------------------------------------------------------
    def _learn_psi0(self, tag_heading_deg, imu_yaw_deg):
        if tag_heading_deg is None or imu_yaw_deg is None:
            return
        a = math.radians(tag_heading_deg - imu_yaw_deg)
        if self._ps is None:
            self._ps, self._pc = math.sin(a), math.cos(a)
        else:
            k = self.cfg.psi_alpha
            self._ps += k * (math.sin(a) - self._ps)
            self._pc += k * (math.cos(a) - self._pc)
        self.psi0 = math.atan2(self._ps, self._pc)

    def _vg_world(self, vg):
        """vgx/vgy (takeoff frame, dm/s) -> (vx, vz) m/s in the tag frame; None until psi0 is known."""
        if self.psi0 is None:
            return None
        f = (math.sin(self.psi0), -math.cos(self.psi0))        # takeoff nose direction in (x, z)
        r = (math.cos(self.psi0), math.sin(self.psi0))         # right of it
        k = self.cfg.vg_scale
        return np.array([k * (vg[0] * f[0] + vg[1] * r[0]), k * (vg[0] * f[1] + vg[1] * r[1])])

    # --- kalman -------------------------------------------------------------------------------------
    def _predict(self, dt):
        c = self.cfg
        dt = min(max(dt, 0.0), c.max_dt * 3)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        F[2, 2] = F[3, 3] = math.exp(-dt / c.vel_tau)      # no free-running velocity through long gaps
        q = c.accel_sigma ** 2
        Q = np.zeros((4, 4))
        for i in (0, 1):
            Q[i, i] = q * dt ** 4 / 4
            Q[i, i + 2] = Q[i + 2, i] = q * dt ** 3 / 2
            Q[i + 2, i + 2] = q * dt ** 2
        self.s = F @ self.s
        sp = math.hypot(self.s[2], self.s[3])
        if sp > c.max_speed:
            self.s[2:] *= c.max_speed / sp
        self.P = F @ self.P @ F.T + Q

    def _update(self, H, z, Rm, gate=True):
        S = H @ self.P @ H.T + Rm
        nu = z - H @ self.s
        if gate and float(nu @ np.linalg.solve(S, nu)) > self.cfg.gate_chi2:
            return False
        K = self.P @ H.T @ np.linalg.inv(S)
        self.s = self.s + K @ nu
        self.P = (np.eye(4) - K @ H) @ self.P
        return True

    def step(self, t, fix_xz=None, fix_range=None, tag_heading_deg=None, imu_yaw_deg=None, vg=None, vg_valid=True):
        c = self.cfg
        if fix_xz is not None:
            self._learn_psi0(tag_heading_deg, imu_yaw_deg)
        if self.s is None:
            if fix_xz is None:
                return None
            sig = c.fix_sigma0 + c.fix_sigma_per_m * (fix_range or 3.0)
            self.s = np.array([fix_xz[0], fix_xz[1], 0.0, 0.0])
            self.P = np.diag([sig ** 2, sig ** 2, 0.5, 0.5])
            self.t = t
            return self._out("fix")
        self._predict(t - self.t)
        self.t = t
        used = []
        H_v = np.array([[0, 0, 1, 0], [0, 0, 0, 1]], float)
        if vg is not None and vg_valid:
            v = self._vg_world(vg)
            if v is not None and self._update(H_v, v, np.eye(2) * c.vel_sigma ** 2):
                used.append("vel")
        if fix_xz is not None:
            sig = c.fix_sigma0 + c.fix_sigma_per_m * (fix_range or 3.0)
            H_p = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
            if self._update(H_p, np.asarray(fix_xz, float), np.eye(2) * sig ** 2):
                used.insert(0, "fix")
                self.rejects = 0
            else:
                self.rejects += 1
                if self.rejects >= c.max_rejects:           # persistent disagreement: believe the tags again
                    self.s = np.array([fix_xz[0], fix_xz[1], 0.0, 0.0])
                    self.P = np.diag([sig ** 2, sig ** 2, 0.5, 0.5])
                    self.rejects, used = 0, ["fix"]
                else:
                    used.append("rejected")
        return self._out("+".join(used) if used else "predict")

    def _out(self, mode):
        sigma = float(math.sqrt(max(self.P[0, 0], self.P[1, 1])))
        return dict(x=float(self.s[0]), z=float(self.s[1]), vx=float(self.s[2]), vz=float(self.s[3]),
                    sigma=sigma, mode=mode, lost=sigma > self.cfg.lost_sigma)
