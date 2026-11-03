"""Aryan's curriculum ladder for AS-Defend-Live-v0 (selected with ``env.curriculum_set=aryan``).

Every ``aryan_*`` policy in ``policies/`` was trained and scored on this table; see ``aryan_defend/docs/CURRICULUM.md``.
The default table (``CURRICULUM`` in ``tello_defend_env_cfg.py``) is untouched.

Stages 0-10  the ladder itself: stationary targets -> non-linear dodgers -> two attackers, ram take-downs ->
             max-aggression attackers (7) -> realistic delayed/noisy feed, 3-4 attackers, defender losses (8-10).
Stages 11-13 first bridge 7 -> 8 (own-state noise, then a mild feed, then the full feed). Collapsed at stage 12.
Stages 14-17 second bridge: feed lag 0.05 -> 0.3 s with attacker difficulty 0.7 -> 1.0 ramped alongside.
Physics: velocity model (fast) while the feed model is off, cascade firmware model once it is on.
Attackers are non-linear and dodge from stage 2 on (``attacker_difficulty`` >= 0.5).
"""

NOISE = dict(feed_model=True, own_pos_noise=0.02, own_vel_noise=0.05)
CLEAN = dict(feed_model=False, own_pos_noise=0.0, own_vel_noise=0.0)
ARYAN_CURRICULUM: dict[int, dict] = {
    # 0-1 stationary target, near then far, so the very first takedowns happen by chance
    0: dict(active_attackers=[1, 1], attacker_difficulty=-1.0, attacker_speed=[0.0, 0.0], spawn_radius=2.0,
            capture_mode="radius", capture_radius=0.8, defender_loss_prob=0.0, **CLEAN),
    1: dict(active_attackers=[1, 1], attacker_difficulty=-1.0, attacker_speed=[0.0, 0.0], spawn_radius=3.5,
            capture_mode="radius", capture_radius=0.7, defender_loss_prob=0.0, **CLEAN),
    # 2-3 one attacker, non-linear + dodging
    2: dict(active_attackers=[1, 1], attacker_difficulty=0.5, spawn_radius=4.5,
            capture_mode="radius", capture_radius=0.6, defender_loss_prob=0.0, **CLEAN),
    3: dict(active_attackers=[1, 2], attacker_difficulty=0.6, spawn_radius=5.0,
            capture_mode="radius", capture_radius=0.5, defender_loss_prob=0.0, **CLEAN),
    # 4-7 two attackers: coordination, then ram take-downs
    4: dict(active_attackers=[2, 2], attacker_difficulty=0.7, spawn_radius=6.0,
            capture_mode="radius", capture_radius=0.4, defender_loss_prob=0.0, **CLEAN),
    5: dict(active_attackers=[2, 2], attacker_difficulty=0.85, spawn_radius=6.0,
            capture_mode="radius", capture_radius=0.35, defender_loss_prob=0.0, **CLEAN),
    6: dict(active_attackers=[2, 2], attacker_difficulty=0.85, spawn_radius=6.0,
            capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0, **CLEAN),
    7: dict(active_attackers=[2, 2], attacker_difficulty=1.0, spawn_radius=6.0,
            capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0, **CLEAN),
    # 8-10 realism: delayed noisy feed + firmware model, more attackers, defender losses
    8: dict(active_attackers=[2, 3], attacker_difficulty=1.0, spawn_radius=6.0,
            capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0, **NOISE),
    9: dict(active_attackers=[2, 3], attacker_difficulty=1.0, spawn_radius=6.0,
            capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.2, **NOISE),
    10: dict(active_attackers=[2, 4], attacker_difficulty=1.0, spawn_radius=6.0,
             capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.3, **NOISE),
    # bridge stages 11-13 (local; run order 7 -> 11 -> 12 -> 13 -> 8 -> 9 -> 10): stage 7 -> 8 switches on feed lag, feed noise,
    # dropout, own-state noise and a 3rd attacker at once, which collapsed takedowns to ~0.2/ep; add them one at a time
    11: dict(active_attackers=[2, 2], attacker_difficulty=1.0, spawn_radius=6.0,
             capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0,
             feed_model=False, own_pos_noise=0.02, own_vel_noise=0.05),
    12: dict(active_attackers=[2, 2], attacker_difficulty=1.0, spawn_radius=6.0,
             capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0,
             feed_model=True, feed_delay_s=[0.0, 0.1], feed_noise=0.01, feed_dropout=0.03,
             own_pos_noise=0.02, own_vel_noise=0.05),
    13: dict(active_attackers=[2, 2], attacker_difficulty=1.0, spawn_radius=6.0,
             capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0, **NOISE),
    # feed-severity ramp 14-17 (2 attackers, difficulty 1.0): 11-13 jumped from a perfect feed to lag+noise+dropout and the
    # policies collapsed (5.9 -> 0.5 takedowns/ep). Lag is the killer (a 1-3 m/s dodger moves 0.3-1 m in 0.1-0.3 s), so
    # grow it 0.05 -> 0.1 -> 0.2 -> 0.3 s with noise/dropout alongside, and ease attacker aggression back in (difficulty 0.7 -> 0.8 -> 0.9 -> 1.0) so the two hardships don't land together; run order 7 -> 14 -> 15 -> 16 -> 17 -> 8 -> 9 -> 10
    14: dict(active_attackers=[2, 2], attacker_difficulty=0.7, spawn_radius=6.0,
             capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0,
             feed_model=True, feed_delay_s=[0.0, 0.05], feed_noise=0.005, feed_dropout=0.0,
             own_pos_noise=0.02, own_vel_noise=0.05),
    15: dict(active_attackers=[2, 2], attacker_difficulty=0.8, spawn_radius=6.0,
             capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0,
             feed_model=True, feed_delay_s=[0.0, 0.1], feed_noise=0.01, feed_dropout=0.03,
             own_pos_noise=0.02, own_vel_noise=0.05),
    16: dict(active_attackers=[2, 2], attacker_difficulty=0.9, spawn_radius=6.0,
             capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0,
             feed_model=True, feed_delay_s=[0.0, 0.2], feed_noise=0.02, feed_dropout=0.06,
             own_pos_noise=0.02, own_vel_noise=0.05),
    17: dict(active_attackers=[2, 2], attacker_difficulty=1.0, spawn_radius=6.0,
             capture_mode="ram", capture_radius=0.25, defender_loss_prob=0.0, **NOISE),
}
