# Lessons and what to try next

## What worked
- **Sparse reward + curriculum** reaches 90-93% stop rate on the hardest *clean-feed* stage (7) against max-aggression, dodging attackers, with no shaping.
- **Warm-starting each stage from the previous best checkpoint** and gating on a scripted baseline keeps every stage a small step.
- **`clip_actions=False`.** skrl's default `clip_actions=True` with a fixed LR of 3e-4 turned the weights NaN around iteration 110-140 in every early run
  (a drone state blow-up fed unbounded values into the net). Fixed by `clip_actions=False` plus the input clamp and `_blown_up()` env reset in `tello_defend_env.py`.
- **Gate relaxation.** On stages 3-4 `downed/ep` saturates (~8.2: the number of attackers that arrive), so a strict "beat the baseline" gate stalls a policy that is already ~98%.

## What did not
- **The realistic feed (stage 8+) is the wall.** Policies at 90% on stage 7 fall to 13-17% the moment the feed has lag, noise and dropout, and a 3rd attacker.
  Bridging in small steps (stages 11-17) lifts that to ~45-55%, but the 90% target is not reached on any realistic stage.
- Bridge 1 (11-13) jumped from a perfect feed to lag+noise+dropout in one step at stage 12: take-downs fell 5.9 -> 0.5/ep. Bridge 2 (14-17) ramped
  lag *and* attacker aggression together and trained 2x longer per rung: it passed every rung but still failed stage 8 (3rd attacker, full noise).
- Retrying the same stage-8 jump from the same checkpoint (k3, l3, d3) climbed only ~0.06-0.09 take-downs/ep per retry (600 iterations): do not just retry.

## Ideas, in the order I would try them
1. **Memory.** The policy is a feed-forward MLP over the alpha-beta estimate; with 0.3 s of lag the ram window (0.25 m) needs prediction. Try a recurrent
   policy (skrl RNN) or stacked past observations.
2. **Train on the target distribution directly** after stage 7: a randomized-severity feed (lag 0-0.3 s sampled per episode, already per-episode in `feed.py`)
   with most mass on the full feed, instead of a monotone ramp that overfits each rung.
3. **More attackers earlier.** Stage 8 adds a 3rd attacker *and* the full feed together; try a 3-attacker stage with the clean feed first.
4. **Longer training per rung** (d3's 32000-step stages gave the best stage-7 policy, 93.1% vs ~90%).
5. Check whether 90% is reachable at all: the best scripted policy gets 54-56% there, and some spawns may be unstoppable.
