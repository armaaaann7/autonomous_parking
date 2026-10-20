# Review of the original blueprint, and what changed while building it

This project started from `autonomous-parking-project-blueprint.md`: a research plan comparing
four parking controllers (rule-based, fuzzy-only, PPO-only, PPO+fuzzy-shield) in a 2D simulation.
The timeline was dropped per request; everything else here is about correctness.

## 1. Bugs found in the blueprint itself (before any code was written)

| # | Location | Issue | Fix |
|---|----------|-------|-----|
| 1 | Part 2 vs Part 32 | Action space defined two different ways: `[steering, throttle, brake]` (3-dim) vs `[steer, accel]` (2-dim) | Used the blueprint's own Part 3 recommendation: 2-dim `[steer, throttle_brake]`, sign of the second value selects throttle vs brake |
| 2 | Part 16 Step 4 skeleton | Observation hard-codes steering angle to `0.0` | Steering is a real, rate-limited actuator state (`vehicle.py`) |
| 3 | Part 16 Step 4 skeleton | Collision detection is a `return False` stub | Implemented SAT (oriented box vs obstacle AABBs), cross-checked against shapely in `tests/test_geometry.py` |
| 4 | Part 10 reward sketch | Progress shaping computes `curr - prev` while the surrounding code already negates it, double-negating and rewarding moving *away* from the goal | Reward uses `prev - curr` directly (= "improvement"), sign-tested |
| 5 | Part 10 reward numbers | Collision penalty (-2) smaller than timeout penalty (-20) in one draft, which pays a fast approach to crash rather than time out | `configs/reward_config.yaml`: collision (20) > timeout (10) |
| 6 | Part 10 success check | Checks the tolerance box on a single step | Requires holding the tolerance for `hold_steps` consecutive steps |
| 7 | Fuzzy rule base (Part 4) | No rule covers `d_obs=medium AND speed=medium` (or `d_obs=far AND speed∈{medium,fast}` for `speed_adj`) — skfuzzy raises when a consequent's aggregated membership is exactly zero | Added the missing rules (`fuzzy/fuzzy_system.py` rules B6, S3, S5); verified with a full 1,500-point grid sweep in `tests/test_fuzzy_and_shield.py` |

## 2. Bugs found while building (these only showed up when the code actually ran)

### 2.1 Fuzzy shield fed the wrong angle (semantic bug, not caught by any unit test)
The fuzzy rule base's `angle_err` input was designed for **obstacle avoidance** (bearing to the
nearest obstacle, clockwise-positive — `large_left → steer large_right` means "steer away from an
obstacle on your left"). An early version of `ParkingEnv.fuzzy_inputs()` instead computed the
bearing to the **slot**, in the opposite sign convention. The result: the "safety shield" was
constantly issuing navigation-toward-slot corrections (not avoidance nudges) whenever it was near
an obstacle — which, with parked cars flanking the slot, was 90%+ of every episode. A controller
evaluation run showed PPO+Fuzzy succeeding on 1.3% of episodes versus PPO-alone's 17.5% — the
shield was actively fighting the policy's fine maneuvering. Fixed by giving `fuzzy_inputs()` the
nearest-obstacle bearing (for the shield) and a separate `slot_bearing_deg()` (for navigation),
and rewriting `fuzzy/fuzzy_only_controller.py` to use a real proportional-navigation steer with
the fuzzy avoidance layered on top, rather than using the avoidance output as the only steering
signal.

### 2.2 Shield trained-vs-tested mismatch
Even after fixing 2.1, the shield's default trigger distance (3m, the fuzzy system's own "medium
distance" range) meant it was still active on >90% of steps near the obstacle-flanked slot. PPO is
trained with **no shield in the loop** (`rl/train_ppo.py` trains the raw policy only), so every
shield intervention at test time is out-of-distribution perturbation to an otherwise-competent
policy. Tightening the trigger to 1.0m (true near-collision only) brought PPO+Fuzzy back in line
with PPO-alone. **This is a stopgap, not a real fix** — see "What a continuation should do" below.

### 2.3 Scenario generator never had an obstacle-free stage
`environment/scenarios.py` always placed two neighbor-car obstacles, even at
`RandomizationLevel.NONE`. The blueprint's own Part 8 curriculum calls for an obstacle-free first
stage; without it, the very first thing PPO's policy ever saw was already the hardest variant.
Added an `include_obstacles` flag, independent of the domain-randomization level.

### 2.4 PPO converged to "full brake, never move" (three separate causes, found in sequence)
1. **No obstacle-free stage** (2.3) meant even the "easiest" setting still had collision risk.
2. Even obstacle-free, the nominal ~6-7m approach distance is far enough that a randomly-exploring
   policy essentially never reaches the success bonus by chance (empirically: random actions
   solve a ~0.3m start distance ~97% of the time, but a ~1m start distance 0% of the time out of
   200 trials — see the distance-scale sweep this was caught with). With no positive reward signal
   to learn from, "sit still" became a stable, low-variance local optimum. **Fix**: an added
   `start_distance_scale` curriculum axis that starts the car almost in the slot and ramps the
   distance back out, mixed via **uniform sampling on `[min_scale, current_scale]` rather than a
   point value** — an earlier version pinned each stage to one exact distance and suffered
   catastrophic forgetting of easier cases the moment training moved to a harder stage (confirmed:
   a checkpoint that passed its curriculum threshold at distance 0.2 scored 0/20 deterministic
   success there after more training at distance 0.35).
3. **`ent_coef` tuning is a knife's edge.** `0.003` collapsed to the frozen policy with zero
   exploration. `0.02` (an attempt to fix #2) caused the action distribution's standard deviation
   to *grow* (observed: 0.87 → 2.05 over training) instead of shrink — stochastic rollouts passed
   the curriculum's success threshold via noise alone while the policy's actual mean action stayed
   frozen at full-brake. SB3's default, `0.0`, combined with the distance curriculum above, is what
   finally produced a policy whose *deterministic* action is a real, non-degenerate parking
   maneuver.

### 2.5 Scenario generator could spawn a car already overlapping an obstacle
Caught via `demo.py`: an episode "collided" on literal step 1, before any action could possibly
matter. `RandomizationLevel.SLOT_POSITION` stacks slot-position jitter (±1m) with start-pose
position jitter (±1m) and heading jitter (±15°); the fixed nominal clearances assumed no jitter,
and occasionally the combination placed the car's initial footprint inside a neighbor obstacle.
Fixed with rejection sampling in `ParkingEnv.reset()` (retry up to 20 times if the fresh scenario's
start pose already collides). This alone improved every controller's measured success rate
(rule-based 1.3%→3%, PPO 17.5%→32%) by removing un-winnable episodes from the evaluation set.

### 2.6 `demo.py` seeds didn't match `evaluation/run_experiments.py` seeds
Two separate causes: (a) `demo.py` used `ParkingEnv`'s own default `RandomizationLevel.NONE`
instead of the evaluation harness's `SLOT_POSITION`, and (b) `demo.py` defaulted to a single
scenario kind while the harness alternates both kinds — since kind selection consumes the episode
RNG before the scenario itself does, a shorter `scenario_kinds` list silently shifts every
downstream random draw. Fixed by making `demo.py`'s defaults match the harness exactly, documented
inline so it doesn't drift back out of sync.

## 3. Final results

100 episodes per controller, both scenario kinds, `SLOT_POSITION` randomization (start pose + slot
position jittered, obstacles fixed relative to the slot, no sensor noise):

| Controller | Success | Collision | Timeout | Mean reward |
|---|---|---|---|---|
| Rule-Based | 3% | 39% | 58% | -9.1 |
| Fuzzy-only | 3% | 14% | 83% | -7.3 |
| **PPO** | **32%** | **11%** | 57% | **32.9** |
| PPO + Fuzzy (shield @ 1.0m) | 32% | 11% | 57% | 32.2 |

Full table: `experiments/results/comparison_table.csv`; plots: `bars_success_collision.png`,
`boxplots_errors.png`, `trajectory_length_hist.png` in the same folder. Demo clips:
`demo_ppo_fuzzy_success.gif` (a genuine, verified park) and `demo_rule_based_collision.gif`.

PPO clearly outperforms both hand-designed baselines on every axis. The shield currently makes no
measurable difference at either threshold tested (3m: actively harmful; 1m: neutral) — see below.

## 4. Honest limitations and what a continuation should do

- **PPO is undertrained, not broken.** 600k-1.1M timesteps on a single CPU core (no GPU was
  available in this sandbox — see the `torch --no-deps` note in `requirements.txt`) is enough to
  learn a real, non-degenerate parking maneuver, but 32% success leaves a lot on the table. Use
  `python -m rl.train_ppo --resume` to keep training the existing checkpoint; it picks up both the
  weights and the curriculum stage from `models/ppo_parking_v1.stage.json`.
- **The shield needs to be trained through, not bolted on at test time.** The methodologically
  correct version of Experiment 4 trains a *separate* policy with `HybridController` wrapped into
  the environment from step one (so the policy learns to work with the shield's corrections,
  rather than being surprised by them at test time). This is a straightforward extension of
  `rl/train_ppo.py` (wrap each `ParkingEnv` in a thin `gym.Wrapper` that runs the actions through
  `HybridController.act` before calling `env.step`) but needs its own training run, which this
  session's compute budget didn't leave room for.
- **Perpendicular parking is harder and less tested.** All of the curriculum tuning above was done
  primarily against parallel scenarios; perpendicular episodes are included in the final 100-episode
  evaluation but weren't separately debugged.
- **Sensor noise (`RandomizationLevel.SENSOR_NOISE`) was never reached by training.** The curriculum
  got through `OBSTACLES`-level position jitter; noisy rays are implemented (`sensors.py`) and unit
  reasoned about, but the policy has not been trained or evaluated under them.

## 5. Parallel parking: root cause and fix

Every controller scored ~3-5% on parallel parking. Two causes, both structural, not tuning:

1. **The car could not reverse.** The action `[steer, throttle_brake]` treated negative values as
   brake only, so no controller (including PPO) could ever drive backwards. Parallel parking needs
   reverse. Fix: opt-in `allow_reverse=True` (signed throttle: braking first if moving the other way).
   Default behaviour is unchanged, so earlier results stay reproducible.
2. **The gap was too small.** Default gap 5.4 m for a 4.2 m car. A gap sweep showed collisions at
   <= 7.2 m (nose clips the front neighbour); the reverse env now defaults to 8.0 m (`parallel_gap`).

Fix delivered: `baselines/parallel_geometric.py` (two equal full-lock reverse arcs, `d = 2R(1-cos a)`)
parks 50/50 in the env; the simple HTML version parks 500/500. Use `--full-approach` in
`run_experiments` so random start distances don't create unwinnable starts.

**Not done:** PPO has not been retrained with reverse. Run
`python -m rl.train_ppo --curriculum --reverse --kinds parallel`. Existing PPO models still score ~3-5% on parallel.
