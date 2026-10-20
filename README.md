# Autonomous Parking Using Reinforcement Learning and Fuzzy Logic

A 2D perpendicular-parking simulator with three controllers (Fuzzy-only, RL-only PPO, and a weighted RL + Fuzzy hybrid), an optional model-based **braking-envelope safety shield**, a reproducible evaluation pipeline, and a three.js replay viewer.

**Read first:** `PRESENTATION_READINESS_AUDIT.md` (verdict, defects, evidence), `DEMO_RUNBOOK.md` (exact demo steps), `PROFESSOR_DEFENSE_QA.md` (20+ hard questions with evidence).

## What the project genuinely does
* **Simulator** (`autonomous-parking/environment/`): Gymnasium env, kinematic bicycle model (rear axle, 4.2 x 1.8 m car, wheelbase 2.5 m, steering slew-limited at 2 rad/s, 10 Hz), 8 centre-cast ray sensors (0-10 m), exact OBB-vs-AABB (separating axis) collision, perpendicular bay 2.4 x 4.8 m between two parked cars.
* **Observation (14):** `[dx, dy, sin(heading err), cos(heading err), v, steer angle, 8 rays]` (dx, dy in the WORLD frame). **Action (2):** `[steer, throttle_brake]` in [-1, 1]; negative second value = brake only (no reverse gear in this project's comparison).
* **Success:** centre within **0.40 m** of slot centre, heading error **< 15 deg**, speed **< 0.20 m/s**, held for **2 steps**, and **no collision**. Failures are collision (OBB overlaps a neighbour, or centre leaves the +-15 m bounds) or 500-step timeout (50 s).
* **Controllers**
  * `fuzzy_only`: zero-order Sugeno heading/distance rules + a proportional speed loop with rolling-resistance feed-forward (`fuzzy/fuzzy_only_controller.py`). *Honest description:* a hand-tuned non-linear proportional navigator; there is no learning and no obstacle-path planning.
  * `ppo_only`: Stable-Baselines3 PPO, 2x128 MLP, **the supplied 100,352-step checkpoint `models/ppo_only_best.zip`, unchanged**. It is an under-trained baseline, not a solved policy.
  * `ppo_fuzzy_bolted` ("RL + Fuzzy"): action = 0.25 x PPO + 0.75 x fuzzy navigation, fuzzy braking priority, Mamdani obstacle shield (18 rules, scikit-fuzzy) fed the *true body clearance*, and a latched fuzzy-recovery mode after 1.2 s stationary. **Measured: it is in recovery for ~71-73 % of steps and gives identical success/collision outcomes to Fuzzy-only on all 100 set-A seeds (paired McNemar: 0 discordant) and 88.5 % vs 89.0 % on set B, so the PPO component has no measurable benefit.** Hand-set weights, not learned.
  * `<controller>+env` (e.g. `fuzzy_only+env`): the same controller wrapped in the **braking-envelope shield** (`integration/envelope_shield.py`). See below.
* **Safety shield, precisely:** every proposed action is simulated one step forward on a copy of the vehicle, followed by full braking with steering held until standstill; the car's true footprint (with a 0.05 m margin) is tested against every obstacle and the bounds. If that fails, the action is replaced by `[held steer, full brake]`. Under the simulator's own dynamics and ground-truth obstacle map this is an inductive safety invariant (0 collisions in 100 + 200 evaluation episodes x 3 controllers, and in an adversarial 40-episode test). **It is a model-based heuristic safety filter, not a formal guarantee for a real vehicle** (no sensor noise, no moving obstacles, no model mismatch, forward-only). It trades completion for safety: a stopped car is safe but may time out.

## Final results (final milestone; pooled n = 500: sets A 3000-3099, B 5000-5199, C 9000-9199 [fresh])
Two success definitions are reported. **Pose success** = legacy criterion. **Strict (contained) success** = pose criterion AND the whole car inside the bay lines. Same trajectories, failures retained. Source: `autonomous-parking/experiments/corrected_evaluation/` (`summary_all.md`, `stratified.md`, per-episode CSVs).

| Controller | Pose success / coll. / timeout | Strict success / coll. / timeout |
|---|---|---|
| Fuzzy-only (legacy, stops 0.32 m out) | 88.2 / 11.8 / 0 | **1.0** / 12.0 / 87.0 |
| Fuzzy-precise (stop radius 0.10 m) | 88.2 / 11.8 / 0 | **57.8** / 13.0 / 29.2 |
| PPO original (100k steps) | 5.8 / 43.4 / 50.8 | 1.8 / 43.4 / 54.8 |
| PPO v2 (retrained, ~420k steps) | 68.2 / 4.2 / 27.6 | **46.4** / 4.2 / 49.4 |
| RL + Fuzzy (original hybrid) | 87.0 / 13.0 / 0 | 1.0 / 13.0 / 86.0 |
| PPO v2 + Fuzzy 25/75 (no latch) | 83.2 / 16.2 / 0.6 | 2.0 / 16.4 / 81.6 |
| any of the above + envelope shield | collisions **0** in every row; success drops (e.g. Fuzzy-only 83.8, PPO v2 66.6, Fuzzy-precise strict 57.6), timeouts rise | |

Key findings (each backed by the files above):
1. **The legacy Fuzzy-only "88 %" is pose success, not parking inside the bay**: strict success is 1 %. Cause: it stops 0.32 m from the slot centre but the bay end clearance is only 0.30 m, so the rear overhangs. Fuzzy-precise fixes that with a smaller stop radius (tuned on seeds 7000-7099 only) and reaches 57.8 % strict; the remaining misses are heading-limited (forward-only car cannot re-manoeuvre).
2. **PPO v2 is much better than the original PPO but not better than Fuzzy** (paired McNemar, pooled: Fuzzy-precise vs PPO v2 contained success 94 vs 37 discordant, p ~ 7e-7). It is strong near the slot and **fails on far starts** (start >= 4.5 m: pose success 11.4 % vs 77.2 % for Fuzzy-only).
3. **Why "RL + Fuzzy" equals Fuzzy-only:** the Mamdani shield fires at spawn in ~77 % of episodes (about 45 % of random starts are within 0.3 m of a neighbour), the car stays still, after 12 stalled steps a stall latch hands control to pure Fuzzy-only; PPO is no longer consulted (latch fires in 96/100 episodes). Without the latch the hybrid gets 3 % success and 97 % timeouts. PPO is queried and its output is state dependent; it simply does not help. Weight sweeps (`experiments/hybrid_diagnosis/`): raising PPO weight trades collisions for timeouts, never success.
4. **The envelope shield removes collisions, not failures:** 0 collisions in all shielded episodes, but 60 of 81 Fuzzy-only+shield timeouts never moved more than 1 m (a deadlock: the shield rejects every proposed action and proposes none of its own). Under a 0.85x brake-strength mismatch the adversarial test produced 49/198 collisions (0.7x: 114, 0.5x: 156). It is a heuristic filter under the simulator's own model, **not formally verified and not universally safe**.
5. **Scenario validity:** 44.6 % of random starts have body clearance < 0.30 m to a neighbour (6.4 % < 0.05 m), and start distance is uniform in [0.05, 1] x 6 m, so many starts are already beside the bay. Absolute rates depend on this generator.

### Earlier audit-round results (pose metric; kept for traceability)
Perpendicular bay, slot jitter +-1 m, start pose +-1 m / +-15 deg, start distance drawn at random in [0.05, 1.0] x 6 m (so many starts are close to the slot), shared seeds, failures retained. 95 % Wilson CI for success in brackets.

| Controller | Set A: seeds 3000-3099 (n=100) success / collision / timeout | Set B: seeds 5000-5199 (n=200) success / collision / timeout |
|---|---|---|
| Fuzzy-only | 88 % [80-93] / 12 / 0 | 89.0 % [84-93] / 11.0 / 0 |
| RL-only (PPO, 100k steps) | 11 % [6-19] / 38 / 51 | 7.0 % [4-11] / 40.0 / 53.0 |
| RL + Fuzzy | 88 % [80-93] / 12 / 0 | 88.5 % [83-92] / 11.5 / 0 |
| Fuzzy-only + envelope shield | 83 / **0** / 17 | 86.0 / **0** / 14.0 |
| RL-only + envelope shield | 10 / **0** / 90 | 7.0 / **0** / 93.0 |
| RL + Fuzzy + envelope shield | 83 / **0** / 17 | 86.0 / **0** / 14.0 |

Set A is the originally published evaluation (seeds 3000-3099); set B is a fresh block that no controller was tuned on. Per-episode CSVs, `comparison_table.csv`, `paired_tests.txt` (exact McNemar) and plots: `autonomous-parking/experiments/audit_evaluation/` and `.../audit_holdout_5000/`. The pre-audit results (`final_evaluation/`, `fusion_development/`, `before_fix/`) are preserved untouched.

**"Success" is not "inside the painted lines".** The 0.40 m centre tolerance is looser than the bay's 0.30 m clearance. Only **5 % (A) / 0 % (B)** of Fuzzy-only episodes end with the whole car inside the bay (`in_bay`); median protrusion of a successful Fuzzy-only park is about 7 cm (max about 30 cm). Say "parked within tolerance", not "parked in the bay".

## Commands
Python **3.11+** is required (pinned `numpy==2.4.4`, `pandas==3.0.2`...). Python 3.10 cannot install the pinned versions.

    cd autonomous-parking
    python3 -m venv .venv && source .venv/bin/activate
    python -m pip install -r requirements.txt

    python -m pytest tests/ -q                              # 131 tests, ~20 s
    python prepare_comparison.py --seed 9170                # regenerate viewer recordings (also --seed 1000)
    python ../viewer/serve.py                               # replay viewer (recorded, NOT live)
    python demo.py --controller fuzzy_only --seed 1000      # LIVE 2D pygame demo
    python demo.py --controller fuzzy_only+env --seed 3009  # shield demo: holds still instead of crashing
    python -m evaluation.run_experiments --episodes 100 --seed 3000 --out-dir experiments/audit_evaluation \
        --controllers fuzzy_only,ppo_only,ppo_fuzzy_bolted,fuzzy_only+env,ppo_only+env,ppo_fuzzy_bolted+env
    python -m evaluation.compare_controllers --dir experiments/audit_evaluation

Controllers for `demo.py` / `export_trajectory.py`: `fuzzy_only`, `fuzzy_precise`, `ppo_only`, `ppo_v2`, `ppo_fuzzy_bolted`, `ppo_v2_fuzzy`, each optionally with `+env`. Strict evaluation: add `--success-mode contained`; full matrix: `python -m evaluation.evaluate_matrix` then `summarize_corrected.py`, `stratify.py`. Shield stress: `python -m evaluation.shield_stress`. A missing/incompatible PPO checkpoint produces a clear error (not a crash trace). PPO v2 was trained in the final milestone (see `autonomous-parking/models/README.md`); it is better than the original PPO but is not claimed to beat Fuzzy.

## Viewer
`viewer/serve.py` serves recorded JSON episodes (the banner says so). 24 episodes: seed 9170, seed 1000, and four shield demos at seed 3009. **Disclosure:** seed 9170 (a held-out far start) was *selected* because it contrasts pose and strict outcomes; it is illustrative, not typical. HUD shows outcome, pose-success step, in-bay status and per-frame protrusion. Rendered in headless Chromium (software GL) and inspected; see the audit. Viewer bay-length bug (drawn 0.2 m too long) fixed.

## Known limitations (also in the audit)
* RL: PPO v2 (~420k steps, 3 seeds, best of 6 candidates picked on validation seeds) is still weaker than Fuzzy and fails far starts; the "PPO < Fuzzy" result is specific to this reward, budget and start distribution. The hybrid's weights are hand-set and its outcome equals Fuzzy-only because of the stall latch. The original PPO's checkpoint selection used seeds 5000-5039 (overlaps set B).
* Fuzzy-only collides in about 11-12 % of episodes, almost entirely starts >= 4 m away with > 0.5 m lateral offset (11 of 27 such starts in set A vs 1 of 73 for smaller offsets).
* Perpendicular parking only; reverse/parallel code exists but is excluded from every reported number.
* Ground-truth geometry and noiseless sensors in evaluation (noise level 4 is never used); kinematic model only.
* Some start poses spawn with only a few millimetres of clearance (rejection sampling only rejects overlap).
* Code comments mention `docs/REVIEW.md`, which is not in this copy (it exists in the GitHub repository).
