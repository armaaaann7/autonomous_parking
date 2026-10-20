# Professor Defense Q&A

Status key: RESOLVED / PARTIALLY MITIGATED / OPEN. Evidence: [RUN] executed in audit, [SRC] source only, [MISSING] no evidence. Paths are relative to `autonomous-parking/` unless stated.

## The five strongest technical objections
1. **"The RL part does nothing."** Status: **OPEN (disclosed).** RL + Fuzzy has identical outcomes to Fuzzy-only on set A (McNemar 0 discordant, `experiments/audit_evaluation/paired_tests.txt`); PPO alone: 7-11 % success. No learned fusion exists.
2. **"The shield was not a shield."** Status: **RESOLVED.** The original Mamdani shield read centre-cast rays (D1); now fed true clearance (`ParkingEnv.shield_inputs`, `environment/geometry.py::obb_aabb_distance`), tests in `tests/test_audit_regressions.py`.
3. **"Safe only in your own simulator."** Status: **PARTIALLY MITIGATED.** Envelope shield: 0/900 collisions + adversarial test, but uses ground-truth geometry and the sim's dynamics (`integration/envelope_shield.py`, `ParkingEnv.braking_envelope_safe`). No noise/model-mismatch evidence [MISSING].
4. **"Your 'success' is not parked."** Status: **PARTIALLY MITIGATED.** Success tolerance 0.40 m > 0.30 m bay clearance; strict in-bay 5 % (A) / 0 % (B). Measured and disclosed, tolerance kept for comparability.
5. **"Results are cherry-picked / underpowered."** Status: **PARTIALLY MITIGATED.** Shared seeds, failures retained, Wilson CIs, exact McNemar, a second unseen block (seeds 5000-5199, n=200), identical re-run. One scenario family, a single PPO checkpoint, start distances skewed close.

## Questions (answers are the honest ones)
1. **What is the observation/action space?** 14-dim `[dx, dy, sin dpsi, cos dpsi, v, delta, 8 rays]`, action `[steer, throttle_brake]` in [-1,1]. `environment/parking_env.py`. [SRC]
2. **What vehicle model?** Kinematic bicycle at rear axle, wheelbase 2.5 m, dt 0.1 s, steering rate-limited. `environment/vehicle.py`; `tests/test_vehicle.py`. No tyre slip/dynamics. [SRC+RUN tests]
3. **How is collision detected?** SAT between car OBB and neighbour AABBs, plus centre-out-of-bounds. `environment/geometry.py`. Bounds test is centre-only (D12).
4. **What counts as success?** Centre < 0.40 m, heading < 15 deg, speed < 0.20 m/s for 2 steps, no collision. Collision gating was added (D2); mutation-checked test.
5. **Why is success 88 % if strict in-bay is 5 %?** Tolerance is looser than the bay (D6). We report both.
6. **How is Fuzzy-only built?** Zero-order Sugeno rules plus a proportional speed loop with rolling-resistance feed-forward and a forward-corridor brake. `fuzzy/fuzzy_only_controller.py`. It is a nonlinear proportional controller, not a planner.
7. **Where is Mamdani used?** 18-rule obstacle shield, scikit-fuzzy, `fuzzy/fuzzy_system.py`; non-finite inputs raise ValueError. Used inside `ppo_fuzzy_bolted`.
8. **What is the PPO setup?** SB3 MlpPolicy 2x128, 100,352 steps, `models/ppo_only_best.zip` (unchanged). Training was not re-run in this audit [MISSING: training curves/seeds].
9. **Why is PPO bad?** Likely undertrained and the reward is sparse for a precision task; not diagnosed experimentally [MISSING]. Don't claim RL cannot work.
10. **How are the controllers blended?** `0.25*PPO + 0.75*fuzzy`, fuzzy braking priority, stall recovery after 12 stalled steps. `integration/hybrid_controller.py`. Weights hand-set.
11. **Is the hybrid better than Fuzzy-only?** No (see objection 1). It spends 71-73 % of steps in recovery [RUN, shield_active/recovery stats].
12. **What exactly does the envelope shield do?** Simulate action one step ahead, then full braking with steering held; if the footprint (+0.05 m) hits anything, replace with `[held steer, -1]`. `integration/envelope_shield.py`.
13. **Is it a formal safety proof?** No. Inductive invariant under the simulator's own model and ground truth; no noise, moving objects, mismatch. Never say "collision-proof".
14. **What does safety cost?** Fuzzy-only 88 -> 83 % (A), 89 -> 86 % (B); collisions 12 -> 0; extra timeouts. Success loss 5 vs 0 discordant, p=0.0625 (n.s. at n=100).
15. **Are results reproducible?** Re-running produced identical outcomes (fixed seeds). Commands in `DEMO_RUNBOOK.md`. [RUN]
16. **How many episodes and which seeds?** Set A 100 (3000-3099), set B 200 (5000-5199); shield demo seed 3009; viewer seed 1000.
17. **Which statistical tests?** Wilson 95 % CI for proportions, exact two-sided McNemar for paired outcomes (`evaluation/metrics.py::wilson_interval, mcnemar_exact`). Multiple comparisons not corrected.
18. **Are the start distances realistic?** Start distance uniform in [0.05,1]x6 m so many starts are close; success is flattering. Failures cluster at >= 4 m with > 0.5 m lateral offset (11/27 vs 1/73).
19. **Is the 3D viewer live?** No: it replays recorded JSON, and its banner says so. Live inference is `demo.py`.
20. **Does it handle sensor noise?** Not in any reported result; noise level 4 exists in code but is unused [SRC]. [MISSING: noisy evaluation].
21. **Does it generalise to parallel/reverse parking?** Code exists but excluded from all numbers; no claim.
22. **What tests exist and what do they not cover?** 81 pass [RUN]. Not covered: viewer visuals, training, noisy sensors, real hardware.
23. **Why do your slides say PPO 32 %?** The deck was made from the GitHub repo, which differs from the local project (7-11 % here). Update the deck. Trust the audit tables.
24. **What if I change the seed and it collides?** Fuzzy-only unshielded collides ~11-12 %; that's the reported rate. With the shield it times out instead.
25. **Did you tune on the evaluation seeds?** Set A seeds were the previously published evaluation; set B is a fresh block. The shield margin 0.05 m was set before running either [SRC; no formal log, treat as plausible not proven].
26. **Why the tiny clearances at start?** Spawn rejection only rejects overlap (D11); worst case 0.0166 m in shielded runs.

## Final-milestone questions
**Q: "Why does RL + Fuzzy equal Fuzzy-only?"** A: Measured, not guessed. The Mamdani shield (threshold 1.0 m on true clearance) fires at spawn in ~77 % of episodes because ~45 % of random starts are within 0.3 m of a neighbour; the car stays still; after 12 stalled steps a latch gives control to pure Fuzzy and PPO is no longer used (96/100 episodes). Without the latch: 3 % success. Evidence: `experiments/hybrid_diagnosis/`. Status: explained, not improved.

**Q: "Did you tune until it looked good?"** A: Fuzzy-precise's stop radius (0.10 m) was chosen on tuning seeds 7000-7099; PPO v2 candidate selection used 7000-7099; reporting uses 3000-3099, 5000-5199 and fresh 9000-9199. Weights were not tuned. Caveat: the original PPO's selection used 5000-5039 (overlaps set B).

**Q: "Is 58 % strict success good?"** A: No, it is a measured limit. Misses are heading-limited; the car is forward-only and cannot re-manoeuvre. Legacy Fuzzy-only was 1 % strict: its 88 % was pose success only.

**Q: "Is PPO v2 better than Fuzzy?"** A: No. It is far better than the original PPO (pose 68 % vs 6 %), competitive near the slot (97.8 % pose for starts < 2.5 m) and fails far starts (11.4 % for >= 4.5 m). Paired McNemar vs Fuzzy-precise on strict success: 94 vs 37 discordant, p ~ 7e-7.

**Q: "Is the shield verified?"** A: No formal verification. It is an inductive-style heuristic under the simulator's own model: 0 collisions in 391 adversarial and 360 sub-step-audited episodes; with real brakes at 0.85x the modelled strength 49/198 adversarial episodes collide. It also deadlocks: shielded timeouts mostly never leave the start (60 of 81).

**Q: "Is the scenario generator fair?"** A: Partly. 44.6 % of starts have < 0.30 m body clearance to a neighbour (6.4 % < 0.05 m) and distance is uniform in [0.05, 1] x 6 m. Rates are generator-dependent; stratified results are in `stratified.md`.

**Q: "Did you look at the 3D viewer?"** A: Yes, rendered in headless Chromium and inspected; found and fixed a 0.2 m bay-length bug and HUD overlap. Not verified: camera framing in top-down/chase modes and behaviour on the presentation GPU.
