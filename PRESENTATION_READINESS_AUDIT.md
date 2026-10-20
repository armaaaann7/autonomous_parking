# Presentation Readiness Audit

Scope: `autonomous-parking-project/` (simulator, three controllers, evaluation, viewer). Audit performed by running code on a copy of the environment (Python 3.12 venv, since the pinned versions need Python >= 3.11). No git repository exists in the folder, so a full backup was taken first (`autonomous-parking-project_BACKUP_*` next to the project). Nothing was deleted; the PPO checkpoint, the original result folders (`final_evaluation/`, `before_fix/`, `fusion_development/`) and training data are untouched. No secrets were read or reported. Nothing was pushed or deployed.

Evidence labels: **[RUN]** executed in this audit and output observed; **[SRC]** source inspection only; **[PRIOR]** result from an earlier run in the repo, not re-executed; **[MISSING]** no evidence.

## 0. FINAL MILESTONE ADDENDUM (supersedes older statements where they differ)
**Verdict after the final milestone: READY WITH CAVEATS** (unchanged tier, but the caveats are now measured, not guessed).

| Risk (P) | Outcome | Evidence |
|---|---|---|
| P1 strict containment | Root cause found (stop radius 0.32 m > bay end clearance 0.30 m). Strict metric added (`success_mode="contained"`), geometry not loosened. Fuzzy-precise: strict 57.8 % vs legacy Fuzzy-only 1.0 %. Remaining 42 % is heading-limited / forward-only, **not solved**. | `experiments/corrected_evaluation/`, `containment_tuning*/` |
| P2 hybrid = Fuzzy-only | Explained: spawn-time Mamdani shield stall (~77 % of episodes) -> latch at step 12 -> pure Fuzzy. PPO is queried but irrelevant. Without latch: 3 % success. Weight sweep: no weight improves success. | `experiments/hybrid_diagnosis/`, `tests/test_hybrid_integrity.py` |
| P3 weak PPO | Retrained (3 seeds, ~420k steps, curriculum). PPO v2: pose 68.2 % / strict 46.4 % pooled (original 5.8 / 1.8). **Still worse than Fuzzy and fails far starts (11.4 % vs 77.2 %).** | `experiments/ppo_retrain/`, `models/README.md` |
| P4 shield | Fail-closed hardening + stress tests: 0 collisions in 391 adversarial and 360 sub-step-audited episodes; **but** a 0.85x brake mismatch gives 49/198 adversarial collisions; shielded runs deadlock (60 of 81 timeouts never moved > 1 m). Heuristic, not formally verified. | `experiments/shield_verification/`, `tests/test_shield_stress.py` |
| P5 viewer | Rendered in headless Chromium (software GL) and **inspected**: 6 frames (Fuzzy-precise inside bay; PPO v2 inside bay; Fuzzy-only stopped outside bay; Fuzzy-only collision; shielded freeze at 0.05 m with BRAKE/ENVELOPE marker; REPLAY banner). Found and fixed: bay drawn 0.2 m too long; banner hidden under panels; HUD text covering the scene at small heights. **Not verified:** top-down/chase camera framing (software rendering ran < 1 fps so the eased camera had not converged), WebGL on the presentation GPU, physical orientation of the car model beyond "nose into bay" as seen. A computer-use inspection of the desktop app was attempted and could not be granted. | this section, `viewer/` |
| P6 deck | `Autonomous_Parking_Final_Evidence.pptx` (new file; originals untouched). | see final report |

Tests: **131 passed** (from `autonomous-parking/` and from the project root); `node --check` OK on all five viewer modules; pyflakes: 2 cosmetic notes (f-string without placeholders in `evaluation/stratify.py`; intentional tensorboard probe in `rl/train_ppo.py`). Two tests were corrected (see CHANGELOG): both encoded a superseded assumption, neither was weakened to hide a defect.

Remaining blockers / caveats to disclose: (1) no PPO beats Fuzzy and the hybrid adds nothing; (2) strict containment is 58 % at best; (3) the scenario generator spawns 44.6 % of starts within 0.30 m of a neighbour, so absolute rates are generator-dependent; (4) the shield is a model-based heuristic with 0 collisions only under its own model; (5) original PPO checkpoint selection used seeds 5000-5039 (overlap with set B; set C 9000-9199 is clean); (6) viewer seed 9170 was selected for contrast; (7) results are one simulator, perpendicular parking only, no reverse; (8) `docs/REVIEW.md` referenced in comments is absent; (9) Python >= 3.11 required.
Claims to avoid: "RL+Fuzzy beats Fuzzy", "collision-free in general", "formally verified", "parks inside the bay" (for legacy Fuzzy-only), "PPO solves parking".

## 1. Executive verdict: READY WITH CAVEATS
The demo runs, the tests pass, the headline numbers reproduce exactly, and the claims in the README now match the measured data. It is **not** a project that can honestly be described as "RL + Fuzzy beats both", "collision-free" or "parks inside the bay". With the corrections below the project can be defended; presenting the old claims would not survive a prepared examiner.

Honest one-paragraph story: a hand-tuned fuzzy-style navigator parks successfully (within tolerance) in ~88-89 % of random perpendicular scenarios; an under-trained 100k-step PPO baseline manages 7-11 %; the weighted hybrid behaves identically to the fuzzy controller; a new model-based braking-envelope shield removes all collisions in the simulator (0 / 900 shielded episodes) but converts some of them into timeouts.

## 2. Initial defects (found) and status
| ID | Sev | Defect | Evidence | Status |
|---|---|---|---|---|
| D1 | P0 | The Mamdani obstacle "shield" was structurally blind: it read ray distances measured from the car centre, so a neighbour's side face looked ~1 m+ away when the body was already touching. It could not prevent the collisions it was claimed to prevent. | [RUN] reproduced; regression tests | **Fixed**: shield inputs now use true body clearance (`ParkingEnv.shield_inputs`, `geometry.obb_aabb_distance`). Plus new envelope shield. |
| D2 | P0 | Success could be declared on the same step as a collision (success gated only by pose/speed). | [RUN] test failed before fix | **Fixed** (`parking_env.py`: success requires `not collided`; test runs hold_steps+1 steps and fails if the gate is removed). |
| D3 | P0 | "RL + Fuzzy" hybrid = 0.25 PPO + 0.75 fuzzy, with fuzzy priority and a recovery latch: it spends ~71-73 % of steps in recovery and gives the same outcomes as Fuzzy-only. The README claimed it improved things (0.361 m error, 87/100). | [RUN] paired McNemar 0 discordant (set A) | **Disclosed, not fixed** (cannot honestly be made better without retraining). README/Q&A state it. |
| D4 | P1 | Reported numbers had no confidence intervals and the paired comparison was not tested; metrics hid collisions inside averages. | [SRC] | **Fixed**: Wilson CIs, exact McNemar, per-episode CSVs, failure rates always shown. |
| D5 | P1 | Non-finite inputs (NaN sensor) could propagate silently into the controller. | [RUN] | **Fixed**: ValueError in fuzzy system; hybrid fails safe to brake. |
| D6 | P1 | "Success" means centre within 0.40 m, but the bay only has 0.30 m side clearance: almost no "successful" park is actually fully inside the lines (strict in-bay 5 % / 0 %). | [RUN] new `in_bay` metric | **Disclosed + measured**; tolerance unchanged to keep results comparable. |
| D7 | P1 | Viewer and README implied live inference; viewer replays recorded JSON. Labels used internal names. | [SRC] | **Fixed**: banner "recorded replay", readable names, shield-intervention marker and clearance display. |
| D8 | P1 | Demo/export scripts could not run the shielded variants; PPO checkpoint load errors were raw traces. | [SRC] | **Fixed** (`make_adapter`, `+env` suffix, validation in `policy_wrapper.py`). |
| D9 | P2 | Pinned dependencies require Python >= 3.11 (undocumented). | [RUN] pip failed on 3.10 | **Documented** in README. |
| D10 | P2 | Code comments reference `docs/REVIEW.md`, absent from this copy. | [SRC] | Open (cosmetic). |
| D11 | P2 | Some random starts spawn with only millimetres of clearance (rejection sampling only rejects overlap). | [RUN] worst-case clearance 0.0166 m in shielded runs | Open; disclosed. |
| D12 | P2 | Out-of-bounds checked on the centre only. | [SRC] | Open. |
| D13 | P2 | Two pre-existing lint warnings in `tests/test_vehicle.py`, `tests/test_geometry.py` (unused import / variable). | [RUN] pyflakes | Open, harmless. |

## 3. Fixes made (files)
`environment/geometry.py` (exact OBB-AABB distance), `environment/parking_env.py` (collision-gated success, `body_clearance`, `shield_inputs`, `bay_protrusion`, `inside_bay`, `braking_envelope_safe`), `integration/envelope_shield.py` (new), `integration/hybrid_controller.py`, `fuzzy/fuzzy_system.py`, `environment/shielded_env.py`, `rl/policy_wrapper.py`, `evaluation/run_experiments.py`, `evaluation/metrics.py`, `evaluation/compare_controllers.py`, `export_trajectory.py`, `prepare_comparison.py`, `demo.py`, `environment/renderer.py`, `viewer/{index.html,src/main.js,src/config.js,assets/data/*}`, `tests/test_audit_regressions.py` (new), `DIAGNOSIS.md` (superseded banner), `README.md` (rewritten).

## 4. Tests: exact outcomes
* `python -m pytest tests -q -p no:cacheprovider` from `autonomous-parking/` **[RUN]**: **81 passed** in ~11 s (2015 warnings, all from scikit-fuzzy/NumPy deprecations). 31 of the 81 are new (`tests/test_audit_regressions.py`); 50 are the original tests, unmodified.
* Regression tests were checked to actually detect their defect: the success-vs-collision test fails with the fix removed (mutation check); the envelope-shield test geometry was corrected after an initial test mistake (the code was right, my stopping-distance arithmetic in the test was wrong).
* `pyflakes` on source and tests **[RUN]**: only the 2 pre-existing test-file warnings (D13).
* `node --check viewer/src/main.js viewer/src/config.js` **[RUN]**: OK.
* Viewer in headless Chromium **[RUN]**: page reaches ready state, no console errors, DOM text correct for several episodes, `?t=` clamped. Visual inspection was completed in the final milestone (section 0). Still open the viewer once on the presentation machine.
* Training (`rl/train_ppo.py`) was **not run**. PPO numbers come from the supplied checkpoint.

## 5. Verified evaluation results [RUN, newly generated, reproducible: re-run gave identical outcomes]
Perpendicular only, shared seeds, failures retained, Wilson 95 % CI.

| Controller | Set A seeds 3000-3099 (n=100): success [CI] / coll / timeout | Set B seeds 5000-5199 (n=200): success [CI] / coll / timeout |
|---|---|---|
| Fuzzy-only | 88 % [80.2-93.0] / 12 / 0 | 89.0 % [83.9-92.6] / 11.0 / 0 |
| RL-only PPO | 11 % [6.3-18.6] / 38 / 51 | 7.0 % [4.2-11.4] / 40.0 / 53.0 |
| RL + Fuzzy | 88 % [80.2-93.0] / 12 / 0 | 88.5 % [83.3-92.2] / 11.5 / 0 |
| Fuzzy-only + envelope | 83 % [74.5-89.1] / **0** / 17 | 86.0 % [80.5-90.1] / **0** / 14.0 |
| RL-only + envelope | 10 % [5.5-17.4] / **0** / 90 | 7.0 % / **0** / 93.0 |
| RL + Fuzzy + envelope | 83 % / **0** / 17 | 86.0 % / **0** / 14.0 |

Paired exact McNemar (set A): Fuzzy-only vs PPO 77 vs 0 discordant, p=1.3e-23; Fuzzy-only vs RL+Fuzzy 0/0 (identical); Fuzzy-only vs Fuzzy-only+envelope: collisions 12 -> 0 (p=4.9e-4), successes lost 5 vs 0 (p=0.0625, not significant at n=100). Files: `autonomous-parking/experiments/audit_evaluation/{comparison_table.csv,paired_tests.txt,*.csv,*.png}` and `audit_holdout_5000/`.
Mean parking error among successes ~0.31 m (Fuzzy-only), heading ~5 deg. **Strict in-bay** (whole footprint inside the bay): Fuzzy-only 5 % (A) / 0 % (B). Median protrusion ~6.6-7.2 cm.

## 6. Remaining limitations
* PPO is a weak baseline (100,352 steps). The "RL loses" conclusion applies to this checkpoint only.
* Hybrid = Fuzzy-only in outcome; hand-set weights; no learned fusion.
* "Fuzzy-only" is a Sugeno proportional navigator with a forward-corridor brake, not a path planner.
* The envelope shield uses the simulator's own dynamics and ground-truth geometry. No sensor noise, moving obstacles, model mismatch, reverse manoeuvres or multi-step planning are covered. It is a heuristic safety filter, **not a formal or real-world guarantee**. 0 collisions observed in 900 episodes + adversarial tests is evidence, not proof.
* Safety costs completion: ~3-5 points of success become timeouts (a stopped car is safe but unfinished).
* Start-distance distribution is skewed toward close starts (uniform fraction of 6 m), which flatters absolute success rates.
* Perpendicular parking only; other modes are excluded from reported numbers.
* Python >= 3.11 required.

## 7. Known failure scenarios
* Fuzzy-only unshielded: starts >= 4 m away with > 0.5 m lateral offset: 11 of 27 such starts collided in set A vs 1 of 73 otherwise. [RUN]
* PPO: circles / stalls (51-53 % timeouts) and clips neighbours (38-40 % collisions).
* Seed 3009: Fuzzy-only collides after 23 steps; with envelope shield it brakes at ~0.05 m clearance and times out (safe, unsuccessful). Used as the shield demo.
* Shield is only as good as its geometry: if the real obstacle map differs from `env`, no protection.

## 8. Launch and verify commands
See `DEMO_RUNBOOK.md`. Quick: `cd autonomous-parking && python -m pytest tests -q` (81 passed); from project root `python3 viewer/serve.py` (replay); `cd autonomous-parking && python demo.py --controller fuzzy_only+env --seed 3009` (live).

## 9. What the professor will most likely challenge
1. "Is RL actually used?" - only as 25 % of a hand-set blend, no measurable effect; PPO alone is weak. (Open / disclosed)
2. "Is the hybrid better than fuzzy alone?" - No. (Open / disclosed)
3. "Is the car actually in the bay?" - Success tolerance is looser than the bay; strict rate 0-5 %. (Partially mitigated by metric + wording)
4. "Is it safe?" - Model-based shield: zero collisions in sim, not a real-world guarantee, costs completion. (Partially mitigated)
5. "Is the viewer live?" - No, recorded replay; the live demo is `demo.py`. (Resolved: disclosed on screen)
Full answers: `PROFESSOR_DEFENSE_QA.md`.

## 10. Scorecard (0-10, evidence-based)
| Category | Score | Justification |
|---|---|---|
| Core functionality | 7 | Runs end-to-end, 88-89 % success for the fuzzy path [RUN]; PPO path weak. |
| Architecture / code quality | 7 | Clear modules, adapters, 81 tests; some legacy comments, un-cleaned lint in tests. |
| Parking correctness | 6 | Exact collision geometry, collision-gated success; but tolerance looser than bay (strict 0-5 %). |
| RL / model integrity | 4 | Checkpoint validated and unmodified; but undertrained, hybrid ineffective. |
| Fuzzy correctness | 7 | Works and is tested; Sugeno navigator is essentially proportional control; Mamdani shield fixed (D1). |
| Shield behaviour | 7 | 0/900 collisions in sim, tested adversarially; model-based, trades completion, not formal. |
| Test coverage | 8 | 81 passing incl. mutation-checked regressions; no tests of the viewer visuals or training. |
| Evaluation validity / reproducibility | 8 | Shared seeds, CIs, paired tests, holdout block, identical re-run; single scenario family. |
| Visualization quality | 6 | Functional, labelled replay; visual check at presentation size not done [MISSING]. |
| Documentation / demo readiness | 8 | README, runbook, Q&A, audit all consistent with measured data. |

**Verdict: READY WITH CAVEATS.** Present the system as "a reproducible fuzzy-control parking simulator with a baseline RL agent and a model-based safety shield", not as a successful RL+Fuzzy fusion.

## 11. Note on the existing slide deck
The earlier decks (`Autonomous_Parking_Premium_Redesigned.pptx` and the artifact deck) were built from the **GitHub** repository (e.g. PPO 32 %, Mamdani-only fuzzy). They do not match this local project's measured numbers above. Update them before presenting.
