# Changelog

## Final milestone (containment, PPO v2, hybrid diagnosis, shield verification, viewer audit)
Nothing was deleted. The original PPO checkpoint (`models/ppo_only_best.zip`), all earlier result folders, and the prior viewer recordings (`viewer/assets/data_prev_audit1/`) are preserved.

### Metric
* `ParkingEnv(success_mode="pose"|"contained")`. `"pose"` is the legacy default (centre <= 0.40 m, heading < 15 deg, speed < 0.20 m/s for 2 steps, no collision). `"contained"` additionally requires the whole 4.2 x 1.8 m car inside the 2.4 x 4.8 m bay lines. The geometric definition was **not** loosened.
* New `info` keys: `contained`, `bay_protrusion`, `pose_ok`, `slot_frame_error`. New CSV columns: `long_error_m`, `lat_error_m`, `success_mode`, `legacy_outcome`, `pose_success_step`, `ps_*`. Legacy outcomes are reconstructed from the same trajectory and were verified to match the stored legacy CSV for 100/100 seeds.
### Controllers
* `FuzzyOnlyController(stop_radius=0.32, heading_gain=0.6)`; new adapter `fuzzy_precise` (stop radius 0.10 m, chosen on tuning seeds 7000-7099 only; held-out seeds were never used for tuning).
* New adapters `ppo_v2` (retrained PPO) and `ppo_v2_fuzzy` (PPO v2 25 % + fuzzy 75 %, **no** stall latch). `make_adapter` accepts `name@r` / `name@rxhg` for tuning.
### PPO
* Retrained with the curriculum recipe: 3 seeds, ~420k steps each (`models/exp_ppo_cur_s{1,2,3}[_best]`). Selected `ppo_v2_curriculum_selected.zip` on validation seeds 7000-7099. See `autonomous-parking/models/README.md`.
* `rl/train_ppo.py`: `--val-seed`, `--val-episodes`, tensorboard fallback, best-so-far persisted across `--resume`.
### Shield
* `integration/envelope_shield.py`: non-finite actions fail closed (brake); wrong shape raises `ValueError`; finite out-of-range values saturate; reverse raises `NotImplementedError`.
* `evaluation/shield_stress.py` (adversarial states, sub-step interpolation audit, brake-mismatch, timeout anatomy).
### Viewer
* Recordings now carry `protrusion`, `pose_success_step`, `success_mode`, `outcome`; 24 episodes (seeds 9170 and 1000 + shield demos at 3009).
* **Bug fixed:** the painted bay was 0.2 m too long (`car.length + 0.8`; env bay is `+0.6`).
* HUD: REPLAY banner moved above the control bar, explanatory text hidden at heights <= 760 px, HUD width capped (all found by actually rendering the page).
### Tests
* 131 tests (was 81): `test_containment_metric.py` (23), `test_hybrid_integrity.py` (8), `test_shield_stress.py` (19).
* `test_trajectory_export_is_consistent...` updated: it asserted the old "success = no collision and no timeout" rule, which is wrong under the strict metric. `test_hybrid_integrity` made independent of the working directory.
* `demo.py --controller` now accepts every adapter (`fuzzy_precise`, `ppo_v2`, `ppo_v2_fuzzy`, each with `+env`); previously only the original three. Verified by running `demo.py --controller fuzzy_precise+env --seed 3009` and `--controller ppo_v2` headless (500 frames each, exit 0).
* New deck: `Autonomous_Parking_Final_Evidence.pptx` (14 slides). Earlier decks untouched.
