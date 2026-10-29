# Autonomous Parking Using Reinforcement Learning and Fuzzy Logic

Three approaches: Fuzzy-only, RL-only (PPO), RL + Fuzzy. Perpendicular parking only in the project comparison.

## Open the included three-controller demonstration
From this autonomous-parking folder in Terminal:

    python3 ../viewer/serve.py

Open the printed URL and choose a controller. The server automatically tries another port if 8000 is occupied.
Three genuine recordings for the same scenario (seed 1000) are bundled. Fuzzy-only and RL + Fuzzy park; the unchanged RL-only baseline times out. The viewer replays exported trajectories; it is not live inference. The result label is the final outcome, while control mode describes the current frame.

## Your model is now included
models/ppo_only_best.zip is your uploaded original checkpoint, byte-for-byte unchanged, with 100,352 training timesteps recorded in its metadata. No copying or retraining is required to view or reproduce these examples. It remains an unreliable baseline, not a fully trained parking solution.

## What was fixed
- Fuzzy-only now uses zero-order Sugeno direction and distance-to-target rules, plus speed feedback and rolling-resistance compensation. It brakes near the goal. Side neighbours no longer trigger blanket braking that stops it outside the slot.
- RL + Fuzzy combines 25% PPO action with 75% Sugeno fuzzy navigation action. Fuzzy braking has priority, followed by the existing Mamdani obstacle shield. If stationary for 1.2 seconds outside the target tolerance, it switches to fuzzy recovery for the rest of that episode. The viewer labels normal fusion as RL 25% + Fuzzy 75%, and fallback as Fuzzy recovery (PPO stalled). This is a weighted hybrid with supervisory fallback, not a newly trained neural model or a shield-only method.
- Raw RL-only is intentionally unchanged for an honest baseline.
- The reward used by future training no longer repeatedly pays a positive docking bonus while stationary outside the success region. This does not retroactively improve saved weights.
- Collision geometry, vehicle physics, action/observation spaces and parking success thresholds are unchanged.

## Re-run all three with your Python environment
Enter autonomous-parking, then activate your existing virtual environment using its absolute path, for example:

    cd autonomous-parking
    source ~/Downloads/autonomous-parking-project/autonomous-parking/.venv/bin/activate
    python prepare_comparison.py --seed 1000
    python ../viewer/serve.py

For a fresh environment use python3 -m venv .venv, activate .venv/bin/activate, then python -m pip install -r requirements.txt.

## Live 2D demos
From autonomous-parking, close each window before running another:

    python demo.py --controller fuzzy_only --seed 1000
    python demo.py --controller ppo_only --model models/ppo_only_best --seed 1000
    python demo.py --controller ppo_fuzzy_bolted --model models/ppo_only_best --seed 1000

The legacy internal identifier ppo_fuzzy_bolted now selects the documented supervisory RL + Fuzzy approach (weighted fusion, obstacle shield and stall recovery).

## Evaluate fairly

    python -m evaluation.run_experiments --episodes 100 --seed 3000 --out-dir experiments/final_evaluation
    python -m evaluation.compare_controllers --dir experiments/final_evaluation

All three use matching seeds, scenario randomization, physics and success conditions. Failures are retained. Bundled results are from 100 scenarios, seeds 3000-3099, not selected successful runs. See DIAGNOSIS.md for results and limits. Fuzzy-only scored 88/100 and the hybrid 87/100; this does not establish a meaningful difference; do not claim the hybrid is the best or guarantees safety.
The shield_active_frac CSV column is retained for compatibility; it now counts ANY hybrid intervention, including weighted fusion and recovery, not only obstacle shielding.

## Optional future learning
No training was performed for this repair. To improve raw RL, train and evaluate separately. Back up all models first. To continue the best weights without overwriting them, copy ppo_only_best.zip to a new model name and resume using that path. A stage sidecar was not supplied with the upload; restore the matching stage file from your Mac before curriculum resumption, or use an explicitly selected curriculum stage. Do not assume more steps guarantee success.

## Tests

    python -m pytest tests/ -q

Tests cover geometry, vehicle dynamics, environment, fuzzy/shield behaviour, the reproduced stall scenario, recovery reset, and the stationary reward issue. Passing tests does not establish real-world safety. Generic reverse/parallel support remains internal for compatibility but is excluded from this project's comparison. No geometric or rule-based controller is part of this deliverable.

## RL audit (architecture, hyper-parameters, reward) - latest
`docs/RL_AUDIT.md` (in `autonomous-parking/`) documents the full audit. Short version: new checkpoint `models/ppo_v3_rlaudit` (256x256 PPO, containment shaping) reaches 56.0 % strict success on 500 held-out episodes vs 46.4 % for PPO v2 (paired p = 2e-6; 54.4 % and 0 collisions with the envelope shield); fuzzy-precise is statistically equal on the strict metric (57.8 %) and still better on the pose metric. Training-seed variance (SD about 0.12) is as large as any recipe effect, so treat v3 as a good checkpoint, not a proven better recipe. "ARC" could not be identified in the code base (see the report). Run `python demo.py --controller ppo_v3`.
