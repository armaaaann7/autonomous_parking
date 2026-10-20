# Autonomous Parking — project layout

| Folder | What it is |
|---|---|
| `simple-parking/parking.html` | Standalone single-file demo. Double-click to open; no install, no internet. Perpendicular + parallel, auto or manual (arrow keys). |
| `autonomous-parking/` | Python project: simulator, controllers (rule-based, fuzzy, PPO, PPO+shield), training, evaluation. |
| `viewer/` | three.js 3D viewer that replays trajectories exported from the Python project. |

## Python setup (python 3.10+; 3.12 recommended)
    cd autonomous-parking
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python -m pytest -q                      # 47 tests

## Train (long; resumable)
    python -m rl.train_ppo --curriculum                       # PPO
    python -m rl.train_ppo --curriculum --shielded            # PPO trained through the shield
    python -m rl.train_ppo --curriculum --reverse --kinds parallel   # parallel (reverse gear)
    # add --resume to continue; best checkpoints are saved as models/<name>_best.zip

## Evaluate
    python -m evaluation.run_experiments --ppo-only-model models/ppo_only_best --ppo-shielded-model models/ppo_shielded_best
    python -m evaluation.run_experiments --reverse --kinds parallel --full-approach
    python -m evaluation.compare_controllers

## 3D viewer
    cd autonomous-parking
    python export_trajectory.py                                   # perpendicular, writes viewer/assets/data/trajectory.json
    python export_trajectory.py --controller parallel_geometric   # parallel
    cd ../viewer && python3 serve.py                              # open http://localhost:8000

`export_trajectory.py` lives in `autonomous-parking/` (not `rl/`); `serve.py` runs from `viewer/`.

## Honest status
- Parallel parking: fixed for the geometric controller (needs reverse gear + 8 m gap). PPO not yet retrained for it.
- The shield shows no statistically significant benefit; baselines are weak, so "PPO beats baselines" is weakly informative.
- Details: `autonomous-parking/docs/REVIEW.md`.
