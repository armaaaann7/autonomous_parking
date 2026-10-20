# Demo Runbook

Everything below was exercised in the audit unless marked **[not tested]**.

## 1. Prepare (once, ~5 min, needs internet for pip)
    cd autonomous-parking
    python3 --version            # must be 3.11 or newer
    python3 -m venv .venv && source .venv/bin/activate
    python -m pip install -r requirements.txt
    python -m pytest tests -q    # expect: 131 passed
Do this the day before, not on stage. Confirm `models/ppo_only_best.zip` exists (do not retrain; do not delete).

## 2. Launch
**A. Recorded 3D viewer (the safe main demo).** From the project root:
    python3 viewer/serve.py
Open the printed address in Chrome. The banner says *recorded replay*: say this aloud, it is not live inference. Dropdown: Fuzzy-only, RL-only (PPO), RL + Fuzzy, and their "+ envelope shield" variants (seed 1000), plus the shield demo (seed 3009). **Look at it once on the presentation screen**: WebGL rendering at presentation size was not visually verified [not tested].

**B. Live 2D demo (real inference, needs a display).** From `autonomous-parking/`:
    python demo.py --controller fuzzy_only --seed 1000
    python demo.py --controller fuzzy_only+env --seed 3009
No display / backup: `SDL_VIDEODRIVER=dummy python demo.py --controller fuzzy_only+env --seed 3009 --save-gif demo.gif` and show the GIF.

## 3. Demonstrate (suggested 5 minutes)
1. Viewer: Fuzzy-only seed 1000 parks; point at distance/heading readouts and the success criterion note ("centre within 0.40 m, heading 15 deg, stopped").
2. RL-only: shows weak behaviour (7-11 % success). Say plainly it is a 100k-step baseline.
3. RL + Fuzzy: same outcome as Fuzzy-only. Say so before you are asked.
4. Shield demo seed 3009: Fuzzy-only collides at step 23; with the envelope shield the car stops at ~5 cm clearance (red marker, "safety shield: INTERVENING"), and the episode times out. This is the honest trade-off: safe, not successful.
5. Show `experiments/audit_evaluation/comparison_table.csv` (or the bar chart PNG) with the CIs.

## 4. Reset
* Viewer: reload the page; or Ctrl+C the server and restart.
* Regenerate recordings if the JSON is damaged: `cd autonomous-parking && python prepare_comparison.py --seed 1000`, then re-export the shield demo: `python export_trajectory.py --controller fuzzy_only+env --seed 3009 --out ../viewer/assets/data/shield_demo_fuzzy_only_env_seed3009.json`.
* Do **not** delete `models/`, `experiments/final_evaluation`, `before_fix`, `fusion_development`.

## 5. Troubleshooting
| Symptom | Cause / fix |
|---|---|
| pip cannot find `numpy==2.4.4` | Python < 3.11. Use 3.11+. |
| `ImportError libcublasLt...` from torch | Install the full torch wheel (`pip install torch`) so its CUDA libs are present; CPU use is fine. |
| Clear error about checkpoint/obs shape | `models/ppo_only_best.zip` missing or not 14-in/2-out; restore the original file. |
| Viewer blank | WebGL disabled or port busy: use Chrome, hardware acceleration on, try another port per `serve.py` output. three.js is vendored (`viewer/vendor/`), so no internet is needed. |
| `pygame` window cannot open | No display; use the dummy driver and `--save-gif`. |
| Results differ from slides | Slides built earlier from the GitHub repo differ from this local project (PPO 32 % there). Quote the numbers in README/audit. |
| Evaluation takes long | 100 episodes x 6 controllers is a few minutes; use the saved CSVs on stage instead. |

## 6. Honesty checklist before presenting
Say: recorded replay not live (viewer); hybrid equals fuzzy; PPO is weak; shield is model-based and costs completion; "success" is within tolerance, not fully inside lines.

## 7. Final-milestone additions
* Viewer now has 24 episodes. Best story order: seed 9170 **Fuzzy-only (legacy)** -> HUD says "stopped within pose tolerance but NOT inside the bay" -> **Fuzzy-precise** (inside the bay) -> **PPO v2** (inside the bay, 0.09 m error) -> shield demos at seed 3009 (collision vs freeze). Say aloud that 9170 was chosen for contrast and is not typical.
* Numbers to quote: `experiments/corrected_evaluation/summary_all.md` (n = 500). Strict success: Fuzzy-only 1.0 %, Fuzzy-precise 57.8 %, PPO v2 46.4 %, original PPO 1.8 %.
* If the 3D scene is slow on the demo machine, use hardware acceleration; headless software rendering ran < 1 fps.
* Reproduce the strict matrix: `python -m evaluation.evaluate_matrix` (resumable) then `python -m evaluation.summarize_corrected`.
