"""Documented tuning of the containment-aware fuzzy stop radius (audit P1).

Procedure: for each candidate stop radius run the STRICT ('contained') success mode on TUNING seeds
7000-7099 only. The chosen radius is then evaluated once on held-out seeds (3000-3099, 5000-5199 and
the untouched block 9000-9199) by evaluation.run_experiments. Nothing here sees the held-out seeds.

    python -m evaluation.tune_precision --radii 0.32,0.2,0.12 --seed 7000 --episodes 100
"""
import argparse
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", default="0.32,0.25,0.2,0.15,0.1,0.06,0.03", help="comma list of specs: r or rxheading_gain (e.g. 0.06x1.2)")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out-dir", default="experiments/containment_tuning")
    a = ap.parse_args()
    procs = []
    for r in a.radii.split(","):
        procs.append(subprocess.Popen([sys.executable, "-m", "evaluation.run_experiments", "--episodes", str(a.episodes),
                                       "--seed", str(a.seed), "--out-dir", a.out_dir, "--success-mode", "contained",
                                       "--controllers", f"fuzzy_precise@{r}"], cwd=_HERE))
    for p in procs:
        p.wait()
