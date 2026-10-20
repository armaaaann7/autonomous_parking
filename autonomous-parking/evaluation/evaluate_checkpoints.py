"""Evaluate one or more PPO checkpoints with the standard protocol (perpendicular, shared seeds,
failures retained), in parallel. Used for the audit's checkpoint selection (validation seeds) and the
single final test on fresh seeds.

    python -m evaluation.evaluate_checkpoints --models models/a,models/b --seed 7000 --episodes 100 \
        --out-dir experiments/ppo_retrain/val [--success-mode contained] [--envelope]
"""
import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent.parent

WORKER = r"""
import sys
from pathlib import Path
from environment.scenarios import Kind
from evaluation.run_experiments import PPOOnlyAdapter, EnvelopeAdapter, run_episodes
model, seed, n, out, mode, env_shield = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6] == "1"
ad = PPOOnlyAdapter(model)
if env_shield:
    ad = EnvelopeAdapter(ad)
run_episodes(ad, n, list(range(seed, seed + n)), [Kind.PERPENDICULAR], Path(out), env_extra={"success_mode": mode})
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="comma list of checkpoint paths (without .zip), relative to the project")
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--out-dir", default="experiments/ppo_retrain/val")
    ap.add_argument("--success-mode", default="pose", choices=["pose", "contained"])
    ap.add_argument("--envelope", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--parts", type=int, default=2, help="split the episodes into this many independent jobs (resumable)")
    a = ap.parse_args()
    out_dir = _HERE / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    import os
    env = {**os.environ, **env}

    part = max(1, a.episodes // a.parts)

    def one(job):
        m, k = job
        out = out_dir / "parts" / f"{Path(m).name}.{a.success_mode}{'.env' if a.envelope else ''}.seed{a.seed + k * part}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:                                                      # finished earlier: resumable across tool-call time limits
            if out.is_file() and len(pd.read_csv(out)) == part:
                return m, out
        except pd.errors.EmptyDataError:
            pass                                                  # partial file from a killed job -> redo
        subprocess.run([sys.executable, "-c", WORKER, str(_HERE / m), str(a.seed + k * part), str(part), str(out), a.success_mode,
                        "1" if a.envelope else "0"], cwd=_HERE, env=env, check=True, capture_output=True)
        return m, out

    jobs = [(m, k) for m in a.models.split(",") for k in range(a.parts)]
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(one, jobs))
    results = []
    for m in a.models.split(","):
        d = pd.concat([pd.read_csv(o) for mm, o in res if mm == m], ignore_index=True)
        out = out_dir / (Path(m).name + f".{a.success_mode}{'.env' if a.envelope else ''}.csv")
        d.to_csv(out, index=False)
        results.append((m, out))
    rows = []
    for m, out in results:
        d = pd.read_csv(out)
        rows.append({"model": m, "n": len(d), "success": d.success.mean(), "collision": d.collision.mean(), "timeout": d.timeout.mean(),
                     "in_bay_end": d.in_bay.mean(), "mean_pos_err_success": d.loc[d.success, "pos_error_m"].mean()})
    tab = pd.DataFrame(rows)
    print(tab.round(3).to_string(index=False))
    tab.to_csv(out_dir / f"summary_seed{a.seed}_{a.success_mode}{'_env' if a.envelope else ''}.csv", index=False)


if __name__ == "__main__":
    main()
