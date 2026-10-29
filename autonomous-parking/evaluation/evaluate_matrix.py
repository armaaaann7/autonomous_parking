"""Resumable controller x seed-block evaluation (audit P1/P4). Every run uses success_mode='contained'
and records the legacy (pose-criterion) outcome from the same trajectory, so one run answers both.

    python -m evaluation.evaluate_matrix --controllers fuzzy_only,fuzzy_only+env --block A:3000:100,B:5000:200,C:9000:200 \
        --out-dir experiments/corrected_evaluation
Re-run the same command after an interruption: finished 50-episode parts are skipped.
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent.parent
PART = 50

WORKER = r"""
import sys
from pathlib import Path
from environment.scenarios import Kind
from evaluation.run_experiments import make_adapter, run_episodes
name, seed, n, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
ad = make_adapter(name, "models/ppo_only_best")
run_episodes(ad, n, list(range(seed, seed + n)), [Kind.PERPENDICULAR], Path(out), env_extra={"success_mode": "contained"})
"""


def part_file(out_dir, block, ctrl, seed):
    return out_dir / "parts" / block / f"{ctrl.replace('+', '_').replace('/', '__').replace(':', '-')}.{seed}.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", required=True)
    ap.add_argument("--block", default="A:3000:100,B:5000:200,C:9000:200", help="name:first_seed:episodes, comma separated")
    ap.add_argument("--out-dir", default="experiments/corrected_evaluation")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out_dir = _HERE / a.out_dir
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    jobs = []
    for b in a.block.split(","):
        bn, s0, n = b.split(":")
        s0, n = int(s0), int(n)
        for c in a.controllers.split(","):
            for k in range(0, n, PART):
                jobs.append((bn, c, s0 + k, min(PART, n - k)))

    def done(job):
        bn, c, seed, n = job
        f = part_file(out_dir, bn, c, seed)
        try:
            return f.is_file() and len(pd.read_csv(f)) == n
        except pd.errors.EmptyDataError:
            return False

    todo = [j for j in jobs if not done(j)]
    print(f"{len(jobs) - len(todo)}/{len(jobs)} parts already done", flush=True)

    def run(job):
        bn, c, seed, n = job
        f = part_file(out_dir, bn, c, seed)
        f.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-c", WORKER, c, str(seed), str(n), str(f)], cwd=_HERE, env=env, check=True,
                       capture_output=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(run, todo))
    # assemble complete (block, controller) CSVs
    for b in a.block.split(","):
        bn, s0, n = b.split(":")
        s0, n = int(s0), int(n)
        for c in a.controllers.split(","):
            fs = [part_file(out_dir, bn, c, s0 + k) for k in range(0, n, PART)]
            if all(f.is_file() for f in fs) and all(done((bn, c, s0 + k, min(PART, n - k))) for k in range(0, n, PART)):
                d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
                d["episode"] = range(len(d))
                (out_dir / f"set_{bn}").mkdir(parents=True, exist_ok=True)
                d.to_csv(out_dir / f"set_{bn}" / f"{c.replace('+', '_').replace('/', '__').replace(':', '-')}.csv", index=False)
    print("assembled", flush=True)


if __name__ == "__main__":
    main()
