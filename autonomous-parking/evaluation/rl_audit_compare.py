"""Held-out comparison for the RL audit: pooled sets A+B+C (500 episodes per checkpoint), strict (whole car in bay)
and legacy pose outcomes, Wilson CIs, per-arm mean/SD across training seeds, a seed-level bootstrap CI for the
arm difference, and paired exact McNemar tests (identical seeds) for chosen checkpoint pairs.

    python -m evaluation.rl_audit_compare --dir experiments/rl_audit/heldout_eval \
        --pairs rl_audit_C2_s2_best:exp_ppo_cur_s3_best --out experiments/rl_audit/heldout_summary.md
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import mcnemar_exact, wilson_interval
from evaluation.rl_audit_summary import split_name, stats
from evaluation.summarize_corrected import md


def load(dirpath: Path) -> dict:
    out = {}
    for f in sorted((dirpath / "set_A").glob("*.csv")):
        parts = [pd.read_csv(dirpath / s / f.name) for s in ("set_A", "set_B", "set_C") if (dirpath / s / f.name).is_file()]
        out[f.stem.replace("ppo-models__", "")] = pd.concat(parts, ignore_index=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--pairs", default="", help="comma list of a:b checkpoint stems (names as in the csv files, without ppo-models__)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    data = load(Path(a.dir))
    rows = []
    for k, d in data.items():
        arm, seed = split_name(k)
        rows.append({"arm": arm, "seed": seed, "ckpt": k, **stats(d)})
    per = pd.DataFrame(rows)
    lines = ["## Per checkpoint (pooled A+B+C, n=500)\n", md(per[["arm", "seed", "ckpt", "pose_succ", "strict_succ", "strict_ci", "strict_coll", "strict_timeout", "steps_succ", "traj_m_succ", "pos_err_m", "head_err_deg", "min_clear_p5"]].round(3).astype(str)), ""]
    agg = per.groupby("arm").agg(seeds=("seed", "count"), pose_succ=("pose_succ", "mean"), strict_mean=("strict_succ", "mean"), strict_sd=("strict_succ", "std"),
                                 strict_min=("strict_succ", "min"), strict_max=("strict_succ", "max"), coll_mean=("strict_coll", "mean"), coll_sd=("strict_coll", "std"),
                                 timeout=("strict_timeout", "mean"), steps=("steps_succ", "mean"), traj=("traj_m_succ", "mean")).reset_index()
    lines += ["## Per arm (mean over training seeds; SD across seeds)\n", md(agg.round(3).astype(str)), ""]
    arms = list(agg.arm)
    rng = np.random.default_rng(0)
    lines.append("## Arm difference, bootstrap over training seeds (strict success, 95% CI; resamples seeds with replacement)\n")
    for i, x in enumerate(arms):
        for y in arms[i + 1:]:
            sx, sy = per[per.arm == x].strict_succ.values, per[per.arm == y].strict_succ.values
            diffs = [rng.choice(sx, len(sx)).mean() - rng.choice(sy, len(sy)).mean() for _ in range(a.boot)]
            lines.append(f"- {x} - {y}: {sx.mean()-sy.mean():+.3f}  CI [{np.percentile(diffs,2.5):+.3f}, {np.percentile(diffs,97.5):+.3f}]  (n={len(sx)} vs {len(sy)} seeds)")
    lines.append("")
    if a.pairs:
        lines.append("## Paired exact McNemar (same 500 episodes)\n")
        for pr in a.pairs.split(","):
            x, y = pr.split(":")
            for col, nm in (("success", "strict success"), ("pose", "pose success")):
                fx = data[x].success.astype(bool).values if col == "success" else (data[x].legacy_outcome == "success").values
                fy = data[y].success.astype(bool).values if col == "success" else (data[y].legacy_outcome == "success").values
                nx, ny, p = mcnemar_exact(fx, fy)
                lines.append(f"- {x} vs {y} [{nm}]: {x} only {nx}, {y} only {ny}, p={p:.4g}; {fx.mean():.3f} vs {fy.mean():.3f}")
            cx, cy = data[x].collision.astype(bool).values, data[y].collision.astype(bool).values
            nx, ny, p = mcnemar_exact(cx, cy)
            lines.append(f"- {x} vs {y} [collision]: {x} only {nx}, {y} only {ny}, p={p:.4g}; {cx.mean():.3f} vs {cy.mean():.3f}")
    txt = "\n".join(lines)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt)
        per.to_csv(str(a.out).replace(".md", "_per_ckpt.csv"), index=False)


if __name__ == "__main__":
    main()
