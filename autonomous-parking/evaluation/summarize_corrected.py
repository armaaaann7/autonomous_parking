"""Summaries for experiments/corrected_evaluation (audit P1/P4): both outcome definitions, per seed block
and pooled, with Wilson 95 % CIs and paired exact McNemar tests on the pooled 500 episodes.

    python -m evaluation.summarize_corrected --dir experiments/corrected_evaluation
"""
import argparse
from pathlib import Path

import pandas as pd

from evaluation.metrics import mcnemar_exact, wilson_interval

LABEL = {
    "fuzzy_only": "Fuzzy-only (legacy, stop 0.32 m)", "fuzzy_precise": "Fuzzy-precise (stop 0.10 m)",
    "ppo_only": "PPO original (100k)", "ppo_v2": "PPO v2 (retrained, 420k)",
    "ppo_fuzzy_bolted": "RL+Fuzzy (original hybrid)", "ppo_v2_fuzzy": "PPO v2 + Fuzzy 25/75 (no latch)",
}
ORDER = list(LABEL)


def label(c):
    base = c[:-4] if c.endswith("_env") else c
    return LABEL.get(base, base) + (" + envelope shield" if c.endswith("_env") else "")


def row(d):
    n = len(d)
    ls = int((d.legacy_outcome == "success").sum())
    lc = int((d.legacy_outcome == "collision").sum())
    lt = int((d.legacy_outcome == "timeout").sum())
    ss, sc, st = int(d.success.sum()), int(d.collision.sum()), int(d.timeout.sum())
    lo, hi = wilson_interval(ls, n)
    slo, shi = wilson_interval(ss, n)
    ok = ~d.collision.astype(bool)
    return {"n": n, "pose_success": ls / n, "pose_ci": f"[{lo*100:.0f}-{hi*100:.0f}]", "pose_collision": lc / n, "pose_timeout": lt / n,
            "contained_success": ss / n, "contained_ci": f"[{slo*100:.0f}-{shi*100:.0f}]", "contained_collision": sc / n,
            "contained_timeout": st / n,
            "ps_in_bay_given_pose_success": float(pd.to_numeric(d.loc[d.legacy_outcome == 'success', 'ps_in_bay']).astype(bool).mean()) if ls else float("nan"),
            "ps_protrusion_median_m": float(pd.to_numeric(d.loc[d.legacy_outcome == 'success', 'ps_protrusion_m']).median()) if ls else float("nan"),
            "final_abs_long_err_m": float(d.loc[ok, "long_error_m"].abs().mean()), "final_abs_lat_err_m": float(d.loc[ok, "lat_error_m"].abs().mean()),
            "final_heading_err_deg": float(d.loc[ok, "heading_error_deg"].mean()),
            "min_clearance_m": float(d.min_clearance_m.min())}


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="experiments/corrected_evaluation")
    a = ap.parse_args()
    root = Path(a.dir)
    sets = sorted(p.name for p in root.glob("set_*"))
    ctrls = sorted({f.stem for s in sets for f in (root / s).glob("*.csv")},
                   key=lambda c: (ORDER.index(c[:-4] if c.endswith("_env") else c) if (c[:-4] if c.endswith("_env") else c) in ORDER else 99, c.endswith("_env")))
    allrows, pooled = [], {}
    for c in ctrls:
        parts = []
        for s in sets:
            f = root / s / f"{c}.csv"
            if f.is_file():
                d = pd.read_csv(f)
                parts.append(d.assign(seed_block=s))
                allrows.append({"set": s, "controller": label(c), "key": c, **row(d)})
        if parts:
            pooled[c] = pd.concat(parts, ignore_index=True)
            allrows.append({"set": "pooled", "controller": label(c), "key": c, **row(pooled[c])})
    tab = pd.DataFrame(allrows)
    tab.to_csv(root / "summary_all.csv", index=False)
    fmt = tab.copy()
    for col in ["pose_success", "pose_collision", "pose_timeout", "contained_success", "contained_collision", "contained_timeout",
                "ps_in_bay_given_pose_success"]:
        fmt[col] = (fmt[col] * 100).round(1)
    for col in ["ps_protrusion_median_m", "final_abs_long_err_m", "final_abs_lat_err_m", "final_heading_err_deg", "min_clearance_m"]:
        fmt[col] = fmt[col].round(3)
    lines = []
    for s in sets + ["pooled"]:
        sub = fmt[fmt.set == s].drop(columns=["set", "key"])
        lines.append(f"\n### {s}\n\n" + md(sub))
    # paired tests on pooled data
    pairs = [("fuzzy_only", "fuzzy_precise"), ("fuzzy_precise", "ppo_v2"), ("fuzzy_only", "ppo_only"), ("ppo_only", "ppo_v2"),
             ("fuzzy_only", "ppo_fuzzy_bolted"), ("fuzzy_precise", "ppo_v2_fuzzy"), ("ppo_v2", "ppo_v2_fuzzy"),
             ("fuzzy_precise", "fuzzy_precise_env"), ("fuzzy_only", "fuzzy_only_env"), ("ppo_v2", "ppo_v2_env")]
    out = ["\n### Paired exact McNemar tests (pooled, same seeds). A-only / B-only = discordant episodes\n",
           "| A | B | outcome | A-only | B-only | p |", "|---|---|---|---|---|---|"]
    for x, y in pairs:
        if x in pooled and y in pooled and len(pooled[x]) == len(pooled[y]):
            for nm, fx in [("pose success", lambda d: d.legacy_outcome == "success"), ("contained success", lambda d: d.success.astype(bool)),
                           ("contained collision", lambda d: d.collision.astype(bool))]:
                na, nb, p = mcnemar_exact(fx(pooled[x]).values, fx(pooled[y]).values)
                out.append(f"| {label(x)} | {label(y)} | {nm} | {na} | {nb} | {p:.3g} |")
    (root / "summary_all.md").write_text("\n".join(lines + out))
    print("\n".join(lines + out))


if __name__ == "__main__":
    main()
