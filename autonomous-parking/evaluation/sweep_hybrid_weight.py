"""Documented sweep of the RL weight in the supervisory blend (audit P2). TUNING seeds only.

Variant under test: PPO + Sugeno-fuzzy blend with the Mamdani obstacle shield and the stall latch
DISABLED (safe_threshold=0, stall_steps=inf) so the PPO contribution is isolated; collision safety is
provided separately by the '+env' braking-envelope shield. Weight 0.0 must reproduce Fuzzy-only.

    python -m evaluation.sweep_hybrid_weight --seed 7000 --episodes 100 --weights 0,0.1,0.25,0.5,0.75,1.0
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from evaluation.diagnose_hybrid import outcome_counts, rollout
from evaluation.run_experiments import PPOPlusShieldAdapter

_HERE = Path(__file__).resolve().parent.parent


def run(args):
    w, seed, n, mode, model = args
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR],
                      "success_mode": mode})
    ad = PPOPlusShieldAdapter(str(_HERE / model), rl_weight=w, safe_threshold=0.0, stall_steps=10 ** 9)
    res = [rollout(ad, env, s) for s in range(seed, seed + n)]
    return w, outcome_counts(res), sum(r["steps"] for r in res) / n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--weights", default="0,0.1,0.25,0.5,0.75,1.0")
    ap.add_argument("--success-mode", default="pose")
    ap.add_argument("--model", default="models/ppo_only_best")
    ap.add_argument("--out", default="experiments/hybrid_diagnosis/weight_sweep.json")
    a = ap.parse_args()
    ws = [float(x) for x in a.weights.split(",")]
    with ProcessPoolExecutor(len(ws)) as ex:
        rows = list(ex.map(run, [(w, a.seed, a.episodes, a.success_mode, a.model) for w in ws]))
    out = {"seeds": f"{a.seed}..{a.seed + a.episodes - 1}", "success_mode": a.success_mode, "model": a.model,
           "variant": "no Mamdani shield, no stall latch", "rows": [{"rl_weight": w, **o, "mean_steps": st} for w, o, st in rows]}
    p = _HERE / a.out
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    for r in out["rows"]:
        print(r)
