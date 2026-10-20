"""Why does 'RL + Fuzzy' (ppo_fuzzy_bolted) match Fuzzy-only? (audit P2)

Traces every intermediate action of the hybrid (PPO -> fuzzy -> blend -> brake priority -> Mamdani
shield -> env) and answers: is PPO queried, is its output non-zero/state dependent, how big is its
influence on the applied action, how soon does the stall latch hand control to pure fuzzy?

    python -m evaluation.diagnose_hybrid --seed 3000 --episodes 100 --out experiments/hybrid_diagnosis
Counterfactuals (stall latch disabled; PPO weight 0.5) are diagnostics on TUNING seeds, not results.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from environment.parking_env import ParkingEnv
from environment.scenarios import Kind, RandomizationLevel
from evaluation.run_experiments import FuzzyOnlyAdapter, PPOPlusShieldAdapter

_HERE = Path(__file__).resolve().parent.parent


def rollout(adapter, env, seed):
    adapter.reset()
    obs, _ = env.reset(seed=seed)
    done, n, rec = False, 0, 0
    first_rec = None
    while not done:
        a, _ = adapter.act(obs, env)
        obs, _, te, tr, info = env.step(a)
        n += 1
        if getattr(adapter, "last_recovery", False):
            rec += 1
            first_rec = n if first_rec is None else first_rec
        done = te or tr
    return {"success": bool(info["success"]), "collision": bool(info["collision"]), "timeout": bool(info["timeout"]),
            "steps": n, "recovery_steps": rec, "first_recovery_step": first_rec, "pos_error": float(info["pos_error"])}


def outcome_counts(rs):
    n = len(rs)
    return {k: sum(r[k] for r in rs) / n for k in ("success", "collision", "timeout")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3000)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--tune-seed", type=int, default=7000)
    ap.add_argument("--model", default="models/ppo_only_best")
    ap.add_argument("--out", default="experiments/hybrid_diagnosis")
    a = ap.parse_args()
    model = str(_HERE / a.model)
    out = _HERE / a.out
    out.mkdir(parents=True, exist_ok=True)
    env = ParkingEnv({"randomization": RandomizationLevel.SLOT_POSITION, "scenario_kinds": [Kind.PERPENDICULAR]})
    seeds = list(range(a.seed, a.seed + a.episodes))

    # 1) trace of the real hybrid
    hyb = PPOPlusShieldAdapter(model)
    hyb.trace = []
    rl_acts, first_rec, rec_frac, hyb_res = [], [], [], []
    diff_final_vs_fuzzy, diff_blend_vs_fuzzy, diff_rl_vs_fuzzy, shield_steps, live_steps = [], [], [], 0, 0
    for s in seeds:
        hyb.trace.clear()
        r = rollout(hyb, env, s)
        hyb_res.append(r)
        rec_frac.append(r["recovery_steps"] / r["steps"])
        first_rec.append(r["first_recovery_step"])
        for t in hyb.trace:
            if t["recovery"]:
                continue
            live_steps += 1
            rl_acts.append(t["a_rl"])
            diff_final_vs_fuzzy.append(np.abs(t["a_final"] - t["a_fuzzy"]))
            diff_blend_vs_fuzzy.append(np.abs(t["a_blend"] - t["a_fuzzy"]))
            diff_rl_vs_fuzzy.append(np.abs(t["a_rl"] - t["a_fuzzy"]))
            shield_steps += t["shield"]
    rl = np.array(rl_acts)
    fo = FuzzyOnlyAdapter()
    fo_res = [rollout(fo, env, s) for s in seeds]
    same_outcome = sum((x["success"], x["collision"], x["timeout"]) == (y["success"], y["collision"], y["timeout"])
                       for x, y in zip(hyb_res, fo_res))
    same_steps = sum(x["steps"] == y["steps"] for x, y in zip(hyb_res, fo_res))

    # 2) PPO query on identical states: states from the fuzzy-only rollouts
    ppo = hyb.policy
    d_rl_fz, d_blend, n_states = [], [], 0
    uniq = set()
    for s in seeds[:30]:
        fo.reset()
        obs, _ = env.reset(seed=s)
        done = False
        while not done:
            a_f = fo.controller.act(obs, env.fuzzy_inputs(), env.slot_bearing_deg())
            a_r = ppo.act(obs)
            uniq.add(tuple(np.round(a_r, 3)))
            d_rl_fz.append(np.abs(a_r - a_f))
            d_blend.append(np.abs(0.25 * a_r + 0.75 * a_f - a_f))
            n_states += 1
            obs, _, te, tr, _ = env.step(a_f)
            done = te or tr

    # 3) counterfactuals on tuning seeds: latch disabled / PPO weight .5
    tseeds = list(range(a.tune_seed, a.tune_seed + a.episodes))
    cf = {}
    cf["fuzzy_only"] = outcome_counts([rollout(FuzzyOnlyAdapter(), env, s) for s in tseeds])
    cf["hybrid_default(w=.25,latch=12)"] = outcome_counts([rollout(PPOPlusShieldAdapter(model), env, s) for s in tseeds])
    cf["hybrid_no_latch(w=.25)"] = outcome_counts([rollout(PPOPlusShieldAdapter(model, stall_steps=10**9), env, s) for s in tseeds])
    cf["hybrid_no_latch(w=.50)"] = outcome_counts([rollout(PPOPlusShieldAdapter(model, rl_weight=.5, stall_steps=10**9), env, s) for s in tseeds])

    report = {
        "episodes": a.episodes, "seeds": f"{a.seed}..{a.seed + a.episodes - 1}",
        "ppo_queried": True, "ppo_mean_action_live_steps": rl.mean(0).round(4).tolist(),
        "ppo_std_action_live_steps": rl.std(0).round(4).tolist(),
        "ppo_min": rl.min(0).round(3).tolist(), "ppo_max": rl.max(0).round(3).tolist(),
        "ppo_distinct_actions_on_fuzzy_states": len(uniq), "ppo_states_checked": n_states,
        "live_steps_before_latch": live_steps, "total_steps": int(sum(r["steps"] for r in hyb_res)),
        "mean_first_recovery_step": float(np.mean([f for f in first_rec if f is not None])),
        "episodes_latched": int(sum(f is not None for f in first_rec)),
        "latch_step_histogram": {str(k): int(v) for k, v in zip(*np.unique([f for f in first_rec if f is not None], return_counts=True))},
        "mean_recovery_fraction": float(np.mean(rec_frac)),
        "mean_abs_diff_ppo_vs_fuzzy_action[steer,thb]": np.mean(diff_rl_vs_fuzzy, 0).round(4).tolist(),
        "mean_abs_diff_blend_vs_fuzzy[steer,thb]_before_braking_priority": np.mean(diff_blend_vs_fuzzy, 0).round(4).tolist(),
        "mean_abs_diff_applied_vs_fuzzy[steer,thb]_live_steps": np.mean(diff_final_vs_fuzzy, 0).round(4).tolist(),
        "mean_abs_diff_ppo_vs_fuzzy_on_fuzzy_states": np.mean(d_rl_fz, 0).round(4).tolist(),
        "mean_abs_diff_blend_vs_fuzzy_on_fuzzy_states": np.mean(d_blend, 0).round(4).tolist(),
        "mamdani_shield_active_live_steps": shield_steps,
        "episodes_same_outcome_as_fuzzy_only": same_outcome, "episodes_same_step_count_as_fuzzy_only": same_steps,
        "hybrid_outcomes": outcome_counts(hyb_res), "fuzzy_outcomes": outcome_counts(fo_res),
        "tuning_seed_counterfactuals": {"seeds": f"{a.tune_seed}..{a.tune_seed + a.episodes - 1}", **cf},
    }
    (out / "hybrid_diagnosis.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
