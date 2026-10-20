"""Mamdani fuzzy safety-shield logic (blueprint Part 4).

Fixes relative to the blueprint's rule base -- see docs/REVIEW.md:

* The blueprint's ``brake_cmd`` rules never cover ``d_obs=medium AND speed=medium``. With
  skfuzzy, if every rule for a consequent evaluates to (near-)zero firing strength for a given
  input, ``ControlSystemSimulation.compute()`` raises rather than returning a default -- this
  combination is a real gap, not just an edge case, because "medium" is the middle third of both
  universes. Rule B6 below closes it.
* Likewise ``speed_adj`` only had rules for ``very_close``/``close``/``far&slow`` -- ``medium``
  d_obs (any speed) and ``far`` combined with ``medium``/``fast`` speed had no rule at all.
  Rules S4-S5 below add an explicit "maintain" default so every input in range produces an
  output.
* The three outputs (brake, steer correction, speed adjustment) are built as three independent
  ``ControlSystem``s rather than one system with multi-consequent rules. This mirrors the
  blueprint's own Part 16 Step 6 code (which only wired up ``brake_cmd``) and keeps each output
  unit-testable in isolation, per Phase 7's test plan.
"""
from __future__ import annotations

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from .membership_configs import CONSERVATIVE

_MF = {"trimf": fuzz.trimf, "trapmf": fuzz.trapmf}


def _build_var(kind, name, universe, spec):
    var = kind(universe, name)
    for label, (mf, params) in spec.items():
        var[label] = _MF[mf](universe, params)
    return var


class FuzzySystem:
    """Builds the three shield outputs (brake, steer correction, speed adjustment) from
    ``(d_obs, angle_err, speed)``. Pass a different ``config`` (see membership_configs.py) to
    run the rule-set-sensitivity ablation (blueprint Part 20)."""

    def __init__(self, config: dict | None = None):
        cfg = config or CONSERVATIVE
        self.d_obs = _build_var(ctrl.Antecedent, "d_obs", np.arange(0, 10.01, 0.05), cfg["d_obs"])
        self.speed = _build_var(ctrl.Antecedent, "speed", np.arange(0, 3.01, 0.02), cfg["speed"])
        self.angle_err = _build_var(ctrl.Antecedent, "angle_err", np.arange(-90, 90.01, 0.5), cfg["angle_err"])
        self.brake_cmd = _build_var(ctrl.Consequent, "brake_cmd", np.arange(0, 1.001, 0.005), cfg["brake_cmd"])
        self.steer_corr = _build_var(ctrl.Consequent, "steer_corr", np.arange(-1, 1.001, 0.005), cfg["steer_corr"])
        self.speed_adj = _build_var(ctrl.Consequent, "speed_adj", np.arange(-1, 1.001, 0.005), cfg["speed_adj"])

        d, s, a = self.d_obs, self.speed, self.angle_err
        b, sc, sa = self.brake_cmd, self.steer_corr, self.speed_adj

        brake_rules = [
            ctrl.Rule(d["very_close"], b["hard"]),                      # B1
            ctrl.Rule(d["close"] & s["fast"], b["hard"]),                # B2
            ctrl.Rule(d["close"] & s["medium"], b["moderate"]),          # B3
            ctrl.Rule(d["close"] & s["slow"], b["moderate"]),            # B4
            ctrl.Rule(d["medium"] & s["fast"], b["moderate"]),           # B5
            ctrl.Rule(d["medium"] & s["medium"], b["moderate"]),         # B6 -- gap fix
            ctrl.Rule(d["medium"] & s["slow"], b["none"]),               # B7
            ctrl.Rule(d["far"], b["none"]),                              # B8
        ]
        steer_rules = [
            ctrl.Rule(a["large_left"], sc["large_right"]),
            ctrl.Rule(a["small_left"], sc["small_right"]),
            ctrl.Rule(a["center"], sc["zero"]),
            ctrl.Rule(a["small_right"], sc["small_left"]),
            ctrl.Rule(a["large_right"], sc["large_left"]),
        ]
        speed_rules = [
            ctrl.Rule(d["very_close"], sa["slow_down"]),                 # S1
            ctrl.Rule(d["close"], sa["slow_down"]),                      # S2
            ctrl.Rule(d["medium"], sa["maintain"]),                      # S3 -- gap fix
            ctrl.Rule(d["far"] & s["slow"], sa["maintain"]),             # S4
            ctrl.Rule(d["far"] & (s["medium"] | s["fast"]), sa["maintain"]),  # S5 -- gap fix
        ]

        self._brake_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(brake_rules))
        self._steer_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(steer_rules))
        self._speed_sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(speed_rules))

    def compute(self, d_obs: float, angle_err_deg: float, speed: float) -> dict:
        d_obs = float(np.clip(d_obs, 0.0, 10.0))
        angle_err_deg = float(np.clip(angle_err_deg, -90.0, 90.0))
        speed = float(np.clip(speed, 0.0, 3.0))

        self._brake_sim.input["d_obs"] = d_obs
        self._brake_sim.input["speed"] = speed
        self._brake_sim.compute()

        self._steer_sim.input["angle_err"] = angle_err_deg
        self._steer_sim.compute()

        self._speed_sim.input["d_obs"] = d_obs
        self._speed_sim.input["speed"] = speed
        self._speed_sim.compute()

        return {
            "brake_cmd": float(self._brake_sim.output["brake_cmd"]),
            "steer_corr": float(self._steer_sim.output["steer_corr"]),
            "speed_adj": float(self._speed_sim.output["speed_adj"]),
        }
