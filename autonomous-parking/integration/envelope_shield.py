"""Model-based braking-envelope safety shield.

What it checks
--------------
For every proposed action the shield simulates, on a scratch copy of the vehicle that uses the
SAME ``Vehicle.step`` dynamics as the environment, (1) one step of the proposed action and then
(2) full braking with the steering command held, until the car is at rest. The car's true
4.2 x 1.8 m footprint (OBB) is tested against every obstacle rectangle and the world bounds at
each simulated step (``ParkingEnv.braking_envelope_safe``).

* Proposed action passes -> it is applied unchanged.
* Proposed action fails  -> it is replaced by [held steering command, full brake].

What it guarantees (and what it does not)
-----------------------------------------
Inductive invariant: if the car starts a step in a state from which "hold steering + full brake"
is collision-free, and the shield accepts only actions after which that is again true, then the
fallback is always available and the car never collides *under the simulator's own deterministic
dynamics and the obstacle map the simulator holds*. This is a verified-by-construction property of
the MODEL, not of a real vehicle. It does not cover: model mismatch, sensor/perception error
(it uses ground-truth obstacle geometry), moving obstacles, actuation delay, an initial state
that already has no safe stop, or reverse gear (not supported). It can stop the car short of the
goal (a stationary car is safe but may time out), so safety is traded against task completion.
"""
from __future__ import annotations

import numpy as np


class BrakingEnvelopeShield:
    def __init__(self, enabled: bool = True, margin: float = 0.05):
        self.enabled = enabled
        self.margin = float(margin)   # metres of clearance the stop trajectory must keep (0 = touching allowed)
        self.reset()

    def reset(self):
        self.last_steer_cmd = 0.0
        self.last_intervened = False
        self.n_interventions = 0
        self.n_calls = 0

    def filter(self, env, action):
        """Return the action that should actually be executed in ``env``."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"action must have shape (2,) = [steer, throttle_brake]; got {action.shape}")
        self.n_calls += 1
        if not np.isfinite(action).all():
            # Fail closed: a NaN/inf command (policy or controller fault) is never executed; brake instead.
            self.last_intervened = True
            self.n_interventions += 1
            return np.array([self.last_steer_cmd, -1.0], dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)      # +-inf is excluded above; out-of-range finite values saturate
        if not self.enabled:
            self.last_intervened = False
            self.last_steer_cmd = float(action[0])
            return action
        safe, _ = env.braking_envelope_safe(action, margin=self.margin)
        if safe:
            self.last_intervened = False
            self.last_steer_cmd = float(action[0])
            return action
        self.last_intervened = True
        self.n_interventions += 1
        # keep the steering command that the safe-stop trajectory was verified with
        return np.array([self.last_steer_cmd, -1.0], dtype=np.float32)
