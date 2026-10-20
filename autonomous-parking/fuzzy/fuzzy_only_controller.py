"""Zero-order Sugeno fuzzy navigation with a low-level speed controller.

Linguistic direction rules steer right/straight/left. Distance rules request
stop/creep/cruise. Rolling-resistance feedforward prevents positive throttle
commands below the vehicle's motion threshold from stranding the car.
"""
import numpy as np


def infer(value, centres, consequents):
    """Triangular memberships with shoulder sets; weighted singleton outputs."""
    membership = np.maximum(0., 1. - np.abs(np.asarray(centres) - value) / (centres[1] - centres[0]))
    membership[0] = max(membership[0], float(value <= centres[0]))
    membership[-1] = max(membership[-1], float(value >= centres[-1]))
    return float(np.dot(membership, consequents) / max(membership.sum(), 1e-9))


class FuzzyOnlyController:
    def __init__(self, fuzzy=None, slow_dist=3., safe_threshold=3., stop_radius=.32, heading_gain=.6):
        # stop_radius: brake to a halt once the centre is this close to the slot centre. The
        # historical value .32 m is larger than the bay's .30 m end clearance, so the car stops
        # with its rear overhanging the bay (audit finding D6). Do not change the default: it
        # defines the legacy 'fuzzy_only' results.
        self.stop_radius = float(stop_radius)
        # heading_gain: weight of the heading-error term in the steering error (legacy .6)
        self.heading_gain = float(heading_gain)
        self.last_mode = 'fuzzy navigation'

    def act(self, obs, fuzzy_inputs, slot_bearing_deg):
        dist = float(np.hypot(obs[0], obs[1]))
        heading = float(np.arctan2(obs[2], obs[3]))
        bearing = float(np.deg2rad(slot_bearing_deg))
        # Heading correction prevents aiming at the position while ignoring orientation.
        error = float(np.clip(1.6 * bearing - self.heading_gain * heading, -1, 1))
        steer = infer(error, [-1., -.5, 0., .5, 1.], [-1., -.5, 0., .5, 1.])
        if dist < self.stop_radius:
            self.last_mode = 'fuzzy stop'
            return np.array([steer, -1.], dtype=np.float32)
        target = infer(dist, [0., .65, 1.3, 1.95], [0., .325, .65, .65])
        target = max(.12, target)
        d_obs, obstacle_bearing, speed = fuzzy_inputs
        # Neighbour cars beside a legal bay are not a reason to brake indefinitely.
        # Restrict emergency braking to the forward corridor, not the nearest side ray.
        if abs(obstacle_bearing) < 30. and d_obs < 2.6:
            target = min(target, max(0., (d_obs - 2.2) * .5))
        self.last_mode = 'fuzzy navigation'
        thb = float(np.clip(.25 + 1.8 * (target - speed), -1, 1))
        if target == 0.:
            thb = -1.
            self.last_mode = 'fuzzy obstacle brake'
        return np.array([steer, thb], dtype=np.float32)
