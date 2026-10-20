"""Membership-function parameters (blueprint Part 4), kept separate from the inference code so
they're easy to tune for the ablation in Part 20 ("two different fuzzy rule sets").
"""

CONSERVATIVE = {
    "d_obs": {
        "very_close": ("trapmf", [0, 0, 0.5, 1.5]),
        "close": ("trimf", [0.5, 1.5, 3]),
        "medium": ("trimf", [2, 4, 6]),
        "far": ("trapmf", [4, 6, 10, 10]),
    },
    "speed": {
        "slow": ("trapmf", [0, 0, 0.3, 0.8]),
        "medium": ("trimf", [0.5, 1.2, 2]),
        "fast": ("trapmf", [1.5, 2.2, 3, 3]),
    },
    "angle_err": {
        "large_left": ("trapmf", [-90, -90, -45, -20]),
        "small_left": ("trimf", [-30, -12, 0]),
        "center": ("trimf", [-8, 0, 8]),
        "small_right": ("trimf", [0, 12, 30]),
        "large_right": ("trapmf", [20, 45, 90, 90]),
    },
    "brake_cmd": {
        "none": ("trapmf", [0, 0, 0.1, 0.3]),
        "moderate": ("trimf", [0.2, 0.5, 0.8]),
        "hard": ("trapmf", [0.6, 0.8, 1, 1]),
    },
    "steer_corr": {
        # same shape family as angle_err, but on the [-1, 1] output universe
        "large_left": ("trapmf", [-1, -1, -0.5, -0.22]),
        "small_left": ("trimf", [-0.33, -0.13, 0]),
        "zero": ("trimf", [-0.09, 0, 0.09]),
        "small_right": ("trimf", [0, 0.13, 0.33]),
        "large_right": ("trapmf", [0.22, 0.5, 1, 1]),
    },
    "speed_adj": {
        "slow_down": ("trapmf", [-1, -1, -0.6, -0.1]),
        "maintain": ("trimf", [-0.2, 0, 0.2]),
        "speed_up": ("trapmf", [0.1, 0.6, 1, 1]),
    },
}

# Aggressive variant: brakes later / less, corrects steering more gently -- used for the
# rule-set-sensitivity ablation in blueprint Part 20.
AGGRESSIVE = {
    **CONSERVATIVE,
    "d_obs": {
        "very_close": ("trapmf", [0, 0, 0.3, 1.0]),
        "close": ("trimf", [0.3, 1.0, 2.2]),
        "medium": ("trimf", [1.5, 3.0, 5.0]),
        "far": ("trapmf", [3.0, 5.0, 10, 10]),
    },
    "brake_cmd": {
        "none": ("trapmf", [0, 0, 0.2, 0.45]),
        "moderate": ("trimf", [0.3, 0.55, 0.8]),
        "hard": ("trapmf", [0.7, 0.85, 1, 1]),
    },
}
