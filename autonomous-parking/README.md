# Autonomous Parking: Rule-Based vs Fuzzy vs RL vs RL+Fuzzy

A 2D simulation comparing four parking controllers, built from (and fixing several bugs in) an
initial project blueprint. See **`docs/REVIEW.md`** for the full list of bugs found and fixed,
the RL-training debugging story, and honest limitations — read that before trusting any number
in here.

## What's here

```
environment/   kinematic bicycle vehicle model, ray-cast sensors, Gymnasium env, pygame renderer
fuzzy/         Mamdani fuzzy safety system + a fuzzy-only baseline controller
baselines/     hand-coded rule-based controller
integration/   RL + fuzzy "shield" hybrid controller
rl/            PPO training (Stable-Baselines3) with a curriculum, + a policy wrapper
evaluation/    N-episode harness + comparison table/plots across all 4 controllers
tests/         pytest suite (40 tests): geometry cross-checked vs shapely, vehicle kinematics,
               fuzzy rule-base completeness, shield safety invariants, env integration
configs/       environment + reward YAML
demo.py        watch any controller drive one episode (live window or exported GIF)
```

## Quickstart

```bash
pip install -r requirements.txt
# On Linux, prefer: pip install torch --no-deps   (see the note in requirements.txt)

python -m pytest tests/ -q                     # 40 tests, ~2s

python -m rl.train_ppo --timesteps 600000 --out models/ppo_parking_v1 --curriculum
# continue training later: add --resume (picks up weights + curriculum stage automatically)

python -m evaluation.run_experiments --episodes 100 --ppo-model models/ppo_parking_v1
python -m evaluation.compare_controllers --dir experiments/results

python demo.py --controller ppo_fuzzy --model models/ppo_parking_v1.zip --seed 1002 \
    --save-gif out.gif          # headless; drop --save-gif for a live window if you have a display
```

## Current results (100 episodes/controller, both scenario kinds, slot+start-pose randomization)

| Controller | Success | Collision | Timeout |
|---|---|---|---|
| Rule-Based | 3% | 39% | 58% |
| Fuzzy-only | 3% | 14% | 83% |
| **PPO** | **32%** | **11%** | 57% |
| PPO + Fuzzy | 32% | 11% | 57% |

PPO clearly beats both hand-designed baselines. It's undertrained (single CPU core, no GPU in the
build environment) — `docs/REVIEW.md` has concrete next steps, including `--resume` for more
training and why the fuzzy shield currently shows no measurable effect (it needs to be trained
through, not bolted on at evaluation time).

## Key design decisions worth knowing before you extend this

- **Action space**: `[steer, throttle_brake]` (2-dim; sign of the second value selects
  throttle vs. brake), resolving an ambiguity in the original blueprint between a 2-dim and 3-dim
  action.
- **Success** requires position (0.4m), heading (15°) and speed (0.2 m/s) tolerances held for 2
  consecutive steps — loosened slightly from the blueprint's 0.3m/10°/3-step figures, which left
  very little room for a continuous controller with rate-limited steering to settle into (these
  were explicitly marked "approximate" in the blueprint).
- **Training curriculum** (`rl/train_ppo.py`) ramps three axes in order: obstacle presence, start
  distance (sampled as a *range* up to the current max, not a fixed point — pinning it to a point
  caused catastrophic forgetting of easier cases), then the blueprint's own Part-9 domain
  randomization (start pose → slot position → obstacle position → sensor noise).
- **The fuzzy shield's `angle_err` input is obstacle bearing, not slot bearing** — it's an
  avoidance signal, not a navigation one. Mixing these up (feeding it slot bearing) was a bug
  found mid-build; see `docs/REVIEW.md` §2.1 if you're adding a new controller that touches the
  shield.
