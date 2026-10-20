> **Superseded in part by `PRESENTATION_READINESS_AUDIT.md`.** The tables below are the pre-audit results, kept unchanged for the record. After the audit fixed the shield's input (defect D1), RL + Fuzzy scores 88/100 (not 87/100) on seeds 3000-3099 and the seed-1000 hybrid error is 0.311 m / 10.09 deg (not 0.361 m / 10.16 deg). The audit also adds an in-bay metric and a safety shield that actually prevents collisions.

# Diagnosis and measured outcomes

## Reproduced failure
The supplied PPO checkpoint records 100,352 training steps. Seed 1000 reproduced a timeout with final position error 1.1567898 m and heading error 13.5081 degrees. Its small throttle near 0.13 cannot overcome the simulator's rolling-resistance threshold (0.5 / 2.0 = 0.25). The old fuzzy-only navigation was also braking for neighbours beside the bay, stranding the vehicle.
The old docking reward could remain positive at rest near the slot while outside success tolerance. This is an incentive problem, though it does not prove the entire cause of the policy's behaviour. Future training now pays the docking bonus only on progress or success. The uploaded PPO weights are unchanged and were not retrained.

## Implemented fixes
Fuzzy navigation uses zero-order Sugeno triangular direction memberships and distance-to-target speed rules. A speed feedback controller includes rolling-resistance feedforward. Side neighbours do not automatically trigger emergency braking; the forward corridor is checked instead. This is heuristic and can still collide.
The hybrid now explicitly fuses 25% PPO action and 75% fuzzy navigation, prioritises fuzzy braking, applies the existing Mamdani obstacle shield, and switches to fuzzy recovery after 1.2 seconds stopped outside the goal tolerance. The viewer displays this control mode. It is not purely RL, not a jointly trained policy, and not merely the old obstacle shield. The fixed weights are hand-selected design parameters, not learned weights or an optimal blend.

## Same scenario as the screenshots: seed 1000

| Controller | Outcome | Position error | Heading error |
|---|---|---:|---:|
| Updated Fuzzy-only | Success | 0.311 m | 10.07 degrees |
| Original RL-only | Timeout | 1.157 m | 13.51 degrees |
| Updated RL + Fuzzy | Success | 0.361 m | 10.16 degrees |

The original success thresholds are unchanged: position <0.4 m, heading <15 degrees, speed <0.2 m/s, held for two steps. Success is not perfect centring or exact alignment. The recordings preserve failures; no favourable-seed search was used.

## Development and evaluation
30 development scenarios (1000-1029) reproduced the old controllers: fuzzy 2 successes, PPO 2, hybrid 2. The first recovery-only fix improved completion but retained too many collisions on exploratory seeds 2000-2099; it was rejected in favour of continuous fuzzy guidance. The final weighted fusion scored 28/30 on development seeds, as did fuzzy-only; PPO remained 2/30.
After selecting the final implementation, 100 different scenarios (3000-3099) were evaluated without further tuning. Each controller faced the same scenario for each seed, with identical physics, observations, success tolerances and 500-step limits.

| Controller | Success | Collision | Timeout |
|---|---:|---:|---:|
| Fuzzy-only | 88/100 | 12/100 | 0/100 |
| RL-only (original PPO) | 11/100 | 38/100 | 51/100 |
| RL + Fuzzy | 87/100 | 13/100 | 0/100 |

Raw per-episode CSVs, comparison table and plots are in autonomous-parking/experiments/final_evaluation. This is evidence of improvement over this particular weak PPO checkpoint under this protocol, not proof that hybrids generally beat RL or fuzzy logic. Fuzzy-only and the hybrid performed similarly. No statistical-significance or real-world-safety claim is made.
The environment samples start distances, including starts already close to the goal. This is not a full-length approach-only benchmark. All results come from one PPO checkpoint. Hybrid assistance uses the same observation-derived navigation and fuzzy inputs, not a geometric path planner. Stronger PPO training and testing across more seeds/scenario families remain future work.
