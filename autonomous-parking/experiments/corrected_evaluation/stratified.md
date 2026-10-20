# Results stratified by start distance (pooled seed blocks A+B+C, n=500)

Start-distance bins: near (<2.5 m): n=181, mid (2.5-4.5 m): n=170, far (>=4.5 m): n=149

Starts with body clearance < 0.30 m to a neighbour: 44.6 %; < 0.05 m: 6.4 %


### near (<2.5 m)

| controller | n | pose_success_% | pose_collision_% | pose_timeout_% | contained_success_% | contained_collision_% |
|---|---|---|---|---|---|---|
| Fuzzy-only (legacy, stop 0.32 m) | 181 | 96.7 | 3.3 | 0.0 | 2.8 | 3.9 |
| Fuzzy-only (legacy, stop 0.32 m) + envelope shield | 181 | 93.9 | 0.0 | 6.1 | 2.8 | 0.0 |
| Fuzzy-precise (stop 0.10 m) | 181 | 96.7 | 3.3 | 0.0 | 66.3 | 6.6 |
| Fuzzy-precise (stop 0.10 m) + envelope shield | 181 | 93.9 | 0.0 | 6.1 | 66.3 | 0.0 |
| PPO original (100k) | 181 | 6.1 | 0.6 | 93.4 | 1.1 | 0.6 |
| PPO original (100k) + envelope shield | 181 | 6.1 | 0.0 | 93.9 | 1.1 | 0.0 |
| PPO v2 (retrained, 420k) | 181 | 97.8 | 1.1 | 1.1 | 60.8 | 1.1 |
| PPO v2 (retrained, 420k) + envelope shield | 181 | 95.6 | 0.0 | 4.4 | 60.8 | 0.0 |
| RL+Fuzzy (original hybrid) | 181 | 96.7 | 3.3 | 0.0 | 2.8 | 3.3 |
| RL+Fuzzy (original hybrid) + envelope shield | 181 | 93.9 | 0.0 | 6.1 | 2.8 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) | 181 | 97.2 | 1.7 | 1.1 | 5.5 | 2.2 |
| PPO v2 + Fuzzy 25/75 (no latch) + envelope shield | 181 | 95.0 | 0.0 | 5.0 | 5.5 | 0.0 |

### mid (2.5-4.5 m)

| controller | n | pose_success_% | pose_collision_% | pose_timeout_% | contained_success_% | contained_collision_% |
|---|---|---|---|---|---|---|
| Fuzzy-only (legacy, stop 0.32 m) | 170 | 88.8 | 11.2 | 0.0 | 0.0 | 11.2 |
| Fuzzy-only (legacy, stop 0.32 m) + envelope shield | 170 | 84.7 | 0.0 | 15.3 | 0.0 | 0.0 |
| Fuzzy-precise (stop 0.10 m) | 170 | 88.8 | 11.2 | 0.0 | 53.5 | 11.2 |
| Fuzzy-precise (stop 0.10 m) + envelope shield | 170 | 84.7 | 0.0 | 15.3 | 52.9 | 0.0 |
| PPO original (100k) | 170 | 7.1 | 50.6 | 42.4 | 3.5 | 50.6 |
| PPO original (100k) + envelope shield | 170 | 6.5 | 0.0 | 93.5 | 3.5 | 0.0 |
| PPO v2 (retrained, 420k) | 170 | 86.5 | 5.9 | 7.6 | 62.9 | 5.9 |
| PPO v2 (retrained, 420k) + envelope shield | 170 | 84.7 | 0.0 | 15.3 | 61.8 | 0.0 |
| RL+Fuzzy (original hybrid) | 170 | 87.6 | 12.4 | 0.0 | 0.0 | 12.4 |
| RL+Fuzzy (original hybrid) + envelope shield | 170 | 84.1 | 0.0 | 15.9 | 0.0 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) | 170 | 88.8 | 10.6 | 0.6 | 0.0 | 10.6 |
| PPO v2 + Fuzzy 25/75 (no latch) + envelope shield | 170 | 85.3 | 0.0 | 14.7 | 0.0 | 0.0 |

### far (>=4.5 m)

| controller | n | pose_success_% | pose_collision_% | pose_timeout_% | contained_success_% | contained_collision_% |
|---|---|---|---|---|---|---|
| Fuzzy-only (legacy, stop 0.32 m) | 149 | 77.2 | 22.8 | 0.0 | 0.0 | 22.8 |
| Fuzzy-only (legacy, stop 0.32 m) + envelope shield | 149 | 70.5 | 0.0 | 29.5 | 0.0 | 0.0 |
| Fuzzy-precise (stop 0.10 m) | 149 | 77.2 | 22.8 | 0.0 | 52.3 | 22.8 |
| Fuzzy-precise (stop 0.10 m) + envelope shield | 149 | 70.5 | 0.0 | 29.5 | 52.3 | 0.0 |
| PPO original (100k) | 149 | 4.0 | 87.2 | 8.7 | 0.7 | 87.2 |
| PPO original (100k) + envelope shield | 149 | 4.7 | 0.0 | 95.3 | 0.7 | 0.0 |
| PPO v2 (retrained, 420k) | 149 | 11.4 | 6.0 | 82.6 | 10.1 | 6.0 |
| PPO v2 (retrained, 420k) + envelope shield | 149 | 10.7 | 0.0 | 89.3 | 9.4 | 0.0 |
| RL+Fuzzy (original hybrid) | 149 | 74.5 | 25.5 | 0.0 | 0.0 | 25.5 |
| RL+Fuzzy (original hybrid) + envelope shield | 149 | 69.8 | 0.0 | 30.2 | 0.0 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) | 149 | 59.7 | 40.3 | 0.0 | 0.0 | 40.3 |
| PPO v2 + Fuzzy 25/75 (no latch) + envelope shield | 149 | 51.7 | 0.0 | 48.3 | 0.0 | 0.0 |

### ALL

| controller | n | pose_success_% | pose_collision_% | pose_timeout_% | contained_success_% | contained_collision_% |
|---|---|---|---|---|---|---|
| Fuzzy-only (legacy, stop 0.32 m) | 500 | 88.2 | 11.8 | 0.0 | 1.0 | 12.0 |
| Fuzzy-only (legacy, stop 0.32 m) + envelope shield | 500 | 83.8 | 0.0 | 16.2 | 1.0 | 0.0 |
| Fuzzy-precise (stop 0.10 m) | 500 | 88.2 | 11.8 | 0.0 | 57.8 | 13.0 |
| Fuzzy-precise (stop 0.10 m) + envelope shield | 500 | 83.8 | 0.0 | 16.2 | 57.6 | 0.0 |
| PPO original (100k) | 500 | 5.8 | 43.4 | 50.8 | 1.8 | 43.4 |
| PPO original (100k) + envelope shield | 500 | 5.8 | 0.0 | 94.2 | 1.8 | 0.0 |
| PPO v2 (retrained, 420k) | 500 | 68.2 | 4.2 | 27.6 | 46.4 | 4.2 |
| PPO v2 (retrained, 420k) + envelope shield | 500 | 66.6 | 0.0 | 33.4 | 45.8 | 0.0 |
| RL+Fuzzy (original hybrid) | 500 | 87.0 | 13.0 | 0.0 | 1.0 | 13.0 |
| RL+Fuzzy (original hybrid) + envelope shield | 500 | 83.4 | 0.0 | 16.6 | 1.0 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) | 500 | 83.2 | 16.2 | 0.6 | 2.0 | 16.4 |
| PPO v2 + Fuzzy 25/75 (no latch) + envelope shield | 500 | 78.8 | 0.0 | 21.2 | 2.0 | 0.0 |