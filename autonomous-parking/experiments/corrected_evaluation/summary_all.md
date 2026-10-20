
### set_A

| controller | n | pose_success | pose_ci | pose_collision | pose_timeout | contained_success | contained_ci | contained_collision | contained_timeout | ps_in_bay_given_pose_success | ps_protrusion_median_m | final_abs_long_err_m | final_abs_lat_err_m | final_heading_err_deg | min_clearance_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fuzzy-only (legacy, stop 0.32 m) | 100 | 88.0 | [80-93] | 12.0 | 0.0 | 5.0 | [2-11] | 13.0 | 82.0 | 5.7 | 0.072 | 0.287 | 0.087 | 5.14 | 0.0 |
| Fuzzy-only (legacy, stop 0.32 m) + envelope shield | 100 | 83.0 | [74-89] | 0.0 | 17.0 | 5.0 | [2-11] | 0.0 | 95.0 | 6.0 | 0.071 | 0.835 | 0.179 | 4.985 | 0.017 |
| Fuzzy-precise (stop 0.10 m) | 100 | 88.0 | [80-93] | 12.0 | 0.0 | 59.0 | [49-68] | 15.0 | 26.0 | 5.7 | 0.072 | 0.175 | 0.056 | 5.574 | 0.0 |
| Fuzzy-precise (stop 0.10 m) + envelope shield | 100 | 83.0 | [74-89] | 0.0 | 17.0 | 59.0 | [49-68] | 0.0 | 41.0 | 6.0 | 0.071 | 0.743 | 0.16 | 5.344 | 0.014 |
| PPO original (100k) | 100 | 11.0 | [6-19] | 38.0 | 51.0 | 6.0 | [3-12] | 38.0 | 56.0 | 54.5 | 0.0 | 0.963 | 0.224 | 4.552 | 0.0 |
| PPO original (100k) + envelope shield | 100 | 10.0 | [6-17] | 0.0 | 90.0 | 6.0 | [3-12] | 0.0 | 94.0 | 60.0 | 0.0 | 1.911 | 0.326 | 5.993 | 0.018 |
| PPO v2 (retrained, 420k) | 100 | 66.0 | [56-75] | 3.0 | 31.0 | 43.0 | [34-53] | 3.0 | 54.0 | 65.2 | 0.0 | 1.65 | 0.233 | 5.994 | 0.0 |
| PPO v2 (retrained, 420k) + envelope shield | 100 | 64.0 | [54-73] | 0.0 | 36.0 | 43.0 | [34-53] | 0.0 | 57.0 | 67.2 | 0.0 | 1.734 | 0.251 | 5.809 | 0.019 |
| RL+Fuzzy (original hybrid) | 100 | 88.0 | [80-93] | 12.0 | 0.0 | 5.0 | [2-11] | 12.0 | 83.0 | 5.7 | 0.072 | 0.286 | 0.088 | 5.152 | 0.0 |
| RL+Fuzzy (original hybrid) + envelope shield | 100 | 83.0 | [74-89] | 0.0 | 17.0 | 5.0 | [2-11] | 0.0 | 95.0 | 6.0 | 0.071 | 0.81 | 0.174 | 4.827 | 0.015 |
| PPO v2 + Fuzzy 25/75 (no latch) | 100 | 85.0 | [77-91] | 15.0 | 0.0 | 5.0 | [2-11] | 16.0 | 79.0 | 5.9 | 0.061 | 0.277 | 0.091 | 4.773 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) + envelope shield | 100 | 76.0 | [67-83] | 0.0 | 24.0 | 5.0 | [2-11] | 0.0 | 95.0 | 6.6 | 0.058 | 1.096 | 0.209 | 4.959 | 0.019 |

### set_B

| controller | n | pose_success | pose_ci | pose_collision | pose_timeout | contained_success | contained_ci | contained_collision | contained_timeout | ps_in_bay_given_pose_success | ps_protrusion_median_m | final_abs_long_err_m | final_abs_lat_err_m | final_heading_err_deg | min_clearance_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fuzzy-only (legacy, stop 0.32 m) | 200 | 89.0 | [84-93] | 11.0 | 0.0 | 0.0 | [0-2] | 11.0 | 89.0 | 0.0 | 0.066 | 0.295 | 0.076 | 5.163 | 0.0 |
| Fuzzy-only (legacy, stop 0.32 m) + envelope shield | 200 | 86.0 | [81-90] | 0.0 | 14.0 | 0.0 | [0-2] | 0.0 | 100.0 | 0.0 | 0.063 | 0.782 | 0.188 | 6.362 | 0.0 |
| Fuzzy-precise (stop 0.10 m) | 200 | 89.0 | [84-93] | 11.0 | 0.0 | 61.5 | [55-68] | 11.5 | 27.0 | 0.0 | 0.064 | 0.179 | 0.049 | 5.559 | 0.0 |
| Fuzzy-precise (stop 0.10 m) + envelope shield | 200 | 86.0 | [81-90] | 0.0 | 14.0 | 61.5 | [55-68] | 0.0 | 38.5 | 0.0 | 0.062 | 0.686 | 0.166 | 6.72 | 0.0 |
| PPO original (100k) | 200 | 7.0 | [4-11] | 40.0 | 53.0 | 1.0 | [0-4] | 40.0 | 59.0 | 14.3 | 0.132 | 1.201 | 0.211 | 6.174 | 0.0 |
| PPO original (100k) + envelope shield | 200 | 7.0 | [4-11] | 0.0 | 93.0 | 1.0 | [0-4] | 0.0 | 99.0 | 14.3 | 0.132 | 1.995 | 0.299 | 7.38 | 0.0 |
| PPO v2 (retrained, 420k) | 200 | 71.5 | [65-77] | 2.5 | 26.0 | 48.0 | [41-55] | 2.5 | 49.5 | 66.4 | 0.0 | 1.364 | 0.18 | 6.077 | 0.0 |
| PPO v2 (retrained, 420k) + envelope shield | 200 | 70.0 | [63-76] | 0.0 | 30.0 | 47.0 | [40-54] | 0.0 | 53.0 | 66.4 | 0.0 | 1.465 | 0.193 | 6.277 | 0.001 |
| RL+Fuzzy (original hybrid) | 200 | 88.5 | [83-92] | 11.5 | 0.0 | 0.0 | [0-2] | 11.5 | 88.5 | 0.0 | 0.068 | 0.297 | 0.077 | 5.217 | 0.0 |
| RL+Fuzzy (original hybrid) + envelope shield | 200 | 86.0 | [81-90] | 0.0 | 14.0 | 0.0 | [0-2] | 0.0 | 100.0 | 0.0 | 0.066 | 0.775 | 0.173 | 5.931 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) | 200 | 87.0 | [82-91] | 11.5 | 1.5 | 1.5 | [1-4] | 11.5 | 87.0 | 0.6 | 0.056 | 0.321 | 0.155 | 5.815 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) + envelope shield | 200 | 84.5 | [79-89] | 0.0 | 15.5 | 1.5 | [1-4] | 0.0 | 98.5 | 0.6 | 0.056 | 0.823 | 0.227 | 5.95 | 0.0 |

### set_C

| controller | n | pose_success | pose_ci | pose_collision | pose_timeout | contained_success | contained_ci | contained_collision | contained_timeout | ps_in_bay_given_pose_success | ps_protrusion_median_m | final_abs_long_err_m | final_abs_lat_err_m | final_heading_err_deg | min_clearance_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fuzzy-only (legacy, stop 0.32 m) | 200 | 87.5 | [82-91] | 12.5 | 0.0 | 0.0 | [0-2] | 12.5 | 87.5 | 0.0 | 0.079 | 0.294 | 0.08 | 5.741 | 0.0 |
| Fuzzy-only (legacy, stop 0.32 m) + envelope shield | 200 | 82.0 | [76-87] | 0.0 | 18.0 | 0.0 | [0-2] | 0.0 | 100.0 | 0.0 | 0.077 | 0.929 | 0.182 | 5.404 | 0.0 |
| Fuzzy-precise (stop 0.10 m) | 200 | 87.5 | [82-91] | 12.5 | 0.0 | 53.5 | [47-60] | 13.5 | 33.0 | 0.0 | 0.079 | 0.168 | 0.049 | 6.219 | 0.0 |
| Fuzzy-precise (stop 0.10 m) + envelope shield | 200 | 82.0 | [76-87] | 0.0 | 18.0 | 53.0 | [46-60] | 0.0 | 47.0 | 0.0 | 0.077 | 0.833 | 0.159 | 5.837 | 0.0 |
| PPO original (100k) | 200 | 2.0 | [1-5] | 49.5 | 48.5 | 0.5 | [0-3] | 49.5 | 50.0 | 25.0 | 0.103 | 1.305 | 0.226 | 6.374 | 0.0 |
| PPO original (100k) + envelope shield | 200 | 2.5 | [1-6] | 0.0 | 97.5 | 0.5 | [0-3] | 0.0 | 99.5 | 20.0 | 0.165 | 2.312 | 0.356 | 7.28 | 0.0 |
| PPO v2 (retrained, 420k) | 200 | 66.0 | [59-72] | 6.5 | 27.5 | 46.5 | [40-53] | 6.5 | 47.0 | 70.5 | 0.0 | 1.58 | 0.217 | 5.932 | 0.0 |
| PPO v2 (retrained, 420k) + envelope shield | 200 | 64.5 | [58-71] | 0.0 | 35.5 | 46.0 | [39-53] | 0.0 | 54.0 | 71.3 | 0.0 | 1.745 | 0.246 | 6.011 | 0.0 |
| RL+Fuzzy (original hybrid) | 200 | 85.0 | [79-89] | 15.0 | 0.0 | 0.0 | [0-2] | 15.0 | 85.0 | 0.0 | 0.08 | 0.294 | 0.08 | 5.687 | 0.0 |
| RL+Fuzzy (original hybrid) + envelope shield | 200 | 81.0 | [75-86] | 0.0 | 19.0 | 0.0 | [0-2] | 0.0 | 100.0 | 0.0 | 0.078 | 0.957 | 0.186 | 5.496 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) | 200 | 78.5 | [72-84] | 21.5 | 0.0 | 1.0 | [0-4] | 21.5 | 77.5 | 0.0 | 0.071 | 0.292 | 0.083 | 5.302 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) + envelope shield | 200 | 74.5 | [68-80] | 0.0 | 25.5 | 1.0 | [0-4] | 0.0 | 99.0 | 0.0 | 0.07 | 1.224 | 0.222 | 5.261 | 0.0 |

### pooled

| controller | n | pose_success | pose_ci | pose_collision | pose_timeout | contained_success | contained_ci | contained_collision | contained_timeout | ps_in_bay_given_pose_success | ps_protrusion_median_m | final_abs_long_err_m | final_abs_lat_err_m | final_heading_err_deg | min_clearance_m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fuzzy-only (legacy, stop 0.32 m) | 500 | 88.2 | [85-91] | 11.8 | 0.0 | 1.0 | [0-2] | 12.0 | 87.0 | 1.1 | 0.073 | 0.293 | 0.08 | 5.388 | 0.0 |
| Fuzzy-only (legacy, stop 0.32 m) + envelope shield | 500 | 83.8 | [80-87] | 0.0 | 16.2 | 1.0 | [0-2] | 0.0 | 99.0 | 1.2 | 0.071 | 0.851 | 0.184 | 5.704 | 0.0 |
| Fuzzy-precise (stop 0.10 m) | 500 | 88.2 | [85-91] | 11.8 | 0.0 | 57.8 | [53-62] | 13.0 | 29.2 | 1.1 | 0.073 | 0.174 | 0.05 | 5.824 | 0.0 |
| Fuzzy-precise (stop 0.10 m) + envelope shield | 500 | 83.8 | [80-87] | 0.0 | 16.2 | 57.6 | [53-62] | 0.0 | 42.4 | 1.2 | 0.07 | 0.756 | 0.162 | 6.091 | 0.0 |
| PPO original (100k) | 500 | 5.8 | [4-8] | 43.4 | 50.8 | 1.8 | [1-3] | 43.4 | 54.8 | 31.0 | 0.053 | 1.186 | 0.219 | 5.89 | 0.0 |
| PPO original (100k) + envelope shield | 500 | 5.8 | [4-8] | 0.0 | 94.2 | 1.8 | [1-3] | 0.0 | 98.2 | 31.0 | 0.053 | 2.105 | 0.327 | 7.062 | 0.0 |
| PPO v2 (retrained, 420k) | 500 | 68.2 | [64-72] | 4.2 | 27.6 | 46.4 | [42-51] | 4.2 | 49.4 | 67.7 | 0.0 | 1.506 | 0.205 | 6.003 | 0.0 |
| PPO v2 (retrained, 420k) + envelope shield | 500 | 66.6 | [62-71] | 0.0 | 33.4 | 45.8 | [41-50] | 0.0 | 54.2 | 68.5 | 0.0 | 1.631 | 0.226 | 6.077 | 0.0 |
| RL+Fuzzy (original hybrid) | 500 | 87.0 | [84-90] | 13.0 | 0.0 | 1.0 | [0-2] | 13.0 | 86.0 | 1.1 | 0.073 | 0.294 | 0.081 | 5.387 | 0.0 |
| RL+Fuzzy (original hybrid) + envelope shield | 500 | 83.4 | [80-86] | 0.0 | 16.6 | 1.0 | [0-2] | 0.0 | 99.0 | 1.2 | 0.072 | 0.854 | 0.179 | 5.536 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) | 500 | 83.2 | [80-86] | 16.2 | 0.6 | 2.0 | [1-4] | 16.4 | 81.6 | 1.4 | 0.061 | 0.301 | 0.115 | 5.413 | 0.0 |
| PPO v2 + Fuzzy 25/75 (no latch) + envelope shield | 500 | 78.8 | [75-82] | 0.0 | 21.2 | 2.0 | [1-4] | 0.0 | 98.0 | 1.5 | 0.06 | 1.038 | 0.221 | 5.476 | 0.0 |

### Paired exact McNemar tests (pooled, same seeds). A-only / B-only = discordant episodes

| A | B | outcome | A-only | B-only | p |
|---|---|---|---|---|---|
| Fuzzy-only (legacy, stop 0.32 m) | Fuzzy-precise (stop 0.10 m) | pose success | 0 | 0 | 1 |
| Fuzzy-only (legacy, stop 0.32 m) | Fuzzy-precise (stop 0.10 m) | contained success | 0 | 284 | 6.43e-86 |
| Fuzzy-only (legacy, stop 0.32 m) | Fuzzy-precise (stop 0.10 m) | contained collision | 0 | 5 | 0.0625 |
| Fuzzy-precise (stop 0.10 m) | PPO v2 (retrained, 420k) | pose success | 106 | 6 | 9.76e-25 |
| Fuzzy-precise (stop 0.10 m) | PPO v2 (retrained, 420k) | contained success | 94 | 37 | 6.72e-07 |
| Fuzzy-precise (stop 0.10 m) | PPO v2 (retrained, 420k) | contained collision | 52 | 8 | 5.21e-09 |
| Fuzzy-only (legacy, stop 0.32 m) | PPO original (100k) | pose success | 413 | 1 | 1.96e-122 |
| Fuzzy-only (legacy, stop 0.32 m) | PPO original (100k) | contained success | 3 | 7 | 0.344 |
| Fuzzy-only (legacy, stop 0.32 m) | PPO original (100k) | contained collision | 8 | 165 | 2.96e-39 |
| PPO original (100k) | PPO v2 (retrained, 420k) | pose success | 6 | 318 | 9.15e-86 |
| PPO original (100k) | PPO v2 (retrained, 420k) | contained success | 1 | 224 | 8.38e-66 |
| PPO original (100k) | PPO v2 (retrained, 420k) | contained collision | 199 | 3 | 4.27e-55 |
| Fuzzy-only (legacy, stop 0.32 m) | RL+Fuzzy (original hybrid) | pose success | 7 | 1 | 0.0703 |
| Fuzzy-only (legacy, stop 0.32 m) | RL+Fuzzy (original hybrid) | contained success | 0 | 0 | 1 |
| Fuzzy-only (legacy, stop 0.32 m) | RL+Fuzzy (original hybrid) | contained collision | 2 | 7 | 0.18 |
| Fuzzy-precise (stop 0.10 m) | PPO v2 + Fuzzy 25/75 (no latch) | pose success | 27 | 2 | 1.62e-06 |
| Fuzzy-precise (stop 0.10 m) | PPO v2 + Fuzzy 25/75 (no latch) | contained success | 279 | 0 | 2.06e-84 |
| Fuzzy-precise (stop 0.10 m) | PPO v2 + Fuzzy 25/75 (no latch) | contained collision | 10 | 27 | 0.00763 |
| PPO v2 (retrained, 420k) | PPO v2 + Fuzzy 25/75 (no latch) | pose success | 4 | 79 | 4e-19 |
| PPO v2 (retrained, 420k) | PPO v2 + Fuzzy 25/75 (no latch) | contained success | 222 | 0 | 2.97e-67 |
| PPO v2 (retrained, 420k) | PPO v2 + Fuzzy 25/75 (no latch) | contained collision | 8 | 69 | 3.14e-13 |
| Fuzzy-precise (stop 0.10 m) | Fuzzy-precise (stop 0.10 m) + envelope shield | pose success | 23 | 1 | 2.98e-06 |
| Fuzzy-precise (stop 0.10 m) | Fuzzy-precise (stop 0.10 m) + envelope shield | contained success | 1 | 0 | 1 |
| Fuzzy-precise (stop 0.10 m) | Fuzzy-precise (stop 0.10 m) + envelope shield | contained collision | 65 | 0 | 5.42e-20 |
| Fuzzy-only (legacy, stop 0.32 m) | Fuzzy-only (legacy, stop 0.32 m) + envelope shield | pose success | 23 | 1 | 2.98e-06 |
| Fuzzy-only (legacy, stop 0.32 m) | Fuzzy-only (legacy, stop 0.32 m) + envelope shield | contained success | 0 | 0 | 1 |
| Fuzzy-only (legacy, stop 0.32 m) | Fuzzy-only (legacy, stop 0.32 m) + envelope shield | contained collision | 60 | 0 | 1.73e-18 |
| PPO v2 (retrained, 420k) | PPO v2 (retrained, 420k) + envelope shield | pose success | 9 | 1 | 0.0215 |
| PPO v2 (retrained, 420k) | PPO v2 (retrained, 420k) + envelope shield | contained success | 3 | 0 | 0.25 |
| PPO v2 (retrained, 420k) | PPO v2 (retrained, 420k) + envelope shield | contained collision | 21 | 0 | 9.54e-07 |