# RL architecture, hyper-parameter and control-rule audit

Scope: PPO policy, reward, curriculum, checkpoint selection, fuzzy/shield integration. Everything below was measured in this repository;
nothing is quoted from memory. Raw outputs: `experiments/rl_audit/` (logs, tuning CSVs, held-out CSVs, summaries). Baselines and old checkpoints are untouched.

## 1. What is actually used (baseline)

| Item | Value (files) |
|---|---|
| Algorithm | Stable-Baselines3 **PPO**, `MlpPolicy` (actor-critic, separate pi/vf heads), `rl/train_ppo.py` |
| Network | `net_arch=[128,128]`, Tanh, Gaussian policy (state-independent log-std) - `rl/configs/ppo_config.yaml` |
| Observation (14) | `[dx, dy, sin(heading err), cos(heading err), v, steer angle]` + 8 range rays (0-10 m); raw, **not normalised** |
| Action (2) | `[steer, throttle_brake]` in [-1,1]; forward only (negative = brake) unless `allow_reverse` |
| PPO hyper-parameters | n_envs 4, n_steps 512, batch 256, epochs 10, gamma .99, lambda .95, lr 3e-4 constant, clip .2, ent_coef 0, vf_coef .5 (SB3 default), max_grad_norm .5 (default). `normalize_reward: true` in the YAML is **not used anywhere** (no VecNormalize) |
| Reward | -0.01/step + 0.5 d(dist) + 0.3 d(|heading err|) - 0.02 |d steer| + docking bonus (progress-gated) - 20 collision + 100 success - 10 timeout. Uses the **centre-pose box**, not bay containment |
| Curriculum | 14 stages (distance 0.05->1.0, then obstacles, start-pose/slot/obstacle/sensor randomisation, then tighter tolerances); advance at rolling success >= 0.6 (window 60) |
| Checkpointing | `*_best` by validation success, ties by lower collision (100 episodes, seeds 7000-7099) |
| Safety | fuzzy `HybridController` (legacy) and `BrakingEnvelopeShield` (`+env`), explicit and unit-tested; never weakened here |
| Training-vs-evaluation distribution | start distance U[0.05,1] x 6 m in both (checked: `ParkingEnv.min_start_distance_scale=0.05`), so no train/eval mismatch there |

### What does "ARC" mean here?
**It cannot be identified.** A case-insensitive search of every `.py/.md/.yaml/.txt/.js/.html` file finds no algorithm, network, paper or class called ARC. The only hits are
`net_arch` (the SB3 policy-architecture argument), geometric "arcs" in `baselines/parallel_geometric.py`, and substrings such as `arctan2`. I therefore did **not** swap the model on a guess.
The most plausible reading is "the network architecture" (`net_arch`), so that is what was tested (128x128 vs 256x256). If you meant something else (a specific paper/algorithm, or "actor-critic"
which PPO already is), tell me the name and I will evaluate it; candidates and my recommendation are in section 8.

## 2. Evaluation-pipeline inspection (done first)
* Success/collision/timeouts are now unambiguous: `success` (strict: whole car inside the bay lines, heading < 15 deg, speed < 0.2 m/s, held 2 steps), `collision`, `timeout`, plus the legacy `pose` outcome reconstructed from the same trajectory. `tests/test_containment_metric.py` pins this.
* **Finding 1 - objective mismatch.** The reward/curriculum/checkpoint selection used the *pose* box, while the headline metric is strict containment. PPO v2: pose 68.2 % but strict 46.4 %.
* **Finding 2 - checkpoint rule can pick unsafe checkpoints** (success-first): e.g. a 0.70-success / 0.29-collision checkpoint beats 0.69 / 0.02. Added an opt-in `safe` rule (success - 2 x collision).
* **Finding 3 - the original checkpoint's validation seeds (5000-5039) overlap evaluation set B.** New runs validate on 7000-7099 and are compared on a separate tuning block (7100-7299); the held-out sets A/B/C (3000-3099, 5000-5199, 9000-9199) were used **only after** the candidates were fixed (except that I looked at held-out numbers before naming the final checkpoint, see 6).
* **Finding 4 - `normalize_reward` is dead config**; documented, not silently "fixed".
* **Finding 5 - SB3 learning-rate schedules restart on every `learn()` chunk.** My first linear-LR attempt therefore collapsed the LR; invalid result kept in `experiments/rl_audit/negative_results/`, fixed with an absolute-step `LinearLR`.
* Min obstacle clearance, trajectory length, shield-activation fraction, smoothness etc. were already recorded per episode by `evaluation/run_experiments.py`; new tools: `rl_audit_summary.py`, `rl_audit_compare.py` (Wilson CIs, seed bootstrap, paired exact McNemar), `rl_audit_stratify.py`, `rl_audit_failures.py`.

## 3. Controlled experiments (tuning seeds only; one change at a time versus baseline)
Training: 4 envs x 512 steps, ~420k steps (6 min CPU), validation seeds 7000-7099; compared on **tuning block T = seeds 7100-7299 (200 episodes/checkpoint, strict metric)**. Training looked deterministic per seed (one rerun, F seed 6 vs C2 seed 6, reproduced identical tuning metrics).

| Arm | Change vs baseline (config key) | Seeds | Strict success (T) | Collision (T) | Verdict |
|---|---|---|---|---|---|
| BASE | original recipe, 128x128 (`exp_ppo_cur_s1..3_best`) | 3 | 0.378 (0.225-0.465) | 0.030 | reference |
| E1obs | fixed observation scaling (`obs_scale=true`) | 2 | 0.182 | 0.338 | **rejected**: more collisions |
| E2lin | linear LR 3e-4 -> 3e-5 (`lr_schedule=linear`) | 2 | 0.378 | 0.025 | no effect; lower seed variance (SD .011, n=2) |
| E3shape | containment shaping `w_contain=1.0` | 2 | 0.458 | 0.132 | better strict, more collisions |
| E4net | `net_arch=[256,256]` | 2 | 0.392 | 0.085 | within noise |
| **C2** | shaping + 256x256 | **6** | **0.461** (0.315-0.575, SD .100) | 0.106 | **chosen recipe** |
| C1 | shaping + linear LR | 2 | 0.555 | 0.125 | 2 seeds only |
| C3 | shaping + 256x256 + linear LR | 2 | 0.442 | 0.035 | no gain over C2 |
| C4 | C2 with `w_contain=2.0` | 2 | 0.412 | 0.075 | no gain |
| B2 | baseline recipe + `--select safe` | 6 | 0.447 (0.250-0.620, SD .124) | 0.057 | selection rule alone = same strict, fewer collisions than C2 |
| F | C2 + `--select safe` | 6 | 0.431 (SD .124) | 0.070 | no gain over C2 |
| G1ent | C2 + `ent_coef=0.003` | 2 | 0.525 | 0.198 | trades timeouts for collisions; rejected |
| G2time | C2 + `time_penalty=0.03` | 2 | 0.480 | 0.032 | no gain over C2 seeds 1-2 (0.565); deferred |
| (E3cont) | train/validate on strict `success_mode=contained` from step 1 | 2 | n/a | n/a | **failed**: never passed curriculum stage 6 in 190k steps (`negative_results/`) |

**Honest reading:** the run-to-run (training-seed) SD of strict success is ~0.12, as large as every effect above. With 2 seeds per arm, differences below ~0.15 are not detectable; resolving a 0.07 difference would need ~45 seeds per arm.
Only C2, B2, F have 6 seeds, and their means (0.46 / 0.45 / 0.43) are **not distinguishable**. The single robust finding is that obs-scaling hurts and entropy 0.003 raises collisions.

Hyper-parameters **not** tested (no budget to resolve them above seed noise; unchanged): gamma, gae_lambda, clip_range, n_steps, batch_size, n_epochs, vf_coef, max_grad_norm, activation, n_envs, curriculum threshold/window. SAC/TD3 not run (see section 8).

### Failure anatomy (what actually limits success)
Stratifying tuning-block outcomes by start distance (`experiments/rl_audit/tuning_stratified.txt`): near starts (<2.5 m, car beside/in the bay mouth, forward-only) succeed only 35-47 % with 50-65 % timeouts; far starts fail by stalling (BASE 77 % timeouts) or colliding (shaped arms). On held-out data (`heldout_failure_taxonomy.txt`) PPO v2 splits as 46.4 % strict success / 4.2 % collision / 21.8 % reaches the pose box but never ends fully inside the lines / 27.6 % never reaches the pose box (median 5 m from the slot = stalled near start). Containment shaping converts about 6 points of "pose-only" into strict success.

## 4. Selected change and held-out results
Selected: **PPO v3 = `models/ppo_v3_rlaudit.zip`** = recipe C2, training seed 1 (copy of `rl_audit_C2_s1_best`; validation 0.75 success / 0.02 collision on seeds 7000-7099 at curriculum stage 13). Rule: highest tuning-block strict success among checkpoints with tuning collision <= 0.07 (C2 seed 1: 0.575 / 0.055).
Changes vs v2: `net_arch` 128x128 -> 256x256; reward gains `w_contain=1.0` (potential-based, capped at 3 m); everything else identical. Metadata: `models/ppo_v3_rlaudit.meta.json` (the `timesteps_requested` field only records the last of the three resumed calls: 150k + 150k + 120k = 420k steps).

Held-out, pooled A+B+C, n = 500 identical episodes, strict metric:

| Controller | Strict success (95 % CI) | Pose success | Collision | Timeout | Steps/successful park | Path (m) | Shield-active steps | Final heading err (deg) |
|---|---|---|---|---|---|---|---|---|
| PPO original (100k) | 1.8 | 5.8 | 43.4 | 54.8 | - | - | - | 5.9 |
| PPO v2 (previous best) | 46.4 [42-51] | 68.2 | 4.2 | 49.4 | 24.0 | 2.67 | - | 6.0 |
| **PPO v3** | **56.0 [52-60]** | 71.6 | 5.4 | 38.6 | 27.3 | 2.98 | - | 3.4 |
| PPO v2 + envelope shield | 45.8 [41-50] | 66.6 | **0.0** | 54.2 | 24.0 | 2.65 | 5.2 % | 6.1 |
| **PPO v3 + envelope shield** | **54.4 [50-59]** | 69.4 | **0.0** | 45.6 | 26.9 | 2.92 | 7.4 % | 3.8 |
| Fuzzy-precise (reference, earlier work) | 57.8 [53-62] | 88.2 | 13.0 | 29.2 | - | - | - | 5.8 |
| Fuzzy-precise + shield (reference) | 57.6 [53-62] | 83.8 | 0.0 | 42.4 | - | - | - | 6.1 |

Paired exact McNemar on the same 500 episodes:
* v3 vs v2, strict success: 75 vs 27 discordant, **p = 2.1e-6**; pose success 33 vs 16, p = 0.021; collision 16 vs 10, p = 0.33 (no significant change).
* v3+shield vs v3: strict 1 vs 9, p = 0.021 - the shield costs ~1.6 points and removes all 27 collisions (p = 1.5e-8); it does not destabilise the policy.
* v3 vs fuzzy-precise: strict 83 vs 92, p = 0.55 (**statistically equal**); fuzzy-precise is much better on the looser pose metric (p = 3.5e-17). So PPO v3 does not beat the fuzzy controller; it closes the strict-metric gap.

### Variability and limits (read before quoting)
* **Training-seed variability dominates.** On held-out data the six C2 seeds range from 0.286 to 0.560 strict (mean 0.427, SD 0.111), the six B2 seeds from 0.256 to 0.576 (mean 0.435); the 3 original-recipe seeds give 0.200-0.464 (mean 0.353). Bootstrap over seeds for C2 - B2 = +0.009, 95 % CI [-0.107, +0.124]; B2 - BASE = +0.083 [-0.059, +0.240]. **The recipe change by itself is not shown to be better; v3 is a good single checkpoint (best-of-seeds), not evidence that the recipe is superior.** Its held-out result is a mild over-estimate of what a freshly trained seed gives (selected using the tuning block; selection effect T 0.575 -> held-out 0.560 was small).
* I looked at held-out numbers of several C2 seeds before naming v3; the selection rule above would also pick seed 1, and the alternative C2 seed 2 (0.540 strict, 4.0 % collision, also p = 1.1e-5 vs v2) tells the same story.
* Single-sample 500-episode CIs are Wilson; one scenario family (perpendicular, forward-only); simulator only.
* The held-out sets were reused to score 15+ checkpoints, so differences between v3 and its sibling seeds are not clean tests; the v3-vs-v2 comparison is clean for the pre-registered-in-spirit candidate only to the extent stated above.
* Minimum obstacle clearance: the 5th-percentile clearance rose from 0.007 m (v2) to 0.032/0.040 m with the shield; absolute minimum is 0 for unshielded runs (collisions).
* Hybrid blend (PPO + fuzzy) was **not** re-tested with v3; earlier diagnosis (`experiments/hybrid_diagnosis/`) showed blends collapse strict success (<= 2 %) for the pre-existing weights.

## 5. Fuzzy rules and shield (inspected, deliberately unchanged)
* Membership functions: the `speed` universe is 0-3 m/s but the vehicle's limit is 2 m/s, so "fast" (trapezoid [1.5, 2.2, 3, 3]) never exceeds membership 0.71; rules B2/B5 can never fire at full strength. The brake rule base covers all (d, speed) combinations (gap B6 already fixed), steering rule `angle_err` is obstacle-bearing based. These are cosmetic for the envelope shield (which does not use them) and the legacy fuzzy shield's weaknesses were already documented (deadlocks). No demonstrated failure mode requires editing them, so they are unchanged (verified results depend on them).
* Shield behaviour on RL actions: v2 5.2 % and v3 7.4 % of steps are shield-active; zero collisions in 500 episodes each; strict success change -0.6 pts (v2) and -1.6 pts (v3). Shield code, thresholds and tests are untouched.

## 6. Reward audit
* Hacking checks: the progress-gated docking bonus cannot be farmed by standing still (existing test `test_parking_recovery.py`); `w_contain` is potential-based so out-and-back dithering nets exactly 0 (`tests/test_rl_audit.py::test_containment_shaping_is_potential_based_and_signed`); penalties (collision 20, timeout 10) and the success bonus are unchanged; the success definition is unchanged.
* Remaining weakness: 23-31 % of held-out episodes never reach the pose box and sit ~5 m from the slot (stall). Raising the time penalty (G2) and entropy (G1) did not fix it without side effects. Candidate fix needing a task change (reverse gear) is deferred.

## 7. Reproduce
```bash
# train v3-style (about 6 min on 4 CPUs per seed; resumable with --resume)
python -m rl.train_ppo --timesteps 420000 --out models/my_v3 --curriculum --seed 1 --chunk 20000 --val-seed 7000 --val-episodes 100 \
    --set net_arch=[256,256] --reward w_contain=1.0 [--select safe]
# evaluate any checkpoint (paired seeds, strict + pose outcomes in one run)
python -m evaluation.evaluate_matrix --controllers "ppo:models/ppo_v3_rlaudit,ppo:models/ppo_v3_rlaudit+env" --block A:3000:100,B:5000:200,C:9000:200 --out-dir experiments/rl_audit/heldout_eval
python -m evaluation.rl_audit_compare --dir experiments/rl_audit/heldout_eval --pairs rl_audit_C2_s1_best:exp_ppo_cur_s3_best
# use it
python demo.py --controller ppo_v3        # or ppo_v3+env ;  PPOPolicy("models/ppo_v3_rlaudit").act(obs)
```
New trainer options: `--set KEY=VAL` (any ppo_config key plus `obs_scale`, `lr_schedule`, `lr_horizon`, `activation`, `vf_coef`, `max_grad_norm`), `--reward KEY=VAL`, `--success-mode`, `--select {success_first,safe}`; every run writes `<model>.meta.json`.

## 8. Other candidates (problem / benefit / complexity / risk / decision)
| Candidate | Problem addressed | Expected benefit | Complexity | Risk | Decision |
|---|---|---|---|---|---|
| Seed ensembling / best-of-N-seeds selection on a >= 200-episode validation block | seed SD 0.12 | largest, cheapest gain (v3 is an instance) | low | winner's curse; needs fresh held-out | **done for v3**; make routine |
| Reverse gear (`allow_reverse`) + curriculum | near-start timeouts (35-47 % success), forward-only geometry | potentially large | medium (changes the task/viewer, retraining) | alters task definition | defer, needs your OK |
| SAC / TD3 | sample efficiency | small: 420k steps already train in 6 min, failure is stalling not data | medium-high (curriculum + shield wrapper with replay buffer) | stale replay vs. changing curriculum | test later |
| Many more seeds per arm (>= 20) to tune gamma/lambda/clip/epochs/vf/n_steps | untested knobs | unknown | compute only (~6 min/seed) | none | test |
| Goal-frame (slot-frame) observation and heading-aware features | world-frame dx,dy force the net to learn rotation | plausible | low-medium (breaks old checkpoints) | new observation contract | test |
| Stall-escape / recovery policy trigger | 23-31 % never-reached episodes | medium | medium | interacts with shield | test |
| Retune fuzzy "fast" membership to the 2 m/s limit | cosmetic rule coverage | negligible | low | re-verification of all fuzzy results | defer |
| VecNormalize / reward normalisation | the dead `normalize_reward` flag | none shown (obs scaling hurt) | low | stats must ship with model | reject for now |

"ARC": if you mean a specific method (e.g. an *adaptive/auxiliary* component, a paper acronym), please name it; the sensible options on this codebase are (a) a larger/other MLP or feature extractor (done: 256x256), (b) a recurrent (LSTM) policy for the stall problem, (c) a slot-frame observation encoder.
