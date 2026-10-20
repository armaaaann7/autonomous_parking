ppo_only_best.zip is the supplied original 100,352-step PPO checkpoint, unchanged. No training occurred during this fix. The original matching stage metadata was not supplied. See the project README before resuming curriculum training.

## PPO v2 (retrained) - training record
* `ppo_v2_curriculum_selected.zip` = copy of `exp_ppo_cur_s3_best.zip` (stage 12, validation success 0.70 / collision 0.03 on seeds 7000-7099, pose mode). **Original `ppo_only_best.zip` was not overwritten.**
* Recipe: `python -m rl.train_ppo --timesteps <N> --out models/exp_ppo_cur_s<k> --curriculum --seed <k> --chunk 40000 --val-seed 7000 --val-episodes 100` (resumable with `--resume`), k = 1, 2, 3, about 420k steps each, CPU, `OMP_NUM_THREADS=1` (thread oversubscription made the first attempt very slow). SB3 PPO, 2x128 MlpPolicy, 14-dim observation, no observation normalisation. `normalize_reward: true` appears in the config but is not used anywhere (no VecNormalize).
* Candidates (validation success / collision): s1_best .55/.04, s1 .60/.04, s2_best .63/.06, s2 .55/.05, **s3_best .70/.03**, s3 .60/.02. The checkpoint was selected on seeds 7000-7099; reported held-out results use seeds 3000-3099, 5000-5199 and 9000-9199.
* Caveat on the original checkpoint: its best-checkpoint selection used seeds 5000-5039, which overlap evaluation set B (5000-5199). Set C (9000-9199) is clean for every controller.
* Logs: `experiments/ppo_retrain/train_cur_s*.log`; validation CSVs: `experiments/ppo_retrain/val/`.
