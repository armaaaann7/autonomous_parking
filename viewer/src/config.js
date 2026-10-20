// All tunables in one place.
export const CONFIG = {
  // Where the real car model lives. Drop any .glb here (see README) and point to it.
  carModelUrl: './assets/models/car.glb',
  // Axis conventions of the downloaded model are unknown a priori. The loader normalises the
  // model's longest horizontal axis onto +X; flip this if the car drives backwards.
  modelForwardSign: 1,
  // Episodes offered in the dropdown (served from assets/data/).
  episodes: [
    { label: "seed 9170 · Fuzzy-only (legacy)", url: "./assets/data/fuzzy_only_seed9170.json" },
    { label: "seed 9170 · Fuzzy-only (legacy) + envelope shield", url: "./assets/data/fuzzy_only_env_seed9170.json" },
    { label: "seed 9170 · Fuzzy-precise", url: "./assets/data/fuzzy_precise_seed9170.json" },
    { label: "seed 9170 · Fuzzy-precise + envelope shield", url: "./assets/data/fuzzy_precise_env_seed9170.json" },
    { label: "seed 9170 · PPO original (100k)", url: "./assets/data/ppo_only_seed9170.json" },
    { label: "seed 9170 · PPO original (100k) + envelope shield", url: "./assets/data/ppo_only_env_seed9170.json" },
    { label: "seed 9170 · PPO v2 (retrained)", url: "./assets/data/ppo_v2_seed9170.json" },
    { label: "seed 9170 · PPO v2 (retrained) + envelope shield", url: "./assets/data/ppo_v2_env_seed9170.json" },
    { label: "seed 9170 · RL + Fuzzy (original hybrid)", url: "./assets/data/ppo_fuzzy_bolted_seed9170.json" },
    { label: "seed 9170 · RL + Fuzzy (original hybrid) + envelope shield", url: "./assets/data/ppo_fuzzy_bolted_env_seed9170.json" },
    { label: "seed 1000 · Fuzzy-only (legacy)", url: "./assets/data/fuzzy_only_seed1000.json" },
    { label: "seed 1000 · Fuzzy-only (legacy) + envelope shield", url: "./assets/data/fuzzy_only_env_seed1000.json" },
    { label: "seed 1000 · Fuzzy-precise", url: "./assets/data/fuzzy_precise_seed1000.json" },
    { label: "seed 1000 · Fuzzy-precise + envelope shield", url: "./assets/data/fuzzy_precise_env_seed1000.json" },
    { label: "seed 1000 · PPO original (100k)", url: "./assets/data/ppo_only_seed1000.json" },
    { label: "seed 1000 · PPO original (100k) + envelope shield", url: "./assets/data/ppo_only_env_seed1000.json" },
    { label: "seed 1000 · PPO v2 (retrained)", url: "./assets/data/ppo_v2_seed1000.json" },
    { label: "seed 1000 · PPO v2 (retrained) + envelope shield", url: "./assets/data/ppo_v2_env_seed1000.json" },
    { label: "seed 1000 · RL + Fuzzy (original hybrid)", url: "./assets/data/ppo_fuzzy_bolted_seed1000.json" },
    { label: "seed 1000 · RL + Fuzzy (original hybrid) + envelope shield", url: "./assets/data/ppo_fuzzy_bolted_env_seed1000.json" },
    { label: "shield demo seed 3009 · Fuzzy-only (collides)", url: "./assets/data/shield_demo_fuzzy_only_seed3009.json" },
    { label: "shield demo seed 3009 · Fuzzy-only + shield (freezes, times out)", url: "./assets/data/shield_demo_fuzzy_only_env_seed3009.json" },
    { label: "shield demo seed 3009 · Fuzzy-precise (collides)", url: "./assets/data/shield_demo_fuzzy_precise_seed3009.json" },
    { label: "shield demo seed 3009 · Fuzzy-precise + shield (freezes, times out)", url: "./assets/data/shield_demo_fuzzy_precise_env_seed3009.json" },
  ],
  names: { fuzzy_only: 'Fuzzy-only (legacy)', fuzzy_precise: 'Fuzzy-precise', ppo_only: 'PPO original (100k)', ppo_v2: 'PPO v2 (retrained)',
    ppo_fuzzy_bolted: 'RL + Fuzzy (original hybrid)',
    'fuzzy_only+env': 'Fuzzy-only (legacy) + envelope shield',
    'fuzzy_precise+env': 'Fuzzy-precise + envelope shield',
    'ppo_only+env': 'PPO original (100k) + envelope shield',
    'ppo_v2+env': 'PPO v2 (retrained) + envelope shield',
    'ppo_fuzzy_bolted+env': 'RL + Fuzzy (original hybrid) + envelope shield',
  },
  fitMode: 'inside',       // 'inside' = visible car fits within the sim's 4.2 x 1.8 m box; 'length' = match length only
  hitboxPad: 0.08,         // hitbox is slightly larger than the car so it is visible around it
  colors: { ego: 0x4fc3f7, hitbox: 0xff5252, trail: 0xffd54f, slot: 0x69f0ae },
  maxSteerRad: 0.61,       // for drawing the steering arrow (matches env_config vehicle.max_steer)
};
