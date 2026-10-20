// All tunables in one place.
export const CONFIG = {
  // Where the real car model lives. Drop any .glb here (see README) and point to it.
  carModelUrl: './assets/models/car.glb',
  // Axis conventions of the downloaded model are unknown a priori. The loader normalises the
  // model's longest horizontal axis onto +X; flip this if the car drives backwards.
  modelForwardSign: 1,
  // Episodes offered in the dropdown (served from assets/data/).
  episodes: [
    { label: 'perpendicular (default)', url: './assets/data/trajectory.json' },
    { label: 'parallel', url: './assets/data/trajectory_parallel.json' },
  ],
  fitMode: 'inside',       // 'inside' = visible car fits within the sim's 4.2 x 1.8 m box; 'length' = match length only
  hitboxPad: 0.08,         // hitbox is slightly larger than the car so it is visible around it
  colors: { ego: 0x4fc3f7, hitbox: 0xff5252, trail: 0xffd54f, slot: 0x69f0ae },
  maxSteerRad: 0.61,       // for drawing the steering arrow (matches env_config vehicle.max_steer)
};
