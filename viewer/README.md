# Parking viewer (three.js)

Replays REAL recorded episodes (`assets/data/*.json`, produced by `autonomous-parking/export_trajectory.py`).

    ./serve.sh            # then open http://localhost:8000

- Real car model: `assets/models/car.glb` (Khronos glTF-Sample-Assets "CarConcept", derived from a
  public-domain Sketchfab model). Swap in any .glb and edit `src/config.js` (`carModelUrl`,
  `modelForwardSign` if the car drives backwards).
- Diagnostics: ghosted collision hitboxes (exact AABBs the sim collides with on parked cars,
  padded box on the ego car), steering + throttle/brake action arrows, shield-active indicator,
  path trail, target slot, orbit / chase / top-down cameras, speed control, scrubbing.
- Export a new episode: `cd autonomous-parking && python export_trajectory.py --controller ppo_shielded --model models/ppo_shielded --kind perpendicular`
- three.js r170 is vendored in `vendor/` (no CDN needed).
