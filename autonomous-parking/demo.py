from __future__ import annotations

import argparse
import os


def build_adapter(name: str, model_path: str):
    from evaluation.run_experiments import make_adapter
    return make_adapter(name, model_path)


def main():
    p = argparse.ArgumentParser()
    from evaluation.run_experiments import ADAPTER_NAMES
    names = list(ADAPTER_NAMES) + [n + "+env" for n in ADAPTER_NAMES]
    p.add_argument("--controller", choices=names, default="fuzzy_only")
    p.add_argument("--model", default="models/ppo_only_best.zip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kind", choices=["perpendicular"], default="perpendicular",
                    help="Project comparison uses perpendicular parking")
    p.add_argument("--save-gif", default=None, help="path to write an animated GIF instead of a live window")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()
    from pathlib import Path
    if not args.controller.startswith('fuzzy_only') and not Path(args.model).with_suffix('.zip').is_file():
        p.error(f'Missing PPO checkpoint {args.model}. Copy models/ppo_only_best.zip into autonomous-parking/models/.')

    if args.save_gif:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import time

    from environment.parking_env import ParkingEnv
    from environment.scenarios import Kind, RandomizationLevel

    kind_map = {"parallel": [Kind.PARALLEL], "perpendicular": [Kind.PERPENDICULAR],
                "both": [Kind.PARALLEL, Kind.PERPENDICULAR]}
    render_mode = "rgb_array" if args.save_gif else "human"
    # Match evaluation.run_experiments' defaults (both kinds, SLOT_POSITION randomization), not
    # ParkingEnv's own defaults, so a --seed here lines up with the same episode in its CSVs --
    # scenario_kinds length changes how many RNG draws reset() makes before picking a start pose,
    # so even "the same seed" silently picks a different episode if this list doesn't match.
    env = ParkingEnv({"scenario_kinds": kind_map[args.kind], "render_mode": render_mode,
                       "randomization": RandomizationLevel.SLOT_POSITION})
    adapter = build_adapter(args.controller, args.model)
    adapter.reset()

    obs, info = env.reset(seed=args.seed)
    frames = []
    done = False
    frame_time = 1.0 / args.fps
    while not done:
        step_start = time.time()
        action, shield_active = adapter.act(obs, env)
        obs, reward, term, trunc, info = env.step(action)
        if env._renderer is None:
            env.render()
        env._renderer.set_shield_active(bool(shield_active))
        frame = env.render()
        if args.save_gif:
            frames.append(frame)
        else:
            # Without this, a live window runs the whole episode as fast as the CPU can step it
            # (a 2-3 second episode would render and close in a fraction of a second) -- pace it
            # to the requested --fps so it's actually watchable.
            elapsed = time.time() - step_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            # also pump the window's event queue so macOS/most window managers don't mark it
            # "not responding" during the sleep
            import pygame
            pygame.event.pump()
        done = term or trunc
    print(f"episode ended: {info}")

    if args.save_gif:
        import imageio
        imageio.mimsave(args.save_gif, frames, fps=args.fps)
        print(f"saved {len(frames)} frames -> {args.save_gif}")
    else:
        print("Episode finished. Close the pygame window (or press Ctrl+C) to exit.")
        import pygame
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
            time.sleep(0.05)
    env.close()


if __name__ == "__main__":
    main()
