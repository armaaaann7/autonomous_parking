"""Watch a controller park (blueprint Part 26).

Live window:
    python demo.py --controller ppo_shielded --model models/ppo_shielded

Headless (saves an animated GIF instead of opening a window -- use this one in this sandbox,
or anywhere without a display):
    python demo.py --controller ppo_shielded --model models/ppo_shielded --save-gif out.gif
"""
from __future__ import annotations

import argparse
import os


def build_adapter(name: str, model_path: str):
    from evaluation.run_experiments import (FuzzyOnlyAdapter, ParallelGeometricAdapter, PPOOnlyAdapter,
                                             PPOPlusShieldAdapter, RuleBasedAdapter)
    return {
        "rule_based": lambda: RuleBasedAdapter(),
        "fuzzy_only": lambda: FuzzyOnlyAdapter(),
        "ppo_only": lambda: PPOOnlyAdapter(model_path),
        "ppo_fuzzy_bolted": lambda: PPOPlusShieldAdapter(model_path, "ppo_fuzzy_bolted"),
        "ppo_shielded": lambda: PPOPlusShieldAdapter(model_path, "ppo_shielded"),
        "parallel_geometric": lambda: ParallelGeometricAdapter(),
    }[name]()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller", choices=["rule_based", "fuzzy_only", "ppo_only", "ppo_fuzzy_bolted", "ppo_shielded", "parallel_geometric"],
                    default="fuzzy_only")
    p.add_argument("--model", default="models/ppo_shielded.zip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--kind", choices=["parallel", "perpendicular", "both"], default="both",
                    help="'both' matches evaluation.run_experiments' default, so --seed lines up with its CSVs")
    p.add_argument("--save-gif", default=None, help="path to write an animated GIF instead of a live window")
    p.add_argument("--fps", type=int, default=20)
    args = p.parse_args()

    if args.save_gif:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import time

    from environment.parking_env import ParkingEnv
    from environment.scenarios import Kind, RandomizationLevel

    kind_map = {"parallel": [Kind.PARALLEL], "perpendicular": [Kind.PERPENDICULAR],
                "both": [Kind.PARALLEL, Kind.PERPENDICULAR]}
    reverse = args.controller == "parallel_geometric"
    if reverse:
        args.kind = "parallel"
    render_mode = "rgb_array" if args.save_gif else "human"
    # Match evaluation.run_experiments' defaults (both kinds, SLOT_POSITION randomization), not
    # ParkingEnv's own defaults, so a --seed here lines up with the same episode in its CSVs --
    # scenario_kinds length changes how many RNG draws reset() makes before picking a start pose,
    # so even "the same seed" silently picks a different episode if this list doesn't match.
    env = ParkingEnv({"scenario_kinds": kind_map[args.kind], "render_mode": render_mode,
                       "randomization": RandomizationLevel.SLOT_POSITION,
                       "allow_reverse": reverse, **({"min_start_distance_scale": 1.0} if reverse else {})})
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
