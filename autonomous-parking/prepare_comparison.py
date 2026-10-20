"""Export every controller on the same scenario (ppo_v2 uses its own selected checkpoint)."""
import argparse
import json
from pathlib import Path
from evaluation.run_experiments import make_adapter
from export_trajectory import CONTROLLERS, run_episode
from environment.scenarios import Kind

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/ppo_only_best')
    parser.add_argument('--seed', type=int, default=1000)
    args = parser.parse_args()
    model = ROOT / args.model
    if not model.with_suffix('.zip').is_file():
        parser.error(f'Missing {model.with_suffix(".zip")}. Copy your previously trained PPO model here; do not retrain unnecessarily.')
    # Build and run all three before replacing any recordings.
    recordings = []
    for name in CONTROLLERS:
        data = run_episode(make_adapter(name, str(model)), args.seed, Kind.PERPENDICULAR, name)
        recordings.append((name, data))
        print(name, data['result'], flush=True)
    out = ROOT.parent / 'viewer/assets/data'
    out.mkdir(parents=True, exist_ok=True)
    for name, data in recordings:
        (out / f'{name.replace("+", "_")}_seed{args.seed}.json').write_text(json.dumps(data))
    print('All recordings saved. Start viewer/serve.py and select each controller.')

if __name__ == '__main__':
    main()
