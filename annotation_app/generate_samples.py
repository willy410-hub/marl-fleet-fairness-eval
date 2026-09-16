"""
Generates the fixed, reproducible decision-sample set used by
annotation_app/app.py. Run once before starting annotation sessions.

Usage:
    python -m annotation_app.generate_samples --n-samples 30
"""
import argparse
import os

from env.config import EnvConfig
from evaluation.episode_runner import random_action_fn
from annotation_app.sample_generator import collect_decision_samples, save_samples

SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "data", "decision_samples.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    config = EnvConfig(steps_per_episode=150)
    print(f"Collecting {args.n_samples} decision samples (seed={args.seed})...")
    samples = collect_decision_samples(config, random_action_fn, n_samples=args.n_samples, seed=args.seed)

    os.makedirs(os.path.dirname(SAMPLES_PATH), exist_ok=True)
    save_samples(samples, SAMPLES_PATH)
    print(f"Saved {len(samples)} samples to {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
