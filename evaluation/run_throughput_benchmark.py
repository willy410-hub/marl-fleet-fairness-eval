"""
CLI entry point for the parallel-rollout throughput benchmark
(Phase 2, addition 3): documents the real, measured before/after
episode-collection throughput of RLlib's distributed rollout workers
on this environment, at whatever worker counts the machine running it
supports.

Usage:
    python -m evaluation.run_throughput_benchmark --worker-counts 1 2 4 --output benchmark_results/throughput.json
"""
import argparse
import json

from env.config import EnvConfig
from evaluation.parallel_rollout import run_scaling_comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-counts", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--rollout-fragment-length", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    config = EnvConfig()
    results = run_scaling_comparison(
        config, worker_counts=args.worker_counts, rollout_fragment_length=args.rollout_fragment_length
    )

    baseline = results[0]
    print(f"{'workers':>8} | {'episodes':>9} | {'seconds':>9} | {'eps/sec':>9} | {'speedup vs 1 worker':>20}")
    for r in results:
        speedup = r.episodes_per_second / baseline.episodes_per_second if baseline.episodes_per_second > 0 else 0.0
        print(f"{r.n_rollout_workers:>8} | {r.total_episodes:>9} | {r.wall_clock_seconds:>9.3f} | "
              f"{r.episodes_per_second:>9.3f} | {speedup:>19.2f}x")

    if args.output:
        payload = [
            {
                "n_rollout_workers": r.n_rollout_workers,
                "wall_clock_seconds": r.wall_clock_seconds,
                "total_episodes": r.total_episodes,
                "total_env_steps": r.total_env_steps,
                "episodes_per_second": r.episodes_per_second,
            }
            for r in results
        ]
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
