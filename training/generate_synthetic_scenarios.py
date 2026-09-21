"""
CLI entry point for the synthetic scenario generator (Phase 2,
addition 5): generates a reproducible batch of synthetic training
scenarios and saves them to disk, so a training run can consume a
fixed, versioned batch instead of regenerating scenarios ad hoc.

Usage:
    python -m training.generate_synthetic_scenarios --n-scenarios 20 --seed 42 \\
        --output benchmark_results/synthetic_scenarios.json
"""
import argparse
import json

from env.config import EnvConfig
from training.synthetic_scenarios import generate_synthetic_scenario_batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-scenarios", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    base_config = EnvConfig()
    batch = generate_synthetic_scenario_batch(base_config, n_scenarios=args.n_scenarios, seed=args.seed)

    print(f"Generated {len(batch)} synthetic scenarios (seed={args.seed}):\n")
    header = f"{'scenario_id':<20} | {'spawn_rate':>10} | {'payout_mult':>11} | {'fairness_mult':>13}"
    print(header)
    print("-" * len(header))
    for scenario in batch:
        m = scenario.perturbation_multipliers
        print(
            f"{scenario.scenario_id:<20} | {m['order_spawn_rate']:>10.3f} | "
            f"{m['base_payout_per_order']:>11.3f} | {m['fairness_weight']:>13.3f}"
        )

    if args.output:
        payload = [scenario.to_dict() for scenario in batch]
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved {len(batch)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
