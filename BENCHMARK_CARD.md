# Benchmark Card: marl-fleet-fairness-eval

A model-card-style document for this project's benchmark suite
(`benchmark/`), following the same "document exactly what was checked,
not just what was built" standard as the rest of this repository (see
[Verification & Rigor](README.md#verification--rigor)).

---

## What This Benchmark Measures

A fixed, versioned set of tasks for scoring *any* dispatch policy on
the fairness/efficiency trade-off described in
[`DESIGN.md`](DESIGN.md) -- not just the policies trained in this
repository. Running the same task ID against two different policies
produces directly comparable numbers, because the environment config,
episode count, and random seeds for that task are pinned and versioned
(`benchmark/tasks.py`).

This turns the project from a personal experiment into something
someone else could run their own policy against and get a real,
apples-to-apples score, in the same spirit as a published benchmark
release (a fixed task suite + documented baseline scores + a
reusable runner), rather than a one-off evaluation script.

---

## How To Run It

```bash
# Score the random-policy reference baseline against every task:
python -m benchmark.run_benchmark --output benchmark_results/my_run.json

# Score a trained RLlib checkpoint:
python -m benchmark.run_benchmark --checkpoint checkpoints/iter_50 \
    --output benchmark_results/my_policy.json

# Score against a subset of tasks:
python -m benchmark.run_benchmark --tasks standard-v1 surge-v1
```

`benchmark.run_benchmark.run_task()` / `run_full_benchmark()` are also
importable directly for programmatic use -- see
`tests/test_benchmark.py` for usage examples.

Every task run also passes through the automated quality gate
(`evaluation/quality_gate.py`, Phase 2 addition 1) — a policy's score
comes with a `PASSED`/`FLAGGED` verdict, so a badly regressed policy's
numbers are never silently reported as if they were trustworthy.

---

## The Fixed Task Suite

| Task ID | Grid / Agents | Episode length | Demand | Episodes | Base seed |
|---|---|---|---|---|---|
| `standard-v1` | 10x10, 8 agents | 200 steps | TLC-calibrated, baseline volume (`order_spawn_rate=0.35`) | 10 | 5000 |
| `surge-v1` | 10x10, 8 agents | 200 steps | Same shape, sustained ~70% higher volume (`order_spawn_rate=0.60`, see [`training/scenarios.py`](training/scenarios.py)) | 10 | 6000 |
| `short-horizon-v1` | 10x10, 8 agents | 60 steps | TLC-calibrated, baseline volume | 10 | 7000 |

`standard-v1` is the headline task. `surge-v1` exists specifically to
test whether a `standard-v1`-trained policy generalizes to the demand
regime that [`training/finetune_ppo.py`](training/finetune_ppo.py)
(Phase 2 addition 2) fine-tunes against. `short-horizon-v1` is a
cheap, fast task for iterating on a policy before spending the compute
on the full-length tasks.

**Versioning:** a task's `env_config` and seeds are frozen once
published under a given task ID (the `-v1` suffix). A breaking change
to a task's definition ships as a new task ID (`standard-v2`, etc.)
rather than silently changing what `standard-v1` means -- otherwise
old scores would stop being comparable to new ones without warning.

---

## Documented Baseline Scores

The **random-policy reference baseline** — `evaluation.episode_runner.random_action_fn`,
i.e. an agent that picks a uniformly random legal action every step —
scored on every task, produced by an actual run of
`python -m benchmark.run_benchmark`, saved at
[`benchmark_results/random_baseline.json`](benchmark_results/random_baseline.json):

| Metric | `standard-v1` | `surge-v1` | `short-horizon-v1` |
|---|---|---|---|
| Completion ratio vs. centralized-optimal baseline | 0.688 | 0.719 | 0.627 |
| Zone Gini (lower = fairer) | 0.064 | 0.046 | 0.214 |
| Rejection rate, lowest-served zone | 0.065 | 0.055 | 0.129 |
| Decision-quality correlation (value-sensitivity) | -0.019 | -0.020 | -0.012 |
| Robustness, 2.0x demand spike (completions ratio) | 1.775 | 1.485 | 11.0* |
| Robustness, 2.5x demand spike (completions ratio) | 2.562 | 1.717 | 5.33* |
| Automated quality gate | **FLAGGED** (rejection_rate) | **FLAGGED** (rejection_rate) | **FLAGGED** (rejection_rate) |

\* The `short-horizon-v1` robustness numbers are noisy (only 60 steps
per episode means very few completions in the "normal" condition,
sometimes near zero, so the ratio swings wildly) -- reported honestly
rather than hidden, but this is exactly why `short-horizon-v1` is
positioned as a *quick-iteration* task, not one to draw robustness
conclusions from.

**Why the random baseline is flagged on every task, correctly:** the
random policy rejects roughly 40-48% of the offers it sees (it has no
value-sensitivity, so it rejects offers essentially by coin flip) --
above the quality gate's `rejection_rate` threshold of 0.40. This is
the gate working as designed: it is supposed to catch and hold back a
policy this degenerate before it reaches a human rater, and the random
baseline is deliberately degenerate (it exists as evaluation
machinery's lower bound, not as a policy anyone would want scored by a
human). A trained policy is expected to clear this check; see
[`benchmark_results/random_baseline.json`](benchmark_results/random_baseline.json)
for the full per-check breakdown.

A learned, converged policy's scores are **not** included here, for
the same reason already documented in the README's
["What Was Deliberately Left Out of Scope"](README.md#what-was-deliberately-left-out-of-scope):
full convergence needs a multi-hour GPU-backed training run, outside
this repository's CI-friendly footprint. What *is* verified end-to-end
in this repository, with real (not fabricated) numbers, is the full
pipeline: `training/train_ppo.py` → `training/finetune_ppo.py` →
`benchmark/run_benchmark.py` all run correctly against real
checkpoints (see
[`benchmark_results/finetune_summary.json`](benchmark_results/finetune_summary.json)
for the fine-tuning run's real reward history).

---

## Statistical Rigor: Replicated Scores

The scores above are each a single run under one fixed seed range --
enough to compare against a regression on the same seeds, but not
enough to say "this number is not just this seed range being
favorable." `benchmark/run_benchmark_replicated.py` (see the
[README](README.md#7-statistically-rigorous-benchmark-scoring)) re-runs
each task 5 times under independent, non-overlapping seed ranges and
reports a Student's-t 95% confidence interval per metric, plus a
**crisp pass/fail** verdict: a task only counts as crisply passing the
quality gate's `MIN_COMPLETION_RATIO_VS_RANDOM` threshold when the
confidence interval's *lower* bound clears it, not just the mean.

A real 5-replicate run of the random-policy baseline, saved at
[`benchmark_results/random_baseline_replicated.json`](benchmark_results/random_baseline_replicated.json):

| Task | Completion ratio (mean, 95% CI) | Zone Gini (mean, 95% CI) | Quality-gate pass rate | Crisp pass? |
|---|---|---|---|---|
| `standard-v1` | 0.7249 [0.6943, 0.7556] | 0.0521 [0.0446, 0.0596] | 0/5 | Yes (vs. the 0.5 threshold) |
| `surge-v1` | 0.7316 [0.7142, 0.7490] | 0.0320 [0.0215, 0.0426] | 4/5 | Yes |
| `short-horizon-v1` | 0.7065 [0.6447, 0.7682] | 0.1364 [0.0744, 0.1985] | 0/5 | Yes |

Reading this correctly: the random policy crisply clears
`MIN_COMPLETION_RATIO_VS_RANDOM` because that threshold is defined
*relative to the random baseline itself* (0.5, i.e. "at least half of
what a random policy manages") -- a policy scoring near 1.0 on this
column is not the bar, clearing 0.5 while a trained policy is expected
to score meaningfully higher is. The quality-gate pass rate (driven by
the separate `rejection_rate` check, not `completion_ratio`) correctly
stays low, for the same reason documented above under "Why the random
baseline is flagged on every task."

---

## Intended Use

- **For this project's own policies:** a fixed regression check --
  re-running `standard-v1` after a training change should not silently
  regress completion ratio or fairness without it showing up here.
- **For someone else's dispatch policy:** implement an
  `evaluation.episode_runner.ActionFn`-compatible function (same
  signature `random_action_fn` and the RLlib checkpoint loader in
  `benchmark/run_benchmark.py` already use) and run it through
  `benchmark.run_benchmark.run_full_benchmark()`.
- **Not intended** as a leaderboard across unrelated environments --
  the tasks are specific to this project's `FleetDispatchEnv` and its
  observation/action space; a policy for a different environment
  cannot be scored against it.

---

## Limitations

- Only 3 tasks exist so far -- deliberately small and fixed rather
  than broad, consistent with this project's "lite" scope (see
  [`DESIGN.md`](DESIGN.md)).
- `n_episodes=10` per task is enough for a smoke-test-grade score, not
  a statistically tight one -- the per-episode numbers in a saved
  results JSON make the variance visible rather than hiding it behind
  a single averaged number.
- The centralized-optimal baseline used for `completion_ratio`
  ([`evaluation/centralized_baseline.py`](evaluation/centralized_baseline.py))
  has full global information, which is a strictly easier problem than
  what the MARL agents solve under partial observability -- so
  `completion_ratio` is an honest ratio against an upper bound, not a
  same-information comparison. See that module's own docstring for
  the full reasoning.
- The replicated scores above use `n_replicates=5` -- enough to widen a
  single-seed point estimate into an honest interval, not enough for a
  tight one; the reported CIs are correspondingly wide, which is the
  t-distribution being honest about a small sample, not a bug.
- `evaluation.run_shielded_evaluation` demonstrates the safety shield
  changing measured behavior (a real 5-episode run recorded 8,102
  shield overrides and moved `completion_ratio` from 0.730 to 0.817 --
  saved at [`benchmark_results/shielded_vs_unshielded.json`](benchmark_results/shielded_vs_unshielded.json))
  but this is the random policy, which attempts the shortcut action far
  more often than a trained policy would; the override count and metric
  shift for a trained checkpoint would be expected to be much smaller.
