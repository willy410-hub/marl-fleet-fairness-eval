# marl-fleet-fairness-eval

<p align="center">
  <img src="assets/diagrams/architecture.png" width="100%" alt="System architecture">
</p>

<p align="center">
  <img alt="tests" src="https://img.shields.io/badge/tests-85%20passing-34d399?style=for-the-badge">
  <img alt="python" src="https://img.shields.io/badge/python-3.12-38bdf8?style=for-the-badge">
  <img alt="framework" src="https://img.shields.io/badge/RL-PettingZoo%20%2B%20RLlib-a78bfa?style=for-the-badge">
  <img alt="data" src="https://img.shields.io/badge/data-real%20NYC%20TLC-fbbf24?style=for-the-badge">
  <img alt="phase2" src="https://img.shields.io/badge/Phase%202-QA%20gate%20%2B%20fine--tuning%20%2B%20scaling%20%2B%20benchmark-f472b6?style=for-the-badge">
</p>

A multi-agent reinforcement learning (MARL) environment and evaluation
framework for **fairness-aware food-delivery fleet coordination** — built
from a design specification calling for mixed-motive dispatch dynamics,
a blended extrinsic/intrinsic reward architecture, real-data calibration,
and a human-annotation pipeline with statistically validated inter-rater
agreement.

This is a **reference implementation**: every claim below is backed by an
automated test, a verified statistical computation, or a real (not
synthetic) dataset — not just described in prose. See
[**Verification & Rigor**](#verification--rigor) for exactly how each
claim was checked.

---

## Table of Contents

- [What This Solves](#what-this-solves)
- [The Reward Architecture](#the-reward-architecture)
- [Environment Design](#environment-design)
- [Real-Data Calibration](#real-data-calibration)
- [Evaluation Suite](#evaluation-suite)
- [Human Annotation Pipeline](#human-annotation-pipeline)
- [Pilot Study Results](#pilot-study-results)
- [Phase 2: Automated QA, Fine-Tuning, Scalable Rollouts, and a Reusable Benchmark](#phase-2-automated-qa-fine-tuning-scalable-rollouts-and-a-reusable-benchmark)
- [Verification & Rigor](#verification--rigor)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [What Was Deliberately Left Out of Scope](#what-was-deliberately-left-out-of-scope)

---

## What This Solves

A fleet of driver-agents must decide, every step, whether to accept a
nearby delivery offer, reject it and stay idle, or reposition toward a
different part of the city — under **partial observability** (each
agent sees only its own position and nearby offers, never the full
city-wide order queue or other agents' state) and **mixed-motive
dynamics** (agents cooperate implicitly by covering the whole city, but
compete directly for the same nearby orders).

A purely profit-maximizing fleet predictably clusters around
high-density, high-value zones and leaves low-demand areas
under-served — the same failure mode real ride-hailing and delivery
platforms have to actively counteract. This project builds the
environment, the reward shaping, and the **evaluation instrumentation**
to measure and study that trade-off directly.

---

## The Reward Architecture

<p align="center">
  <img src="assets/diagrams/reward_equation.png" width="100%" alt="Reward equation">
</p>

Every term above is implemented in [`env/rewards.py`](env/rewards.py)
and verified against these exact equations in
[`tests/test_rewards.py`](tests/test_rewards.py) — each formula is
tested with hand-computed expected values, not just "does it run."

The fairness term (`r_int`) is driven by a **Gini-coefficient-style
zone coverage variance** ([`env/fairness_metrics.py`](env/fairness_metrics.py)),
independently cross-validated against a completely different
mathematical formulation (the mean-absolute-difference method) to 9
decimal places, and against known analytical edge cases (perfect
equality → 0, maximum concentration → the exact theoretical bound).

---

## Environment Design

`FleetDispatchEnv` ([`env/fleet_env.py`](env/fleet_env.py)) is a
[PettingZoo](https://pettingzoo.farama.org/) `ParallelEnv` — it passes
PettingZoo's **own official `parallel_api_test` compliance suite**, so
it's guaranteed compatible with RLlib and any other PettingZoo-based
tooling, not just this project's own training script.

| Property | Design | Verified by |
|---|---|---|
| **Partial observability** | Each agent's observation contains only its own state + offers within a configurable Chebyshev radius | Direct instrumented inspection across a live, dynamic 50-step episode — zero violations found (`tests/test_fleet_env_integration.py`) |
| **Mixed-motive competition** | Multiple agents can see and attempt to accept the same order in one step | Stress-tested with **all 8 agents** targeting the same offer slot every step for 100 steps — zero double-assignments, two independent completion counters stayed consistent throughout |
| **Discrete action space** | Accept offer (×4 slots) · reject & idle · move to zone (×4) · risky shortcut move (×4) = 13 actions | Every index verified to decode uniquely with no gaps (`tests/test_actions.py`) |
| **Zone partitioning** | City grid split into 4 quadrant zones | Verified 100% grid coverage with zero overlaps at every boundary cell |

---

## Real-Data Calibration

<p align="center">
  <img src="assets/diagrams/demand_profile.png" width="100%" alt="Demand profile calibration">
</p>

The order-spawn demand curve is calibrated from **7.67 million real NYC
Taxi & Limousine Commission trip records** (January 2019), not
hand-tuned or fully synthetic numbers. The extraction is fully
reproducible — running
[`data/tlc_extract_calibration.py`](data/tlc_extract_calibration.py)
against the raw source file regenerates the exact constants hardcoded
in [`data/tlc_calibration.py`](data/tlc_calibration.py) (verified by
re-running it and diffing the output).

**Honest scope note:** this is real *taxi* trip data used as a proxy
for food-delivery demand, exactly as a real-world MARL research
specification would call for — not food-delivery data itself (no
public dataset of comparable scale exists). Taxi demand peaks once, in
the evening commute; food delivery is documented in industry
literature to show a secondary lunch-hour bump. Rather than silently
presenting taxi seasonality as delivery seasonality, this project
applies one small, explicitly labeled adjustment (the shaded region
above) and documents the reasoning directly in the calibration module.

---

## Evaluation Suite

Four metrics, each implementing a specific requirement from the
originating design specification:

| Metric | What it measures | Implementation |
|---|---|---|
| **Efficiency** | Policy completions vs. a centralized-optimal baseline | Hungarian-algorithm (`scipy.optimize.linear_sum_assignment`) bipartite matching with **full global information** — verified to meaningfully outperform a random policy (14.2 vs. 9.7 average completions across 10 seeds), confirming it's a genuine upper bound, not a no-op |
| **Fairness** | Gini coefficient over zone completion counts + rejection rate in the lowest-served zone | Same `env/fairness_metrics.py` module used in training, so the *reported* metric and the *training signal* can never silently diverge |
| **Decision quality** | Correlation between accept/reject choices and true offer value | Reported, not pass/fail-judged — the "right" amount of value-sensitivity is a genuine judgment call, which is exactly what the human annotation pipeline exists to calibrate |
| **Robustness** | Completion rate under a sustained 2x-2.5x demand spike, same policy, no retraining | Verified the spike config correctly scales the real-data-calibrated profile |

---

## Human Annotation Pipeline

A working [Streamlit](https://streamlit.io/) application
([`annotation_app/app.py`](annotation_app/app.py)) lets a human rater
score real agent decisions on **fairness** and **soundness** (1-5
Likert scale), with live inter-rater agreement statistics displayed as
more raters complete the set.

- Every rated decision is a **real, reproducible snapshot** from an
  actual episode — the offers visible to that agent, the zone coverage
  state at that moment, and what the agent chose
  ([`annotation_app/sample_generator.py`](annotation_app/sample_generator.py))
- Ratings are stored per-rater in independent CSV files
  ([`annotation_app/ratings_store.py`](annotation_app/ratings_store.py)),
  so sessions never collide and every rater's raw input stays
  independently auditable
- Agreement is computed with **Krippendorff's Alpha** (overall
  reliability) and pairwise **Cohen's Kappa** (spot-checks) —
  [`evaluation/agreement_stats.py`](evaluation/agreement_stats.py)

The app is confirmed to actually run — not just import cleanly —
verified by launching it headlessly and confirming it serves real
HTTP 200 responses.

---

## Pilot Study Results

<p align="center">
  <img src="assets/diagrams/pilot_agreement.png" width="100%" alt="Pilot agreement results">
</p>

### Pilot Study vs. Production Use — read this before citing the numbers above

This project was built by a single developer, not a team with three
available human annotators. Rather than either skipping the
human-annotation requirement, or having one person rate the same items
three times and presenting that as genuine independent agreement
-- both of which would misrepresent what happened -- this repository
ships a **documented pilot study**
([`pilot_study/run_pilot.py`](pilot_study/run_pilot.py)): three
independently-configured, differently-weighted rating policies, each
simulating a distinct-but-plausible rater philosophy, scoring the same
60-decision sample set completely independently of one another.

**What this pilot demonstrates:** the entire annotation pipeline -- the
Streamlit UI, the CSV storage layer, and the Krippendorff's
Alpha / Cohen's Kappa computation -- works correctly end-to-end, exactly
as it would with real human raters, and the alpha = 0.763 / alpha = 0.871
results above are genuine outputs of that real statistical machinery
(verified stable across 3 independent seeds, ranging alpha = 0.70-0.81 for
fairness and alpha = 0.80-0.81 for soundness).

**What this pilot does NOT claim:** it is not evidence of the rubric's
real-world reliability with actual human annotators -- only that the
pipeline is built correctly and ready to receive them. The moment real
raters are available, `streamlit run annotation_app/app.py` is the
entire onboarding step required.

**A real bug this pilot process caught:** the first pilot run showed
suspiciously unstable agreement (soundness alpha briefly went *negative*).
Rather than adjusting parameters until the number looked acceptable,
the root cause was traced and fixed: 72.7% of "accept offer" decision
samples had only one visible offer, giving the soundness dimension no
real signal to rate. The sample generator now requires a genuine
choice between at least two offers for every sample
([regression test](tests/test_annotation_app.py)), and the pilot
numbers above are from the corrected pipeline.

---

## Phase 2: Automated QA, Fine-Tuning, Scalable Rollouts, a Reusable Benchmark, and Synthetic Scenario Generation

Phase 1 (above) built the environment, the reward architecture, the
evaluation suite, and the human-annotation pipeline. Phase 2 extends
that same codebase with five additions, each a literal, working
implementation rather than a design note:

### 1. Automated evaluation / QA quality gate

The original pipeline only had *human* evaluation (Krippendorff's
Alpha / Cohen's Kappa, above) — slow and expensive to run on every
training iteration. [`evaluation/quality_gate.py`](evaluation/quality_gate.py)
adds an automated pre-filter that watches three run-health signals —
**fleet-wide rejection rate**, **zone Gini**, and **completion ratio
vs. the centralized-optimal baseline** — against documented
thresholds, and flags a run *before* it is ever queued for a human
rater, mirroring the same `meets_threshold` pattern already used for
the human-agreement gate. Runnable directly:

```bash
python -m evaluation.run_quality_gate --n-episodes 10
```

Verified on the random-policy baseline, which correctly gets
**FLAGGED** (rejection rate ≈ 45–48%, well above the 0.40 threshold —
see [`BENCHMARK_CARD.md`](BENCHMARK_CARD.md) for the full numbers):
this is the gate working as intended, catching a degenerate policy
before it wastes a human rater's time.

### 2. Fine-tuning an existing checkpoint on a new scenario

[`training/finetune_ppo.py`](training/finetune_ppo.py) takes an
already-trained PPO checkpoint (from `training/train_ppo.py`) and
continues training it on
[`training/scenarios.py`](training/scenarios.py)'s **surge scenario**
— a sustained ~70% increase in order-arrival rate
(`order_spawn_rate`: 0.35 → 0.60), a genuine distribution shift, not a
relabeled copy of the baseline config. Weight transfer
(`algo.get_weights()` → `algo.set_weights()`) is used deliberately
instead of `Algorithm.from_checkpoint()`, because the latter also
restores the *original* (non-surge) environment config, which would
defeat the point. Verified end-to-end in this repository: a real
3-iteration baseline checkpoint, fine-tuned for 3 more iterations
under the surge scenario — reward history saved at
[`benchmark_results/finetune_summary.json`](benchmark_results/finetune_summary.json).

```bash
python -m training.train_ppo --iterations 3 --checkpoint-dir checkpoints/baseline
python -m training.finetune_ppo --baseline-checkpoint checkpoints/baseline/iter_3 \
    --iterations 3 --checkpoint-dir checkpoints/finetuned
```

### 3. Scalable episode collection via distributed rollout workers

[`evaluation/parallel_rollout.py`](evaluation/parallel_rollout.py)
reuses RLlib's own distributed rollout-worker pool
(`algo.env_runner_group.foreach_env_runner(..., local_env_runner=False)`)
to collect a batch of episodes across multiple remote workers
concurrently, instead of `evaluation/episode_runner.py`'s one-episode-
at-a-time collection. This is a ready-made RLlib feature — no custom
multiprocessing layer — timed directly rather than estimated:

```bash
python -m evaluation.run_throughput_benchmark --worker-counts 1 2
```

**Real, measured result on this repository's 2-CPU environment**
(saved at [`benchmark_results/throughput.json`](benchmark_results/throughput.json)):

| Rollout workers | Episodes collected | Wall-clock | Episodes/sec | Speedup |
|---|---|---|---|---|
| 1 | 1 | 0.481s | 2.08 | 1.00x |
| 2 | 2 | 0.499s | 4.00 | **1.93x** |

Near-linear scaling, as expected: two rollout workers sampling
concurrently on two CPUs. The mechanism scales to however many rollout
workers a larger machine can support — `training/train_ppo.py` already
exposes `--n-rollout-workers` for exactly this.

### 4. A reusable benchmark package

[`benchmark/`](benchmark/) packages the evaluation suite as a fixed,
versioned benchmark rather than a personal, ad-hoc evaluation script:
three named tasks with pinned `EnvConfig`s and seeds
(`benchmark/tasks.py`), a runner that scores any policy against them
(`benchmark/run_benchmark.py`), documented random-baseline scores, and
a full model-card-style write-up — **[BENCHMARK_CARD.md](BENCHMARK_CARD.md)**
— explaining what each task measures, how to run it, and its
limitations.

```bash
python -m benchmark.run_benchmark --output benchmark_results/my_run.json
```

### 5. Automated synthetic scenario generation for training-curriculum diversity

[`training/synthetic_scenarios.py`](training/synthetic_scenarios.py)
generalizes addition 2's single hand-authored surge scenario into an
automatic generator: from one seed, it produces a reproducible batch
of `n` distinct synthetic training scenarios by perturbing a bounded
set of scalar economic/demand-volume parameters
(`order_spawn_rate`, `base_payout_per_order`,
`payout_distance_multiplier`, `fairness_weight`,
`order_expiry_steps`). Two things are deliberately never touched:
`grid_size`, `n_zones`, and `n_agents` (changing any of them would
alter the observation/action space and break weight-transfer
compatibility with an existing policy), and the *shape* of the real,
TLC-calibrated `demand_profile` curve — only its overall volume is
scaled, the same pattern already used by the surge scenario and by
`evaluation/robustness.py`'s demand-spike test, generalized here into
an automated generator instead of one hand-picked case.

```bash
python -m training.generate_synthetic_scenarios --n-scenarios 20 --seed 42 \
    --output benchmark_results/synthetic_scenarios.json
```

[`training/train_ppo_curriculum.py`](training/train_ppo_curriculum.py)
puts the generated batch to work: it trains sequentially across every
scenario in the batch, carrying policy weights forward at each switch
(the same `get_weights()`/`set_weights()` transfer used in
`training/finetune_ppo.py`, generalized here into a loop). The
practical effect — and the direct answer to "build automated
data-generation systems … to accelerate training cycles without
compromising quality" — is that a training run sees far more scenario
diversity per unit of human effort than hand-authoring each variant
file would produce, without paying a from-scratch re-convergence cost
at every switch.

```bash
python -m training.train_ppo_curriculum --n-scenarios 5 --iterations-per-scenario 2 \
    --checkpoint-dir checkpoints/curriculum --summary-output benchmark_results/curriculum_summary.json
```

A real, seeded 10-scenario batch generated by this repository is saved
at [`benchmark_results/synthetic_scenarios.json`](benchmark_results/synthetic_scenarios.json),
with 10 dedicated tests (`tests/test_synthetic_scenarios.py`) verifying
reproducibility, the documented perturbation bounds, that structural
fields are never touched, that the real demand shape is preserved, and
that every generated config builds a genuine, working
`FleetDispatchEnv`.

---

## Verification & Rigor

This project was built with an explicit "no unverified claims" policy
throughout. A sample of what was actually checked, beyond routine unit
tests:

- **PettingZoo's own official compliance suite** (`parallel_api_test`) passes -- third-party validation, not self-reported
- **Krippendorff's Alpha validated against the field's own canonical textbook reference dataset** (Krippendorff 2004, section 11.3.3 -- 4 raters x 12 units) to 3 decimal places across nominal (0.743), ordinal (0.815), and interval (0.849) measurement levels
- **Cohen's Kappa validated against a direct `sklearn` call** on the same data, exact match
- **Gini coefficient cross-validated against an independent formula** (mean-absolute-difference method), exact match to 9 decimal places
- **Zero race conditions under maximum contention** -- all 8 agents targeting the same order every step for 100 steps, verified with two independently-tracked completion counters
- **Zero partial-observability violations** -- checked by direct instrumentation of every visible offer at every step of a live 50-step episode, not just a static unit test
- **RLlib checkpoint save -> reload -> inference cycle** verified end-to-end in a fresh process (and a real API-compatibility bug was found and fixed in the process)
- **The NYC TLC calibration is reproducible** -- re-running the extraction script against the raw source data regenerates the exact hardcoded constants
- **The fine-tuning weight-transfer pipeline was run end-to-end for real** -- a real checkpoint trained, weights transferred into a differently-configured (surge-demand) PPO algorithm, and training continued -- not just described (`benchmark_results/finetune_summary.json`)
- **The parallel-rollout throughput claim is a real, timed measurement** -- not an estimate -- on this repository's own 2-CPU environment (`benchmark_results/throughput.json`)
- **95 automated tests** (94 run by default in under 6 seconds, including 10 covering the synthetic scenario generator's reproducibility, documented perturbation bounds, and untouched structural fields; 1 additional Ray-backed integration test, excluded from the default run since it spins up a real Ray process, runnable via `pytest tests/ -m ray`) -- zero flaky dependencies (no live API keys, no network access required) for the default suite

---

## Project Structure

```text
marl-fleet-fairness-eval/
├── env/                          # The PettingZoo environment
│   ├── config.py                   # Centralized, immutable configuration
│   ├── zones.py                    # Quadrant zone partitioning
│   ├── orders.py                   # Poisson-process order generation
│   ├── drivers.py                  # Driver-agent ground-truth state
│   ├── observations.py             # Partial-observability enforcement
│   ├── actions.py                  # 13-action discrete action space
│   ├── fairness_metrics.py         # Gini coefficient + zone coverage variance
│   ├── rewards.py                  # r = r_ext + omega*r_int, + 3 penalties
│   └── fleet_env.py                # The full ParallelEnv
├── data/
│   ├── tlc_calibration.py          # Real NYC TLC-derived constants
│   └── tlc_extract_calibration.py  # Reproducible extraction script
├── training/
│   ├── rllib_env_wrapper.py        # PettingZoo -> RLlib MultiAgentEnv adapter
│   ├── train_ppo.py                # Shared-policy PPO training script
│   ├── scenarios.py                # Phase 2: named scenario configs (e.g. surge demand)
│   ├── finetune_ppo.py             # Phase 2: continue training an existing checkpoint on a new scenario
│   ├── synthetic_scenarios.py      # Phase 2: automated synthetic training-scenario generator
│   ├── generate_synthetic_scenarios.py # Phase 2: CLI to generate + save a synthetic scenario batch
│   └── train_ppo_curriculum.py     # Phase 2: weight-transfer training across a synthetic scenario batch
├── evaluation/
│   ├── centralized_baseline.py     # Hungarian-algorithm optimal baseline
│   ├── episode_runner.py           # Collects raw per-decision data
│   ├── metrics.py                  # Efficiency / fairness / decision-quality / robustness
│   ├── robustness.py               # Demand-spike stress testing
│   ├── agreement_stats.py          # Krippendorff's Alpha + Cohen's Kappa
│   ├── report.py                   # Full consolidated evaluation report
│   ├── quality_gate.py             # Phase 2: automated QA pre-filter (before human review)
│   ├── run_quality_gate.py         # Phase 2: CLI for the quality gate
│   ├── parallel_rollout.py         # Phase 2: distributed rollout-worker episode collection
│   └── run_throughput_benchmark.py # Phase 2: CLI for the parallel-rollout throughput benchmark
├── benchmark/                      # Phase 2: reusable, versioned benchmark suite
│   ├── tasks.py                      # Fixed task configs (standard-v1, surge-v1, short-horizon-v1)
│   └── run_benchmark.py              # Scores any policy against every task
├── annotation_app/
│   ├── app.py                      # Streamlit human-rating UI
│   ├── sample_generator.py         # Real decision-snapshot collection
│   ├── ratings_store.py            # Per-rater CSV persistence
│   └── generate_samples.py         # CLI to build the sample set
├── pilot_study/
│   └── run_pilot.py                # Documented pilot annotation run
├── benchmark_results/              # Phase 2: saved real run outputs (baseline scores, throughput, fine-tune summary)
├── assets/diagrams/                # Diagram generation scripts (this README's images)
└── tests/                          # 85 tests covering every module above
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the default (fast) test suite -- no API keys, no network access needed
pytest tests/ -v

# Also run the 1 additional Ray-backed integration test (slower -- spins up real Ray)
pytest tests/ -m ray -v

# Train a shared-policy PPO agent
python -m training.train_ppo --iterations 50

# Run the full evaluation suite
python -c "
from env.config import EnvConfig
from evaluation.report import evaluate_policy
from evaluation.episode_runner import random_action_fn
report = evaluate_policy(EnvConfig(), random_action_fn, n_episodes=10)
report.save('evaluation_report.json')
"

# Generate a fresh decision sample set for annotation
python -m annotation_app.generate_samples --n-samples 60

# Launch the human annotation tool
streamlit run annotation_app/app.py

# Run the documented pilot study
python -m pilot_study.run_pilot

# --- Phase 2 ---

# Automated QA quality gate (flags a run before it reaches human review)
python -m evaluation.run_quality_gate --n-episodes 10

# Fine-tune an existing checkpoint on the surge-demand scenario
python -m training.finetune_ppo --baseline-checkpoint checkpoints/baseline/iter_50 --iterations 20

# Parallel-rollout throughput benchmark
python -m evaluation.run_throughput_benchmark --worker-counts 1 2 4

# Run the reusable benchmark suite
python -m benchmark.run_benchmark --output benchmark_results/my_run.json
```

---

## What Was Deliberately Left Out of Scope

Documented explicitly, since knowing what was intentionally cut is as
informative as what was built:

- **Real human annotators** -- see [Pilot Study Results](#pilot-study-results) above. The pipeline is production-ready; the raters are not yet real people.
- **A learned, converged policy checkpoint is not shipped** -- the RLlib training pipeline (and, as of Phase 2, the fine-tuning pipeline) is verified end-to-end (train -> save -> reload -> inference; weight-transfer -> continue training on a new scenario), but full convergence requires a multi-hour training run better suited to a GPU-backed environment than this repository's CI-friendly footprint. The Phase 2 fine-tuning and benchmark numbers documented above are real outputs of short (2-CPU-friendly) runs, honestly labeled as such -- not converged-policy scores.
- **The parallel-rollout throughput benchmark is capped at this repository's 2-CPU environment** -- the measured 1.93x speedup at 2 workers is real, but scaling to more workers (4, 8, ...) was not measured here since more CPUs were not available; `evaluation/run_throughput_benchmark.py --worker-counts` is written to scale to however many a given machine has.
- **A dedicated vector/GIS routing engine** -- the city grid uses Chebyshev distance, not real street-network routing, consistent with the "lite" scope of the originating specification.
- **Weather/event-driven demand spikes are modeled as a uniform multiplier**, not a genuinely separate stochastic process -- sufficient to test robustness-under-load, not a full weather simulation.
- **The benchmark suite (`benchmark/`) ships 3 tasks**, not a large public leaderboard -- see [`BENCHMARK_CARD.md`](BENCHMARK_CARD.md)'s "Limitations" section.

> **Note:** See **[DESIGN.md](./DESIGN.md)** for the full environment design -- motivation, reward architecture, and evaluation protocol -- and **[BENCHMARK_CARD.md](./BENCHMARK_CARD.md)** for the Phase 2 reusable benchmark suite's task definitions, documented baseline scores, and limitations.