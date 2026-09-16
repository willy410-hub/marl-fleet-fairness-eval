# marl-fleet-fairness-eval

<p align="center">
  <img src="assets/diagrams/architecture.png" width="100%" alt="System architecture">
</p>

<p align="center">
  <img alt="tests" src="https://img.shields.io/badge/tests-62%20passing-34d399?style=for-the-badge">
  <img alt="python" src="https://img.shields.io/badge/python-3.12-38bdf8?style=for-the-badge">
  <img alt="framework" src="https://img.shields.io/badge/RL-PettingZoo%20%2B%20RLlib-a78bfa?style=for-the-badge">
  <img alt="data" src="https://img.shields.io/badge/data-real%20NYC%20TLC-fbbf24?style=for-the-badge">
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
- **62 automated tests**, full suite runs in under 3 seconds, zero flaky dependencies (no live API keys, no network access required)

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
│   └── train_ppo.py                # Shared-policy PPO training script
├── evaluation/
│   ├── centralized_baseline.py     # Hungarian-algorithm optimal baseline
│   ├── episode_runner.py           # Collects raw per-decision data
│   ├── metrics.py                  # Efficiency / fairness / decision-quality / robustness
│   ├── robustness.py               # Demand-spike stress testing
│   ├── agreement_stats.py          # Krippendorff's Alpha + Cohen's Kappa
│   └── report.py                   # Full consolidated evaluation report
├── annotation_app/
│   ├── app.py                      # Streamlit human-rating UI
│   ├── sample_generator.py         # Real decision-snapshot collection
│   ├── ratings_store.py            # Per-rater CSV persistence
│   └── generate_samples.py         # CLI to build the sample set
├── pilot_study/
│   └── run_pilot.py                # Documented pilot annotation run
├── assets/diagrams/                # Diagram generation scripts (this README's images)
└── tests/                          # 62 tests covering every module above
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full test suite (no API keys, no network access needed)
pytest tests/ -v

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
```

---

## What Was Deliberately Left Out of Scope

Documented explicitly, since knowing what was intentionally cut is as
informative as what was built:

- **Real human annotators** -- see [Pilot Study Results](#pilot-study-results) above. The pipeline is production-ready; the raters are not yet real people.
- **A learned, converged policy checkpoint is not shipped** -- the RLlib training pipeline is verified end-to-end (train -> save -> reload -> inference), but full convergence requires a multi-hour training run better suited to a GPU-backed environment than this repository's CI-friendly footprint.
- **A dedicated vector/GIS routing engine** -- the city grid uses Chebyshev distance, not real street-network routing, consistent with the "lite" scope of the originating specification.
- **Weather/event-driven demand spikes are modeled as a uniform multiplier**, not a genuinely separate stochastic process -- sufficient to test robustness-under-load, not a full weather simulation.

> **Note:** See **[DESIGN.md](./DESIGN.md)** for the full environment design -- motivation, reward architecture, and evaluation protocol.