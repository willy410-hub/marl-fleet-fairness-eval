# Design Document: MARL Fleet Fairness Eval

**Domain:** Agentic Food Delivery Fleet Coordination — Evaluating Agent Decision Quality in Real-Time Dispatch

---

## 1. Motivation

This project frames an everyday delivery-dispatch scenario as a multi-agent RL environment for evaluating agent decision quality, not just simulating traffic.

- Partial observability: each driver-agent sees its own location and nearby order offers, but not the full city order queue or other drivers' state.
- Mixed-motive dynamics: agents optimize personal earnings vs. system-wide fairness/coverage.
- Underexplored angle: most dispatch research is centralized assignment optimization, not decentralized learned agents whose decisions need quality/fairness evaluation.
- This directly mirrors real agentic-evaluation work: judging whether an agent's operational decision is sound, fair, and non-exploitative — not just efficient.
- Real-world relevance: platforms increasingly use autonomous agents for dispatch decisions, and need evaluation frameworks for those decisions before deploying them.

---

## 2. Environment Specification

### 2.1 Agents & Observation Space

- Heterogeneous driver-agents (vehicle type, speed, current order load).
- Each agent observes: own location, nearby order offers, own earnings so far, local time/demand signal.
- Each agent does NOT observe: the full city-wide order queue, or other agents' positions or decisions.
- State dimensions: location, active order count, time-of-day, local demand density.

### 2.2 Action Space & Interdependence

- Interdependence type: Mixed-Motive (Cooperation + Competition).
- Action: accept/reject an order offer, or choose a zone to idle in.
- Interdependence: accepting an order removes it from other agents' option pool; clustering in one zone starves coverage elsewhere.
- Temporal effect: accepting a long order delays an agent's availability for the next offer.

---

## 3. Reward Architecture

**Approach:** Intrinsic Motivation + Extrinsic Rewards.

- Extrinsic (profit-driven), per delivery: `r_ext = payout − (time_penalty * delivery_time) − (fuel_cost * distance)`
- Intrinsic (fairness-driven), per step: `r_int = coverage_bonus * (1 − zone_coverage_variance)`
- Total reward: `r = r_ext + omega * r_int`, where omega tunes how much fairness matters vs. raw profit.
- Penalized behaviors: cherry-picking only high-value orders, abandoning low-demand zones, unsafe shortcuts (risk penalty).
- This extrinsic/intrinsic split is the core agentic-evaluation angle: it separates raw efficiency from judged decision quality.

---

## 4. Evaluation Protocol

- Efficiency: average delivery time vs. a centralized-optimal dispatch baseline.
- Fairness: coverage variance across zones (Gini-style), rejection rate in low-order areas.
- Decision-quality flag: correlation between accept/reject choices and true order value vs. bias signals — a proxy scoring rubric similar to grading real agent decisions.
- Annotator agreement requirement: 3+ human raters score a sample of agent decisions on fairness/soundness; require Krippendorff's Alpha ≥ 0.7 before trusting the rubric's labels (pairwise spot-checks use Cohen's Kappa).
- Robustness: performance under demand spikes (rush hour, weather) without retraining.

---

## 5. Required Expertise & Data Sources

- Expertise: logistics/dispatch operations, urban demand patterns, rubric design for judging agent decisions.
- Data: public ride-hailing trip data (e.g., NYC TLC) as a proxy, with synthetic order generation calibrated per zone.
- Tools: PettingZoo/RLlib, plus a lightweight city-grid simulator.

---

## 6. Resourcing & Feasibility

- Estimated compute budget: $5K–$20K (moderate complexity).
- Feasibility confidence: 4/5 (High) — a well-understood dispatch problem with abundant public transportation data to build on.

---

## 7. Known Challenges & Mitigations

- Defining "fairness" quantitatively → use coverage/Gini metrics, validated against a human-judgment rubric with a required Krippendorff's Alpha ≥ 0.7 across annotators before trusting it.
- Balancing profit realism vs. the fairness objective → addressed by the intrinsic/extrinsic reward split above.
- Limited hyperlocal delivery data → use ride-hailing data as a proxy plus a synthetic order overlay.
