"""
Generates a fixed, reproducible sample of agent decisions for human
annotation, per the design document's "3+ human raters score a sample
of agent decisions on fairness/soundness."

Each sample item is one specific decision point in an episode --
snapshotted with enough surrounding context (what offers were visible,
what the fleet's zone coverage looked like at that moment, what the
agent chose) that a human rater can judge it without needing to run
the simulator themselves.
"""
import json
from dataclasses import asdict, dataclass

import numpy as np

from env.actions import ActionKind, decode_action
from env.config import EnvConfig
from env.fleet_env import FleetDispatchEnv
from evaluation.episode_runner import ActionFn


@dataclass
class DecisionSample:
    """One decision point, snapshotted for human review."""

    sample_id: str
    episode_seed: int
    step: int
    agent_id: str
    agent_position: tuple
    action_taken: str
    visible_offers: list
    chosen_offer_payout: float | None
    mean_visible_payout: float | None
    zone_completion_counts_before: list
    zone_id_of_action: int | None


def collect_decision_samples(
    config: EnvConfig, action_fn: ActionFn, n_samples: int, seed: int = 777
) -> list[DecisionSample]:
    """
    Run episodes until `n_samples` ACCEPT_OFFER or REJECT_AND_IDLE
    decisions have been collected, evenly spaced across the run so the
    sample isn't biased toward the very start of episodes.
    """
    samples: list[DecisionSample] = []
    episode_idx = 0

    while len(samples) < n_samples:
        env = FleetDispatchEnv(config)
        obs, infos = env.reset(seed=seed + episode_idx)
        episode_idx += 1

        rng = np.random.default_rng(seed + episode_idx)

        while env.agents and len(samples) < n_samples:
            actions = action_fn(env, obs)

            for aid, action_index in actions.items():
                if rng.random() > 0.3:
                    continue
                kind, param = decode_action(action_index, env.config)
                if kind not in (ActionKind.ACCEPT_OFFER, ActionKind.REJECT_AND_IDLE):
                    continue

                driver = env.get_driver_states()[aid]
                visible_ids = env._last_offer_slots.get(aid, [])
                # Require at least 2 visible offers: a decision with only one
                # option carries no real "was this the best choice among
                # alternatives" signal for a human rater to judge on soundness
                # -- the accepted/rejected payout would trivially equal the
                # mean of a single value, collapsing the soundness dimension
                # to a constant and producing spuriously unstable agreement
                # statistics (this was found and fixed during pilot testing --
                # see README.md's pilot study section).
                if len(visible_ids) < 2:
                    continue

                # Skip no-op ACCEPT_OFFER actions that point at an empty/padded
                # slot (param >= number of actually visible offers) -- these are
                # effectively "did nothing" from the environment's own handling
                # in FleetDispatchEnv._try_accept_offer, and carry no meaningful
                # fairness/soundness signal for a human rater to judge.
                if kind == ActionKind.ACCEPT_OFFER and param >= len(visible_ids):
                    continue

                visible_offers = []
                for oid in visible_ids:
                    order = env._orders.get(oid)
                    if order is None:
                        continue
                    visible_offers.append(
                        {
                            "pickup": list(order.pickup),
                            "payout": order.payout,
                            "distance": driver.distance_to(order.pickup),
                        }
                    )
                if not visible_offers:
                    continue

                mean_payout = float(np.mean([o["payout"] for o in visible_offers]))
                chosen_payout = None
                zone_id = None
                if kind == ActionKind.ACCEPT_OFFER and param < len(visible_ids):
                    order = env._orders.get(visible_ids[param])
                    if order is not None:
                        chosen_payout = order.payout
                        zone_id = order.zone_id
                elif kind == ActionKind.REJECT_AND_IDLE:
                    nearest_oid = min(
                        visible_ids,
                        key=lambda oid: driver.distance_to(env._orders[oid].pickup)
                        if oid in env._orders
                        else 1e9,
                    )
                    order = env._orders.get(nearest_oid)
                    if order is not None:
                        zone_id = order.zone_id

                samples.append(
                    DecisionSample(
                        sample_id=f"sample_{len(samples):04d}",
                        episode_seed=seed + episode_idx - 1,
                        step=env._step_count,
                        agent_id=aid,
                        agent_position=driver.position,
                        action_taken=kind.name,
                        visible_offers=visible_offers,
                        chosen_offer_payout=chosen_payout,
                        mean_visible_payout=mean_payout,
                        zone_completion_counts_before=env.get_zone_completion_counts().tolist(),
                        zone_id_of_action=zone_id,
                    )
                )
                if len(samples) >= n_samples:
                    break

            obs, rewards, terms, truncs, infos = env.step(actions)

    return samples


def save_samples(samples: list[DecisionSample], path: str) -> None:
    with open(path, "w") as f:
        json.dump([asdict(s) for s in samples], f, indent=2)


def load_samples(path: str) -> list[DecisionSample]:
    with open(path) as f:
        raw = json.load(f)
    return [DecisionSample(**item) for item in raw]
