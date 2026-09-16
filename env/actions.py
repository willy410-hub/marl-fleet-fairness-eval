"""
Action space definition.

Per the design document: "accept/reject an order offer, or choose a
zone to idle in." Implemented as a single discrete action space per
agent so it plugs directly into PettingZoo's Discrete space and
RLlib's default action distributions without custom policy code.
"""
from enum import IntEnum

from env.config import EnvConfig


class ActionKind(IntEnum):
    """The semantic category of a chosen discrete action index."""

    ACCEPT_OFFER = 0
    REJECT_AND_IDLE = 1
    MOVE_TO_ZONE = 2
    MOVE_TO_ZONE_SHORTCUT = 3  # the "unsafe shortcut" risk-taking variant


def action_space_size(config: EnvConfig) -> int:
    """
    Total number of discrete actions:
      - one ACCEPT action per visible offer slot
      - one REJECT_AND_IDLE action
      - one MOVE_TO_ZONE action per zone
      - one MOVE_TO_ZONE_SHORTCUT (risky) action per zone
    """
    accept_actions = config.max_open_offers_per_agent
    reject_action = 1
    move_actions = config.n_zones
    shortcut_actions = config.n_zones
    return accept_actions + reject_action + move_actions + shortcut_actions


def decode_action(action_index: int, config: EnvConfig) -> tuple[ActionKind, int]:
    """
    Decode a flat action index into (ActionKind, parameter).

    For ACCEPT_OFFER, parameter is the offer slot index (0-based, into
    the agent's own visible-offer list, matching the order used in
    env/observations.py's build_observation). For MOVE_TO_ZONE and
    MOVE_TO_ZONE_SHORTCUT, parameter is the target zone_id. For
    REJECT_AND_IDLE, parameter is unused (0).
    """
    n_accept = config.max_open_offers_per_agent
    if action_index < n_accept:
        return ActionKind.ACCEPT_OFFER, action_index

    action_index -= n_accept
    if action_index == 0:
        return ActionKind.REJECT_AND_IDLE, 0
    action_index -= 1

    if action_index < config.n_zones:
        return ActionKind.MOVE_TO_ZONE, action_index
    action_index -= config.n_zones

    if action_index < config.n_zones:
        return ActionKind.MOVE_TO_ZONE_SHORTCUT, action_index

    raise ValueError(f"Action index out of range for action_space_size={action_space_size(config)}.")
