"""
Driver-agent state: position, active orders, and earnings.

Each DriverState is the ground-truth state of one agent. What that
agent actually *observes* (partial observability) is computed
separately in env/observations.py -- this module only tracks the true
underlying state, not what any agent is allowed to see.
"""
from dataclasses import dataclass, field


@dataclass
class DriverState:
    """Ground-truth state of one driver-agent."""

    agent_id: str
    position: tuple[int, int]
    active_order_ids: list[int] = field(default_factory=list)
    total_earnings: float = 0.0
    completed_orders: int = 0
    rejected_offers: int = 0
    steps_idle: int = 0

    @property
    def is_available(self) -> bool:
        return len(self.active_order_ids) == 0

    def distance_to(self, cell: tuple[int, int]) -> int:
        """Chebyshev distance from this driver's current position to `cell`."""
        return max(abs(self.position[0] - cell[0]), abs(self.position[1] - cell[1]))
