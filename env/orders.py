"""
Order generation and lifecycle.

Orders spawn stochastically (Poisson process) at random locations,
weighted toward the current hour's demand profile, and expire if no
agent accepts them within order_expiry_steps.
"""
from dataclasses import dataclass, field

import numpy as np

from env.config import EnvConfig
from env.zones import Zone, zone_of


@dataclass
class Order:
    """A single delivery order, from spawn to completion or expiry."""

    order_id: int
    pickup: tuple[int, int]
    dropoff: tuple[int, int]
    spawn_step: int
    zone_id: int
    payout: float
    assigned_agent: str | None = None
    picked_up: bool = False
    accepted_step: int | None = None

    @property
    def distance(self) -> int:
        """Chebyshev distance between pickup and dropoff (grid-move distance)."""
        return max(abs(self.pickup[0] - self.dropoff[0]), abs(self.pickup[1] - self.dropoff[1]))

    def is_expired(self, current_step: int, expiry_steps: int) -> bool:
        return self.assigned_agent is None and (current_step - self.spawn_step) >= expiry_steps


class OrderGenerator:
    """Stateful generator that spawns new orders each step according to the demand profile."""

    def __init__(self, config: EnvConfig, zones: list[Zone], rng: np.random.Generator):
        self.config = config
        self.zones = zones
        self.rng = rng
        self._next_order_id = 0

    def hourly_multiplier(self, step: int) -> float:
        """Return the demand multiplier for the hour-of-day implied by `step`."""
        hour = (step // self.config.steps_per_hour) % 24
        return self.config.demand_profile[hour]

    def spawn_orders(self, step: int) -> list[Order]:
        """Spawn zero or more new orders for this step, via a Poisson draw."""
        lam = self.config.order_spawn_rate * self.hourly_multiplier(step)
        n_new = self.rng.poisson(lam)

        orders = []
        for _ in range(n_new):
            pickup = self._random_cell()
            dropoff = self._random_cell()
            while dropoff == pickup:
                dropoff = self._random_cell()

            zid = zone_of(pickup[0], pickup[1], self.zones)
            distance = max(abs(pickup[0] - dropoff[0]), abs(pickup[1] - dropoff[1]))
            payout = (
                self.config.base_payout_per_order
                + self.config.payout_distance_multiplier * distance
            )

            orders.append(
                Order(
                    order_id=self._next_order_id,
                    pickup=pickup,
                    dropoff=dropoff,
                    spawn_step=step,
                    zone_id=zid,
                    payout=round(payout, 2),
                )
            )
            self._next_order_id += 1

        return orders

    def _random_cell(self) -> tuple[int, int]:
        return (
            int(self.rng.integers(0, self.config.grid_size)),
            int(self.rng.integers(0, self.config.grid_size)),
        )
