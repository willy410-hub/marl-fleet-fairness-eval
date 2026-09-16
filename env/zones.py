"""
Zone partitioning of the city grid.

The grid is divided into quadrant zones so that "coverage" and
"fairness" have a concrete, measurable meaning: how evenly are orders
being served across the four quadrants of the city, not just in
aggregate.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    """One rectangular region of the city grid."""

    zone_id: int
    row_start: int
    row_end: int  # exclusive
    col_start: int
    col_end: int  # exclusive

    def contains(self, row: int, col: int) -> bool:
        return self.row_start <= row < self.row_end and self.col_start <= col < self.col_end

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.row_start + self.row_end - 1) / 2,
            (self.col_start + self.col_end - 1) / 2,
        )


def build_zones(grid_size: int) -> list[Zone]:
    """
    Partition a grid_size x grid_size grid into 4 quadrant zones.

    Kept simple and deterministic (4 quadrants) so zone assignment is
    easy to verify by hand and easy to visualize -- the fairness
    metrics computed over these zones (see evaluation/fairness.py) are
    only meaningful if the zone boundaries are unambiguous.
    """
    mid = grid_size // 2
    return [
        Zone(zone_id=0, row_start=0, row_end=mid, col_start=0, col_end=mid),          # NW
        Zone(zone_id=1, row_start=0, row_end=mid, col_start=mid, col_end=grid_size),   # NE
        Zone(zone_id=2, row_start=mid, row_end=grid_size, col_start=0, col_end=mid),   # SW
        Zone(zone_id=3, row_start=mid, row_end=grid_size, col_start=mid, col_end=grid_size),  # SE
    ]


def zone_of(row: int, col: int, zones: list[Zone]) -> int:
    """Return the zone_id containing (row, col). Assumes zones partition the grid exactly."""
    for zone in zones:
        if zone.contains(row, col):
            return zone.zone_id
    raise ValueError(f"Cell ({row}, {col}) is not covered by any zone.")
