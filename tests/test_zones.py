import pytest

from env.zones import build_zones, zone_of


def test_zones_fully_cover_the_grid_with_no_gaps_or_overlaps():
    grid_size = 10
    zones = build_zones(grid_size)
    covered = set()
    for r in range(grid_size):
        for c in range(grid_size):
            zid = zone_of(r, c, zones)
            assert (r, c) not in covered
            covered.add((r, c))
    assert len(covered) == grid_size * grid_size


def test_zone_of_boundary_cells():
    zones = build_zones(10)
    assert zone_of(0, 0, zones) == 0
    assert zone_of(4, 4, zones) == 0
    assert zone_of(0, 5, zones) == 1
    assert zone_of(5, 0, zones) == 2
    assert zone_of(5, 5, zones) == 3
    assert zone_of(9, 9, zones) == 3


def test_zone_of_raises_outside_grid():
    zones = build_zones(10)
    with pytest.raises(ValueError):
        zone_of(-1, 0, zones)
    with pytest.raises(ValueError):
        zone_of(10, 10, zones)
