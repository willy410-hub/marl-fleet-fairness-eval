import pytest

from env.actions import ActionKind, action_space_size, decode_action
from env.config import EnvConfig


@pytest.fixture
def config():
    return EnvConfig()


def test_action_space_size_matches_component_counts(config):
    expected = config.max_open_offers_per_agent + 1 + config.n_zones + config.n_zones
    assert action_space_size(config) == expected


def test_every_action_index_decodes_without_error_and_uniquely(config):
    size = action_space_size(config)
    seen = set()
    for i in range(size):
        decoded = decode_action(i, config)
        assert decoded not in seen, f"action index {i} decoded to a duplicate of an earlier index"
        seen.add(decoded)
    assert len(seen) == size


def test_out_of_range_action_raises(config):
    with pytest.raises(ValueError):
        decode_action(action_space_size(config), config)


def test_accept_offer_slots_decode_correctly(config):
    for slot in range(config.max_open_offers_per_agent):
        kind, param = decode_action(slot, config)
        assert kind == ActionKind.ACCEPT_OFFER
        assert param == slot


def test_reject_action_decodes_correctly(config):
    reject_index = config.max_open_offers_per_agent
    kind, _ = decode_action(reject_index, config)
    assert kind == ActionKind.REJECT_AND_IDLE
