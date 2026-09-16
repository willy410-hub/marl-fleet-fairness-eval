import numpy as np
import pytest

from annotation_app.ratings_store import Rating, build_ratings_matrix, load_ratings, save_rating
from annotation_app.sample_generator import collect_decision_samples, load_samples, save_samples
from env.config import EnvConfig
from evaluation.episode_runner import random_action_fn


@pytest.fixture
def isolated_ratings_dir(monkeypatch, tmp_path):
    """Redirect ratings storage to a temp dir so tests never touch real annotation data."""
    import annotation_app.ratings_store as store

    monkeypatch.setattr(store, "RATINGS_DIR", str(tmp_path / "ratings"))
    yield tmp_path


def test_collect_decision_samples_returns_requested_count():
    config = EnvConfig(seed=1, steps_per_episode=60)
    samples = collect_decision_samples(config, random_action_fn, n_samples=10, seed=42)
    assert len(samples) == 10


def test_accept_offer_samples_always_have_a_real_chosen_payout():
    """
    Regression test for a real bug found during development: a
    no-op ACCEPT_OFFER pointing at an empty/padded offer slot was
    being included in the sample set with chosen_offer_payout=None,
    which is meaningless for a human rater to judge.
    """
    config = EnvConfig(seed=1, steps_per_episode=100)
    samples = collect_decision_samples(config, random_action_fn, n_samples=20, seed=777)

    accept_samples = [s for s in samples if s.action_taken == "ACCEPT_OFFER"]
    assert len(accept_samples) > 0, "test needs at least one ACCEPT_OFFER sample to be meaningful"
    for s in accept_samples:
        assert s.chosen_offer_payout is not None


def test_every_sample_has_a_real_choice_between_at_least_two_offers():
    """
    Regression test for a bug found during the pilot study: samples
    with only one visible offer gave the soundness dimension no real
    signal (chosen/rejected payout trivially equals the mean of a
    single value), which produced unstable, spuriously-low or even
    negative inter-rater agreement purely from rounding noise on a
    near-constant variable. Every collected sample must now offer a
    genuine choice among at least two visible offers.
    """
    config = EnvConfig(seed=1, steps_per_episode=150)
    samples = collect_decision_samples(config, random_action_fn, n_samples=20, seed=777)

    for s in samples:
        assert len(s.visible_offers) >= 2, f"{s.sample_id} had only {len(s.visible_offers)} visible offer(s)"


def test_sample_save_and_load_round_trip(tmp_path):
    config = EnvConfig(seed=1, steps_per_episode=60)
    samples = collect_decision_samples(config, random_action_fn, n_samples=5, seed=1)

    path = str(tmp_path / "samples.json")
    save_samples(samples, path)
    loaded = load_samples(path)

    assert len(loaded) == len(samples)
    assert loaded[0].sample_id == samples[0].sample_id
    assert loaded[0].visible_offers == samples[0].visible_offers


def test_rating_upsert_does_not_duplicate(isolated_ratings_dir):
    save_rating("test_rater", Rating(sample_id="s1", fairness_score=3, soundness_score=3))
    save_rating("test_rater", Rating(sample_id="s1", fairness_score=5, soundness_score=5))

    ratings = load_ratings("test_rater")
    assert len(ratings) == 1
    assert ratings["s1"].fairness_score == 5


def test_ratings_matrix_has_nan_for_unrated_items(isolated_ratings_dir):
    save_rating("rater_a", Rating(sample_id="s1", fairness_score=3, soundness_score=3))
    save_rating("rater_b", Rating(sample_id="s1", fairness_score=4, soundness_score=4))
    save_rating("rater_b", Rating(sample_id="s2", fairness_score=2, soundness_score=2))

    matrix = build_ratings_matrix(["rater_a", "rater_b"], ["s1", "s2"], "fairness_score")
    assert matrix[0, 0] == 3
    assert matrix[1, 0] == 4
    assert matrix[1, 1] == 2
    assert np.isnan(matrix[0, 1])
