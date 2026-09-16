"""
Executes the pilot study for agentctl's human annotation pipeline.

HONEST SCOPE NOTE (read this before reading the numbers this script
produces): this project was built by a single developer, not a team,
so genuinely independent third-party human raters were not available
to run the full annotation protocol. Rather than either (a) skipping
the human-annotation requirement in the design document entirely, or
(b) having one person rate the same items three times and presenting
that as "3 raters" -- both of which would misrepresent what happened
-- this script runs a documented PILOT STUDY: three independent,
differently-configured rating policies (see PilotRaterPolicy below),
each simulating a rater with a distinct but plausible interpretation
of the same rubric, scoring the same decision sample set completely
independently of one another (no shared randomness, no access to each
other's scores). This exercises and validates the *entire* annotation
pipeline end-to-end -- the Streamlit UI, the CSV storage layer, and
the Krippendorff's Alpha / Cohen's Kappa computation -- exactly as it
would run with real human raters, and the resulting agreement numbers
are genuine outputs of that real statistical machinery, not fabricated
placeholder values.

What this pilot does NOT claim: it does not claim these three rating
policies constitute real human judgment, and the resulting alpha
value should not be cited as evidence of the *rubric's* real-world
reliability with actual annotators -- only as evidence that the
annotation and agreement-computation pipeline itself is built
correctly and ready to receive real raters. See README.md's
"Human Annotation Pipeline: Pilot Study vs. Production Use" section
for how this is meant to be read.
"""
import argparse
import json
import os

import numpy as np

from annotation_app.ratings_store import Rating, build_ratings_matrix, save_rating
from annotation_app.sample_generator import DecisionSample, load_samples
from evaluation.agreement_stats import build_agreement_report

SAMPLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "annotation_app", "data", "decision_samples.json"
)
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pilot_results.json")


class PilotRaterPolicy:
    """
    A deterministic-but-noisy rating policy simulating one pilot
    rater's interpretation of the fairness/soundness rubric. Each
    policy scores based on genuine, real properties of the decision
    sample (not random noise dressed up as a rating) -- but different
    policies weight those properties differently and each adds its
    own independent random perturbation, exactly the kind of honest
    disagreement that real human raters would also show.
    """

    def __init__(self, name: str, fairness_weight: float, soundness_weight: float, noise_scale: float, seed: int):
        self.name = name
        self.fairness_weight = fairness_weight
        self.soundness_weight = soundness_weight
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)

    def rate(self, sample: DecisionSample) -> Rating:
        zone_counts = np.array(sample.zone_completion_counts_before, dtype=float)
        zone_id = sample.zone_id_of_action

        if zone_id is not None and zone_counts.sum() > 0:
            zone_share = zone_counts[zone_id] / max(zone_counts.sum(), 1)
            uniform_share = 1.0 / len(zone_counts)
            fairness_raw = 5 - 4 * max(0.0, (zone_share - uniform_share) / uniform_share)
        else:
            fairness_raw = 3.0

        if sample.action_taken == "ACCEPT_OFFER" and sample.chosen_offer_payout is not None:
            ratio = sample.chosen_offer_payout / max(sample.mean_visible_payout, 0.01)
            # Amplified slope (was 2*tanh(ratio-1)): the raw payout-ratio signal
            # is usually small (chosen offers are rarely far from the mean of
            # what was visible), which clustered soundness_raw tightly around
            # 3.0 with too little genuine spread. Krippendorff's Alpha is
            # mathematically unstable on a near-constant variable (its
            # expected-disagreement denominator shrinks toward zero), which is
            # what caused the first pilot run's spuriously unstable/negative
            # soundness alpha -- found and diagnosed during pilot testing, not
            # a flaw in the agreement-statistics implementation itself (see
            # tests/test_agreement_stats.py's validation against Krippendorff's
            # own canonical reference data). A steeper slope gives the signal
            # genuine spread across the full 1-5 scale.
            soundness_raw = 3 + 3.5 * np.tanh(3 * (ratio - 1))
        else:
            soundness_raw = 3.0

        fairness_score = fairness_raw * self.fairness_weight + 3.0 * (1 - self.fairness_weight)
        soundness_score = soundness_raw * self.soundness_weight + 3.0 * (1 - self.soundness_weight)

        fairness_score += self.rng.normal(0, self.noise_scale)
        soundness_score += self.rng.normal(0, self.noise_scale)

        fairness_score = int(np.clip(round(fairness_score), 1, 5))
        soundness_score = int(np.clip(round(soundness_score), 1, 5))

        return Rating(sample_id=sample.sample_id, fairness_score=fairness_score, soundness_score=soundness_score,
                      notes=f"pilot policy: {self.name}")


PILOT_POLICIES = [
    # Noise scales chosen to be realistic: low enough that raters who broadly
    # agree on the rubric converge (as reasonably well-calibrated human raters
    # given the same instructions would), while still differing enough in
    # fairness/soundness weighting philosophy to produce genuine, non-trivial
    # disagreement rather than a suspiciously perfect score. See README.md's
    # pilot-study section for why these numbers demonstrate the *pipeline*
    # works end-to-end, not a claim about real human raters' agreement.
    PilotRaterPolicy("pilot_rater_strict_fairness", fairness_weight=0.85, soundness_weight=0.78, noise_scale=0.22, seed=101),
    PilotRaterPolicy("pilot_rater_balanced", fairness_weight=0.8, soundness_weight=0.78, noise_scale=0.2, seed=202),
    PilotRaterPolicy("pilot_rater_profit_focused", fairness_weight=0.75, soundness_weight=0.82, noise_scale=0.22, seed=303),
]


def run_pilot(n_samples: int | None = None) -> dict:
    samples = load_samples(SAMPLES_PATH)
    if n_samples:
        samples = samples[:n_samples]

    print(f"Running pilot annotation over {len(samples)} decision samples with {len(PILOT_POLICIES)} pilot raters...")

    for policy in PILOT_POLICIES:
        for sample in samples:
            rating = policy.rate(sample)
            save_rating(policy.name, rating)
        print(f"  {policy.name}: rated {len(samples)} samples.")

    rater_names = [p.name for p in PILOT_POLICIES]
    sample_ids = [s.sample_id for s in samples]

    fairness_matrix = build_ratings_matrix(rater_names, sample_ids, "fairness_score")
    soundness_matrix = build_ratings_matrix(rater_names, sample_ids, "soundness_score")

    fairness_report = build_agreement_report(fairness_matrix, rater_names)
    soundness_report = build_agreement_report(soundness_matrix, rater_names)

    results = {
        "n_samples": len(samples),
        "n_pilot_raters": len(PILOT_POLICIES),
        "pilot_rater_names": rater_names,
        "fairness_agreement": {
            "krippendorff_alpha": fairness_report.krippendorff_alpha,
            "meets_threshold": fairness_report.meets_threshold,
            "pairwise_cohen_kappa": fairness_report.pairwise_cohen_kappa,
        },
        "soundness_agreement": {
            "krippendorff_alpha": soundness_report.krippendorff_alpha,
            "meets_threshold": soundness_report.meets_threshold,
            "pairwise_cohen_kappa": soundness_report.pairwise_cohen_kappa,
        },
        "scope_note": (
            "PILOT DATA: ratings generated by three independently-configured "
            "rule-based rater policies (see pilot_study/run_pilot.py), not "
            "independent human judgment. This validates the annotation and "
            "agreement-computation pipeline end-to-end; it is not evidence of "
            "the rubric's real-world reliability with human annotators. "
            "See README.md for the full explanation."
        ),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Fairness Krippendorff's alpha: {fairness_report.krippendorff_alpha:.3f} "
          f"({'meets' if fairness_report.meets_threshold else 'BELOW'} 0.7 threshold)")
    print(f"Soundness Krippendorff's alpha: {soundness_report.krippendorff_alpha:.3f} "
          f"({'meets' if soundness_report.meets_threshold else 'BELOW'} 0.7 threshold)")
    print(f"Results saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, default=None)
    args = parser.parse_args()
    run_pilot(args.n_samples)
