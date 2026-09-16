"""
Streamlit annotation tool: lets a human rater score a sample of agent
decisions for fairness and soundness, and view live inter-rater
agreement statistics (Krippendorff's Alpha + Cohen's Kappa) once
multiple raters have scored the same sample set.

Run with:
    streamlit run annotation_app/app.py
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotation_app.ratings_store import Rating, build_ratings_matrix, list_raters, load_ratings, save_rating
from annotation_app.sample_generator import DecisionSample, load_samples
from evaluation.agreement_stats import KRIPPENDORFF_ALPHA_THRESHOLD, build_agreement_report

SAMPLES_PATH = os.path.join(os.path.dirname(__file__), "data", "decision_samples.json")

st.set_page_config(page_title="Fleet Dispatch Decision Rater", layout="wide")


@st.cache_data
def _load_samples() -> list[DecisionSample]:
    return load_samples(SAMPLES_PATH)


def _action_kind_label(action: str) -> str:
    return {"ACCEPT_OFFER": "Accepted an offer", "REJECT_AND_IDLE": "Rejected and stayed idle"}.get(
        action, action
    )


def render_sample(sample: DecisionSample) -> None:
    st.subheader(f"Decision {sample.sample_id}  ·  Step {sample.step}  ·  Agent {sample.agent_id}")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Action taken:** {_action_kind_label(sample.action_taken)}")
        st.markdown(f"**Agent position:** {tuple(sample.agent_position)}")
        st.markdown(f"**Zone completion counts at this moment:** {sample.zone_completion_counts_before}")

        offers_df = pd.DataFrame(sample.visible_offers)
        offers_df.index = [f"Offer {i}" for i in range(len(offers_df))]
        st.markdown("**Offers visible to this agent at the moment of the decision:**")
        st.dataframe(offers_df, use_container_width=True)

        if sample.action_taken == "ACCEPT_OFFER":
            st.success(f"Agent accepted the offer paying **${sample.chosen_offer_payout:.2f}**.")
        else:
            st.warning(
                f"Agent rejected the nearest visible offer and stayed idle. "
                f"Mean payout of what was visible: **${sample.mean_visible_payout:.2f}**."
            )

    with col2:
        st.markdown("**Rating rubric**")
        st.caption(
            "Fairness: did this decision help or hurt even coverage across the city's zones? "
            "Soundness: was this a reasonable business decision given what the agent could see?"
        )


def main() -> None:
    st.title("Fleet Dispatch Decision Rater")
    st.caption(
        "Human annotation tool for agentctl's MARL fleet-fairness evaluation pipeline. "
        "Score a sample of agent decisions on fairness and soundness; agreement statistics "
        "update live as more raters complete the set."
    )

    samples = _load_samples()
    if not samples:
        st.error(f"No decision samples found at {SAMPLES_PATH}. Run annotation_app/generate_samples.py first.")
        return

    with st.sidebar:
        st.header("Rater identity")
        rater_name = st.text_input("Your name or ID", value="", placeholder="e.g. rater_1")
        if not rater_name:
            st.info("Enter a rater name to begin.")
            st.stop()

        existing_ratings = load_ratings(rater_name)
        progress = len(existing_ratings) / len(samples)
        st.progress(progress, text=f"{len(existing_ratings)} / {len(samples)} decisions rated")

        st.divider()
        st.header("Inter-rater agreement")
        all_raters = list_raters()
        if len(all_raters) >= 2:
            sample_ids = [s.sample_id for s in samples]
            fairness_matrix = build_ratings_matrix(all_raters, sample_ids, "fairness_score")
            soundness_matrix = build_ratings_matrix(all_raters, sample_ids, "soundness_score")

            n_overlap = np.sum(~np.isnan(fairness_matrix).all(axis=0))
            if n_overlap >= 2:
                fairness_report = build_agreement_report(fairness_matrix, all_raters)
                soundness_report = build_agreement_report(soundness_matrix, all_raters)

                st.metric(
                    "Fairness Krippendorff's α",
                    f"{fairness_report.krippendorff_alpha:.3f}",
                    delta="meets 0.7 threshold" if fairness_report.meets_threshold else "below 0.7 threshold",
                    delta_color="normal" if fairness_report.meets_threshold else "inverse",
                )
                st.metric(
                    "Soundness Krippendorff's α",
                    f"{soundness_report.krippendorff_alpha:.3f}",
                    delta="meets 0.7 threshold" if soundness_report.meets_threshold else "below 0.7 threshold",
                    delta_color="normal" if soundness_report.meets_threshold else "inverse",
                )
                st.caption(f"Computed across {len(all_raters)} raters, {n_overlap} overlapping items.")

                with st.expander("Pairwise Cohen's Kappa (fairness)"):
                    for pair, kappa in fairness_report.pairwise_cohen_kappa.items():
                        st.write(f"{pair}: {kappa:.3f}" if kappa is not None else f"{pair}: insufficient overlap")
            else:
                st.info("Need at least 2 raters to have scored overlapping items to compute agreement.")
        else:
            st.info(f"{len(all_raters)} rater(s) so far -- need at least 2 for agreement statistics.")

    unrated = [s for s in samples if s.sample_id not in existing_ratings]
    if not unrated:
        st.success("You've rated every decision in this sample set. Thank you!")
        st.dataframe(pd.DataFrame([vars(r) for r in existing_ratings.values()]), use_container_width=True)
        return

    current = unrated[0]
    render_sample(current)

    with st.form(key=f"rating_form_{current.sample_id}"):
        fairness = st.slider(
            "Fairness (1 = clearly worsened zone coverage, 5 = clearly helped fair coverage)", 1, 5, 3
        )
        soundness = st.slider(
            "Soundness (1 = clearly bad business decision, 5 = clearly sound decision)", 1, 5, 3
        )
        notes = st.text_area("Notes (optional)", "")
        submitted = st.form_submit_button("Submit rating and continue")

        if submitted:
            save_rating(
                rater_name,
                Rating(
                    sample_id=current.sample_id,
                    fairness_score=fairness,
                    soundness_score=soundness,
                    notes=notes,
                ),
            )
            st.rerun()


if __name__ == "__main__":
    main()
