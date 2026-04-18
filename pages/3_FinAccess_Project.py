from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app_utils import (
    FIGURE_DIR,
    build_prediction_story,
    configure_page,
    get_default_label,
    label_to_value,
    load_feature_importance,
    load_model_comparison,
    load_prediction_artifacts,
    load_top_excluded_counties,
    load_weighted_summary,
    predict_profile,
    render_fact_grid,
    render_hero,
    safe_read_bytes,
    value_to_label,
)


def select_from_context(context: dict[str, object], feature: str) -> object:
    options = context["display_options"][feature]
    labels = [option["label"] for option in options]
    default_label = get_default_label(context, feature)
    selected_label = st.selectbox(
        context["feature_labels"][feature],
        labels,
        index=labels.index(default_label),
        help=context["feature_help"][feature],
    )
    return label_to_value(context, feature, selected_label)


configure_page("Specific Project Page")

render_hero(
    title="Predicting Financial Access Profiles in Kenya",
    subtitle=(
        "Interactive capstone page for the FinAccess 2024 survey. The model predicts whether an "
        "adult respondent is most likely to be Excluded, Mobile money only, or Banked."
    ),
)

try:
    pipeline, context = load_prediction_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

comparison_df = load_model_comparison()
weighted_summary = load_weighted_summary()
feature_importance = load_feature_importance()
top_counties = load_top_excluded_counties()

gradient_row = comparison_df.loc[comparison_df["model"] == "Gradient Boosting"].iloc[0]
summary_lookup = weighted_summary.set_index("financial_access_profile")["weighted_percent"].to_dict()

metric_cols = st.columns(5)
metric_cols[0].metric("Model", "Gradient Boosting")
metric_cols[1].metric("Macro F1", f"{gradient_row['test_macro_f1']:.4f}")
metric_cols[2].metric("Balanced accuracy", f"{gradient_row['test_balanced_accuracy']:.4f}")
metric_cols[3].metric("Excluded recall", f"{gradient_row['test_excluded_recall']:.4f}")
metric_cols[4].metric("Adult sample size", f"{int(weighted_summary['sample_count'].sum()):,}")

st.markdown("### Project overview")
overview_left, overview_right = st.columns([1.1, 0.9], gap="large")

with overview_left:
    st.write(
        "This capstone uses the Kenya FinAccess 2024 survey to predict whether an adult respondent is most likely "
        "to be financially Excluded, Mobile money only, or Banked. The project focuses on prediction and "
        "interpretation, not causal claims."
    )
    st.write(
        "The current app is driven by a compact feature set selected from the master notebook, including county, "
        "sex, age, household size, education, marital status, household composition, livelihood, internet access, "
        "internet frequency, and financial health."
    )

with overview_right:
    render_fact_grid(
        [
            ("Dataset", "FinAccess 2024 public survey"),
            ("Problem type", "Multiclass classification"),
            ("Deployment", "Streamlit"),
            ("Current best model", "Gradient Boosting"),
        ],
        columns=2,
    )

left, right = st.columns([1.0, 1.0], gap="large")

with left:
    st.subheader("Try the model")
    st.write(
        "Use the form below to enter a survey-style respondent profile. The output is meant to explain the "
        "classification in plain language, not to give personal financial advice."
    )

    with st.form("prediction_form"):
        county = select_from_context(context, "county")
        sex = select_from_context(context, "sex")

        age_bounds = context["numeric_bounds"]["age_years"]
        age_years = st.slider(
            context["feature_labels"]["age_years"],
            min_value=age_bounds["min"],
            max_value=age_bounds["max"],
            value=context["defaults"]["age_years"],
            step=1,
            help=context["feature_help"]["age_years"],
        )

        household_bounds = context["numeric_bounds"]["household_size"]
        household_size = st.number_input(
            context["feature_labels"]["household_size"],
            min_value=household_bounds["min"],
            max_value=household_bounds["max"],
            value=context["defaults"]["household_size"],
            step=1,
            help=context["feature_help"]["household_size"],
        )

        education = select_from_context(context, "education")
        marital_status = select_from_context(context, "marital_status")

        children_bounds = context["numeric_bounds"]["children_in_household"]
        children_in_household = st.number_input(
            context["feature_labels"]["children_in_household"],
            min_value=children_bounds["min"],
            max_value=children_bounds["max"],
            value=context["defaults"]["children_in_household"],
            step=1,
            help=context["feature_help"]["children_in_household"],
        )

        livelihood = select_from_context(context, "livelihood")
        can_access_internet = select_from_context(context, "can_access_internet")
        internet_frequency = select_from_context(context, "internet_frequency")
        financial_health = select_from_context(context, "financial_health")

        submitted = st.form_submit_button("Predict financial access profile", use_container_width=True)

    if submitted:
        form_data = {
            "county": county,
            "sex": sex,
            "age_years": age_years,
            "household_size": household_size,
            "education": education,
            "marital_status": marital_status,
            "children_in_household": children_in_household,
            "livelihood": livelihood,
            "can_access_internet": can_access_internet,
            "internet_frequency": internet_frequency,
            "financial_health": financial_health,
        }

        try:
            prediction, probabilities = predict_profile(pipeline, form_data)
        except RuntimeError as exc:
            st.error(str(exc))
            st.stop()

        story = build_prediction_story(context, form_data, prediction, probabilities, top_counties)

        st.success(f"Predicted profile: {prediction}")
        st.write(story)

        probability_frame = probabilities.rename_axis("profile").reset_index()
        probability_frame.columns = ["profile", "probability"]
        st.bar_chart(probability_frame.set_index("profile"))

        st.caption(
            f"County selected: {value_to_label(context, 'county', county)} | "
            f"Education selected: {value_to_label(context, 'education', education)}"
        )

with right:
    st.subheader("Quick context from the weighted EDA")
    render_fact_grid(
        [
            ("Excluded", f"{summary_lookup.get('Excluded', 0.0):.2f}%"),
            ("Mobile money only", f"{summary_lookup.get('Mobile money only', 0.0):.2f}%"),
            ("Banked", f"{summary_lookup.get('Banked', 0.0):.2f}%"),
        ],
        columns=1,
    )
    st.caption("These weighted shares describe the adult survey sample, not an individual prediction.")

    chart_path = FIGURE_DIR / "weighted_target_distribution_adults.png"
    if chart_path.exists():
        st.image(str(chart_path), caption="Adult weighted target distribution", use_container_width=True)

tabs = st.tabs(["Model interpretation", "Model comparison", "County view", "Downloads and limitations"])

with tabs[0]:
    st.write(
        "The chart below summarizes the strongest features in the final notebook workflow. Age, education, "
        "internet frequency, and county are the largest signals in the current model."
    )
    importance_path = FIGURE_DIR / "adult_feature_importance.png"
    if importance_path.exists():
        st.image(str(importance_path), caption="Top feature importance", use_container_width=True)
    st.dataframe(feature_importance, use_container_width=True, hide_index=True)

with tabs[1]:
    st.write(
        "The current demo uses Gradient Boosting because it provided the strongest overall balance in the final "
        "notebook comparison, though excluded-group recall remains the hardest part of the task."
    )
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    comparison_chart = pd.DataFrame(
        {
            "model": comparison_df["model"],
            "macro_f1": comparison_df["test_macro_f1"],
            "balanced_accuracy": comparison_df["test_balanced_accuracy"],
        }
    ).set_index("model")
    st.bar_chart(comparison_chart)

with tabs[2]:
    county_path = FIGURE_DIR / "weighted_top_excluded_counties_adults.png"
    if county_path.exists():
        st.image(str(county_path), caption="Top counties by weighted exclusion share", use_container_width=True)
    st.dataframe(top_counties, use_container_width=True, hide_index=True)

with tabs[3]:
    proposal_path = ROOT / "Mutiga Proposal revised after feedback.pdf"
    proposal_bytes = safe_read_bytes(proposal_path)
    notebook_path = ROOT / "FINACCESS_MASTER_PROJECT_NOTEBOOK.ipynb"
    notebook_bytes = safe_read_bytes(notebook_path)

    download_cols = st.columns(2)
    with download_cols[0]:
        if proposal_bytes is not None:
            st.download_button(
                "Download revised proposal",
                data=proposal_bytes,
                file_name=proposal_path.name,
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning("Planning document is not bundled in this deployment yet.")

    with download_cols[1]:
        if notebook_bytes is not None:
            st.download_button(
                "Download master notebook",
                data=notebook_bytes,
                file_name=notebook_path.name,
                mime="application/x-ipynb+json",
                use_container_width=True,
            )
        else:
            st.warning("Capstone notebook is not bundled in this deployment yet.")

    st.write(
        "This app reflects patterns in one public survey wave. It is designed for explanation and interaction, "
        "not causal inference, institutional decision-making, or personal financial advice."
    )
    st.write(
        "The classes are intentionally simplified: the Banked group can include respondents who use both banks "
        "and mobile money, and the app predicts current access status rather than long-term financial outcomes."
    )
