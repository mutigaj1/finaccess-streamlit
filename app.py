from __future__ import annotations

import streamlit as st

from app_utils import (
    configure_page,
    load_model_comparison,
    load_weighted_summary,
    render_fact_grid,
    render_hero,
)


configure_page("FinAccess Story")

comparison_df = load_model_comparison()
weighted_summary = load_weighted_summary()

best_row = comparison_df.sort_values(
    by=["test_macro_f1", "test_balanced_accuracy"],
    ascending=False,
).iloc[0]
summary_lookup = weighted_summary.set_index("financial_access_profile")["weighted_percent"].to_dict()

render_hero(
    title="Predicting Financial Access Profiles in Kenya Using FinAccess 2024",
    subtitle=(
        "A DTSC 691 capstone that uses the FinAccess 2024 public survey to predict whether an adult respondent is "
        "most likely to be Excluded, Mobile money only, or Banked."
    ),
)

st.caption(
    "This page tells the full project story. When you are ready to try the live model, open the FinAccess Project page."
)
st.page_link("pages/3_FinAccess_Project.py", label="Open the FinAccess Project page")

st.markdown("### 1. What this project is")
st.write(
    "This project is a machine learning capstone built around a simple goal: use the FinAccess 2024 survey to "
    "estimate which financial access group a person is most likely to belong to. Instead of leaving the work inside "
    "a notebook, the project turns the analysis into a clear Streamlit app that explains the problem, shows the "
    "results, and lets a visitor try the model interactively."
)
render_fact_grid(
    [
        ("Dataset", "FinAccess 2024 public survey for Kenya"),
        ("Responses", "20,871"),
        ("Fields in the full public dataset", "3,816"),
        ("Counties covered", "47"),
    ],
    columns=4,
)

st.markdown("### 2. The problem")
st.write(
    "The project asks a practical question: based on a small set of interpretable respondent characteristics, can we "
    "estimate the financial access profile a person is most likely to fall into? The aim is not to label people for a "
    "real-world decision system. The aim is to use survey data to understand patterns in financial access and to "
    "communicate those patterns clearly."
)

st.markdown("### 3. Why this matters in Kenya")
st.write(
    "Kenya is often discussed as a strong mobile money success story, but access to formal financial services is still "
    "not evenly distributed. County context, education, household conditions, and digital access can all shape how "
    "people participate in the financial system. A tool like this helps turn a large national survey into a clearer "
    "story about inclusion, exclusion, and uneven access."
)

st.markdown("### 4. The data")
st.write(
    "This capstone uses the FinAccess 2024 public survey for Kenya. The public dataset contains 20,871 responses and "
    "3,816 fields. For the core modeling workflow, the project focuses on adult respondents ages 18 and above and "
    "keeps the interactive app aligned to a compact 11-feature workflow that is easier to explain and deploy."
)
st.write(
    "The project combines weighted exploratory data analysis with a streamlined modeling workflow so the final app can "
    "connect the notebook results to a public-facing interface."
)

st.markdown("### 5. The target and features")
target_col, feature_col = st.columns([0.8, 1.2], gap="large")

with target_col:
    st.markdown("**Target classes**")
    st.write("- Excluded")
    st.write("- Mobile money only")
    st.write("- Banked")
    st.caption(
        "These classes describe current financial access status in the survey. They are simplified categories for the "
        "purposes of the capstone."
    )

with feature_col:
    st.markdown("**11-feature compact workflow**")
    st.write(
        "county, sex, age, household size, education, marital status, number of children in household, livelihood, "
        "internet access, internet frequency, and financial health."
    )
    st.caption(
        "The app intentionally avoids target-defining variables such as current mobile money use and current bank use "
        "as predictors."
    )

st.markdown("### 6. The modeling approach")
st.write(
    "At a high level, the workflow uses weighted EDA, preprocessing, model comparison, tuning, evaluation, "
    "interpretation, and deployment. Missing values coded as survey placeholders are handled during preprocessing, "
    "numeric fields are imputed with medians, categorical fields are imputed with most-frequent values, and multiple "
    "classification models are compared before selecting the strongest current workflow for the app."
)
st.write(
    "The project is built around prediction and interpretation. It is not designed to make causal claims about why a "
    "person has a given financial access outcome."
)

st.markdown("### 7. What the analysis showed")
metric_cols = st.columns(5)
metric_cols[0].metric("Current best model", str(best_row["model"]))
metric_cols[1].metric("Macro F1", f"{best_row['test_macro_f1']:.4f}")
metric_cols[2].metric("Balanced accuracy", f"{best_row['test_balanced_accuracy']:.4f}")
metric_cols[3].metric("Excluded recall", f"{best_row['test_excluded_recall']:.4f}")
metric_cols[4].metric("Adult sample size", f"{int(weighted_summary['sample_count'].sum()):,}")

st.write(
    f"In the current saved workflow, **{best_row['model']}** is the strongest model overall. The results suggest the "
    "problem is learnable, but not trivial. Some classes are much easier to identify than others, which is why the "
    "project emphasizes metrics such as macro F1, balanced accuracy, class-level recall, subgroup summaries, and "
    "feature importance rather than overall accuracy alone."
)
render_fact_grid(
    [
        ("Excluded share", f"{summary_lookup.get('Excluded', 0.0):.2f}%"),
        ("Mobile money only share", f"{summary_lookup.get('Mobile money only', 0.0):.2f}%"),
        ("Banked share", f"{summary_lookup.get('Banked', 0.0):.2f}%"),
    ],
    columns=3,
)
st.write(
    "Predictions from this project should be read as pattern-based estimates from the survey. They are useful for "
    "explanation and interpretation, not as proof of cause and effect."
)

st.markdown("### 8. What the app does")
st.write(
    "The multipage app turns the notebook workflow into something a visitor can actually use. This story page explains "
    "the capstone. The Resume page stays focused on background and qualifications. The General Projects page holds "
    "supporting assets. The FinAccess Project page is the actual prediction tool."
)

st.markdown("### 9. How to use the FinAccess Project page")
st.write("1. Open the **FinAccess Project** page.")
st.write("2. Enter a respondent profile using the survey-style inputs.")
st.write("3. Run the prediction to estimate the most likely class: Excluded, Mobile money only, or Banked.")
st.write("4. Read the plain-language interpretation and probability chart.")
st.write("5. Use the tabs to review model interpretation, model comparison, county context, and downloads.")

st.markdown("### 10. Open the project tool")
st.page_link("pages/3_FinAccess_Project.py", label="Go to FinAccess Project")
