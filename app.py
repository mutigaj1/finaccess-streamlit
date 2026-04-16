from __future__ import annotations

from pathlib import Path

import streamlit as st

from app_utils import (
    FIGURE_DIR,
    configure_page,
    load_weighted_summary,
    render_hero,
    render_section_card,
    safe_read_bytes,
)
from profile_content import PROFILE


ROOT = Path(__file__).resolve().parent

configure_page("FinAccess Home")

summary = load_weighted_summary()
metrics = {row["financial_access_profile"]: row["weighted_percent"] for _, row in summary.iterrows()}

render_hero(
    title="Who Gets Left Out?",
    subtitle=(
        "A DTSC 691 Streamlit capstone on financial inclusion in Kenya, built from the "
        "FinAccess 2024 public survey."
    ),
)

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader(f"{PROFILE['name']} | {PROFILE['title']}")
    st.caption(PROFILE["location"])
    for paragraph in PROFILE["home_intro"]:
        st.write(paragraph)

    render_section_card(
        "Project objective",
        "This app explores a three-class prediction problem: Excluded, Mobile money only, and Banked. "
        "The current workflow uses adult respondents, a compact feature set, weighted exploratory analysis, "
        "and a Gradient Boosting classifier for the interactive demo.",
    )

    render_section_card(
        "What the app includes",
        "A project home page, a resume page, a general-projects page, and a specific project page with live "
        "prediction, supporting charts, model metrics, and plain-language interpretation.",
    )

    st.markdown("### Navigate the app")
    st.page_link("app.py", label="Home")
    st.page_link("pages/1_Resume.py", label="Resume")
    st.page_link("pages/2_General_Projects.py", label="General Projects")
    st.page_link("pages/3_FinAccess_Project.py", label="Specific Project Page")

with right:
    image_path = FIGURE_DIR / "weighted_target_distribution_adults.png"
    if image_path.exists():
        st.image(
            str(image_path),
            caption="Weighted target distribution for the adult analytic sample",
            use_container_width=True,
        )

metric_cols = st.columns(3)
metric_cols[0].metric("Excluded", f"{metrics.get('Excluded', 0):.2f}%")
metric_cols[1].metric("Mobile money only", f"{metrics.get('Mobile money only', 0):.2f}%")
metric_cols[2].metric("Banked", f"{metrics.get('Banked', 0):.2f}%")

st.markdown("### Project snapshot")
snapshot_cols = st.columns(4)
snapshot_cols[0].metric("Survey responses", "20,871")
snapshot_cols[1].metric("Counties", "47")
snapshot_cols[2].metric("Target classes", "3")
snapshot_cols[3].metric("Selected features", "11")

st.markdown("### Why this project matters")
st.write(
    "Kenya is well known for mobile money adoption, but that does not mean financial access looks the same across "
    "all counties or social groups. This capstone turns survey patterns into an interactive tool that supports "
    "clear explanation, public-interest storytelling, and a more practical final presentation."
)

st.markdown("### Key project files")
download_cols = st.columns(2)
proposal_path = ROOT / "Mutiga Proposal revised after feedback.pdf"
proposal_bytes = safe_read_bytes(proposal_path)
if proposal_bytes is not None:
    download_cols[0].download_button(
        "Download revised proposal",
        data=proposal_bytes,
        file_name=proposal_path.name,
        mime="application/pdf",
        use_container_width=True,
    )

notebook_path = ROOT / "FINACCESS_MASTER_PROJECT_NOTEBOOK.ipynb"
notebook_bytes = safe_read_bytes(notebook_path)
if notebook_bytes is not None:
    download_cols[1].download_button(
        "Download master notebook",
        data=notebook_bytes,
        file_name=notebook_path.name,
        mime="application/x-ipynb+json",
        use_container_width=True,
    )
