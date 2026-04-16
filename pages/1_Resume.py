from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app_utils import FIGURE_DIR, configure_page, render_hero, render_skill_pills
from profile_content import PROFILE


configure_page("Resume")

render_hero(
    title="Resume",
    subtitle="Course-required profile page with education, experience, technical skills, and a supporting visual.",
)

highlight_cols = st.columns(3)
highlight_cols[0].metric("Primary focus", "Applied ML")
highlight_cols[1].metric("Project area", "Financial inclusion")
highlight_cols[2].metric("Deployment", "Streamlit")

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("Education")
    for item in PROFILE["education"]:
        with st.container(border=True):
            st.markdown(f"**{item['program']}**")
            st.write(f"{item['institution']} | {item['period']}")
            for detail in item["details"]:
                st.write(f"- {detail}")

    st.subheader("Project Experience")
    for item in PROFILE["experience"]:
        with st.container(border=True):
            st.markdown(f"**{item['role']}**")
            st.write(f"{item['organization']} | {item['period']}")
            for detail in item["details"]:
                st.write(f"- {detail}")

    st.subheader("Skills")
    render_skill_pills(PROFILE["skills"])

with right:
    st.subheader("Visual")
    visual_path = FIGURE_DIR / "adult_feature_importance.png"
    if visual_path.exists():
        st.image(
            str(visual_path),
            caption="Model feature importance from the FinAccess capstone workflow",
            use_container_width=True,
        )

    with st.container(border=True):
        st.markdown("### Capstone highlights")
        st.write("- Built a three-class financial access prediction workflow")
        st.write("- Used weighted survey summaries and subgroup checks")
        st.write("- Compared multiple classifiers before selecting Gradient Boosting")
        st.write("- Deployed results in a course-aligned multipage Streamlit app")
