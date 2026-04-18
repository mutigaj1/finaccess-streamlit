from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app_utils import configure_page, render_hero, render_skill_pills
from profile_content import PROFILE


configure_page("Resume")

render_hero(
    title=PROFILE["name"],
    subtitle=f"{PROFILE['title']} | {PROFILE['location']}",
)

education_col, experience_col = st.columns([0.95, 1.05], gap="large")

with education_col:
    st.subheader("Education")
    for item in PROFILE["education"]:
        with st.container(border=True):
            st.markdown(f"**{item['program']}**")
            st.write(f"{item['institution']} | {item['period']}")
            for detail in item["details"]:
                st.write(f"- {detail}")

with experience_col:
    st.subheader("Work Experience / Project Experience")
    for item in PROFILE["experience"]:
        with st.container(border=True):
            st.markdown(f"**{item['role']}**")
            st.write(f"{item['organization']} | {item['period']}")
            for detail in item["details"]:
                st.write(f"- {detail}")

st.subheader("Technical Skills")
render_skill_pills(PROFILE["skills"])

certifications = PROFILE.get("certifications", [])
if certifications:
    st.subheader("Certifications")
    for item in certifications:
        with st.container(border=True):
            st.write(item)
