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
    subtitle=PROFILE["title"],
)

contact = PROFILE.get("contact", {})
contact_parts = []
if contact.get("phone"):
    contact_parts.append(contact["phone"])
if contact.get("email"):
    contact_parts.append(contact["email"])
if contact_parts:
    st.caption(" | ".join(contact_parts))

education_col, experience_col = st.columns([0.9, 1.1], gap="large")

with education_col:
    st.subheader("Education")
    for item in PROFILE["education"]:
        with st.container(border=True):
            st.markdown(f"**{item['program']}**")
            institution_line = item["institution"]
            if item.get("period"):
                institution_line = f"{institution_line} | {item['period']}"
            st.write(institution_line)
            for detail in item.get("details", []):
                st.write(f"- {detail}")

    certifications = PROFILE.get("certifications", [])
    if certifications:
        st.subheader("Certifications")
        with st.container(border=True):
            for item in certifications:
                st.write(f"- {item}")

with experience_col:
    st.subheader("Work Experience / Project Experience")
    for item in PROFILE["experience"]:
        with st.container(border=True):
            st.markdown(f"**{item['role']}**")
            organization_line = item["organization"]
            if item.get("period"):
                organization_line = f"{organization_line} | {item['period']}"
            st.write(organization_line)
            for detail in item["details"]:
                st.write(f"- {detail}")

st.subheader("Technical Skills")
render_skill_pills(PROFILE["skills"])
