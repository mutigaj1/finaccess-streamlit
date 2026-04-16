from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app_utils import configure_page, render_hero, safe_read_bytes
from profile_content import PROFILE


configure_page("General Projects")

render_hero(
    title="General Projects",
    subtitle="Course portfolio page highlighting the main DTSC 691 project artifacts and supporting materials.",
)

st.info(PROFILE["general_projects_note"])

for project in PROFILE["general_projects"]:
    with st.container(border=True):
        st.markdown(f"### {project['title']}")
        st.write(project["summary"])
        st.caption(project["status"])

        if project.get("page_path"):
            st.page_link(project["page_path"], label="Open in this app")

        if project.get("download_path"):
            file_path = ROOT / project["download_path"]
            file_bytes = safe_read_bytes(file_path)
            if file_bytes is not None:
                st.download_button(
                    label="Download local file",
                    data=file_bytes,
                    file_name=file_path.name,
                    mime="application/octet-stream",
                    key=f"download-{file_path.name}",
                )

        if not project.get("page_path") and not project.get("download_path"):
            st.write("Add a private GitHub, Google Drive, or portfolio link here when you are ready.")
