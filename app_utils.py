from __future__ import annotations

from html import escape
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ROOT / "figures"
PIPELINE_PATH = ARTIFACT_DIR / "finaccess_streamlit_pipeline.joblib"
CONTEXT_PATH = ARTIFACT_DIR / "finaccess_streamlit_context.joblib"

TOP_COUNTIES_PATH = ARTIFACT_DIR / "weighted_top_excluded_counties_adults.csv"
MODEL_COMPARISON_PATH = ARTIFACT_DIR / "final_model_comparison.csv"
FEATURE_IMPORTANCE_PATH = ARTIFACT_DIR / "final_model_feature_importance.csv"
WEIGHTED_SUMMARY_PATH = ARTIFACT_DIR / "weighted_target_summary_adults.csv"


THEME_CSS = """
<style>
    #MainMenu,
    footer,
    .stDeployButton,
    [data-testid="stStatusWidget"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        display: none !important;
    }
    header[data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(232, 183, 98, 0.18), transparent 28%),
            linear-gradient(180deg, #f8f3e8 0%, #f4ecdd 52%, #efe6d6 100%);
        color: #203238;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(248, 243, 232, 0.98), rgba(244, 236, 221, 0.98));
        border-right: 1px solid rgba(27, 75, 63, 0.12);
    }
    [data-testid="stSidebarNav"] {
        padding-top: 1.1rem;
    }
    h1, h2, h3 {
        font-family: "Palatino Linotype", "Book Antiqua", Georgia, serif;
        color: #163a33;
        letter-spacing: -0.02em;
    }
    p, li, div, label, [data-testid="stMarkdownContainer"] {
        font-family: "Trebuchet MS", "Gill Sans", "Segoe UI", sans-serif;
    }
    .hero-card {
        background: linear-gradient(135deg, rgba(22, 58, 51, 0.97), rgba(39, 102, 88, 0.92));
        border: 1px solid rgba(22, 58, 51, 0.18);
        border-radius: 24px;
        box-shadow: 0 20px 45px rgba(22, 58, 51, 0.14);
        color: #f8f3e8;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1rem;
    }
    .hero-card p {
        color: rgba(248, 243, 232, 0.9);
        margin-bottom: 0;
    }
    .section-card {
        background: rgba(255, 252, 246, 0.9);
        border: 1px solid rgba(27, 75, 63, 0.12);
        border-radius: 20px;
        box-shadow: 0 14px 30px rgba(22, 58, 51, 0.07);
        padding: 1.1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .pill {
        display: inline-block;
        margin: 0 0.4rem 0.45rem 0;
        padding: 0.32rem 0.72rem;
        border-radius: 999px;
        background: rgba(198, 92, 43, 0.12);
        color: #8d431c;
        font-size: 0.88rem;
        font-weight: 600;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 252, 246, 0.92);
        border: 1px solid rgba(27, 75, 63, 0.12);
        border-radius: 18px;
        padding: 0.8rem 0.95rem;
        box-shadow: 0 10px 25px rgba(22, 58, 51, 0.06);
    }
    [data-testid="stMetricValue"] {
        color: #163a33;
    }
    .fact-grid {
        display: grid;
        grid-template-columns: repeat(var(--fact-cols), minmax(0, 1fr));
        gap: 0.8rem;
        margin: 0.2rem 0 0.8rem;
    }
    .fact-card {
        background: rgba(255, 252, 246, 0.94);
        border: 1px solid rgba(27, 75, 63, 0.12);
        border-radius: 18px;
        box-shadow: 0 10px 25px rgba(22, 58, 51, 0.06);
        padding: 0.9rem 1rem;
    }
    .fact-label {
        color: #5e736f;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.22rem;
    }
    .fact-value {
        color: #163a33;
        font-size: 1.02rem;
        font-weight: 700;
        line-height: 1.35;
    }
</style>
"""


def configure_page(title: str) -> None:
    st.set_page_config(page_title=title, layout="wide")
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def render_hero(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
            <h3>{title}</h3>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_skill_pills(items: list[str]) -> None:
    html = "".join(f'<span class="pill">{item}</span>' for item in items)
    st.markdown(html, unsafe_allow_html=True)


def render_fact_grid(items: list[tuple[str, str]], columns: int = 2) -> None:
    safe_columns = max(1, columns)
    cards = "".join(
        (
            '<div class="fact-card">'
            f'<div class="fact-label">{escape(label)}</div>'
            f'<div class="fact-value">{escape(value)}</div>'
            '</div>'
        )
        for label, value in items
    )
    st.markdown(
        f'<div class="fact-grid" style="--fact-cols: {safe_columns};">{cards}</div>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_prediction_artifacts() -> tuple[object, dict[str, object]]:
    if not PIPELINE_PATH.exists() or not CONTEXT_PATH.exists():
        raise FileNotFoundError(
            "The Streamlit prediction artifacts are missing. Run "
            "`C:\\Users\\jesse\\anaconda3\\python.exe prepare_streamlit_assets.py` "
            "from the finaccess folder first."
        )
    return joblib.load(PIPELINE_PATH), joblib.load(CONTEXT_PATH)


@st.cache_data(show_spinner=False)
def load_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_weighted_summary() -> pd.DataFrame:
    return load_table(WEIGHTED_SUMMARY_PATH)


def load_model_comparison() -> pd.DataFrame:
    return load_table(MODEL_COMPARISON_PATH)


def load_feature_importance() -> pd.DataFrame:
    return load_table(FEATURE_IMPORTANCE_PATH)


def load_top_excluded_counties() -> pd.DataFrame:
    return load_table(TOP_COUNTIES_PATH)


def predict_profile(pipeline: object, form_data: dict[str, object]) -> tuple[str, pd.Series]:
    input_frame = pd.DataFrame([form_data])
    try:
        predicted_class = pipeline.predict(input_frame)[0]
        probabilities = pd.Series(
            pipeline.predict_proba(input_frame)[0],
            index=pipeline.classes_,
            name="probability",
        ).sort_values(ascending=False)
    except AttributeError as exc:
        if "_fill_dtype" in str(exc):
            raise RuntimeError(
                "The saved model artifact is incompatible with the scikit-learn version running this app. "
                "Pin scikit-learn to 1.7.2 in requirements.txt, rebuild the Streamlit artifacts, and redeploy."
            ) from exc
        raise
    return predicted_class, probabilities


def get_default_label(context: dict[str, object], feature: str) -> str:
    default_value = context["defaults"][feature]
    for option in context["display_options"][feature]:
        if option["value"] == default_value:
            return option["label"]
    return context["display_options"][feature][0]["label"]


def label_to_value(context: dict[str, object], feature: str, label: str) -> object:
    for option in context["display_options"][feature]:
        if option["label"] == label:
            return option["value"]
    raise KeyError(f"Could not map {label!r} back to a value for feature {feature!r}.")


def value_to_label(context: dict[str, object], feature: str, value: object) -> str:
    return context["value_labels"].get(feature, {}).get(value, str(value))


def build_prediction_story(
    context: dict[str, object],
    form_data: dict[str, object],
    prediction: str,
    probabilities: pd.Series,
    top_counties: pd.DataFrame,
) -> str:
    county_name = value_to_label(context, "county", form_data["county"])
    education_name = value_to_label(context, "education", form_data["education"])
    internet_name = value_to_label(context, "internet_frequency", form_data["internet_frequency"])
    health_name = value_to_label(context, "financial_health", form_data["financial_health"])
    top_county_names = set(top_counties["county_name"].tolist())

    reasons: list[str] = []
    if prediction == "Excluded":
        if form_data["can_access_internet"] == 0.0:
            reasons.append("the profile reports no internet access")
        if health_name == "Low":
            reasons.append("financial health is in the low category")
        if county_name in top_county_names:
            reasons.append(f"{county_name} appears among the higher-exclusion counties in the weighted county view")
        if education_name in {"None", "Some primary", "Primary completed"}:
            reasons.append("education is in one of the lower-completion categories used by the model")
    elif prediction == "Mobile money only":
        if form_data["can_access_internet"] == 1.0:
            reasons.append("the profile has internet access")
        if internet_name in {"Daily", "Weekly"}:
            reasons.append(f"internet use is relatively frequent ({internet_name.lower()})")
        if education_name in {
            "Some secondary",
            "Secondary completed",
            "Some technical training after secondary school",
            "Completed technical training after secondary school",
        }:
            reasons.append("the profile sits in a middle band of education and digital access")
    else:
        if education_name in {
            "Some university",
            "University completed",
            "Completed technical training after secondary school",
        }:
            reasons.append("education is in one of the stronger completion categories")
        if form_data["can_access_internet"] == 1.0:
            reasons.append("the profile includes internet access")
        if health_name in {"High", "Medium"}:
            reasons.append(f"financial health is recorded as {health_name.lower()}")

    if not reasons:
        reasons.append("the combined pattern of age, county, education, and internet behavior is closest to this class")

    reason_text = "; ".join(reasons[:3])
    confidence = probabilities.iloc[0]
    return (
        f"The model leans toward **{prediction}** with an estimated probability of "
        f"**{confidence:.1%}**. In this profile, {reason_text}. "
        f"This is a pattern-based survey prediction, not a causal claim or financial advice."
    )


def safe_read_bytes(path: Path) -> bytes | None:
    try:
        return path.read_bytes()
    except OSError:
        return None
