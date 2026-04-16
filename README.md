# FinAccess Streamlit App

DTSC 691 capstone project for predicting financial access profiles in Kenya using the FinAccess 2024 public survey.

## App pages

- `app.py` - home page
- `pages/1_Resume.py` - resume page
- `pages/2_General_Projects.py` - project portfolio page
- `pages/3_FinAccess_Project.py` - interactive prediction page

## Model summary

The app predicts whether an adult respondent is most likely to be:
- Excluded
- Mobile money only
- Banked

The current deployed demo uses a Streamlit-compatible Gradient Boosting pipeline artifact.

## Run locally

```bash
python -m streamlit run app.py
```

## Deploy

This repository is prepared for Streamlit Community Cloud deployment.

Main entry file: `app.py`
