# FinAccess Streamlit App

This folder now includes a multipage Streamlit scaffold for the DTSC 691 capstone.

## Files

- `app.py`: homepage
- `pages/1_Resume.py`: resume page
- `pages/2_General_Projects.py`: general projects page
- `pages/3_FinAccess_Project.py`: interactive project page
- `prepare_streamlit_assets.py`: rebuilds a Streamlit-compatible model artifact and form metadata

## Run locally

From `C:\Users\jesse\OneDrive - eastern.edu\DTSC 691\finaccess`:

```powershell
C:\Users\jesse\anaconda3\python.exe prepare_streamlit_assets.py
C:\Users\jesse\anaconda3\python.exe -m streamlit run app.py
```

## Notes

- The prediction page uses a rebuilt pipeline artifact because the older saved joblib file was created with a different scikit-learn version.
- Resume and portfolio sections include placeholders where the project folder did not provide personal details.
- Replace placeholder cards and work-history text before the final submission.
