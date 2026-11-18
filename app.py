"""Streamlit app for the Medical Insurance Cost model (ADSE Project 3).

It loads best_insurance_model.joblib (saved at the end of the notebook) and builds
the input form automatically from the trained pipeline, so it works whatever the
dataset's columns are. If the original CSV sits next to this file, it is only used
to give the number fields sensible min/max ranges.
"""
import glob

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Medical Insurance Cost Predictor", page_icon="🏥")


@st.cache_resource
def load_model():
    return joblib.load("best_insurance_model.joblib")


@st.cache_data
def load_reference():
    files = [f for f in glob.glob("*.csv") if "checkpoint" not in f.lower()]
    return pd.read_csv(files[0]) if files else None


pipe = load_model()
ref = load_reference()
prep = pipe.named_steps["prep"]
model_name = type(pipe.named_steps["model"]).__name__

# Read the expected inputs straight from the fitted preprocessor
num_info, cat_info = {}, {}
for name, trans, cols in prep.transformers_:
    if name == "num":
        medians = trans.named_steps["impute"].statistics_
        num_info = dict(zip(cols, medians))
    elif name == "cat":
        cats = trans.named_steps["onehot"].categories_
        cat_info = dict(zip(cols, cats))

st.title("🏥 Medical Insurance Cost Predictor")
st.write(
    f"Enter the person's details and press **Predict**. "
    f"The model used is **{model_name}**, trained in the Project 3 notebook."
)

values = {}
with st.form("inputs"):
    left, right = st.columns(2)
    i = 0
    for col in prep.feature_names_in_:
        box = left if i % 2 == 0 else right
        i += 1
        label = col.replace("_", " ").title()
        with box:
            if col in num_info:
                default = float(num_info[col])
                if ref is not None and col in ref.columns:
                    s = ref[col].dropna()
                    lo, hi = float(s.min()), float(s.max())
                    if (s % 1 == 0).all():
                        values[col] = st.number_input(
                            label, int(lo), int(hi), int(np.clip(round(default), lo, hi)), step=1
                        )
                    else:
                        values[col] = st.number_input(
                            label, lo, hi, float(np.clip(default, lo, hi))
                        )
                else:
                    values[col] = st.number_input(label, value=default)
            else:
                values[col] = st.selectbox(label, list(cat_info[col]))
    submitted = st.form_submit_button("Predict")

if submitted:
    row = pd.DataFrame([values], columns=list(prep.feature_names_in_))
    prediction = float(pipe.predict(row)[0])
    st.success("Prediction ready")
    st.metric("Predicted annual medical cost", f"{prediction:,.2f}")
    st.caption("Shown in the same unit as the cost column in the dataset.")
