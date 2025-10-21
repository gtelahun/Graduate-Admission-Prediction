import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Graduate Admission Predictor 🌟", layout="wide")

MODEL_PATH = Path("reg_admission.pickle")
DATA_PATH = Path("Admission_Predict.csv")

CANON_COLS = {
    "GRE Score": "GRE Score",
    "TOEFL Score": "TOEFL Score",
    "CGPA": "CGPA",
    "University Rating": "University Rating",
    "SOP": "SOP",
    "LOR ": "LOR",
    "LOR": "LOR",
    "Research": "Research",
    "Chance of Admit ": "Chance",
    "Chance of Admit": "Chance",
    "Chance": "Chance",
}

# ---------- helpers ----------
def as_percent01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0)) * 100.0

@st.cache_resource
def load_model(pth: Path):
    with open(pth, "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data(pth: Path):
    if not pth.exists(): return None
    df = pd.read_csv(pth)
    for old, new in list(CANON_COLS.items()):
        if old in df.columns and old != new:
            df.rename(columns={old: new}, inplace=True)
    return df

def expected_feature_names(model):
    # Try several places (plain estimator, Mapie wrappers, pipelines)
    for attr in ["feature_names_in_", "features_"]:
        if hasattr(model, attr):
            return list(getattr(model, attr))
    for attr in ["estimator_", "single_estimator_"]:
        if hasattr(model, attr) and hasattr(getattr(model, attr), "feature_names_in_"):
            return list(getattr(model, attr).feature_names_in_)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)
    return []

def row_with_one_hot(gre,toefl,cgpa,research_yes,uni,sop,lor,expected):
    X = pd.DataFrame([{
        "GRE Score":gre, "TOEFL Score":toefl, "CGPA":cgpa,
        "University Rating":uni, "SOP":sop, "LOR":lor,
        "Research_Yes": 1 if research_yes else 0,
        "Research_No":  0 if research_yes else 1,
    }])
    # align to expected
    if expected:
        for c in expected:
            if c not in X.columns: X[c] = 0
        X = X[expected]
    return X

def row_with_raw_research(gre,toefl,cgpa,research_yes,uni,sop,lor,expected):
    X = pd.DataFrame([{
        "GRE Score":gre, "TOEFL Score":toefl, "CGPA":cgpa,
        "University Rating":uni, "SOP":sop, "LOR":lor,
        "Research": 1 if research_yes else 0
    }])
    if expected:
        for c in expected:
            if c not in X.columns: X[c] = 0
        X = X[expected]
    return X

def predict01_with_retry(model, X_one_hot, X_raw):
    """
    Try to predict using one-hot version first (since your error shows the model
    was trained with Research_No/Research_Yes). If it fails, retry with raw.
    """
    try:
        y = model.predict(X_one_hot)
        val = float(np.asarray(y).ravel()[0])
    except Exception:
        y = model.predict(X_raw)
        val = float(np.asarray(y).ravel()[0])
    if 1.0 < val <= 100.0: val /= 100.0
    return float(np.clip(val, 0.0, 1.0))

def residual_pack(model, df, expected):
    if df is None: return None
    df = df.copy()
    # ensure canonical names
    for old, new in list(CANON_COLS.items()):
        if old in df.columns and old != new:
            df.rename(columns={old: new}, inplace=True)

    # Build both matrices
    X1 = df.copy()
    if "Research" in X1.columns:
        X1["Research_Yes"] = (X1["Research"] == 1).astype(int)
        X1["Research_No"]  = (X1["Research"] == 0).astype(int)
    # prefer one-hot
    feat_oh = ["GRE Score","TOEFL Score","CGPA","University Rating","SOP","LOR","Research_No","Research_Yes"]
    feat_raw = ["GRE Score","TOEFL Score","CGPA","University Rating","SOP","LOR","Research"]

    if expected:
        X_oh = pd.DataFrame(columns=expected)
        for c in expected: X_oh[c] = X1[c] if c in X1.columns else 0
        X_try = X_oh
    else:
        X_try = X1[[c for c in feat_oh if c in X1.columns]] if "Research_No" in X1.columns else X1[[c for c in feat_raw if c in X1.columns]]

    y = None
    for ycol in ["Chance"]:
        if ycol in df.columns:
            y = df[ycol].astype(float).to_numpy()
            if y.max() > 1.0: y = y/100.0
            break
    if y is None or len(X_try)==0: return None

    try:
        yhat = model.predict(X_try)
    except Exception:
        # fallback to raw schema
        X_raw = df[[c for c in feat_raw if c in df.columns]]
        yhat = model.predict(X_raw)

    yhat = np.where(yhat > 1.0, yhat/100.0, yhat)
    res = y - yhat
    return res, yhat, y

def prediction_interval(yhat01, residuals, z=1.645):
    if residuals is None or len(residuals) < 5:
        half = 0.10
    else:
        half = float(np.std(residuals, ddof=1)) * z
    lo = np.clip(yhat01 - half, 0.0, 1.0)
    hi = np.clip(yhat01 + half, 0.0, 1.0)
    return lo, hi

# ---------- load model/data ----------
if not MODEL_PATH.exists():
    st.error("Model file **reg_admission.pickle** not found in this folder.")
    st.stop()

model = load_model(MODEL_PATH)
df_data = load_data(DATA_PATH)
expected = expected_feature_names(model)

# ---------- sidebar inputs ----------
with st.sidebar:
    st.header("Enter Your Profile Details")
    gre = st.number_input("GRE Score", 260, 340, 323, 1)
    toefl = st.number_input("TOEFL Score", 0, 120, 100, 1)
    cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0, 0.1)
    research = st.selectbox("Research Experience", ["No", "Yes"])
    uni = st.slider("University Rating", 1, 5, 3)
    sop = st.slider("Statement of Purpose (SOP)", 1.0, 5.0, 3.5, 0.5)
    lor = st.slider("Letter of Recommendation (LOR)", 1.0, 5.0, 3.5, 0.5)
    predict_btn = st.button("Predict", type="primary")

# ---------- header ----------
st.markdown("<h1 style='color:#14532d;'>Graduate Admission Predictor 🌟</h1>", unsafe_allow_html=True)
st.write("Predict your chances of admission based on your profile.")
cols = st.columns([1.2, 2.0])

with cols[0]:
    if Path("admission.jpg").exists():
        st.image("admission.jpg", use_container_width=True)

with cols[1]:
    st.subheader("Predicting Admission Chance...")

    # Build both schemas, then predict with retry
    X_one_hot = row_with_one_hot(gre,toefl,cgpa,research=="Yes",uni,sop,lor,expected)
    X_raw     = row_with_raw_research(gre,toefl,cgpa,research=="Yes",uni,sop,lor,expected)
    yhat01 = predict01_with_retry(model, X_one_hot, X_raw)

    pack = residual_pack(model, df_data, expected)
    residuals = pack[0] if pack else None
    lo, hi = prediction_interval(yhat01, residuals)

    st.markdown("### Predicted Admission Probability")
    st.markdown(f"<div style='font-size:34px;font-weight:800;color:#1d4ed8;'>{as_percent01(yhat01):.2f}%</div>", unsafe_allow_html=True)
    st.progress(float(yhat01))
    st.caption(f"With a 90% confidence level: **[{as_percent01(lo):.2f}%, {as_percent01(hi)::.2f}%]**")

st.markdown("---")
st.markdown("<h2 style='color:#b91c1c;'>Model Insights</h2>", unsafe_allow_html=True)
tabs = st.tabs(["Feature Importance", "Histogram of Residuals", "Predicted vs. Actual", "Coverage Plot"])

# ---- Feature Importance
with tabs[0]:
    # Try several candidates for feature importance
    importance, labels = None, None
    cands = [model]
    for a in ["estimator_", "single_estimator_"]:
        if hasattr(model, a): cands.append(getattr(model, a))
    if hasattr(model, "named_steps"):
        cands += list(model.named_steps.values())
    for est in cands:
        if hasattr(est, "feature_importances_"):
            importance = np.asarray(est.feature_importances_).ravel()
            labels = expected if expected else (list(getattr(est, "feature_names_in_", [])) or None)
            break
        if hasattr(est, "coef_"):
            importance = np.abs(np.asarray(est.coef_).ravel())
            labels = expected if expected else (list(getattr(est, "feature_names_in_", [])) or None)
            break

    if importance is None or labels is None or len(importance) != len(labels):
        st.info("Feature importance not available for this model.")
    else:
        order = np.argsort(importance)[::-1]
        fig = plt.figure()
        plt.barh(range(len(order)), importance[order])
        plt.gca().invert_yaxis()
        plt.yticks(range(len(order)), [labels[i] for i in order])
        plt.xlabel("Importance")
        plt.title("Feature Importance")
        st.pyplot(fig, clear_figure=True)

# ---- Residuals
with tabs[1]:
    if pack is None:
        st.info("Provide Admission_Predict.csv to see residuals.")
    else:
        res, _, _ = pack
        fig = plt.figure()
        plt.hist(res, bins=20)
        plt.title("Histogram of Residuals")
        plt.xlabel("Residual (Actual - Predicted)")
        plt.ylabel("Count")
        st.pyplot(fig, clear_figure=True)

# ---- Predicted vs Actual
with tabs[2]:
    if pack is None:
        st.info("Provide Admission_Predict.csv to view this plot.")
    else:
        _, yhat_all, y_all = pack
        fig = plt.figure()
        plt.scatter(y_all, yhat_all)
        lo_ax = float(min(y_all.min(), yhat_all.min()))
        hi_ax = float(max(y_all.max(), yhat_all.max()))
        plt.plot([lo_ax, hi_ax], [lo_ax, hi_ax], linestyle="--")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Predicted vs Actual Values")
        st.pyplot(fig, clear_figure=True)

# ---- Coverage
with tabs[3]:
    if pack is None:
        st.info("Provide Admission_Predict.csv to compute coverage.")
    else:
        res, yhat_all, y_all = pack
        z = 1.645
        sigma = float(np.std(res, ddof=1))
        lower = np.clip(yhat_all - z * sigma, 0.0, 1.0)
        upper = np.clip(yhat_all + z * sigma, 0.0, 1.0)
        covered = (y_all >= lower) & (y_all <= upper)
        coverage = covered.mean() * 100.0
        st.write(f"Estimated coverage of 90% PI: **{coverage:.1f}%**")
        fig = plt.figure()
        plt.bar(["Inside PI", "Outside PI"], [covered.sum(), (~covered).sum()])
        plt.title("Coverage Count (90% PI)")
        st.pyplot(fig, clear_figure=True)

st.caption("If you still see a feature-name error, check the model’s expected features with "
           "`feature_names_in_` and ensure your pickle matches the CSV preprocessing.")
