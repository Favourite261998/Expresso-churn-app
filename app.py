# app.py
import os
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Expresso Churn Prediction", layout="centered")

# ---------- CONFIG ----------
CSV_NAME = "Expresso_churn_dataset.csv"
MODEL_NAME = "espreso_churn_model.pkl"

# ---------- utilities ----------
@st.cache_data(show_spinner=False)
def load_dataset(out=CSV_NAME):
    if not os.path.exists(out):
        st.error(f"❌ Dataset file '{out}' not found in folder.")
        st.stop()
    df = pd.read_csv(out)
    return df

@st.cache_data(show_spinner=False)
def prepare_dataset_and_mappings(df):
    df = df.copy()
    df = df.drop(columns=["user_id"], errors="ignore")

    # Fill categorical
    cat_cols = ["REGION", "TENURE", "MRG", "TOP_PACK"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("Unknown").astype(str)
        else:
            df[c] = "Unknown"

    # Fill numeric
    num_cols = [c for c in df.columns if df[c].dtype in ("float64", "int64") and c != "CHURN"]
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())

    # Create mappings
    mappings = {}
    for c in cat_cols:
        cats = sorted(df[c].unique().tolist())
        mappings[c] = {cat: idx for idx, cat in enumerate(cats)}

    numeric_defaults = {c: float(df[c].median()) for c in num_cols if c in df.columns}
    return mappings, numeric_defaults

def map_cat(value, mapping):
    if value in mapping:
        return mapping[value]
    if "Unknown" in mapping:
        return mapping["Unknown"]
    return 0

# ---------- Load model ----------
@st.cache_resource(show_spinner=False)
def load_model(path=MODEL_NAME):
    if not os.path.exists(path):
        st.error(f"❌ Model file '{path}' not found in folder.")
        st.stop()
    try:
        return joblib.load(path)
    except Exception as e:
        st.error("❌ Failed to load model.")
        st.exception(e)
        st.stop()

# ---------- Main App ----------
st.title("📱 Expresso Churn Prediction (Streamlit)")

model = load_model()
df = load_dataset()
mappings, numeric_defaults = prepare_dataset_and_mappings(df)

FEATURES = [
    "REGION","TENURE","MONTANT","FREQUENCE_RECH","REVENUE",
    "ARPU_SEGMENT","FREQUENCE","DATA_VOLUME","ON_NET","ORANGE",
    "TIGO","ZONE1","ZONE2","MRG","REGULARITY","TOP_PACK","FREQ_TOP_PACK"
]

st.markdown("Enter customer values (defaults come from dataset medians/categories).")

with st.form("predict_form"):
    REGION = st.selectbox("REGION", options=list(mappings["REGION"].keys()), index=0)
    TENURE = st.selectbox("TENURE", options=list(mappings["TENURE"].keys()), index=0)
    MRG = st.selectbox("MRG", options=list(mappings["MRG"].keys()), index=0)
    TOP_PACK = st.selectbox("TOP_PACK", options=list(mappings["TOP_PACK"].keys()), index=0)

    def nd(name, fallback=0.0):
        return numeric_defaults.get(name, fallback)

    MONTANT = st.number_input("MONTANT", min_value=0.0, value=nd("MONTANT"))
    FREQUENCE_RECH = st.number_input("FREQUENCE_RECH", min_value=0.0, value=nd("FREQUENCE_RECH"))
    REVENUE = st.number_input("REVENUE", min_value=0.0, value=nd("REVENUE"))
    ARPU_SEGMENT = st.number_input("ARPU_SEGMENT", min_value=0.0, value=nd("ARPU_SEGMENT"))
    FREQUENCE = st.number_input("FREQUENCE", min_value=0.0, value=nd("FREQUENCE"))
    DATA_VOLUME = st.number_input("DATA_VOLUME", min_value=0.0, value=nd("DATA_VOLUME"))
    ON_NET = st.number_input("ON_NET", min_value=0.0, value=nd("ON_NET"))
    ORANGE = st.number_input("ORANGE", min_value=0.0, value=nd("ORANGE"))
    TIGO = st.number_input("TIGO", min_value=0.0, value=nd("TIGO"))
    ZONE1 = st.number_input("ZONE1", min_value=0.0, value=nd("ZONE1"))
    ZONE2 = st.number_input("ZONE2", min_value=0.0, value=nd("ZONE2"))
    REGULARITY = st.number_input("REGULARITY", min_value=0, value=int(nd("REGULARITY") if nd("REGULARITY") is not None else 0))
    FREQ_TOP_PACK = st.number_input("FREQ_TOP_PACK", min_value=0.0, value=nd("FREQ_TOP_PACK"))

    submit = st.form_submit_button("Predict")

if submit:
    try:
        row = [
            map_cat(REGION, mappings["REGION"]),
            map_cat(TENURE, mappings["TENURE"]),
            float(MONTANT), float(FREQUENCE_RECH), float(REVENUE),
            float(ARPU_SEGMENT), float(FREQUENCE), float(DATA_VOLUME),
            float(ON_NET), float(ORANGE), float(TIGO), float(ZONE1), float(ZONE2),
            map_cat(MRG, mappings["MRG"]), int(REGULARITY),
            map_cat(TOP_PACK, mappings["TOP_PACK"]), float(FREQ_TOP_PACK)
        ]
        input_df = pd.DataFrame([row], columns=FEATURES)

        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None

        if pred == 1:
            st.error(f"⚠️ Likely to CHURN (p = {proba:.2%})" if proba is not None else "⚠️ Likely to CHURN")
        else:
            st.success(f"✅ Not likely to churn (p = {proba:.2%})" if proba is not None else "✅ Not likely to churn")

    except Exception as e:
        st.error("Prediction failed — see details below.")
        st.exception(e)
