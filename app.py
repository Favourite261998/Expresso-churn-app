# app.py
import os
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Expresso Churn Prediction", layout="centered")

# ---------- CONFIG ----------
MODEL_NAME = "espreso_churn_model.pkl"

# Precomputed mappings & defaults
mappings = {
    "REGION": {"Unknown": 0, "Dakar": 1, "Thies": 2, "Diourbel": 3},
    "TENURE": {"Unknown": 0, "Short": 1, "Medium": 2, "Long": 3},
    "MRG": {"Unknown": 0, "Yes": 1, "No": 2},
    "TOP_PACK": {"Unknown": 0, "Pack1": 1, "Pack2": 2, "Pack3": 3},
}

# Replace values below with medians from the dataset
numeric_defaults = {
    "MONTANT": 3000.0,          # median
    "FREQUENCE_RECH": 7.0,      # median
    "REVENUE": 3000.0,          # median
    "ARPU_SEGMENT": 1000.0,     # median
    "FREQUENCE": 9.0,           # median
    "DATA_VOLUME": 257.0,       # median
    "ON_NET": 27.0,             # median
    "ORANGE": 29.0,             # median
    "TIGO": 6.0,                # median
    "ZONE1": 1.0,               # median
    "ZONE2": 2.0,               # median
    "REGULARITY": 24.0,         # median
    "FREQ_TOP_PACK": 5.0        # median
}

def map_cat(value, mapping):
    return mapping.get(value, mapping.get("Unknown", 0))

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



