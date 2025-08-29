# Rotten_fruit.py — clean & fixed
import os, base64, pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.metrics import classification_report
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize


# ----------------------------
# FIXED PATHS
# ----------------------------
BG_PATH               = "fruit-modrl/main_background.jpg"
MODEL_PATH            = "fruit-modrl/model.pkl"
SCALER_PATH           = "fruit-modrl/scaler.pkl"           # optional
FRUIT_ENCODER_PATH    = "fruit-modrl/fruit_encoder.pkl"    # optional (if you used it)
STAGE_ENCODER_PATH    = "fruit-modrl/stage_encoder.pkl"    # optional (if you used it)
EVAL_CSV_PATH         = "fruit-modrl/fruits_data.csv"      # optional

SENSOR_IMAGE_MAP = {
    "nir_850":       "sensors/nir-all.png",
    "nir_940":       "sensors/nir-all.png",
    "R":             "sensors/R G B.jpg",
    "G":             "sensors/R G B.jpg",
    "B":             "sensors/R G B.jpg",
    "temp_c":        "sensors/Temp.jpg",
    "humidity_pct":  "sensors/dh22.jpg",
    "voc_ppm":       "sensors/voc.jpg",
}

# If your eval CSV includes "fruit" feature and "stage" label:
FEATURE_COLS = ["fruit","nir_850","nir_940","R","G","B","temp_c","humidity_pct","voc_ppm"]
LABEL_COL    = "stage"

# ----------------------------
# PAGE CONFIG + BACKGROUND
# ----------------------------
st.set_page_config(
    page_title='Classify Fruits',
    layout='centered',
    initial_sidebar_state='expanded',
    page_icon=':watermelon:',
)

def set_background(path: str):
    if not os.path.exists(path):
        return
    ext = os.path.splitext(path)[1].replace(".", "") or "jpg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
    f"""
    <style>
    .stApp {{
        background: url(data:image/{ext};base64,{b64});
        background-size: cover;
    }}

    /* Glass effect for cards */
    .glass {{
        background: rgba(255,255,255,0.72);
        border-radius: 16px;
        padding: 1.0rem 1.25rem;
        border: 1px solid rgba(255,255,255,0.35);
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }}

    .metric-card {{
        background: rgba(255,255,255,0.88);
        border-radius: 14px;
        padding: .75rem 1rem;
        border: 1px solid rgba(0,0,0,0.05);
    }}

    /* Transparent Sidebar */
    section[data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.12) !important;
        backdrop-filter: blur(12px);
        border-right: 1px solid rgba(255,255,255,0.25);
    }}

    /* Remove/transparent top header */
    header[data-testid="stHeader"] {{
        background: rgba(0,0,0,0) !important;
    }}

    /* Make main content container transparent */
    .block-container {{
        background: transparent !important;
    }}

    /* Optional: make tabs container transparent */
    div[data-baseweb="tab-list"] {{
        background: rgba(255,255,255,0.20) !important;
        border-radius: 12px;
        backdrop-filter: blur(10px);
        padding: 0.3rem 0.6rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)


set_background(BG_PATH)

st.sidebar.image("logo.png")
st.sidebar.title("About Rotten Fruit")
st.sidebar.info(
    "Rotten Fruit is a machine learning project that detects the freshness stage of fruits (Raw, Fresh, Spoiled)"

)

# ----------------------------
# LOADER (robust: joblib/pickle)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_bytes_as_obj(b: bytes):
    # Try joblib first then pickle
    try:
        import joblib
        return joblib.load(BytesIO(b))
    except Exception:
        return pickle.loads(b)

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    with open(MODEL_PATH, "rb") as f:
        model = load_bytes_as_obj(f.read())

    scaler = None
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = load_bytes_as_obj(f.read())

    fruit_enc = None
    if os.path.exists(FRUIT_ENCODER_PATH):
        with open(FRUIT_ENCODER_PATH, "rb") as f:
            fruit_enc = load_bytes_as_obj(f.read())

    stage_enc = None
    if os.path.exists(STAGE_ENCODER_PATH):
        with open(STAGE_ENCODER_PATH, "rb") as f:
            stage_enc = load_bytes_as_obj(f.read())

    return model, scaler, fruit_enc, stage_enc
# --- Utilities for consistent preprocessing ---
def _get_fruit_options():
    # Priority 1: encoder classes_
    if FRUIT_ENCODER is not None and hasattr(FRUIT_ENCODER, "classes_"):
        return list(FRUIT_ENCODER.classes_)
    # Priority 2: read from eval CSV
    if os.path.exists(EVAL_CSV_PATH):
        try:
            _df = pd.read_csv(EVAL_CSV_PATH)
            if "fruit" in _df.columns:
                return sorted(list(pd.Series(_df["fruit"].dropna().unique()).astype(str)))
        except Exception:
            pass
    # Fallback options
    return ["apple", "banana", "orange", "watermelon", "grape", "mango"]

def plot_roc_pr_multiclass(y_true_labels, y_proba, classes):
    """
    y_true_labels: array of original class labels (strings or ints) aligned to `classes`
    y_proba: (n_samples, n_classes) predicted probabilities in the SAME class order as `classes`
    classes: model.classes_ (array-like)
    """
    # binarize truth to match class order
    y_bin = label_binarize(y_true_labels, classes=list(classes))  # shape: (n_samples, n_classes)

    # --- micro-avg ROC ---
    fpr, tpr, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, label=f"micro-avg AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--")
    ax_roc.set_title("ROC Curve (micro-average, multiclass)")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # --- micro-avg PR ---
    precision, recall, _ = precision_recall_curve(y_bin.ravel(), y_proba.ravel())
    fig_pr, ax_pr = plt.subplots()
    ax_pr.plot(recall, precision)
    ax_pr.set_title("Precision–Recall Curve (micro-average, multiclass)")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    st.pyplot(fig_pr)

def preprocess_features(df_in: pd.DataFrame) -> np.ndarray:
    """
    Returns a numeric 2D array ready for MODEL.predict_proba().
    Encodes 'fruit' using saved encoder or a deterministic mapping, then applies SCALER if present.
    """
    df = df_in.copy()

    # Encode categorical 'fruit' if present
    if "fruit" in df.columns:
        if FRUIT_ENCODER is not None:
            try:
                df["fruit"] = FRUIT_ENCODER.transform(df["fruit"])
            except Exception:
                # Build mapping from encoder classes if available; unseen -> -1
                classes = list(getattr(FRUIT_ENCODER, "classes_", []))
                mapping = {name: i for i, name in enumerate(classes)}
                df["fruit"] = df["fruit"].map(mapping).fillna(-1).astype(int)
        else:
            # Deterministic mapping from list of fruits
            mapping = {name: i for i, name in enumerate(_get_fruit_options())}
            df["fruit"] = df["fruit"].map(mapping).fillna(-1).astype(int)

    # Coerce numerics
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Apply scaler (some scalers/pipelines accept DataFrame, some need ndarray)
    if SCALER is not None:
        try:
            X = SCALER.transform(df)
        except Exception:
            X = SCALER.transform(df.values)
        return X

    return df.values

try:
    MODEL, SCALER, FRUIT_ENCODER, STAGE_ENCODER = load_artifacts()
    st.sidebar.success("Model loaded ✓")
except Exception as e:
    st.sidebar.error(f"Model load error: {e}")
    st.stop()

# ----------------------------
# UI
# ----------------------------
st.title(':rainbow[Classify Freshness Detection]')
TAB_OVERVIEW, TAB_SENSORS, TAB_EVAL, TAB_TEST = st.tabs(
    ["Overview", "Sensors", "Model Evaluation", "Test the Project"]
)

def show_card(title: str, img_path: str, desc: str):
        st.subheader(title)
        if img_path and os.path.exists(img_path):
            st.image(img_path, caption=title)
        else:
            st.image(f"https://placehold.co/600x360?text={title.replace(' ', '+')}", caption=f"{title} (placeholder)")
        st.caption(desc)

# OVERVIEW
with TAB_OVERVIEW:
    st.markdown(
        """
        <div class='glass'>
            <h3>Project Summary</h3>
            <p>This app predicts fruit freshness using a Logistic Regression model trained on eight multi-modal sensors (NIR bands, RGB color, temperature, humidity, and VOC).</p>
            <p>Explore sensor descriptions, review evaluation metrics and diagnostic charts, and test the preloaded model by entering the 8 sensor values interactively.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    a1,a2,a3 = st.columns(3)

    with a1:
        show_card(
            "Raw Mango",
            "fruit-modrl/Raw.png",   
            ""
        )

    with a2:
        show_card(
            "Fresh Apple",
            "fruit-modrl/Fresh.jpg",    
            ""
        )

    with a3:
        show_card(
            "Spoiled Mango",
            "fruit-modrl/spoiled.png",    
            ""
        )

# ----------------------------
# ------ Sensors (grouped)
# ----------------------------
with TAB_SENSORS:
    st.markdown(
        "<div class='glass'><h3>Sensor Explanations</h3><p>Grouped views: NIR (850 & 940) together, RGB (R+G+B) together.</p></div>",
        unsafe_allow_html=True
    )
    st.write("")
    c1, c2 = st.columns(2)

    with c1:
        show_card(
            "NIR (850 nm + 940 nm)",
            "sensors/nir-all.png",   # one combined image you already have
            "Near-infrared pair: 850 nm (higher camera sensitivity) + 940 nm (deeper IR). Together they highlight water content and internal tissue changes related to ripeness/spoilage."
        )

    with c2:
        show_card(
            "RGB (R + G + B)",
            "sensors/R_G_B.png",    # one combined image you already have
            "Visible color channels combined. Color shifts track chlorophyll breakdown, browning, and surface defects."
        )

    st.markdown("---")

    c3, c4, c5 = st.columns(3)
    with c3:
        show_card(
            "Temperature (°C)",
            "sensors/Temp.png",
            "Higher temperature accelerates respiration and decay; cooling slows deterioration."
        )
    with c4:
        show_card(
            "Humidity (%)",
            "sensors/dt22.png",
            "Relative humidity affects dehydration and mold growth risk, influencing shelf life."
        )
    with c5:
        show_card(
            "VOC (ppm)",
            "sensors/voc.png",
            "Volatile organics emitted during ripening/spoilage (e.g., ethylene) provide a gas-based freshness cue."
        )


# EVALUATION
with TAB_EVAL:
    st.markdown(
        "<div class='glass'><h3>Evaluation</h3><p>This tab uses a predefined CSV if available (<code>fruit-modrl/fruits_data.csv</code>). It can contain either (<code>y_true</code>,<code>y_pred_proba</code>) or full features + label.</p></div>",
        unsafe_allow_html=True
    )
    st.write("")
    st.write("")
    df_eval = None
    if os.path.exists(EVAL_CSV_PATH):
        try:
            df_eval = pd.read_csv(EVAL_CSV_PATH)
            if "Unnamed: 0" in df_eval.columns:
                df_eval = df_eval.drop(columns=["Unnamed: 0"])
            st.write("Preview:", df_eval.head())
        except Exception as e:
            st.error(f"Failed to read {EVAL_CSV_PATH}: {e}")
    st.write("")
    y_true = None
    y_pred_proba = None

    if df_eval is not None:
        if {"y_true", "y_pred_proba"}.issubset(df_eval.columns):
            y_true = df_eval["y_true"].values
            y_pred_proba = df_eval["y_pred_proba"].values
        elif set(FEATURE_COLS).issubset(df_eval.columns) and (LABEL_COL in df_eval.columns):
            X = df_eval[FEATURE_COLS].values
            # Optional categorical encoders if you actually used them:
            # if FRUIT_ENCODER is not None and "fruit" in df_eval.columns:
            #     df_eval["fruit"] = FRUIT_ENCODER.transform(df_eval["fruit"])
            X = df_eval[FEATURE_COLS].values
            X_df = df_eval[FEATURE_COLS].copy()

            if df_eval[LABEL_COL].dtype == object:
                if STAGE_ENCODER is not None:
                    y_true = STAGE_ENCODER.transform(df_eval[LABEL_COL])
                else:
                    # fallback: map the first sorted class to 0, second to 1
                    classes = sorted(df_eval[LABEL_COL].astype(str).unique())
                    mapping = {cls: i for i, cls in enumerate(classes)}
                    y_true = df_eval[LABEL_COL].map(mapping).astype(int).values
            else:
                y_true = df_eval[LABEL_COL].values

            X_pre = preprocess_features(X_df)
            y_pred_proba = MODEL.predict_proba(X_pre) 

        else:
            st.info("Place an eval CSV with either (y_true,y_pred_proba) or the feature columns + 'stage' label.")
        st.write("")
    # Keep original labels (strings) for multiclass curves
    y_true_labels_str = None
    if df_eval is not None and LABEL_COL in df_eval.columns:
        if df_eval[LABEL_COL].dtype == object:
            y_true_labels_str = df_eval[LABEL_COL].astype(str).values
        else:
            # If numeric labels, try to recover strings via STAGE_ENCODER if available
            if STAGE_ENCODER is not None and hasattr(STAGE_ENCODER, "inverse_transform"):
                try:
                    y_true_labels_str = STAGE_ENCODER.inverse_transform(df_eval[LABEL_COL])
                except Exception:
                    y_true_labels_str = df_eval[LABEL_COL].astype(str).values
            else:
                y_true_labels_str = df_eval[LABEL_COL].astype(str).values
    st.write("")

    if (y_true is not None) and (y_pred_proba is not None):
        # y_true may be ints (encoded) or strings. We'll compute y_pred accordingly.
        # Detect number of classes from model if possible; otherwise from y_pred_proba shape.
        n_classes = None
        classes = getattr(MODEL, "classes_", None)
        if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
            n_classes = y_pred_proba.shape[1]
        elif classes is not None:
            n_classes = len(classes)

        # -------------------------
        # BINARY CASE
        # -------------------------
        if n_classes in (None, 2):
            # ensure 1D proba for positive class
            if isinstance(y_pred_proba, np.ndarray) and y_pred_proba.ndim == 2:
                y_proba_bin = y_pred_proba[:, 1]
            else:
                y_proba_bin = y_pred_proba

            threshold = st.slider("Decision Threshold", 0.05, 0.95, 0.50, 0.01)
            y_pred = (y_proba_bin >= threshold).astype(int)

            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)

            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-card'><b>Accuracy</b><br><h2>{acc:.3f}</h2></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><b>Precision</b><br><h2>{prec:.3f}</h2></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><b>Recall</b><br><h2>{rec:.3f}</h2></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><b>F1-Score</b><br><h2>{f1:.3f}</h2></div>", unsafe_allow_html=True)

            st.markdown("---")
            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots()
            ax_cm.imshow(cm, interpolation='nearest')
            ax_cm.set_title('Confusion Matrix')
            ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
            ax_cm.set_xticks([0,1]); ax_cm.set_yticks([0,1])
            for (i, j), z in np.ndenumerate(cm):
                ax_cm.text(j, i, str(z), ha='center', va='center')
            st.pyplot(fig_cm)
            # ROC
            fpr, tpr, _ = roc_curve(y_true, y_proba_bin)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0,1], [0,1], linestyle='--')
            ax_roc.set_title('ROC Curve')
            ax_roc.set_xlabel('False Positive Rate'); ax_roc.set_ylabel('True Positive Rate')
            ax_roc.legend(loc='lower right')
            st.pyplot(fig_roc)
            # PR
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba_bin)
            fig_pr, ax_pr = plt.subplots()
            ax_pr.plot(recall_vals, precision_vals)
            ax_pr.set_title('Precision–Recall Curve')
            ax_pr.set_xlabel('Recall'); ax_pr.set_ylabel('Precision')
            st.pyplot(fig_pr)

        # -------------------------
        # MULTICLASS CASE
        # -------------------------
        else:
            # y_pred via argmax
            y_pred = np.argmax(y_pred_proba, axis=1)
            # Weighted metrics + report
            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            #st.text(classification_report(y_true, y_pred))
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"<div class='metric-card'><b>Accuracy</b><br><h2>{acc:.3f}</h2></div>", unsafe_allow_html=True)
            c2.markdown(f"<div class='metric-card'><b>Precision</b><br><h2>{prec:.3f}</h2></div>", unsafe_allow_html=True)
            c3.markdown(f"<div class='metric-card'><b>Recall</b><br><h2>{rec:.3f}</h2></div>", unsafe_allow_html=True)
            c4.markdown(f"<div class='metric-card'><b>F1-Score</b><br><h2>{f1:.3f}</h2></div>", unsafe_allow_html=True)

            st.markdown("---")

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots()
            ax_cm.imshow(cm, interpolation='nearest')
            ax_cm.set_title('Confusion Matrix')
            ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('Actual')
            ax_cm.set_xticks(range(n_classes)); ax_cm.set_yticks(range(n_classes))
            for (i, j), z in np.ndenumerate(cm):
                ax_cm.text(j, i, str(z), ha='center', va='center')
            st.pyplot(fig_cm)

            # Micro-avg ROC/PR for multiclass
            # Need class labels for binarization; prefer MODEL.classes_ else unique y_true
            classes_for_curves = classes if classes is not None else np.unique(y_true)
            # If your y_true are encoded ints, that's fine; binarize will still work.
            y_bin = label_binarize(y_true, classes=list(classes_for_curves))
            proba_2d = y_pred_proba  # already (n_samples, n_classes)
            st.write("")
            # ROC (micro-average)
            fpr, tpr, _ = roc_curve(y_bin.ravel(), proba_2d.ravel())
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"micro-avg AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], linestyle="--")
            ax_roc.set_title("ROC Curve (micro-average, multiclass)")
            ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)
            st.write("")
            # PR (micro-average)
            precision_vals, recall_vals, _ = precision_recall_curve(y_bin.ravel(), proba_2d.ravel())
            fig_pr, ax_pr = plt.subplots()
            ax_pr.plot(recall_vals, precision_vals)
            ax_pr.set_title("Precision–Recall Curve (micro-average, multiclass)")
            ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
            st.pyplot(fig_pr)

    elif set(FEATURE_COLS).issubset(df_eval.columns) and (LABEL_COL in df_eval.columns):
        X_df = df_eval[FEATURE_COLS].copy()
        y_true = df_eval[LABEL_COL].values

        try:
            X_pre = X_df
            if SCALER is not None:
                X_pre = SCALER.transform(X_pre)
        except Exception as e1:
            # fallback: encode fruit then scale
            if FRUIT_ENCODER is not None:
                X_df = X_df.copy()
                X_df["fruit"] = FRUIT_ENCODER.transform(X_df["fruit"])
            else:
                # map fruit strings consistently across dataset
                uniq = {name: i for i, name in enumerate(sorted(X_df["fruit"].astype(str).unique()))}
                X_df["fruit"] = X_df["fruit"].map(uniq).astype(int)

            X_pre = X_df.values
            if SCALER is not None:
                X_pre = SCALER.transform(X_pre)

        y_pred_proba = MODEL.predict_proba(X_pre)[:, 1]

    else:
        st.info("No evaluation available. Add CSV at fruit-modrl/fruits_data.csv.")

# ----------------------------
# ------ Test the Project
# ----------------------------
with TAB_TEST:
    st.markdown(
        "<div class='glass'><h3>Test a Single Prediction</h3><p>Enter the eight sensor values; the preloaded model will return probability and label.</p></div>",
        unsafe_allow_html=True
    )
    st.write("")
    # 1) Fruit options
    def _get_fruit_options():
        # Priority 1: encoder classes_
        if FRUIT_ENCODER is not None and hasattr(FRUIT_ENCODER, "classes_"):
            return list(FRUIT_ENCODER.classes_)
        # Priority 2: values from eval CSV
        if os.path.exists(EVAL_CSV_PATH):
            try:
                _df = pd.read_csv(EVAL_CSV_PATH)
                if "fruit" in _df.columns:
                    return sorted(list(pd.Series(_df["fruit"].dropna().unique()).astype(str)))
            except Exception:
                pass
        # Fallback default list
        return ["apple", "banana", "orange", "watermelon", "grape", "mango"]
    st.write("")
    fruit_options = _get_fruit_options()
    c0, _, _, _ = st.columns(4)
    fruit = c0.selectbox("fruit", fruit_options, index=fruit_options.index("watermelon") if "watermelon" in fruit_options else 0)

    c1, c2, c3, c4 = st.columns(4)
    nir_850 = c1.number_input("nir_850", value=0.50, min_value=0.0, max_value=10.0, step=0.01)
    nir_940 = c2.number_input("nir_940", value=0.50, min_value=0.0, max_value=10.0, step=0.01)
    R       = c3.number_input("R", value=128.0, min_value=0.0, max_value=255.0, step=1.0)
    G       = c4.number_input("G", value=128.0, min_value=0.0, max_value=255.0, step=1.0)

    c5, c6, c7, c8 = st.columns(4)
    B        = c5.number_input("B", value=128.0, min_value=0.0, max_value=255.0, step=1.0)
    temp_c   = c6.number_input("temp_c (°C)", value=20.0, min_value=-20.0, max_value=60.0, step=0.1)
    humidity = c7.number_input("humidity_pct (%)", value=60.0, min_value=0.0, max_value=100.0, step=0.5)
    voc_ppm  = c8.number_input("voc_ppm", value=10.0, min_value=0.0, max_value=50.0, step=1.0)

    run = st.button("🔮 Predict Freshness")
    st.write("")
    if run:
        # 2) Build a DataFrame with ALL training columns, including 'fruit'
        row = {
            "fruit": fruit,
            "nir_850": nir_850,
            "nir_940": nir_940,
            "R": R,
            "G": G,
            "B": B,
            "temp_c": temp_c,
            "humidity_pct": humidity,
            "voc_ppm": voc_ppm,
        }
        df_input = pd.DataFrame([row], columns=FEATURE_COLS)
        st.write("")
        try:
            # 3A) First try: let SCALER (likely a ColumnTransformer/Pipeline) handle strings directly
            X_pre = df_input
            if SCALER is not None:
                X_pre = SCALER.transform(X_pre)

        except Exception as e1:
            # 3B) Fallback: label-encode 'fruit' if encoder is available
            try:
                df_enc = df_input.copy()
                if FRUIT_ENCODER is not None:
                    df_enc["fruit"] = FRUIT_ENCODER.transform(df_enc["fruit"])
                else:
                    # Fallback quick mapping if no encoder saved
                    # Build mapping from eval CSV or the current fruit list
                    mapping = {name: i for i, name in enumerate(fruit_options)}
                    df_enc["fruit"] = df_enc["fruit"].map(mapping).astype(int)

                X_pre = df_enc[FEATURE_COLS].values
                if SCALER is not None:
                    X_pre = SCALER.transform(X_pre)
            except Exception as e2:
                st.error(f"Preprocessing failed.\nPrimary error: {e1}\nFallback error: {e2}")
                st.stop()

       # 4) Predict (multiclass: Raw, Fresh, Spoiled)
    # 4) Predict (multiclass: Raw, Fresh, Spoiled)
    try:
        proba = MODEL.predict_proba(X_pre)[0]   # shape: (n_classes,)
        pred_class_index = np.argmax(proba)

        # Decode class name properly
        if STAGE_ENCODER is not None:
            # Use encoder inverse_transform if available
            pred_class = STAGE_ENCODER.inverse_transform([pred_class_index])[0]
            class_labels = STAGE_ENCODER.classes_
        elif hasattr(MODEL, "classes_"):
            class_labels = MODEL.classes_
            pred_class = class_labels[pred_class_index]
        else:
            # fallback default
            class_labels = ["raw", "fresh", "spoiled"]
            pred_class = class_labels[pred_class_index]

        # Emojis mapping for nice UI
        emoji_map = {
            "raw": "🥭",
            "fresh": "🍏",
            "spoiled": "🍂"
        }

        # Show result card
        st.markdown(
            f"""
            <div class='glass'>
                <h3>Prediction Result</h3>
                <h2>{emoji_map.get(pred_class, '❓')} {pred_class}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.write("")
        # Build probability table
        prob_df = pd.DataFrame({
            "Class": class_labels,
            "Probability": proba
        })

        st.bar_chart(prob_df.set_index("Class"))

    except AttributeError:
        st.error("Loaded model does not support predict_proba. Ensure it is LogisticRegression or compatible.")
    except Exception as e:
        st.error(f"Prediction error: {e}")


# Footer
st.markdown(
    """
    <div style='text-align:center; opacity:.7; margin-top:6rem;'>
      <small>© 2025 Fruit Freshness </small>
    </div>
    """,
    unsafe_allow_html=True
)
