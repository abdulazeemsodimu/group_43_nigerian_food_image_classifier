import os
import json
import traceback
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Naija Food AI",
    page_icon="🇳🇬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CONSTANTS ─────────────────────────────────────────────────
MODELS_DIR = "models"
MODEL_FILE  = "EfficientNetB0_model.keras"
CLASS_JSON  = os.path.join(MODELS_DIR, "class_info.json")
IMG_SIZE    = (224, 224)
TOP_K       = 5

FOOD_EMOJIS = {
    "jollof rice": "🍚", "fried rice":  "🍛", "egusi soup":  "🥘",
    "suya":        "🥩", "moi moi":     "🫕", "puff puff":   "🍩",
    "pepper soup": "🍜", "asaro":       "🍠", "akara":       "🫘",
    "banga soup":  "🍲",
}

def get_emoji(name: str) -> str:
    return FOOD_EMOJIS.get(name.strip().lower(), "🍽️")

# ── STYLING ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: #0a0f1e;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,135,81,0.25) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(14,165,233,0.18) 0%, transparent 60%);
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03) !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.hero {
    padding: 2.5rem 2rem; border-radius: 20px;
    background: linear-gradient(135deg, #008751 0%, #006e42 40%, #0ea5e9 100%);
    color: white; text-align: center; margin-bottom: 1.5rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: 2.2rem;
    font-weight: 800; margin: 0 0 0.4rem; letter-spacing: -0.5px;
}
.hero p { font-size: 1rem; opacity: 0.88; margin: 0; }

.result-card {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(20px);
    padding: 1.8rem; border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.10);
}
.food-name {
    font-family: 'Syne', sans-serif; font-size: 1.9rem;
    font-weight: 800; color: #f1f5f9; margin: 0.3rem 0 1rem; line-height: 1.2;
}
.conf-label {
    font-size: 0.78rem; letter-spacing: 0.1em;
    text-transform: uppercase; color: #94a3b8; margin-bottom: 0.3rem;
}
.conf-value {
    font-family: 'Syne', sans-serif; font-size: 2.4rem;
    font-weight: 700; color: #4ade80; line-height: 1;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03);
    border: 2px dashed rgba(255,255,255,0.15);
    border-radius: 16px; padding: 1rem;
}
[data-testid="stFileUploader"] * { color: #cbd5e1 !important; }
[data-testid="stMetricValue"] { color: #4ade80 !important; font-family: 'Syne', sans-serif; }
.stMarkdown, p, span, label { color: #cbd5e1; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
header    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── CACHED LOADERS ────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    path = os.path.join(MODELS_DIR, MODEL_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return tf.keras.models.load_model(path, compile=False)

@st.cache_data(show_spinner=False)
def load_classes():
    if not os.path.exists(CLASS_JSON):
        raise FileNotFoundError(f"class_info.json not found: {CLASS_JSON}")
    with open(CLASS_JSON) as f:
        return json.load(f)["class_names"]

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)          # scale to [-1, 1]
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

# ── SESSION STATE INIT ────────────────────────────────────────
if "image"  not in st.session_state: st.session_state.image  = None
if "pred"   not in st.session_state: st.session_state.pred   = None
if "error"  not in st.session_state: st.session_state.error  = None

# ── LOAD CLASSES ──────────────────────────────────────────────
try:
    classes = load_classes()
except Exception as exc:
    st.error(f"❌ Could not load class list: {exc}")
    st.stop()

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='font-family:Syne,sans-serif;color:#4ade80;margin-bottom:0'>"
        "🇳🇬 Naija Food AI</h2>",
        unsafe_allow_html=True,
    )
    st.caption("EfficientNetB0 · Deep Learning Classifier")
    st.divider()
    st.markdown("**🍽️ Recognised Foods**")
    for food in sorted(classes):
        st.markdown(f"&nbsp;&nbsp;{get_emoji(food)} {food.title()}")
    st.divider()
    st.info(f"Model knows **{len(classes)}** Nigerian food classes.")

    if st.button("🔄 Upload New Image", use_container_width=True):
        st.session_state.image = None
        st.session_state.pred  = None
        st.session_state.error = None

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🍲 Nigerian Food Recognition AI</h1>
    <p>Upload a photo — instant identification powered by EfficientNetB0</p>
</div>
""", unsafe_allow_html=True)

# ── FILE UPLOADER ─────────────────────────────────────────────
# Always render the uploader so Streamlit tracks widget state.
# Hide it visually once an image is already loaded.
uploaded = st.file_uploader(
    "Drop your food photo here",
    type=["jpg", "jpeg", "png"],
    label_visibility="visible" if st.session_state.image is None else "collapsed",
)

# New file uploaded → read bytes immediately into session state
if uploaded is not None:
    try:
        raw  = uploaded.read()
        img  = Image.open(BytesIO(raw)).copy()
        st.session_state.image = img
        st.session_state.pred  = None
        st.session_state.error = None
    except Exception as exc:
        st.error(f"❌ Could not open image: {exc}")
        st.stop()

# ── PREDICTION ────────────────────────────────────────────────
if st.session_state.image is not None and \
   st.session_state.pred  is None and \
   st.session_state.error is None:

    with st.spinner("Analysing your food image…"):
        try:
            model  = load_model()
            tensor = preprocess(st.session_state.image)
            output = model.predict(tensor, verbose=0)
            # Store as plain Python list — avoids numpy serialisation issues
            st.session_state.pred = output[0].tolist()
        except Exception:
            st.session_state.error = traceback.format_exc()

# ── ERROR DISPLAY ─────────────────────────────────────────────
if st.session_state.error:
    st.error("❌ Prediction failed. See traceback below:")
    st.code(st.session_state.error, language="python")
    if st.button("Try another image"):
        st.session_state.image = None
        st.session_state.error = None
    st.stop()

# ── RESULTS ───────────────────────────────────────────────────
if st.session_state.image is not None and st.session_state.pred is not None:
    image = st.session_state.image
    pred  = np.array(st.session_state.pred)
    idx   = int(np.argmax(pred))
    conf  = float(pred[idx])
    food  = classes[idx]

    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image, use_container_width=True, caption="Uploaded image")

    with col_info:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### {get_emoji(food)}")
        st.markdown(f'<p class="food-name">{food.title()}</p>', unsafe_allow_html=True)
        st.divider()
        st.markdown('<p class="conf-label">Confidence</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="conf-value">{conf*100:.1f}%</p>', unsafe_allow_html=True)
        st.progress(float(conf))
        st.divider()
        if conf >= 0.80:
            st.success("High confidence prediction ✅")
        elif conf >= 0.50:
            st.warning("Moderate confidence — result may vary ⚠️")
        else:
            st.error("Low confidence — try a clearer photo 📷")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── TOP-K CHART ───────────────────────────────────────────
    st.markdown("### 📊 Top Predictions")
    top_idx = np.argsort(pred)[::-1][:TOP_K]
    labels  = [f"{get_emoji(classes[i])} {classes[i].title()}" for i in top_idx]
    values  = [round(float(pred[i]) * 100, 2) for i in top_idx]
    colors  = ["#008751" if i == 0 else "#0ea5e9" for i in range(len(top_idx))]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.1)", width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=13),
    ))
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=60, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        xaxis=dict(
            showgrid=True, gridcolor="rgba(255,255,255,0.07)",
            tickfont=dict(color="#64748b"),
            range=[0, max(values) * 1.18],
            title=dict(text="Confidence (%)", font=dict(color="#64748b")),
        ),
        yaxis=dict(tickfont=dict(color="#e2e8f0", size=13), autorange="reversed"),
        font=dict(family="DM Sans, sans-serif"),
    )
    st.plotly_chart(fig, use_container_width=True)

elif st.session_state.image is None:
    st.markdown(
        "<p style='text-align:center;color:#64748b;margin-top:1rem;'>"
        "Supported formats: JPG · JPEG · PNG</p>",
        unsafe_allow_html=True,
    )
