import os
import json
import traceback

# ── TENSORFLOW MEMORY & CPU LIMITS (MUST BE BEFORE TF LOADS) ──
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress unnecessary warnings

import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Throttle TensorFlow threads to prevent Out-Of-Memory (OOM) crashes on deployment
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass


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
    font-weight: 800; margin: 0 0 0.4rem;
}
.hero p { font-size: 1rem; opacity: 0.88; margin: 0; }
.result-card {
    background: rgba(255,255,255,0.05); backdrop-filter: blur(20px);
    padding: 1.8rem; border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.10);
}
.food-name {
    font-family: 'Syne', sans-serif; font-size: 1.9rem;
    font-weight: 800; color: #f1f5f9; margin: 0.3rem 0 1rem;
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
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

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

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🍲 Nigerian Food Recognition AI</h1>
    <p>Upload a photo — instant identification powered by EfficientNetB0</p>
</div>
""", unsafe_allow_html=True)

# ── FILE UPLOADER ─────────────────────────────────────────────
uploaded = st.file_uploader(
    "Drop your food photo here",
    type=["jpg", "jpeg", "png"],
)

# ── MAIN LOGIC ────────────────────────────────────────────────
if uploaded is None:
    st.info("👆 Upload a Nigerian food image to get started.")
    st.stop()

# ── OPEN IMAGE ────────────────────────────────────────────────
try:
    image = Image.open(uploaded).convert("RGB")
except Exception as exc:
    st.error(f"❌ Could not open image: {exc}")
    st.stop()

# ── SHOW IMAGE IMMEDIATELY ────────────────────────────────────
col_img, col_info = st.columns([1, 1], gap="large")
with col_img:
    st.image(image, use_container_width=True, caption=uploaded.name)

# ── RUN PREDICTION ────────────────────────────────────────────
with col_info:
    try:
        with st.spinner("Analysing image…"):
            model  = load_model()
            tensor = preprocess(image)
            pred   = model.predict(tensor, verbose=0)[0]

        idx  = int(np.argmax(pred))
        conf = float(pred[idx])
        food = classes[idx]

        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown(f"### {get_emoji(food)}")
        st.markdown(f'<p class="food-name">{food.title()}</p>', unsafe_allow_html=True)
        st.divider()
        st.markdown('<p class="conf-label">Confidence</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="conf-value">{conf*100:.1f}%</p>', unsafe_allow_html=True)
        st.progress(float(conf))
        st.divider()
        if conf >= 0.80:
            st.success("High confidence ✅")
        elif conf >= 0.50:
            st.warning("Moderate confidence ⚠️")
        else:
            st.error("Low confidence — try a clearer photo 📷")
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception:
        st.error("❌ Prediction failed. Full error:")
        st.code(traceback.format_exc(), language="python")
        st.stop()

# ── TOP-K CHART ───────────────────────────────────────────────
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
