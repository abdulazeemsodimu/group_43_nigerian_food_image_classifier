import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import tensorflow as tf

st.set_page_config(
    page_title="Naija Food AI",
    page_icon="🇳🇬",
    layout="wide"
)

MODELS_DIR = "models"
MODEL_FILE = "EfficientNetB0_model.keras"
CLASS_JSON = os.path.join(MODELS_DIR, "class_info.json")
IMG_SIZE = (224, 224)
TOP_K = 5

FOOD_EMOJIS = {
    "jollof rice":"🍚","fried rice":"🍛","egusi soup":"🥘","suya":"🥩",
    "moi moi":"🫕","puff puff":"🍩","pepper soup":"🍜","asaro":"🍠"
}

def emoji(name):
    return FOOD_EMOJIS.get(name.lower(), "🍽️")

st.markdown("""
<style>
.stApp{
background:linear-gradient(135deg,#0b1220,#111827);
}
.hero{
padding:2rem;border-radius:24px;
background:linear-gradient(135deg,#008751,#0ea5e9);
color:white;text-align:center;margin-bottom:1rem;
}
.card{
background:rgba(255,255,255,.06);
backdrop-filter:blur(16px);
padding:1.5rem;border-radius:20px;
border:1px solid rgba(255,255,255,.12);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        os.path.join(MODELS_DIR, MODEL_FILE),
        compile=False,
        safe_mode=False
    )

@st.cache_data
def load_classes():
    with open(CLASS_JSON) as f:
        data = json.load(f)
    return data["class_names"]

def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

classes = load_classes()


with st.sidebar:
    st.title("🇳🇬 Naija Food AI")

    st.success("EfficientNetB0")

    st.write("Nigerian Food Image Classifier")

    st.markdown("---")

    st.subheader("🍽️ Available Classes")

    for food in sorted(classes):
        st.write(f"• {food.title()}")

    st.markdown("---")

    st.info(
        f"Model can recognize {len(classes)} Nigerian food classes."
    )

st.markdown("""
<div class="hero">
<h1>🍲 Nigerian Food Recognition AI</h1>
<p>Powered by EfficientNetB0 Deep Learning</p>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "Upload Food Image",
    type=["jpg","jpeg","png","webp"]
)

if uploaded:
    image = Image.open(uploaded)

    with st.spinner("Analyzing image..."):
        model = load_model()
        pred = model.predict(preprocess(image), verbose=0)[0]

    idx = int(np.argmax(pred))
    conf = float(pred[idx])
    food = classes[idx]

    c1, c2 = st.columns(2)

    with c1:
        st.image(image, use_container_width=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"# {emoji(food)} {food.title()}")
        st.metric("Confidence", f"{conf*100:.1f}%")
        st.progress(conf)
        st.markdown("</div>", unsafe_allow_html=True)

    top_idx = np.argsort(pred)[::-1][:TOP_K]

    labels = [classes[i] for i in top_idx]
    values = [float(pred[i])*100 for i in top_idx]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h"
        )
    )

    fig.update_layout(
        title="Top Predictions",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload a Nigerian food image to begin.")
