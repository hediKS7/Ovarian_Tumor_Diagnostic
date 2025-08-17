
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from PIL import Image
import io
from streamlit.components.v1 import html as components_html

# Optional libs for explainability
_SHAP_AVAILABLE = True
try:
    import shap  # model-agnostic SHAP (KernelExplainer)
except Exception:
    _SHAP_AVAILABLE = False

# matplotlib for Grad-CAM coloring (no figure shown to user; we convert to PIL)
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm

# -------------------------
# Helper: convert PIL image to base64 (for inline preview)
# -------------------------
def _pil_image_to_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    import base64
    return base64.b64encode(img_bytes).decode('utf-8')

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="SeleneX — Modern UI", page_icon="🩺", layout="wide")

# -------------------------
# Styles + animations (injected as raw HTML/CSS)
# -------------------------
CUSTOM_CSS = r"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<style>
:root{
  --bg1: #0f172a; /* deep navy */
  --bg2: #06122e; /* darker */
  --card: rgba(255,255,255,0.04);
  --accent: linear-gradient(90deg,#7c3aed,#06b6d4);
  --glass: rgba(255,255,255,0.03);
}
html,body,[data-testid='stAppViewContainer']{
  height:100%;
  background: radial-gradient(1200px 800px at 10% 10%, rgba(99,102,241,0.12), transparent 8%),
              radial-gradient(1000px 600px at 90% 90%, rgba(6,182,212,0.07), transparent 10%),
              linear-gradient(180deg,var(--bg1),var(--bg2));
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
  color: #e6eef8;
}

/* Card */
.stream-card{
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border-radius: 14px;
  padding: 18px;
  box-shadow: 0 6px 30px rgba(2,6,23,0.6);
  border: 1px solid rgba(255,255,255,0.03);
}
.header-row{ display:flex; align-items:center; gap:18px; }
.app-title{ font-size:1.6rem; font-weight:700; margin:0 }
.app-sub{ color: #bcd6f7; margin:0; font-size:0.95rem }

/* Fancy predict button */
.stButton>button{
  background: linear-gradient(90deg,#7c3aed,#06b6d4);
  color:white; padding:10px 18px; border-radius:10px; border: none; font-weight:600;
  box-shadow: 0 6px 18px rgba(7,10,60,0.35);
}
.stButton>button:hover{ transform: translateY(-2px); }

/* Floating blobs */
.floating-blob{
  position: absolute; border-radius: 50%; filter: blur(36px); opacity:0.65; z-index:0;
}
.blob-a{ width:360px; height:360px; left:-120px; top:-80px; background: linear-gradient(135deg,#7c3aed,#06b6d4); }
.blob-b{ width:240px; height:240px; right:-80px; bottom:-60px; background: linear-gradient(135deg,#ef4444,#f97316); }

@keyframes floaty { 0%{ transform: translateY(0px) } 50%{ transform: translateY(-18px) } 100%{ transform: translateY(0px) } }
.icon-float{ animation: floaty 6s ease-in-out infinite; }

/* Input image preview */
.preview-img{ max-width:100%; border-radius:10px; border:1px solid rgba(255,255,255,0.06); }

/* small responsive tweaks */
@media (max-width:800px){ .header-row{ flex-direction:column; align-items:flex-start } }

/* small translucent badges */
.badge { background: rgba(255,255,255,0.03); padding:6px 10px; border-radius:999px; display:inline-block; font-weight:600; }
</style>

<!-- floating blobs markup -->
<div class="floating-blob blob-a" aria-hidden></div>
<div class="floating-blob blob-b" aria-hidden></div>
"""
components_html(CUSTOM_CSS, height=0)

# -------------------------
# Load fused model
# -------------------------
@st.cache_resource
def load_fused_model():
    try:
        model = load_model("fused_model.h5")
        return model
    except Exception:
        return None

model = load_fused_model()

# -------------------------
# Preprocessing functions
# -------------------------
IMG_SIZE = (224, 224)

def preprocess_image(img: Image.Image):
    img = img.resize(IMG_SIZE)
    img = img.convert("RGB")
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def preprocess_tabular(age, ca125, brca):
    tab = np.array([[age, ca125, brca]], dtype=np.float32)
    return tab

# -------------------------
# Grad-CAM utilities
# -------------------------
def _find_last_conv_layer(m: tf.keras.Model):
    # Find last Conv2D-like layer in the graph (handles nested models)
    last_conv = None
    for layer in m.layers[::-1]:
        try:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer
                break
        except Exception:
            pass
    return last_conv

def make_gradcam_heatmap(model, X_img, X_tab, conv_layer=None):
    """
    Compute Grad-CAM heatmap for the fused model.
    - model: Keras model with two inputs [image, tabular] and scalar output.
    - X_img: shape (1, H, W, 3)
    - X_tab: shape (1, 3)
    """
    if conv_layer is None:
        conv_layer = _find_last_conv_layer(model)
    if conv_layer is None:
        raise RuntimeError("No Conv2D layer found for Grad-CAM.")

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([X_img, X_tab])
        # If model output is shape (1,1), select scalar
        pred = predictions[:, 0]

    grads = tape.gradient(pred, conv_outputs)  # shape (1, Hc, Wc, C)

    # Global average pooling on the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

    # Weight conv feature maps
    conv_outputs = conv_outputs[0]  # (Hc, Wc, C)
    heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)

    # Apply ReLU and normalize to [0,1]
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap) + 1e-8
    heatmap = heatmap / max_val
    heatmap = heatmap.numpy()
    return heatmap  # (Hc, Wc)

def overlay_heatmap_on_image(pil_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.35) -> Image.Image:
    # Resize heatmap to original image size
    heatmap_resized = Image.fromarray(np.uint8(cm.jet(heatmap)[:, :, :3] * 255)).resize(pil_img.size)
    heatmap_resized = np.array(heatmap_resized).astype(np.float32) / 255.0
    base = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    overlay = (1 - alpha) * base + alpha * heatmap_resized
    overlay = np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)

# -------------------------
# SHAP utilities (KernelExplainer on tabular features)
# -------------------------
def compute_tabular_shap(model, X_img, x_tab, background_tab=None, nsamples=200):
    """
    Model-agnostic SHAP for the 3 tabular features given the current image.
    We keep the image fixed and explain variation due to tabular features.

    - model: fused model
    - X_img: (1, H, W, 3) fixed
    - x_tab: (1, 3) instance to explain
    - background_tab: (B, 3) background for KernelExplainer
    """
    if not _SHAP_AVAILABLE:
        raise RuntimeError("shap is not installed")

    if background_tab is None:
        # Simple background set around typical ranges
        # age [25, 45, 65], ca125 [20, 100, 400], brca [0, 1]
        background_tab = np.array([
            [45, 35.0, 0],
            [55, 100.0, 0],
            [35, 20.0, 0],
            [60, 400.0, 0],
            [45, 35.0, 1],
        ], dtype=np.float32)

    def f_tab(T):
        # T: (n, 3)
        T = np.array(T, dtype=np.float32)
        imgs = np.repeat(X_img, T.shape[0], axis=0)
        preds = model.predict([imgs, T], verbose=0)
        return preds.reshape(-1)

    explainer = shap.KernelExplainer(f_tab, background_tab, link="identity")
    shap_vals = explainer.shap_values(x_tab, nsamples=nsamples)
    # KernelExplainer returns 1D array for regression-like output
    shap_vals = np.array(shap_vals).reshape(-1)  # (3,)
    base = explainer.expected_value
    if isinstance(base, (list, tuple, np.ndarray)):
        base = np.array(base).reshape(-1)[0]
    return shap_vals, float(base)

# -------------------------
# Header
# -------------------------
header_html = '''
<div class="stream-card" style="display:flex;align-items:center;gap:16px;">
  <div style="width:84px;height:84px;border-radius:14px;background:linear-gradient(135deg,#0ea5a2,#7c3aed);display:flex;align-items:center;justify-content:center;">
    <i class="fa-solid fa-stethoscope" style="font-size:36px;color:white;"></i>
  </div>
  <div>
    <div class="app-title">SeleneX — Ovarian Tumor Diagnostic</div>
    <div class="app-sub">Upload an ultrasound image + biomarkers to predict benign vs malignant</div>
  </div>
  <div style="margin-left:auto;display:flex;gap:10px;align-items:center">
    <div class="badge">AI Prototype</div>
    <div class="badge">Explainable</div>
  </div>
</div>
'''
st.markdown(header_html, unsafe_allow_html=True)
st.write("")

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="stream-card">', unsafe_allow_html=True)
    uploaded_img = st.file_uploader(
        "Upload Ultrasound Image",
        type=["jpg", "jpeg", "png"],
        help="High-quality transverse/longitudinal ultrasound preferred"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="stream-card">', unsafe_allow_html=True)
    age = st.number_input("Age", min_value=18, max_value=100, value=45, step=1)

    # CA-125 with step = 0.01
    ca125 = st.number_input(
        "CA-125 Level (U/mL)",
        min_value=0.0,
        max_value=5000.0,
        value=50.0,
        step=0.01
    )

    # BRCA: Positive / Negative mapped to 1 / 0
    brca_choice = st.selectbox(
        "BRCA Mutation Status",
        options=["Negative", "Positive"],
        index=0,
        help="Select BRCA mutation status"
    )
    brca = 1 if brca_choice == "Positive" else 0
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    predict_clicked = st.button("🔍 Predict")

with col2:
    st.markdown('<div class="stream-card">', unsafe_allow_html=True)
    st.subheader("Preview")
    if uploaded_img is not None:
        try:
            image = Image.open(uploaded_img)
            st.image(image, caption="Uploaded Ultrasound", use_column_width=True, output_format="PNG")
        except Exception:
            st.error("Unable to open the uploaded image. Make sure file is an image.")
    else:
        placeholder = """
        <div style='display:flex;flex-direction:column;align-items:center;justify-content:center;padding:18px'>
          <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
          <lottie-player src="https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json"  background="transparent"  speed="1"  style="width:280px; height:280px;"  loop  autoplay></lottie-player>
          <div style='color:#cfe8ff;margin-top:8px;font-weight:600'>Drop an ultrasound image to begin</div>
        </div>
        """
        components_html(placeholder, height=360)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    st.markdown('<div class="stream-card">', unsafe_allow_html=True)

# -------------------------
# Predict + Explain
# -------------------------
if predict_clicked:
    if model is None:
        st.error("Model not loaded. Please check your fused_model.h5 file.")
    elif uploaded_img is None:
        st.warning("Please upload an ultrasound image before predicting.")
    else:
        try:
            image = Image.open(uploaded_img)
            X_img = preprocess_image(image)
            X_tab = preprocess_tabular(age, ca125, brca)

            with st.spinner('Running prediction — analyzing image and biomarkers...'):
                components_html(
                    '<div style="height:6px"></div>'
                    '<div style="width:100%;height:8px;background:rgba(255,255,255,0.03);border-radius:999px;overflow:hidden;">'
                    '<div style="width:70%;height:100%;background:linear-gradient(90deg,#7c3aed,#06b6d4);animation:progress 1.2s ease-in-out;"></div>'
                    '</div>'
                    '<style>@keyframes progress{0%{width:0%}100%{width:70%}}</style>',
                    height=30
                )
                prob = float(model.predict([X_img, X_tab], verbose=0)[0][0])
            
            pred = "Malignant" if prob > 0.5 else "Benign"

            # Result card
            result_html = f"""
            <div class='stream-card' style='display:flex;gap:18px;align-items:center'>
              <div style='flex:1'>
                <div style='font-size:1.2rem;font-weight:700'>{pred}</div>
                <div style='margin-top:6px'>Probability of malignancy: <span style='font-weight:800'>{prob:.2f}</span></div>
                <div style='margin-top:10px;color:#bcd6f7'>This is a prototype — clinical decisions require specialist review.</div>
              </div>
              <div style='width:110px;height:110px;border-radius:10px;overflow:hidden;border:1px solid rgba(255,255,255,0.04)'>
                <img src='data:image/png;base64,{_pil_image_to_base64(image.resize((110,110)))}' style='width:100%;height:100%;object-fit:cover'>
              </div>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)

            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            bar_html = f"""
            <div style='background:rgba(255,255,255,0.03);border-radius:999px;padding:10px'>
              <div style='font-weight:700;margin-bottom:6px'>Malignancy score</div>
              <div style='width:100%;height:14px;background:rgba(255,255,255,0.02);border-radius:999px;overflow:hidden'>
                <div style='height:100%;width:{prob*100:.1f}%;background:linear-gradient(90deg,#ef4444,#7c3aed);'></div>
              </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)

            # ---------- Grad-CAM ----------
            try:
                heatmap = make_gradcam_heatmap(model, X_img, X_tab, conv_layer=None)
                gradcam_img = overlay_heatmap_on_image(image.resize(IMG_SIZE), heatmap, alpha=0.35)
                st.markdown('<div style="height:14px"></div>', unsafe_allow_html=True)
                st.markdown("### Grad-CAM (Image Explanation)")
                st.image(gradcam_img, caption="Grad-CAM overlay", use_column_width=True)
            except Exception as e:
                st.warning(f"Grad-CAM not available: {e}")

            # ---------- SHAP on tabular ----------
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            st.markdown("### Top Tabular SHAP Features")
            if _SHAP_AVAILABLE:
                try:
                    shap_vals, base = compute_tabular_shap(model, X_img, X_tab, background_tab=None, nsamples=200)
                    feature_names = ["Age", "CA-125 (U/mL)", "BRCA (1=Positive)"]
                    contributions = list(zip(feature_names, shap_vals, np.abs(shap_vals)))

                    # Sort by absolute impact
                    contributions.sort(key=lambda x: x[2], reverse=True)

                    # Show top 3
                    top3 = contributions[:3]
                    # Display as simple HTML bars (keeps style consistent)
                    shap_html = "<div class='stream-card'>"
                    shap_html += "<div style='font-weight:600;margin-bottom:8px'>Feature contributions (local SHAP)</div>"
                    for name, val, mag in top3:
                        width = min(100, max(0, float(mag) / (max(1e-6, max(c[2] for c in contributions))) * 100))
                        sign = "positive" if val >= 0 else "negative"
                        grad = "linear-gradient(90deg,#22c55e,#16a34a)" if val >= 0 else "linear-gradient(90deg,#ef4444,#dc2626)"
                        shap_html += f"""
                        <div style='margin:10px 0'>
                          <div style='display:flex;justify-content:space-between'>
                            <span>{name}</span>
                            <span style='opacity:0.8'>{val:+.4f}</span>
                          </div>
                          <div style='width:100%;height:10px;background:rgba(255,255,255,0.05);border-radius:999px;overflow:hidden'>
                            <div style='height:100%;width:{width:.1f}%;background:{grad};'></div>
                          </div>
                        </div>
                        """
                    shap_html += "<div style='opacity:0.8;margin-top:8px;font-size:0.9rem'>Explanation conditioned on the uploaded image; shows how tabular inputs push the probability.</div>"
                    shap_html += "</div>"
                    st.markdown(shap_html, unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"SHAP computation failed: {e}. Try installing/updating shap (e.g., pip install shap).")
            else:
                st.info("Install SHAP to see tabular attributions: `pip install shap`")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# small footer
st.markdown('<div style="height:26px"></div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;color:#9fb9e8;font-size:0.9rem">Prototype for research/education. Not for clinical use.</div>', unsafe_allow_html=True)
# End of file
