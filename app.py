from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from inference import load_model


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "convnext_tiny.pth"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No tumor",
    "pituitary": "Pituitary tumor",
}

st.set_page_config(
    page_title="NeuroScan AI | MRI classifier",
    page_icon="🧠",
    layout="centered",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 900px; padding-top: 2.1rem;}
      .hero {padding: 1.15rem 1.35rem; border: 1px solid rgba(120,130,150,.25);
        border-radius: 18px; background: linear-gradient(120deg, rgba(72,110,245,.10), rgba(44,190,170,.08));}
      .hero h1 {margin: 0 0 .35rem 0;}
      .result {padding: 1rem 1.2rem; border-radius: 14px; background: rgba(58,120,95,.12);
        border-left: 5px solid #36a879; margin: .7rem 0 1rem 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the ConvNeXt-Tiny model…")
def get_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes, image_size = load_model(MODEL_PATH, device)
    return model, classes, image_size, device


def preprocess(image: Image.Image, size: int) -> torch.Tensor:
    pipeline = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    return pipeline(image).unsqueeze(0)


st.markdown(
    """
    <div class="hero">
      <h1>🧠 NeuroScan AI</h1>
      <div>Brain MRI research classifier · 4 classes</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("Upload an MRI image to see the model's predicted class and probability scores.")

with st.sidebar:
    st.subheader("Model")
    st.write("**ConvNeXt-Tiny** — the strongest of the three trained models in the current evaluation.")
    st.metric("Held-out test accuracy", "93.16%")
    st.caption("The same 2,414-image test set was used for all three models, so this comparison is exploratory.")
    st.divider()
    st.caption("Accepted image types: JPG, JPEG, PNG, BMP, TIFF, WEBP")

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"],
    help="Images are processed in memory by this app and are not written to disk.",
)

if uploaded_file is not None:
    try:
        image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("I couldn't read that file as an image. Try a JPG or PNG MRI image.")
    else:
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        try:
            model, classes, image_size, device = get_model()
            tensor = preprocess(image, image_size).to(device)
            with st.spinner("Analyzing image…"):
                with torch.inference_mode():
                    probabilities = model(tensor).softmax(dim=1)[0].cpu()
            order = torch.argsort(probabilities, descending=True).tolist()
            predicted_index = order[0]
            predicted_class = classes[predicted_index]
            confidence = float(probabilities[predicted_index])

            st.markdown(
                f'<div class="result"><strong>Predicted class: {DISPLAY_NAMES.get(predicted_class, predicted_class)}</strong>'
                f'<br>Model score: {confidence:.1%}</div>',
                unsafe_allow_html=True,
            )
            chart_data = pd.DataFrame(
                {
                    "Class": [DISPLAY_NAMES.get(classes[index], classes[index]) for index in order],
                    "Probability": [float(probabilities[index]) for index in order],
                }
            ).set_index("Class")
            st.subheader("Class probability scores")
            st.bar_chart(chart_data, x_label="Class", y_label="Probability", height=260)
            st.dataframe(
                chart_data.style.format({"Probability": "{:.1%}"}),
                use_container_width=True,
            )
            st.caption(f"Inference device: {device}")
        except FileNotFoundError:
            st.error(f"Model checkpoint not found at `{MODEL_PATH}`. Keep the `models` folder beside `app.py`.")
        except Exception as exc:
            st.error(f"Could not run the classifier: {exc}")

st.warning(
    "Research use only. This model is not clinically validated and must not be used to diagnose,"
    " rule out, or guide treatment for a brain tumor. A model score is not a medical certainty."
)
