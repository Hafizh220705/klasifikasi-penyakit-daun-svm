import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision.models import ResNet50_Weights, resnet50

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "svm_pipeline.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

st.set_page_config(
    page_title="Potato Leaf Disease Classification",
    page_icon="🥔",
    layout="centered",
)


@st.cache_resource(show_spinner="Memuat model klasifikasi...")
def load_model_assets():
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError(
            "Artefak model belum tersedia. Jalankan seluruh notebook terlebih dahulu "
            "untuk membuat models/svm_pipeline.pkl dan models/model_metadata.json."
        )

    with METADATA_PATH.open("r", encoding="utf-8") as metadata_file:
        metadata = json.load(metadata_file)

    pipeline = joblib.load(MODEL_PATH)
    weights_name = metadata.get("resnet50_weights", "IMAGENET1K_V2")
    try:
        weights = ResNet50_Weights[weights_name]
    except KeyError as error:
        raise ValueError(
            f"ResNet50 weights pada metadata tidak dikenali: {weights_name}"
        ) from error

    preprocess = weights.transforms()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_extractor = resnet50(weights=weights)
    feature_extractor.fc = nn.Identity()
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    return pipeline, metadata, feature_extractor, preprocess, device


def predict_uploaded_image(
    image: Image.Image,
    pipeline,
    feature_extractor,
    preprocess,
    device,
) -> dict:
    rgb_image = image.convert("RGB")
    input_tensor = preprocess(rgb_image).unsqueeze(0).to(device)

    feature_extractor.eval()
    with torch.inference_mode():
        feature = feature_extractor(input_tensor).flatten(start_dim=1)

    feature_array = feature.cpu().numpy().astype(np.float32)
    prediction = pipeline.predict(feature_array)[0]
    probabilities = pipeline.predict_proba(feature_array)[0]
    probability_map = {
        class_name: float(probability)
        for class_name, probability in zip(pipeline.classes_, probabilities)
    }

    return {
        "label": prediction,
        "confidence": probability_map[prediction],
        "probabilities": probability_map,
    }


st.title("Potato Leaf Disease Classification")
st.caption(
    "Klasifikasi penyakit daun kentang menggunakan ResNet50 Feature Extraction "
    "dan Support Vector Machine"
)

try:
    pipeline, metadata, feature_extractor, preprocess, device = load_model_assets()
except (FileNotFoundError, ValueError, OSError) as error:
    st.error(str(error))
    st.info("Buka notebook.ipynb, lalu jalankan seluruh sel dari atas ke bawah.")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload gambar daun kentang",
    type=["jpg", "jpeg", "png"],
    help="Gunakan gambar yang jelas dan menampilkan daun sebagai objek utama.",
)

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file)
        uploaded_image.load()
        uploaded_image = uploaded_image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"File tidak dapat dibaca sebagai gambar: {error}")
        st.stop()

    st.image(uploaded_image, caption="Gambar yang di-upload", use_container_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Mengekstrak fitur dan membuat prediksi..."):
            try:
                result = predict_uploaded_image(
                    uploaded_image,
                    pipeline,
                    feature_extractor,
                    preprocess,
                    device,
                )
            except Exception as error:
                st.error(f"Prediksi gagal: {error}")
                st.stop()

        prediction_column, confidence_column = st.columns(2)
        prediction_column.metric("Prediction", result["label"])
        confidence_column.metric("Confidence", f"{result['confidence']:.2%}")

        st.subheader("Probabilitas per Kelas")
        probability_df = (
            pd.DataFrame(
                {
                    "Kelas": list(result["probabilities"].keys()),
                    "Probabilitas": list(result["probabilities"].values()),
                }
            )
            .sort_values("Probabilitas", ascending=False)
            .reset_index(drop=True)
        )
        st.bar_chart(probability_df.set_index("Kelas"), y="Probabilitas")

        for row in probability_df.itertuples(index=False):
            st.write(f"{row.Kelas}: {row.Probabilitas:.2%}")
            st.progress(int(round(row.Probabilitas * 100)))

with st.expander("Tentang Model"):
    st.markdown(
        """
        - ResNet50 pretrained ImageNet digunakan hanya sebagai feature extractor.
        - Support Vector Machine digunakan sebagai classifier akhir.
        - Model membedakan tiga kelas: Early Blight, Late Blight, dan Healthy.
        - Preprocessing inference sama dengan preprocessing saat ekstraksi fitur training.
        """
    )
    st.caption(f"Perangkat inference saat ini: {device}")

st.divider()
