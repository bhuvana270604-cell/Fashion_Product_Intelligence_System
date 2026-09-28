import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="Fashion Product Intelligence",
    page_icon="👕",
    layout="wide"
)

# ==============================
# MODEL
# ==============================

MODEL_PATH = "fashion_classifier_mobilenetv2.keras"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = [
    "Casual Shoes",
    "Shirts",
    "Socks",
    "Sports Shoes",
    "Tshirts"
]

# ==============================
# TITLE
# ==============================

st.title("👕 Fashion Product Intelligence System")

st.write(
    "Upload a fashion product image to identify its product category."
)

st.divider()

# ==============================
# IMAGE UPLOAD
# ==============================

uploaded_file = st.file_uploader(
    "Upload a fashion product image",
    type=["jpg", "jpeg", "png"]
)

# ==============================
# PREDICTION
# ==============================

if uploaded_file is not None:

    image = tf.keras.utils.load_img(
        uploaded_file,
        target_size=(224, 224)
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    predicted_class = class_names[predicted_index]

    confidence = predictions[0][predicted_index] * 100

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            uploaded_file,
            caption="Uploaded Product",
            use_container_width=True
        )

    with col2:
        st.subheader("Prediction")

        st.success(
            f"Product Category: {predicted_class}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

st.divider()

st.caption(
    "Powered by MobileNetV2 Transfer Learning"
)