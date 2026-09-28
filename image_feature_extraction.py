import os
import numpy as np
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ==============================
# SETTINGS
# ==============================

CSV_FILE = "men_fashion_products.csv"

OUTPUT_FEATURES = "image_features.npy"
OUTPUT_METADATA = "image_feature_metadata.csv"

IMAGE_SIZE = (224, 224)

# Start with a small number for testing
MAX_IMAGES = 22160


# ==============================
# LOAD DATA
# ==============================

print("Loading dataset...")
df = pd.read_csv(CSV_FILE)

print("Total products:", len(df))


# ==============================
# CHECK IMAGE URL
# ==============================

df = df.dropna(subset=["imageURL"])

print("Products with image URLs:", len(df))


# ==============================
# LOAD MOBILENETV2
# ==============================

print("Loading MobileNetV2...")

model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

print("MobileNetV2 loaded successfully!")


# ==============================
# FEATURE EXTRACTION
# ==============================

features = []
metadata = []

processed = 0

print("Starting image feature extraction...")


for index, row in df.iterrows():

    if processed >= MAX_IMAGES:
        break

    try:

        image_url = row["imageURL"]

        response = requests.get(
            image_url,
            timeout=10
        )

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        image = image.resize(IMAGE_SIZE)

        image_array = np.array(image)

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        image_array = preprocess_input(
            image_array
        )

        feature = model.predict(
            image_array,
            verbose=0
        )

        features.append(
            feature[0]
        )

        metadata.append(
            row.to_dict()
        )

        processed += 1

        if processed % 10 == 0:
            print(
                f"Processed {processed} images..."
            )

    except Exception as e:

        print(
            f"Error processing product {row['id']}: {e}"
        )


# ==============================
# SAVE FEATURES
# ==============================

if len(features) == 0:

    print("No images were processed.")
    print("Please check the image URLs.")

else:

    features = np.array(features)

    metadata_df = pd.DataFrame(metadata)

    np.save(
        OUTPUT_FEATURES,
        features
    )

    metadata_df.to_csv(
        OUTPUT_METADATA,
        index=False
    )

    print()
    print("==============================")
    print("FEATURE EXTRACTION COMPLETED")
    print("==============================")
    print("Images processed:", len(features))
    print("Feature shape:", features.shape)
    print("Saved:", OUTPUT_FEATURES)
    print("Saved:", OUTPUT_METADATA)