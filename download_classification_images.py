import os
import requests
import pandas as pd
from urllib.parse import urlparse

# Load classification dataset
df = pd.read_csv("classification_dataset.csv")

# Load image mapping
images = pd.read_csv("images.csv")

# Create filename column from product id
df["filename"] = df["id"].astype(str) + ".jpg"

# Merge with image URLs
df = df.merge(images, on="filename", how="left")

# Main image folder
os.makedirs("classification_images", exist_ok=True)

downloaded = 0
failed = 0

for _, row in df.iterrows():

    category = row["articleType"]
    filename = row["filename"]
    url = row["link"]

    if pd.isna(url):
        failed += 1
        continue

    category_folder = os.path.join(
        "classification_images",
        category
    )

    os.makedirs(category_folder, exist_ok=True)

    save_path = os.path.join(
        category_folder,
        filename
    )

    try:
        response = requests.get(
            url,
            timeout=15
        )

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)

            downloaded += 1

        else:
            failed += 1

    except Exception:
        failed += 1

print("==============================")
print("IMAGE DOWNLOAD COMPLETED")
print("==============================")
print("Downloaded:", downloaded)
print("Failed:", failed)