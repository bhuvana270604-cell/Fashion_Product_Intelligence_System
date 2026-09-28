import os
import shutil
import random

SOURCE_DIR = "classification_images"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.8

categories = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Socks"
]

random.seed(42)

for category in categories:

    source_folder = os.path.join(SOURCE_DIR, category)

    train_folder = os.path.join(
        OUTPUT_DIR, "train", category
    )

    validation_folder = os.path.join(
        OUTPUT_DIR, "validation", category
    )

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(validation_folder, exist_ok=True)

    images = [
        file for file in os.listdir(source_folder)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    validation_images = images[split_index:]

    for image in train_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(train_folder, image)
        )

    for image in validation_images:
        shutil.copy2(
            os.path.join(source_folder, image),
            os.path.join(validation_folder, image)
        )

    print(
        category,
        "-> Train:",
        len(train_images),
        "| Validation:",
        len(validation_images)
    )

print("\n==============================")
print("DATASET SPLIT COMPLETED")
print("==============================")