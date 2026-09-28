import os
import json
import pandas as pd

json_folder = os.path.join("styles", "styles")

print("Starting JSON processing...")
print("Folder:", os.path.abspath(json_folder))

files = [f for f in os.listdir(json_folder) if f.endswith(".json")]

print("JSON files found:", len(files))

records = []

for i, file_name in enumerate(files, start=1):

    file_path = os.path.join(json_folder, file_name)

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            json_data = json.load(file)

        # Actual product data is inside "data"
        data = json_data.get("data", {})

        record = {
            "id": data.get("id"),
            "productDisplayName": data.get("productDisplayName"),
            "brandName": data.get("brandName"),
            "gender": data.get("gender"),
            "masterCategory": data.get("masterCategory", {}).get("typeName"),
            "subCategory": data.get("subCategory", {}).get("typeName"),
            "articleType": data.get("articleType", {}).get("typeName"),
            "baseColour": data.get("baseColour"),
            "usage": data.get("usage"),
            "season": data.get("season"),
            "year": data.get("year")
        }

        # Get default image URL
        style_images = data.get("styleImages", {})

        image_url = None

        if isinstance(style_images, dict):
            default_image = style_images.get("default", {})

            if isinstance(default_image, dict):
                image_url = default_image.get("imageURL")

        record["imageURL"] = image_url

        records.append(record)

    except Exception as e:
        print("Error:", file_name, e)

    if i % 1000 == 0:
        print(f"Processed {i} files...")

# Create DataFrame
df = pd.DataFrame(records)

# Save CSV
df.to_csv("fashion_products.csv", index=False)

print()
print("===================================")
print("JSON files processed:", len(records))
print("CSV created successfully!")
print("CSV rows:", len(df))
print("===================================")

print(df.head())