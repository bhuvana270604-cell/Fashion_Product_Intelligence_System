import pandas as pd

df = pd.read_csv("image_feature_metadata.csv")

# Men category மட்டும்
df = df[df["gender"] == "Men"].copy()

# Classification-க்கு top 5 categories
selected_categories = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Socks"
]

df = df[df["articleType"].isin(selected_categories)].copy()

classification_df = df[
    ["id", "productDisplayName", "articleType", "imageURL"]
].copy()

classification_df.to_csv(
    "classification_dataset.csv",
    index=False
)

print("==============================")
print("CLASSIFICATION DATASET CREATED")
print("==============================")

print("Total images:", len(classification_df))

print("\nCategory distribution:")
print(classification_df["articleType"].value_counts())

print("\nSaved: classification_dataset.csv")