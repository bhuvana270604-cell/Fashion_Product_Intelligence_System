import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD FEATURES AND METADATA
# ==============================

FEATURE_FILE = "image_features.npy"
METADATA_FILE = "image_feature_metadata.csv"

print("Loading image features...")

features = np.load(FEATURE_FILE)
metadata = pd.read_csv(METADATA_FILE)

print("Feature shape:", features.shape)
print("Metadata shape:", metadata.shape)


# ==============================
# SELECT PRODUCT
# ==============================

product_index = 0

query_feature = features[product_index].reshape(1, -1)


# ==============================
# CALCULATE SIMILARITY
# ==============================

similarity_scores = cosine_similarity(
    query_feature,
    features
)[0]


# ==============================
# GET TOP 5 SIMILAR PRODUCTS
# ==============================

similarity_indices = similarity_scores.argsort()[::-1]

# Remove the selected product itself
similarity_indices = [
    i for i in similarity_indices
    if i != product_index
]

top_indices = similarity_indices[:5]


# ==============================
# DISPLAY RESULTS
# ==============================

print()
print("==============================")
print("SIMILAR PRODUCTS")
print("==============================")

for rank, index in enumerate(top_indices, start=1):

    product = metadata.iloc[index]

    print()
    print("Rank:", rank)
    print("Product ID:", product["id"])
    print("Product Name:", product["productDisplayName"])
    print("Brand:", product["brandName"])
    print("Article Type:", product["articleType"])
    print("Colour:", product["baseColour"])
    print("Similarity Score:", round(similarity_scores[index], 4))
    print("Image URL:", product["imageURL"])