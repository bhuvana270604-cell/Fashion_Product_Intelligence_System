import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# LOAD IMAGE FEATURES
# ==========================================

features = np.load("image_features.npy")

metadata = pd.read_csv("image_feature_metadata.csv")

print("Features shape:", features.shape)
print("Metadata shape:", metadata.shape)


# ==========================================
# COSINE SIMILARITY FUNCTION
# ==========================================

def find_similar_products(query_index, top_n=5):

    # Query image feature
    query_feature = features[query_index].reshape(1, -1)

    # Compare query image with all images
    similarity_scores = cosine_similarity(
        query_feature,
        features
    )[0]

    # Sort from highest similarity to lowest
    similar_indices = np.argsort(similarity_scores)[::-1]

    # Remove the query image itself
    similar_indices = [
        index for index in similar_indices
        if index != query_index
    ]

    # Select Top-N
    top_indices = similar_indices[:top_n]

    # Create result
    results = metadata.iloc[top_indices].copy()

    results["similarity_score"] = similarity_scores[top_indices]

    return results


# ==========================================
# TEST RECOMMENDATION
# ==========================================

query_index = 0

recommendations = find_similar_products(
    query_index,
    top_n=5
)

print("\n==============================")
print("TOP 5 SIMILAR PRODUCTS")
print("==============================")

print(recommendations)