import pandas as pd

df = pd.read_csv("fashion_products_cleaned.csv")

men = df[df["gender"] == "Men"].copy()

print("Total Men products:", len(men))

print("\nMen products by Article Type:")
print(men["articleType"].value_counts().head(20))

print("\nMen products with images:", men["imageURL"].notna().sum())

print("\nMen products without images:", men["imageURL"].isna().sum())