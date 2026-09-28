import pandas as pd

# Load cleaned dataset
df = pd.read_csv("fashion_products_cleaned.csv")

print("Dataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nGender Distribution:")
print(df["gender"].value_counts())

print("\nMaster Category:")
print(df["masterCategory"].value_counts())

print("\nSub Category:")
print(df["subCategory"].value_counts().head(20))

print("\nArticle Type:")
print(df["articleType"].value_counts().head(20))

print("\nBase Colour:")
print(df["baseColour"].value_counts())

print("\nUsage:")
print(df["usage"].value_counts())

print("\nSeason:")
print(df["season"].value_counts())

print("\nYear:")
print(df["year"].value_counts().sort_index())

print("\nTop 20 Brands:")
print(df["brandName"].value_counts().head(20))