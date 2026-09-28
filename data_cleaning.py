import pandas as pd

# Load flattened dataset
df = pd.read_csv("fashion_products.csv")

print("Original shape:", df.shape)

print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate products
df = df.drop_duplicates(subset="id")

# Remove records without image URL
df = df.dropna(subset=["imageURL"])

# Fill missing categorical values
categorical_columns = [
    "brandName",
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "usage",
    "season"
]

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown")

# Fill missing year
df["year"] = df["year"].fillna(0)

# Save cleaned dataset
df.to_csv("fashion_products_cleaned.csv", index=False)

print("\nCleaned shape:", df.shape)

print("\nCleaning completed successfully!")

print("\nFirst 5 rows:")
print(df.head())