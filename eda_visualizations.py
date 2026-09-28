import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("fashion_products_cleaned.csv")

# 1. Gender Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="gender")
plt.title("Product Distribution by Gender")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 2. Master Category Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, y="masterCategory")
plt.title("Product Distribution by Master Category")
plt.tight_layout()
plt.show()


# 3. Top 15 Sub Categories
top_sub = df["subCategory"].value_counts().head(15)

plt.figure(figsize=(9, 6))
top_sub.sort_values().plot(kind="barh")
plt.title("Top 15 Sub Categories")
plt.xlabel("Number of Products")
plt.tight_layout()
plt.show()


# 4. Top 15 Article Types
top_article = df["articleType"].value_counts().head(15)

plt.figure(figsize=(9, 6))
top_article.sort_values().plot(kind="barh")
plt.title("Top 15 Article Types")
plt.xlabel("Number of Products")
plt.tight_layout()
plt.show()


# 5. Top 15 Colours
top_colors = df["baseColour"].value_counts().head(15)

plt.figure(figsize=(9, 6))
top_colors.sort_values().plot(kind="barh")
plt.title("Top 15 Product Colours")
plt.xlabel("Number of Products")
plt.tight_layout()
plt.show()


# 6. Usage Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="usage")
plt.title("Product Distribution by Usage")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 7. Season Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="season")
plt.title("Product Distribution by Season")
plt.tight_layout()
plt.show()


# 8. Year-wise Distribution
year_counts = df[df["year"] > 0]["year"].value_counts().sort_index()

plt.figure(figsize=(10, 5))
year_counts.plot(kind="line", marker="o")
plt.title("Year-wise Product Distribution")
plt.xlabel("Year")
plt.ylabel("Number of Products")
plt.grid(True)
plt.tight_layout()
plt.show()


# 9. Top 15 Brands
top_brands = df["brandName"].value_counts().head(15)

plt.figure(figsize=(9, 6))
top_brands.sort_values().plot(kind="barh")
plt.title("Top 15 Brands")
plt.xlabel("Number of Products")
plt.tight_layout()
plt.show()


# 10. Gender vs Master Category
gender_category = pd.crosstab(
    df["gender"],
    df["masterCategory"]
)

plt.figure(figsize=(10, 6))
gender_category.plot(kind="bar", figsize=(10, 6))
plt.title("Gender vs Master Category")
plt.xlabel("Gender")
plt.ylabel("Number of Products")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 11. Gender vs Usage
gender_usage = pd.crosstab(
    df["gender"],
    df["usage"]
)

gender_usage.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Gender vs Usage")
plt.xlabel("Gender")
plt.ylabel("Number of Products")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 12. Gender vs Season
gender_season = pd.crosstab(
    df["gender"],
    df["season"]
)

gender_season.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Gender vs Season")
plt.xlabel("Gender")
plt.ylabel("Number of Products")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()














