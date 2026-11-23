import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
csv_file = r"..\results\ocr_evaluation_detail.csv"

df = pd.read_csv(csv_file)

df["Model"] = df["profile"].map({
    "baseline": "EasyOCR",
    "large": "Florence-2"
})


summary = (
    df.groupby("Model")
      .agg(
          Samples=("image", "count"),
          Mean_CER=("normalized_cer", "mean"),
          Median_CER=("normalized_cer", "median"),
          Exact_Match_Rate=("exact_match_normalized", "mean"),
          Mean_Runtime=("analysis_seconds", "mean"),
      )
      .reset_index()
)

print("\n=== OCR Model Summary ===\n")
print(summary)

print("\n=== OCR Model Summary Baseline ===\n")
print(df[df["profile"] == "baseline"]["normalized_cer"].describe())

print("\n=== OCR Model Summary Large  ===\n")
print(df[df["profile"] == "large"]["normalized_cer"].describe())


sns.set_theme(style="whitegrid", palette="deep")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(
    data=df,
    x="Model",
    y="normalized_cer",
    ax=axes[0, 0]
)
axes[0, 0].set_title("Normalized CER Distribution")
axes[0, 0].set_ylabel("Normalized CER")

sns.violinplot(
    data=df,
    x="Model",
    y="normalized_cer",
    ax=axes[0, 1]
)
axes[0, 1].set_title("Normalized CER Density")

exact_match = (
    df.groupby("Model")["exact_match_normalized"]
      .mean()
      .reset_index()
)

sns.barplot(
    data=exact_match,
    x="Model",
    y="exact_match_normalized",
    ax=axes[1, 0]
)

axes[1, 0].set_ylim(0, 1)
axes[1, 0].set_title("Exact Match Rate")
axes[1, 0].set_ylabel("Fraction Correct")

sns.boxplot(
    data=df,
    x="Model",
    y="analysis_seconds",
    ax=axes[1, 1]
)
axes[1, 1].set_title("Analysis Time")
axes[1, 1].set_ylabel("Seconds")

plt.tight_layout()
plt.savefig("ocr_model_comparison.png", dpi=300)
plt.show()


category_stats = (
    df.groupby(["category", "Model"])
      .agg(
          Mean_CER=("normalized_cer", "mean"),
          Exact_Match=("exact_match_normalized", "mean")
      )
      .reset_index()
)

print("\n=== Category Performance ===\n")
print(category_stats)

plt.figure(figsize=(10, 5))

sns.barplot(
    data=category_stats,
    x="category",
    y="Mean_CER",
    hue="Model"
)

plt.title("Normalized CER by Category")
plt.ylabel("Mean Normalized CER")
plt.xlabel("")
plt.tight_layout()
plt.savefig("ocr_category_cer.png", dpi=300)
plt.show()


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="raw_cer",
    y="normalized_cer",
    hue="Model",
    style="category",
    s=80
)

plt.title("Raw vs. Normalized CER")
plt.xlabel("Raw CER")
plt.ylabel("Normalized CER")
plt.tight_layout()
plt.savefig("ocr_raw_vs_normalized.png", dpi=300)
plt.show()

# ------------------------------------------------------------------
# Improvement Calculation
# ------------------------------------------------------------------
pivot = (
    df.pivot(
        index="image",
        columns="Model",
        values="normalized_cer"
    )
)

pivot["CER_Improvement"] = (
    pivot["EasyOCR"] - pivot["Florence-2"]
)

print("\n=== Improvement Statistics ===\n")
print(pivot["CER_Improvement"].describe())