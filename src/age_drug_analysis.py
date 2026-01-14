import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from data_cleaning import aggregate_drug_families

# Load data
df = pd.read_csv("data/processed/Drug_Consumption_Cleaned.csv")
df = aggregate_drug_families(df)

# --- FIX: Do NOT use to_numeric on Age in this dataset ---
# Define the correct chronological order
age_order = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
df["Age"] = pd.Categorical(df["Age"], categories=age_order, ordered=True)

# 1) Compute mean usage per age group
# We use the existing "Age" column which already has the groups
grouped = df.groupby("Age")[
    ["Score_Stimulants", "Score_Depressants", "Score_Hallucinogens"]
].mean()

# 2) STATISTICAL ANALYSIS (Gender difference across all ages)
print("\n" + "="*50)
print("STATISTICAL ANALYSIS: GENDER DIFFERENCES")
print("="*50)
for score in ["Score_Stimulants", "Score_Depressants", "Score_Hallucinogens"]:
    male = df[df['Gender'] == 'Male'][score]
    female = df[df['Gender'] == 'Female'][score]
    t_stat, p_val = stats.ttest_ind(male, female, nan_policy='omit')
    result = "Significant ✅" if p_val < 0.05 else "Not Significant ❌"
    print(f"{score:18} | P-Value: {p_val:.4f} | {result}")
print("="*50 + "\n")

# 3) Plot grouped bar chart
x = np.arange(len(grouped.index))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, grouped["Score_Stimulants"], width, label="Stimulants")
plt.bar(x, grouped["Score_Depressants"], width, label="Depressants")
plt.bar(x + width, grouped["Score_Hallucinogens"], width, label="Hallucinogens")

plt.xlabel("Age Group")
plt.ylabel("Mean Usage Score")
plt.title("Average Drug Family Usage by Age Group")
plt.xticks(x, grouped.index)
plt.legend(title="Drug Families")

plt.tight_layout()
plt.show()