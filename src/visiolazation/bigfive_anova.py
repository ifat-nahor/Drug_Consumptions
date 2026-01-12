import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# --- 1. DATA LOADING & SETUP ---
# PURPOSE: Load the cleaned data and ensure we are in the correct directory.
current_dir = os.path.dirname(os.path.abspath(__file__))
# Going up to the root folder to find the processed data
data_path = os.path.join(current_dir, '..', '..', 'data', 'processed', 'Drug_Consumption_Cleaned.csv')

if not os.path.exists(data_path):
    print(f"❌ Error: Could not find the file at {data_path}")
    exit()

df_cleaned = pd.read_csv(data_path)

# --- 2. DEFINE COLUMNS ---
# PURPOSE: Explicitly define which columns are drugs and which are traits.
# This prevents the 'NameError' you encountered.
big_five_traits = ['Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore']

drug_columns = [
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 
    'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 
    'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]

# --- 3. ANOVA STATISTICAL ANALYSIS ---
# PURPOSE: To perform a One-Way ANOVA (Analysis of Variance).
# GOAL: To test if personality trait scores (e.g., Neuroticism) vary significantly 
# between different levels of drug consumption (CL0 - CL6).
print("\n" + "="*80)
print("RUNNING BIG FIVE PERSONALITY TRAITS ANOVA")
print("="*80)

big_five_results = []

for drug in drug_columns:
    if drug not in df_cleaned.columns:
        continue
        
    result_row = {'Drug': drug}
    for trait in big_five_traits:
        if trait in df_cleaned.columns:
            # Grouping the personality scores based on drug usage levels
            unique_levels = sorted(df_cleaned[drug].unique())
            groups = [df_cleaned[df_cleaned[drug] == lvl][trait].dropna() for lvl in unique_levels]
            
            # Executing ANOVA: Comparing means of multiple groups
            if len(groups) > 1 and all(len(g) > 1 for g in groups):
                f_stat, p_val = stats.f_oneway(*groups)
                result_row[trait] = p_val
            else:
                result_row[trait] = np.nan
                
    big_five_results.append(result_row)

# Convert results to a DataFrame for easier visualization
big_five_df = pd.DataFrame(big_five_results).set_index('Drug')

# --- 4. VISUALIZATION: SIGNIFICANCE HEATMAP ---
# PURPOSE: To visualize the strength of relationship using p-values.
# EXPLANATION: We use -log10 transformation. 
# WHY: A very small p-value (e.g., 0.0001) becomes a larger number (4.0).
# This makes significant results appear darker/more intense on the map.

plt.figure(figsize=(12, 10))
significance_matrix = -np.log10(big_five_df.astype(float))

sns.heatmap(
    significance_matrix, 
    annot=big_five_df,     # Display actual p-values inside cells
    cmap='YlOrRd',         # Yellow to Red color scale
    fmt=".4f", 
    cbar_kws={'label': 'Significance Level (-log10 p-value)'}
)

plt.title('Big Five Traits vs Drug Consumption: ANOVA Significance Map', fontsize=16)
plt.xlabel('Personality Traits (Big Five)', fontsize=12)
plt.ylabel('Substances / Drugs', fontsize=12)
plt.tight_layout()

# Save the final visualization
output_path = os.path.join(current_dir, 'big_five_anova_heatmap.png')
plt.savefig(output_path)

print(f"\n✅ Analysis complete!")
print(f"✓ Summary plot saved as: {output_path}")
plt.show()