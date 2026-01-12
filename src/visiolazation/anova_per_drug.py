import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# Load the cleaned dataset
# Ensure the path is correct for your local setup
df_cleaned = pd.read_csv('data/processed/Drug_Consumption_Cleaned.csv')

def run_universal_anova(df, group_column, target_columns=None):
    """
    Performs ANOVA test and creates visualizations for each target variable 
    against a grouping column (independent variable).
    """
    
    # 1. Automatically detect numerical columns if target_columns is not provided
    if target_columns is None:
        target_columns = df.select_dtypes(include=['number']).columns.tolist()
        if group_column in target_columns:
            target_columns.remove(group_column)

    print(f"\n" + "="*60)
    print(f"UNIVERSAL ANOVA SUMMARY: Grouping by '{group_column}'")
    print("="*60)

    results = []

    # 2. Iterate through each trait (e.g., Impulsive, SS) and run the ANOVA test
    for col in target_columns:
        # Create a clean subset of the data containing only the drug and the trait
        clean_data = df[[group_column, col]].dropna()
        
        # Prepare groups based on consumption levels (CL0 to CL6)
        unique_groups = sorted(clean_data[group_column].unique())
        groups = [clean_data[clean_data[group_column] == g][col] for g in unique_groups]
        
        # Check if there are enough groups and samples for a valid ANOVA
        if len(groups) > 1 and all(len(g) > 1 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            sig_level = "Significant" if p_val < 0.05 else "Not Significant"
            
            results.append({
                'Variable': col,
                'F-Statistic': f_stat,
                'p-value': p_val,
                'Status': sig_level
            })
            
            print(f"Variable: {col:15} | p-value: {p_val:.4e} | {sig_level}")

            # 3. Visualization: Generate and save a Boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=group_column, y=col, data=clean_data, palette='viridis', hue=group_column, legend=False)
            plt.title(f'ANOVA: {col} by {group_column}\np-value: {p_val:.4e}', fontsize=14)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save the figure to the current folder
            plt.tight_layout()
            plt.savefig(f'anova_{group_column}_{col}.png')
            plt.close() 
        else:
            print(f"Variable: {col:15} | Skipped (insufficient data/groups)")

    summary_df = pd.DataFrame(results)
    return summary_df

# --- Execution Block ---

# 1. Define the list of drugs (Independent Variables)
drug_columns = [
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 
    'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 
    'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]

# 2. Define the personality traits (Dependent Variables)
# Changed from BIS11/ImpSS to Impulsive/SS to match your cleaned data
my_traits = ['Impulsive', 'SS']

print("🚀 Starting ANOVA Analysis Pipeline...")

# 3. Loop through all drugs and perform ANOVA against Impulsive and SS
for drug in drug_columns:
    if drug in df_cleaned.columns:
        # Passing 'Impulsive' and 'SS' as the target columns
        run_universal_anova(df_cleaned, group_column=drug, target_columns=my_traits)
    else:
        print(f"Skipping {drug} - column name not found in CSV.")

print("\n✅ Analysis Complete! All plots have been generated and saved.")