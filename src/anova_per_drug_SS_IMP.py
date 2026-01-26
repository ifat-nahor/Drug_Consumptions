import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
from pathlib import Path
# =============================================================================
# STAGE 1: Data Loading & Environment Setup
# =============================================================================
# Locate the current script's absolute path
current_file = Path(__file__).resolve()
# Navigate 3 levels up to reach the project root directory 
project_root = current_file.parent.parent
# Define path to the processed dataset using universal path separators
file_path = project_root / 'data' / 'processed' / 'Drug_Consumption_Cleaned.csv'
print(f"Searching for data in: {file_path}")
# Load the dataset into a DataFrame
if file_path.exists():
    df = pd.read_csv(file_path)
    print("Data successfully loaded from the project directory")
else:
    print(f"Critical Error: Dataset not found at {file_path}")
    exit()

# =============================================================================
# STAGE 2: Factor Preparation (Feature Binning)
# =============================================================================
# Discretize continuous personality traits into 'Low' and 'High' categories 
# using a median split (qcut) to facilitate ANOVA factor analysis.
df['Impulsive_Bin'] = pd.qcut(df['Impulsive'], 2, labels=['Low', 'High'])
df['SS_Bin'] = pd.qcut(df['SS'], 2, labels=['Low', 'High'])
# Define the list of substances to be analyzed
drug_cols = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 
             'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 
             'Meth', 'Mushrooms', 'Nicotine', 'VSA']

# =============================================================================
# STAGE 3: Statistical Modeling (Two-Way ANOVA)
# =============================================================================
# Filter for drugs actually present in the dataframe to prevent execution errors
valid_drugs = [d for d in drug_cols if d in df.columns]
results = []

for drug in valid_drugs:
    # Model Formula: Consumption Frequency as a function of Impulsivity, 
    # Sensation Seeking, and their interaction effect.
    formula = f'{drug} ~ C(Impulsive_Bin) * C(SS_Bin)'
    model = ols(formula, data=df).fit()
    
    # Run Type II ANOVA 
    aov_table = anova_lm(model, typ=2)
    
    # Extract p-values for main effects and interaction
    results.append({
        'Drug': drug,
        'P_Impulsivity': aov_table.loc['C(Impulsive_Bin)', 'PR(>F)'],
        'P_Sensation_Seeking': aov_table.loc['C(SS_Bin)', 'PR(>F)'],
        'P_Interaction': aov_table.loc['C(Impulsive_Bin):C(SS_Bin)', 'PR(>F)']
    })

results_df = pd.DataFrame(results)

# =============================================================================
# STAGE 4: Multiple Comparison Correction
# =============================================================================
# Apply Bonferroni correction to mitigate the Family-Wise Error Rate (FWER) 
# resulting from testing multiple hypotheses (18 drugs x 3 effects).
p_cols = ['P_Impulsivity', 'P_Sensation_Seeking', 'P_Interaction']
all_p_values = results_df[p_cols].values.flatten()
_, corrected_p, _, _ = multipletests(all_p_values, method='bonferroni')

# Reshape corrected p-values back into the results dataframe structure
results_df[['P_Imp_Corr', 'P_SS_Corr', 'P_Inter_Corr']] = corrected_p.reshape(results_df[p_cols].shape)

# =============================================================================
# STAGE 5: VISUALIZATION 1 - Statistical Significance Heatmap
# =============================================================================
# Define and create output directory at the project root
output_dir = project_root / 'plot_pics_for_groups_vs_imp_ss'
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare data for -log10 transformation to highlight significance levels
plot_df = results_df.set_index('Drug')[['P_Imp_Corr', 'P_SS_Corr', 'P_Inter_Corr']]
log_p_df = -np.log10(plot_df.astype(float).clip(lower=1e-10))

plt.figure(figsize=(12, 8))
sns.heatmap(log_p_df, annot=plot_df.round(4), cmap='YlOrRd', cbar_kws={'label': '-log10(p-value)'})
plt.title('ANOVA Significance: Personality Traits & Interaction Effects (Bonferroni Corrected)')
plt.tight_layout()

# Save visualization using Pathlib integration
plt.savefig(output_dir / 'anova_significance_heatmap.png')
plt.show()

# =============================================================================
# STAGE 6: VISUALIZATION 2 - Mean Consumption Profile Heatmap
# =============================================================================
# Calculate mean consumption scores across the four personality profiles
mean_scores = df.groupby(['Impulsive_Bin', 'SS_Bin'], observed=True)[valid_drugs].mean().T

plt.figure(figsize=(12, 10))
sns.heatmap(mean_scores, annot=True, cmap='YlOrBr', fmt='.2f')
plt.title('Mean Drug Consumption Scores per Personality Profile')
plt.xlabel('Personality Cohorts (Impulsivity | Sensation Seeking)')
plt.ylabel('Substance Type')
plt.tight_layout()

plt.savefig(output_dir / 'mean_consumption_heatmap.png')
plt.show()
print(f"Visualizations exported to: {output_dir}")