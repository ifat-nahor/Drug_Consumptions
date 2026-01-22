import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
# =============================================================================
# STAGE 1: Data Loading and Filtering
file_path = r'C:\Users\inbal\Desktop\pyton things\final_proj\data\processed\Drug_Consumption_Cleaned.csv'
df = pd.read_csv(file_path)
df = df[df['Semer'] == 0].copy()
# =============================================================================
# STAGE 2: Preparing Factors 
# Using 'qcut' for efficient median splitting into categorical groups.
df['Impulsive_Bin'] = pd.qcut(df['Impulsive'], 2, labels=['Low', 'High'])
df['SS_Bin'] = pd.qcut(df['SS'], 2, labels=['Low', 'High'])
drug_cols = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 
             'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 
             'Meth', 'Mushrooms', 'Nicotine', 'VSA']
# =============================================================================
# STAGE 3: Running Two-Way ANOVAs
# Efficiency Note: We pre-filter columns to avoid 'if' checks inside the loop.
valid_drugs = [d for d in drug_cols if d in df.columns]
results = []
for drug in valid_drugs:
    formula = f'{drug} ~ C(Impulsive_Bin) * C(SS_Bin)'
    # Running the Ordinary Least Squares model and ANOVA
    model = ols(formula, data=df).fit()
    aov_table = anova_lm(model, typ=2)
    
    results.append({
        'Drug': drug,
        'P_Impulsivity': aov_table.loc['C(Impulsive_Bin)', 'PR(>F)'],
        'P_Sensation_Seeking': aov_table.loc['C(SS_Bin)', 'PR(>F)'],
        'P_Interaction': aov_table.loc['C(Impulsive_Bin):C(SS_Bin)', 'PR(>F)']
    })
results_df = pd.DataFrame(results)
# =============================================================================
# STAGE 4: Bonferroni Multiple Comparison Correction
p_cols = ['P_Impulsivity', 'P_Sensation_Seeking', 'P_Interaction']
# Flattening and correcting all p-values at once for efficiency
all_p_values = results_df[p_cols].values.flatten()
_, corrected_p, _, _ = multipletests(all_p_values, method='bonferroni')
# Reshaping back to the original table structure
results_df[['P_Imp_Corr', 'P_SS_Corr', 'P_Inter_Corr']] = corrected_p.reshape(results_df[p_cols].shape)
# =============================================================================
# STAGE 5: GRAPH 1 - Significance Heatmap (-log10)
plot_df = results_df.set_index('Drug')[['P_Imp_Corr', 'P_SS_Corr', 'P_Inter_Corr']]
# Using clip to avoid infinity values in log transformation
log_p_df = -np.log10(plot_df.astype(float).clip(lower=1e-10))
plt.figure(figsize=(12, 8))
sns.heatmap(log_p_df, annot=plot_df.round(4), cmap='YlOrRd', cbar_kws={'label': '-log10(p-value)'})
plt.title('Graph 1: Statistical Significance (Bonferroni Corrected p-values)')
plt.tight_layout()
plt.savefig('plot_pics/anova_significance_heatmap.png')
plt.show()
# =============================================================================
# STAGE 6: GRAPH 2 - Mean Consumption Heatmap
# Grouping and calculating means for all drugs simultaneously
mean_scores = df.groupby(['Impulsive_Bin', 'SS_Bin'], observed=True)[valid_drugs].mean().T
plt.figure(figsize=(12, 10))
sns.heatmap(mean_scores, annot=True, cmap='YlOrBr', fmt='.2f')
plt.title('Graph 2: Mean Drug Consumption Score by Personality Profile')
plt.xlabel('Personality Groups (Impulsivity | Sensation Seeking)')
plt.ylabel('Drug Type')
plt.tight_layout()
plt.savefig('plot_pics/mean_consumption_heatmap.png')
plt.show()