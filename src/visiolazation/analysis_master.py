import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING
# ==============================================================================
# We use the path logic from your main.py to locate the cleaned data
current_dir = os.path.dirname(os.path.abspath(__file__))
# Adjusted to find the processed data as defined in your project structure
data_path = os.path.join(current_dir, '..', 'data', 'processed', 'Drug_Consumption_Cleaned.csv')

if not os.path.exists(data_path):
    # Fallback for different execution environments
    data_path = 'Drug_Consumption_Cleaned.csv'

try:
    df = pd.read_csv(data_path)
    print(f"✅ Data loaded successfully from {data_path}")
except FileNotFoundError:
    print("❌ Error: Could not find the cleaned CSV file. Please check the path.")
    exit()

# Renaming columns for clarity (Mapping original Kaggle names to research terms)
# BIS11 -> Impulsive, ImpSS -> SS (Sensation Seeking)
name_map = {'BIS11': 'Impulsive', 'ImpSS': 'SS'}
df = df.rename(columns=name_map)

# Defining our primary drug families for pharmacological comparison
stimulants = ['Amphet', 'Coke', 'Crack', 'Ecstasy', 'Nicotine']
depressants = ['Alcohol', 'Benzos', 'Cannabis', 'Heroin', 'Meth']
hallucinogens = ['LSD', 'Mushrooms', 'Ketamine']

# ==============================================================================
# 2. STATISTICAL ANALYSIS: ANOVA (Analysis of Variance)
# ==============================================================================
def run_anova_analysis(df, drug_list, traits):
    """
    PURPOSE: This function performs an ANOVA test.
    WHY: It determines if there is a statistically significant difference in 
    personality scores (e.g., Impulsivity) across the 7 levels of drug consumption (CL0-CL6).
    """
    print("\n--- Running ANOVA Analysis ---")
    results = []
    for drug in drug_list:
        if drug in df.columns:
            for trait in traits:
                # Group data by drug usage level and extract trait scores
                groups = [df[df[drug] == lvl][trait].dropna() for lvl in sorted(df[drug].unique())]
                if len(groups) > 1:
                    f_stat, p_val = stats.f_oneway(*groups)
                    results.append({'Drug': drug, 'Trait': trait, 'p_value': p_val})
    
    return pd.DataFrame(results)

# ==============================================================================
# 3. DRUG FAMILY COMPARISON
# ==============================================================================
def analyze_drug_families(df):
    """
    PURPOSE: Aggregates individual drugs into families (Stimulants, etc.).
    WHY: To see if regular users of specific types of drugs have 
    different personality profiles compared to others.
    """
    print("--- Analyzing Drug Families ---")
    # Calculate average consumption score per family
    df['Score_Stimulants'] = df[[c for c in stimulants if c in df.columns]].mean(axis=1)
    df['Score_Depressants'] = df[[c for c in depressants if c in df.columns]].mean(axis=1)
    df['Score_Hallucinogens'] = df[[c for c in hallucinogens if c in df.columns]].mean(axis=1)
    
    family_data = []
    for family, col in [('Stimulants', 'Score_Stimulants'), 
                        ('Depressants', 'Score_Depressants'), 
                        ('Hallucinogens', 'Score_Hallucinogens')]:
        # Filter for "Regular Users" (mean score >= 3)
        regular_users = df[df[col] >= 3]
        family_data.append({
            'Family': family,
            'Impulsive_Mean': regular_users['Impulsive'].mean(),
            'SS_Mean': regular_users['SS'].mean()
        })
    return pd.DataFrame(family_data)

# ==============================================================================
# 4. UNSUPERVISED LEARNING: K-MEANS CLUSTERING
# ==============================================================================
def run_clustering(df):
    """
    PURPOSE: Identifies hidden patterns using the K-Means algorithm.
    WHY: To group participants into 'Personality Profiles' (e.g., High-Risk vs Low-Risk)
    based on their trait scores, regardless of their drug usage labels.
    """
    print("--- Running K-Means Clustering ---")
    features = ['Impulsive', 'SS', 'Nscore', 'Cscore'] # Using Big Five + Research Traits
    X = df[features].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    return clusters

# ==============================================================================
# 5. MAIN EXECUTION & VISUALIZATION
# ==============================================================================
# Running the components
traits_to_test = ['Impulsive', 'SS']
drug_columns = [c for c in df.columns if any(drug in c for drug in stimulants + depressants + hallucinogens)]

# Perform ANOVA
anova_summary = run_anova_analysis(df, drug_columns, traits_to_test)

# Perform Family Analysis
family_summary = analyze_drug_families(df)

# Visualization
plt.figure(figsize=(14, 6))

# Subplot 1: Personality Profile by Drug Family
plt.subplot(1, 2, 1)
family_melted = family_summary.melt(id_vars='Family', var_name='Trait', value_name='Mean_Score')
sns.barplot(x='Family', y='Mean_Score', hue='Trait', data=family_melted, palette='viridis')
plt.title('Personality Profile by Drug Family\n(ANOVA-based comparison)')

# Subplot 2: Correlation Heatmap of p-values
plt.subplot(1, 2, 2)
anova_pivot = anova_summary.pivot(index='Drug', columns='Trait', values='p_value')
sns.heatmap(-np.log10(anova_pivot.astype(float)), annot=anova_pivot, cmap='YlOrRd', fmt=".3f")
plt.title('Significance Heatmap (-log10 p-value)\nDarker = More Significant')

plt.tight_layout()
plt.savefig('comprehensive_neuro_analysis.png')
print("\n✅ Analysis Complete! Summary plot saved as 'comprehensive_neuro_analysis.png'")
plt.show()