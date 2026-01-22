import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# 1. LOAD DATA
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
file_path = os.path.join(project_root, 'data', 'processed', 'Drug_Consumption_Cleaned.csv')

try:
    df = pd.read_csv(file_path)
    print("✅ Data loaded successfully!")
except Exception as e:
    print(f"❌ Error loading data: {e}")

def plot_top_drugs_education_with_insights(df_input):
    """
    Analyzes top 5 drugs by education, prints statistical ANOVA, 
    and generates automated insights based on medians.
    """
    
    top_drugs = ['Alcohol', 'Cannabis', 'Nicotine', 'Coke', 'Ecstasy']
    
    edu_order = [
        "Left school at 16 years", "Left school at 17 years", "Left school at 18 years",
        "Some college or university, no degree", "Professional certificate/ diploma",
        "University degree", "Masters degree", "Doctorate degree"
    ]
    
    df_plot = df_input[df_input['Education'].isin(edu_order)].copy()
    df_plot['Education'] = pd.Categorical(df_plot['Education'], categories=edu_order, ordered=True)

    # --- PART 1: STATISTICAL ANALYSIS (ANOVA) ---
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS: IMPACT OF EDUCATION (ANOVA)")
    print("="*60)
    for drug in top_drugs:
        groups_list = [df_plot[df_plot['Education'] == edu][drug] for edu in edu_order]
        f_stat, p_val = stats.f_oneway(*[g for g in groups_list if not g.empty])
        status = "SIGNIFICANT ✅" if p_val < 0.05 else "NOT SIGNIFICANT ❌"
        print(f"Drug: {drug:10} | P-Value: {p_val:.4f} | {status}")

    # --- PART 2: AUTOMATED INSIGHTS ---
    print("\n" + "="*60)
    print("KEY INSIGHTS (Based on Median Usage)")
    print("="*60)
    for drug in top_drugs:
        # Calculate median for each education level
        medians = df_plot.groupby('Education', observed=True)[drug].median()
        max_edu = medians.idxmax()
        max_val = medians.max()
        
        # Determine intensity description
        intensity = "High" if max_val > 0.5 else "Moderate"
        print(f"• For {drug:8}: The highest median usage is found among: '{max_edu}' ({intensity}).")
    print("="*60 + "\n")

    # --- PART 3: VISUALIZATION ---
    df_long = pd.melt(df_plot, id_vars=['Education'], value_vars=top_drugs,
                      var_name='Drug_Type', value_name='Usage_Value')

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 8))

    # Using 'quartile' for a cleaner look as discussed
    sns.violinplot(
        data=df_long, x='Education', y='Usage_Value', hue='Drug_Type',
        inner="quartile", linewidth=1.2, cut=0, alpha=0.8
    )

    plt.title("Distribution of Top 5 Drugs by Education Level", fontsize=16)
    plt.xticks(rotation=35, ha='right')
    plt.ylabel("Standardized Usage Score (Median marked by center line)")
    plt.legend(title="Drug Type", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

# EXECUTION
if 'df' in locals():
    plot_top_drugs_education_with_insights(df)