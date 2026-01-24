import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy import stats # Added for statistical analysis
from data_cleaning import aggregate_drug_families

# 1. ROBUST DATA LOADING
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file_path))
file_path = os.path.join(project_root, 'data', 'processed', 'Drug_Consumption_Cleaned.csv')

try:
    df = pd.read_csv(file_path)
    df_processed = aggregate_drug_families(df)
    print("✅ Data loaded and processed successfully!")
except FileNotFoundError:
    print(f"❌ Error: Could not find the file at {file_path}")

# 2. ANALYSIS FUNCTION: GENDER x AGE x DRUG FAMILY WITH STATS
def plot_gender_age_drug_analysis(df_input):
    """
    Creates a faceted line plot and performs T-tests to analyze
    gender differences in drug usage across age groups.
    """
    
    # Define the chronological order for age groups
    age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df_input['Age'] = pd.Categorical(df_input['Age'], categories=age_order, ordered=True)
    
    # List of score columns
    score_cols = ["Score_Stimulants", "Score_Depressants", "Score_Hallucinogens"]
    
    # MELTING: Converting from wide format to long format
    df_long = pd.melt(df_input, 
                      id_vars=['Age', 'Gender'], 
                      value_vars=score_cols,
                      var_name='Drug_Family', 
                      value_name='Usage_Score')

    # CLEANING LABELS: Removing 'Score_' prefix
    df_long['Drug_Family'] = df_long['Drug_Family'].str.replace('Score_', '')

    # --- NEW: STATISTICAL ANALYSIS SECTION ---
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS: GENDER DIFFERENCES")
    print("="*50)
    
    for family in df_long['Drug_Family'].unique():
        # Separate data by Gender for the T-test
        male_grp = df_long[(df_long['Drug_Family'] == family) & (df_long['Gender'] == 'Male')]['Usage_Score']
        female_grp = df_long[(df_long['Drug_Family'] == family) & (df_long['Gender'] == 'Female')]['Usage_Score']
        
        # Independent T-test
        t_stat, p_val = stats.ttest_ind(male_grp, female_grp, nan_policy='omit')
        
        # Determine significance (p < 0.05)
        significance = "Significant ✅" if p_val < 0.05 else "Not Significant ❌"
        print(f"Family: {family:15} | P-Value: {p_val:.4f} | {significance}")
    print("="*50 + "\n")
    # ----------------------------------------

    # SETTING THE VISUAL STYLE
    sns.set_theme(style="whitegrid")

    # CREATING THE FACETED PLOT
    g = sns.relplot(
        data=df_long,
        x='Age', 
        y='Usage_Score',
        hue='Drug_Family', 
        col='Gender',
        kind='line', 
        marker='o',
        errorbar=('ci', 95), # Added error bars for scientific accuracy
        height=5, 
        aspect=1.2,
        facet_kws={'sharey': True} 
    )

    # TITLES & LABELS
    g.set_axis_labels("Age Group", "Average Usage Score")
    g.set_titles("Gender: {col_name}")
    g.fig.suptitle("Gender Differences in Drug Usage Patterns Across Ages", y=1.05, fontsize=16)

    # Rotate X-axis labels to avoid overlap
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)

    plt.show()

# 3. EXECUTION
if 'df_processed' in locals():
    plot_gender_age_drug_analysis(df_processed)