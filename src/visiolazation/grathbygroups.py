import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. DATA LOADING & SETUP ---
# Defining the path to the cleaned data
# Using a relative path from the project root
data_path = 'data/processed/Drug_Consumption_Cleaned.csv'

if not os.path.exists(data_path):
    print(f"Error: Could not find the file at {data_path}")
    exit()

# Loading the dataset into df_cleaned
df_cleaned = pd.read_csv(data_path)

# Defining the directory to save plots
output_dir = 'plots'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. DRUG FAMILY DEFINITIONS ---
# Grouping individual drug columns into pharmacological families
stimulants = ['Amphet', 'Coke', 'Crack', 'Ecstasy', 'Nicotine']
depressants = ['Alcohol', 'Benzos', 'Cannabis', 'Heroin', 'Meth']
hallucinogens = ['LSD', 'Mushrooms', 'Ketamine']

# --- 3. CALCULATING FAMILY SCORES ---
# Calculating the mean consumption score for each drug family
# This creates a general usage profile per category
df_cleaned['Score_Stimulants'] = df_cleaned[[c for c in stimulants if c in df_cleaned.columns]].mean(axis=1)
df_cleaned['Score_Depressants'] = df_cleaned[[c for c in depressants if c in df_cleaned.columns]].mean(axis=1)
df_cleaned['Score_Hallucinogens'] = df_cleaned[[c for c in hallucinogens if c in df_cleaned.columns]].mean(axis=1)

# --- 4. PREPARING DATA FOR VISUALIZATION ---
# We focus on "Heavy Users" (mean score >= 3) to see the strongest personality links
family_comparison = []

families = {
    'Stimulants': 'Score_Stimulants',
    'Depressants': 'Score_Depressants',
    'Hallucinogens': 'Score_Hallucinogens'
}

# Mapping standardized personality trait names (updated to match your data)
traits = {'Impulsivity': 'Impulsive', 'Sensation Seeking': 'SS'}

for family_name, score_col in families.items():
    # Filter for heavy users within each drug family
    heavy_users = df_cleaned[df_cleaned[score_col] >= 3]
    
    for trait_label, trait_col in traits.items():
        if trait_col in df_cleaned.columns:
            # Calculate the average trait score for this user group
            mean_val = heavy_users[trait_col].mean()
            family_comparison.append({
                'Drug Family': family_name,
                'Personality Trait': trait_label,
                'Mean Score': mean_val
            })

# Convert the results list into a DataFrame for Seaborn plotting
plot_df = pd.DataFrame(family_comparison)

# --- 5. GRAPH GENERATION ---
plt.figure(figsize=(12, 7))
sns.set_style("whitegrid")

# Create a Grouped Bar Chart to compare traits across drug families
ax = sns.barplot(
    x='Drug Family', 
    y='Mean Score', 
    hue='Personality Trait', 
    data=plot_df, 
    palette='muted'
)

# Adding titles, labels, and visual formatting
plt.title('Neuro-Personality Profile by Drug Family\n(Average Scores for Heavy Users)', fontsize=16, pad=20)
plt.ylabel('Average Standardized Score (Z-Score)', fontsize=12)
plt.xlabel('Drug Family', fontsize=12)

# Dynamic Y-axis limits based on data range
plt.ylim(plot_df['Mean Score'].min() - 0.2, plot_df['Mean Score'].max() + 0.3)

# Adding numeric labels on top of each bar for precision
for p in ax.patches:
    if p.get_height() != 0:
        ax.annotate(format(p.get_height(), '.2f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points',
                    fontsize=10, fontweight='bold')

plt.tight_layout()

# Saving the final plot
output_file = os.path.join(output_dir, 'personality_by_drug_family.png')
plt.savefig(output_file)

print(f"✓ Analysis by drug family completed.")
print(f"✓ Comparison graph saved as: {output_file}")