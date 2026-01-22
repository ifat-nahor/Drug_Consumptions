import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# =============================================================================
# STAGE 1: Data Loading, Cleaning, and Categorical Grouping
file_path = r'C:\Users\inbal\Desktop\pyton things\final_proj\data\processed\Drug_Consumption_Cleaned.csv'
df = pd.read_csv(file_path)
# Removing subjects who claimed usage of the non-existent drug 'Semer'
df = df[df['Semer'] == 0].copy()
# Defining the 4 groups
groups = {
    'Hard Drugs': ['Heroin', 'Crack', 'Meth', 'Coke', 'VSA'],
    'Psychedelics': ['Ecstasy', 'LSD', 'Mushrooms'],
    'Prescription': ['Benzos', 'Amphet', 'Ketamine', 'Cannabis', 'Amyl'],
    'Daily Addictive': ['Alcohol', 'Nicotine', 'Caff', 'Choc']
}
# Feature Engineering: Determining the maximum usage frequency (0-6 scale) 
# for each drug category per participant.
for group_name, drugs in groups.items():
    existing_drugs = [d for d in drugs if d in df.columns]
    df[f'{group_name}_Max_Freq'] = df[existing_drugs].max(axis=1)
# =============================================================================
# STAGE 2: Personality-Usage Escalation Trends
# analyze whether an increase in consumption frequency is accompanied by a significant rise in specific personality trait scores.
plot_data = []
for group_name in groups.keys():
    for freq in range(7):
        subset = df[df[f'{group_name}_Max_Freq'] == freq]
        # Data Stability: Only processing levels with a sufficient number of participants
        if len(subset) > 5: 
            plot_data.append({
                'Group': group_name,
                'Frequency': freq,
                'Avg_Impulsive': subset['Impulsive'].mean(),
                'Avg_SS': subset['SS'].mean()
            })
summary_df = pd.DataFrame(plot_data)
plt.figure(figsize=(15, 6))
# Plot 1A: Mean Impulsivity vs. Frequency Level
plt.subplot(1, 2, 1)
sns.lineplot(data=summary_df, x='Frequency', y='Avg_Impulsive', hue='Group', marker='o', palette='Set1')
plt.title('Impulsivity vs. Usage Frequency (0-6)', fontsize=13)
plt.xlabel('Frequency (0=Never, 6=Daily)')
plt.ylabel('Mean Impulsivity Score (Z-Score)')
plt.grid(True, linestyle='--', alpha=0.6)
# Plot 1B: Mean Sensation Seeking (SS) vs. Frequency Level
plt.subplot(1, 2, 2)
sns.lineplot(data=summary_df, x='Frequency', y='Avg_SS', hue='Group', marker='o', palette='Set1')
plt.title('Sensation Seeking vs. Usage Frequency (0-6)', fontsize=13)
plt.xlabel('Frequency (0=Never, 6=Daily)')
plt.ylabel('Mean Sensation Seeking Score (Z-Score)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
#ensuring the output folder exists
if not os.path.exists('plot_pics'): os.makedirs('plot_pics')
plt.savefig('plot_pics/personality_frequency_trends.png')
plt.show()
# =============================================================================
# STAGE 3: Consumption Distribution Profiling
#  visualization compares the "usage signature" of each group.
# identifing the proportion of daily users vs. occasional users across the groups
freq_dist = []
for group_name in groups.keys():
    # Calculating the percentage distribution of frequency levels 0-6
    counts = df[f'{group_name}_Max_Freq'].value_counts(normalize=True).sort_index() * 100
    for freq, pct in counts.items():
        freq_dist.append({'Group': group_name, 'Frequency': freq, 'Percentage': pct})
dist_df = pd.DataFrame(freq_dist)
plt.figure(figsize=(12, 6))
# Using a 'Yellow to Red' gradient to emphasize intensity of usage
sns.barplot(data=dist_df, x='Frequency', y='Percentage', hue='Group', palette='YlOrRd')
plt.title('Consumption Distribution: Percentage Analysis (0-6 Scale)', fontsize=14)
plt.ylabel('Percentage of Participants (%)')
plt.xlabel('Frequency Level (0=Never, 1=Decade, 2=Year, 3=Month, 4=Week, 5=Day, 6=Daily)')
plt.legend(title='Drug Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('plot_pics/usage_frequency_distribution.png')
plt.show()
print("Processing complete. Visualizations exported to 'plot_pics'")