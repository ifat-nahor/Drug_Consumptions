import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# =============================================================================
# STAGE 1: Environment Setup & Data Loading
# Identify the absolute path of the current script
current_file = Path(__file__).resolve()
# Navigate 2 levels up to reach the project root directory
project_root = current_file.parent.parent
# Define paths for input data and output visualizations
file_path = project_root / 'data' / 'processed' / 'Drug_Consumption_Cleaned.csv'
output_dir = project_root / 'plot_pics_for_groups_vs_imp_ss'
# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)
logger.info(f"Searching for data file at: {file_path}")

# Load dataset
if file_path.exists():
    df = pd.read_csv(file_path)
    logger.info("Dataset successfully loaded.")
else:
    logger.error(f"File not found: {file_path}")
    exit()
# Define drug categories based on neuro-pharmacological properties
groups = {
    'Hard Drugs': ['Heroin', 'Crack', 'Meth', 'Coke', 'VSA'],
    'Psychedelics': ['Ecstasy', 'LSD', 'Mushrooms'],
    'Prescription': ['Benzos', 'Amphet', 'Ketamine', 'Cannabis', 'Amyl'],
    'Daily Addictive': ['Alcohol', 'Nicotine', 'Caff', 'Choc']
}
# Feature Engineering: Identify maximum usage frequency (0-6) per category
for group_name, drugs in groups.items():
    existing_drugs = [d for d in drugs if d in df.columns]
    df[f'{group_name}_Max_Freq'] = df[existing_drugs].max(axis=1)

# =============================================================================
# STAGE 2: Personality-Usage Escalation Trends
logger.info("Analyzing personality trait trends across usage frequencies...")
plot_data = []
for group_name in groups.keys():
    for freq in range(7):
        subset = df[df[f'{group_name}_Max_Freq'] == freq]
        # Ensure statistical stability by filtering groups with insufficient participants
        if len(subset) > 5: 
            plot_data.append({
                'Group': group_name,
                'Frequency': freq,
                'Avg_Impulsive': subset['Impulsive'].mean(),
                'Avg_SS': subset['SS'].mean()
            })

summary_df = pd.DataFrame(plot_data)
# Visualization 1: Personality Trait Slopes
plt.figure(figsize=(15, 6))
# Subplot A: Impulsivity Trends
plt.subplot(1, 2, 1)
sns.lineplot(data=summary_df, x='Frequency', y='Avg_Impulsive', hue='Group', marker='o', palette='Set1')
plt.title('Impulsivity vs. Usage Frequency (0-6)', fontsize=13)
plt.xlabel('Frequency (0=Never, 6=Daily)')
plt.ylabel('Mean Impulsivity Score (Z-Score)')
plt.grid(True, linestyle='--', alpha=0.6)
# Subplot B: Sensation Seeking Trends
plt.subplot(1, 2, 2)
sns.lineplot(data=summary_df, x='Frequency', y='Avg_SS', hue='Group', marker='o', palette='Set1')
plt.title('Sensation Seeking vs. Usage Frequency (0-6)', fontsize=13)
plt.xlabel('Frequency (0=Never, 6=Daily)')
plt.ylabel('Mean Sensation Seeking Score (Z-Score)')
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(output_dir / 'personality_frequency_trends.png')
logger.info(f"Personality trend visualization saved to {output_dir}")
plt.show()

# =============================================================================
# STAGE 3: Consumption Distribution Profiling
logger.info("Profiling consumption distribution across drug categories...")
freq_dist = []

for group_name in groups.keys():
    # Calculate percentage distribution per frequency level (0-6)
    counts = df[f'{group_name}_Max_Freq'].value_counts(normalize=True).sort_index() * 100
    for freq, pct in counts.items():
        freq_dist.append({'Group': group_name, 'Frequency': freq, 'Percentage': pct})

dist_df = pd.DataFrame(freq_dist)
# Visualization 2: Distribution Bar Chart
plt.figure(figsize=(12, 6))
sns.barplot(data=dist_df, x='Frequency', y='Percentage', hue='Group', palette='YlOrRd')

plt.title('Consumption Distribution: Percentage Analysis (0-6 Scale)', fontsize=14)
plt.ylabel('Percentage of Participants (%)')
plt.xlabel('Frequency Level (0=Never, 1=Decade, 2=Year, 3=Month, 4=Week, 5=Day, 6=Daily)')
plt.legend(title='Drug Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(output_dir / 'usage_frequency_distribution.png')
logger.info(f"Usage distribution visualization saved to {output_dir}")
plt.show()
logger.info("Processing complete. All visualizations exported.")