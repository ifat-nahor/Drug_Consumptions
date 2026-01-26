import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. LOAD DATA
# Ensure the path points to your processed CSV file
df = pd.read_csv("data/processed/Drug_Consumption_Cleaned.csv")

def analyze_age_group_drivers(df_input):
    """
    Identifies which specific drugs 'drive' the average usage within 
    different age categories and drug families.
    """
    
    # Define drug families/groups
    groups = {
        'Stimulants': ['Coke', 'Crack', 'Ecstasy', 'Amphet', 'Meth'],
        'Depressants': ['Alcohol', 'Benzos', 'Heroin', 'VSA'],
        'Hallucinogens': ['LSD', 'Mushrooms', 'Ketamine']
    }
    
    # Set logical order for the Age categories
    age_order = ['18-24', '25-34', '35-44', '45-54', '55-64', '65+']
    df_plot = df_input.copy()
    df_plot['Age'] = pd.Categorical(df_plot['Age'], categories=age_order, ordered=True)

    # Iterate through each drug family to create separate visualizations
    for group_name, drugs in groups.items():
        # Melt data to long format: transforms drug columns into a single 'Drug' column
        df_long = pd.melt(df_plot, id_vars=['Age'], value_vars=drugs,
                          var_name='Drug', value_name='Usage')
        
        # Calculate the global mean for this drug group (used as a benchmark line)
        group_overall_mean = df_long['Usage'].mean()

        plt.figure(figsize=(14, 7))
        
        # Create a grouped bar chart showing usage of each drug per age group
        ax = sns.barplot(data=df_long, x='Age', y='Usage', hue='Drug', palette='muted')
        
        # Add a horizontal dashed line representing the overall group mean
        plt.axhline(group_overall_mean, color='red', linestyle='--', 
                    label=f'Overall {group_name} Mean')
        
        # Chart labeling and formatting
        plt.title(f"Drug Drivers in {group_name} by Age Group", fontsize=16)
        plt.ylabel("Standardized Usage Score")
        plt.legend(title="Specific Drug", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Visual annotation to help interpret the 'Drivers'
        plt.annotate('Above this line = Drivers of the average', 
                     xy=(0, group_overall_mean), xytext=(0.5, group_overall_mean + 0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

        plt.tight_layout()
        plt.show()

        # Print statistical insights to the console
        print(f"\n>>> Insight for {group_name}:")
        # Group by Age and Drug to find the highest contributor per category
        avg_by_age_drug = df_long.groupby(['Age', 'Drug'], observed=True)['Usage'].mean()
        
        for age in age_order:
            # Identify the drug with the maximum mean usage in this age group
            top_drug = avg_by_age_drug[age].idxmax()
            top_val = avg_by_age_drug[age].max()
            
            # Check if this top drug exceeds the global average for this family
            if top_val > group_overall_mean:
                print(f"Age {age:5}: {top_drug} is a clear DRIVER (Value: {top_val:.2f}, Group Mean: {group_overall_mean:.2f})")
            else:
                print(f"Age {age:5}: All drugs in this group are below the overall average.")

# EXECUTION
if __name__ == "__main__":
    analyze_age_group_drivers(df)