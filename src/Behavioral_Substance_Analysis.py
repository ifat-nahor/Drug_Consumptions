import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import warnings
from sklearn.cluster import KMeans

# --- PROJECT INTEGRATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    import Personality_Based_Drug_Usage_Predictor as predictor
except ImportError:
    print("ERROR: Could not find 'Personality_Based_Drug_Usage_Predictor.py'")
    sys.exit(1)

# --- CONFIGURATION ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)
logger = logging.getLogger("BehavioralAnalysis")

def run_behavioral_analysis(df):
    """
    Analyzes substance usage patterns and diversity across personality clusters.
    Ensures cluster naming aligns with psychological profiles (Impulsivity levels).
    """
    output_folder = "plots_pics_for_behavioral_and_gateway"
    os.makedirs(output_folder, exist_ok=True)

    # List of key substances for behavioral signature
    substances = ['Alcohol', 'Cannabis', 'Nicotine', 'Coke', 'Ecstasy', 'LSD', 'Mushrooms', 'Benzos']
    
    # Ensure all substance columns are numeric for calculation
    for col in substances:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate Poly-drug Use Diversity (Count of substances used > 0)
    df['Usage_Diversity'] = (df[substances] > 0).sum(axis=1)
    
    # --- CLUSTER IDENTIFICATION LOGIC ---
    # We identify clusters based on their mean Impulsivity score to match Yifat's PCA findings.
    # Lowest Impulsivity = Stable, Highest Impulsivity = Anxious-Impulsive.
    personality_rank = df.groupby('Cluster')['Impulsive'].mean().sort_values()
    
    mapping = {
        personality_rank.index[0]: "C1: Stable/Conservative", # Lowest Impulsivity
        personality_rank.index[1]: "C2: Extroverted",         # Moderate/Social Impulsivity
        personality_rank.index[2]: "C0: Anxious-Impulsive"    # Highest Impulsivity
    }
    
    df['Profile'] = df['Cluster'].map(mapping)

    # Log cluster verification to terminal
    logger.info("--- Cluster-to-Profile Verification ---")
    for cid, name in mapping.items():
        avg_imp = df[df['Cluster'] == cid]['Impulsive'].mean()
        avg_div = df[df['Cluster'] == cid]['Usage_Diversity'].mean()
        logger.info(f"Cluster {cid} mapped to {name} | Avg Impulsivity: {avg_imp:.2f} | Avg Diversity: {avg_div:.2f}")

    # 1. HEATMAP: Substance Usage Signatures
    plt.figure(figsize=(12, 6))
    summary_data = df.groupby('Profile')[substances].mean()
    # Reindexing ensures the plot follows the logical order of your report
    order = ["C0: Anxious-Impulsive", "C1: Stable/Conservative", "C2: Extroverted"]
    summary_data = summary_data.reindex(order)
    
    sns.heatmap(summary_data, annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title("Substance Usage Signatures by Personality Profile\n(Mean Usage Level)")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/substance_heatmap.png")
    plt.close()

    # 2. BOXPLOT: Diversity Distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Profile', y='Usage_Diversity', data=df, 
                order=order,
                palette='Set2')
    plt.title("Poly-drug Use Diversity (Substance Count) across Personality Clusters")
    plt.ylabel("Number of Different Substances Used")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"{output_folder}/drug_diversity_boxplot.png")
    plt.close()

    logger.info(f"Behavioral analysis complete. Visuals saved in: {output_folder}")

if __name__ == "__main__":
    # Integration with shared predictor model (Fixed Seed = 1)
    df_raw, scaled = predictor.load_and_scale_data(predictor.DATA_FILE, predictor.PERSONALITY_COLS)
    pca_res = predictor.apply_pca(scaled, predictor.PCA_COMPONENTS)
    
    # Executing KMeans with fixed parameters for reproducibility
    kmeans = KMeans(n_clusters=3, random_state=1, n_init=10)
    df_raw['Cluster'] = kmeans.fit_predict(pca_res)
    
    run_behavioral_analysis(df_raw)