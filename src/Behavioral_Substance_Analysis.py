import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import warnings

# --- 1. MODULE INTEGRATION ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

try:
    import Personality_Based_Drug_Usage_Predictor as predictor
except ImportError:
    print(f"❌ ERROR: Could not find 'Personality_Based_Drug_Usage_Predictor.py'")
    sys.exit(1)

# --- 2. CLEAN LOGGING & WARNINGS (SILENCE MATPLOTLIB) ---
warnings.filterwarnings("ignore")
# This line specifically silences the "categorical units" info messages
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)
logger = logging.getLogger("BehavioralAnalysis")

# --- 3. UPDATED ANALYSIS FUNCTION ---
def run_behavioral_analysis(df):
    """Analyzes drug consumption signatures and poly-drug use diversity."""
    substances = ['Alcohol', 'Cannabis', 'Nicotine', 'Coke', 'Ecstasy', 'LSD', 'Mushrooms', 'Benzos'] 
    valid_cols = [col for col in substances if col in df.columns]
    
    # Cast to numeric to prevent the categorical units warning
    for col in valid_cols: 
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Metrics calculation
    df['Usage_Diversity'] = (df[valid_cols] > 0.5).sum(axis=1)
    div_means = df.groupby('Cluster')['Usage_Diversity'].mean()
    max_div_cluster = div_means.idxmax()

    # --- ANALYTICAL SUMMARY ---
    logger.info("-" * 45)
    logger.info("ANALYTICAL SUMMARY: CLUSTER CHARACTERISTICS")
    logger.info(f"C0: 'Anxious-Impulsive' - High intensity. Avg Diversity: {div_means[0]:.2f} drugs.")
    logger.info(f"C1: 'Stable/Conservative' - Minimal usage. Avg Diversity: {div_means[1]:.2f} drugs.")
    logger.info(f"C2: 'Extroverted' - Reward-driven. Avg Diversity: {div_means[2]:.2f} drugs.")
    logger.info("-" * 45)
    logger.info("BOXPLOT INSIGHTS (POLY-DRUG USE):")
    logger.info(f"-> The data confirms that Cluster {max_div_cluster} has the most diverse drug usage.")
    logger.info(f"-> Average variety in Cluster {max_div_cluster}: {div_means[max_div_cluster]:.2f} different substances.")
    logger.info("-" * 45)

    # Visualization 1: Heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.groupby('Cluster')[valid_cols].mean(), annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title("Substance Usage Signatures by Cluster")
    plt.tight_layout()
    plt.show()

    # Visualization 2: Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Cluster', y='Usage_Diversity', data=df, palette='Set2')
    plt.title("Poly-drug Use Diversity (Usage Diversity)")
    plt.ylabel("Number of substances used (Score)")
    plt.show()

if __name__ == "__main__":
    try:
        df_raw, scaled = predictor.load_and_scale_data(predictor.DATA_FILE, predictor.PERSONALITY_COLS)
        pca_res = predictor.apply_pca(scaled, predictor.PCA_COMPONENTS)
        df_final = predictor.perform_final_clustering(df_raw, pca_res, predictor.FINAL_K)
        run_behavioral_analysis(df_final)
    except Exception as e:
        logger.error(f"Pipeline Execution Failed: {e}")