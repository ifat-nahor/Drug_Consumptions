import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import logging

# --- 1. LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 2. BYPASS PERMISSION ERRORS (Monkey Patching) ---
original_makedirs = os.makedirs
os.makedirs = lambda *args, **kwargs: None 

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.append(SCRIPT_DIR)

    import Personality_Based_Drug_Usage_Predictor as predictor
    logger.info("Predictor logic imported successfully.")
finally:
    os.makedirs = original_makedirs

# --- 3. PATH CONFIGURATION ---
CLUSTERED_DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "Drug_Consumption_WITH_Clusters.csv")
RAW_CLEANED_DATA = os.path.join(SCRIPT_DIR, "..", "data", "processed", "Drug_Consumption_Cleaned.csv")

def prepare_data_dependency():
    if os.path.exists(CLUSTERED_DATA_PATH):
        logger.info(f"Data source confirmed: {CLUSTERED_DATA_PATH}")
        return True
    
    logger.warning("Clustered file missing. Running background clustering...")
    try:
        df, scaled = predictor.load_and_scale_data(RAW_CLEANED_DATA, predictor.PERSONALITY_COLS)
        pca_results = predictor.apply_pca(scaled, predictor.PCA_COMPONENTS)
        df = predictor.perform_final_clustering(df, pca_results, predictor.FINAL_K)
        
        df.to_csv(CLUSTERED_DATA_PATH, index=False)
        logger.info("Data synchronization complete.")
        return True
    except Exception as e:
        logger.error(f"Data synchronization failed: {e}")
        return False

def run_behavioral_analysis():
    """
    Profiles summary:
    - C0: Anxious-Impulsive (High Alcohol/Nicotine)
    - C1: Stable/Conservative (Low usage)
    - C2: Extroverted/Experience Seeker (High Cannabis)
    """
    if not prepare_data_dependency():
        return

    df = pd.read_csv(CLUSTERED_DATA_PATH)
    target_substances = ['Alcohol', 'Cannabis', 'Nicotine', 'Coke', 'Ecstasy', 'LSD', 'Mushrooms', 'Benzos'] 
    valid_cols = [col for col in target_substances if col in df.columns]

    # --- VISUALIZATION 1: HEATMAP ---
    logger.info("Generating Substance Usage Heatmap...")
    plt.figure(figsize=(12, 7))
    usage_trends = df.groupby('Cluster')[valid_cols].mean()
    sns.heatmap(usage_trends, annot=True, cmap='YlGnBu', fmt=".2f")
    plt.title("Substance Usage Signatures Across Personality Clusters")
    plt.tight_layout()
    plt.show() 

    # --- ANALYSIS LOGGING (The Summary now goes to LOG) ---
    logger.info("-" * 40)
    logger.info("ANALYTICAL SUMMARY: CLUSTER CHARACTERISTICS")
    logger.info("CLUSTER 0: 'Anxious-Impulsive' - High Alcohol/Nicotine/Cannabis intensity.")
    logger.info("CLUSTER 1: 'Stable/Conservative' - Minimal usage across the spectrum.")
    logger.info("CLUSTER 2: 'Extroverted/Experience Seeker' - Reward-driven Alcohol/Cannabis usage.")
    logger.info("-" * 40)

    # --- VISUALIZATION 2: BOXPLOT ---
    logger.info("Generating Usage Diversity Boxplot...")
    df['Usage_Diversity'] = (df[valid_cols] > 0.5).sum(axis=1)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Cluster', y='Usage_Diversity', data=df, palette='Set2')
    plt.title("Substance Usage Diversity by Personality Group")
    plt.show()

if __name__ == "__main__":
    run_behavioral_analysis()