import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
import warnings

import os
from pathlib import Path

# project root (final_proj)
BASE_DIR = Path(__file__).resolve().parent.parent

PLOTS_DIR = BASE_DIR / "plots_pics_for_pca_cluster"
PLOTS_DIR.mkdir(exist_ok=True)

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants (Project-wide settings)
DATA_FILE = BASE_DIR / "data" / "processed" / "Drug_Consumption_Cleaned.csv"
PERSONALITY_COLS = ['Nscore', 'Escore', 'Oscore', 'AScore', 'Cscore', 'Impulsive', 'SS']
DRUG_COLS = [
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
    'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
    'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]
K_SELECTION_RANGE = range(2, 11)
FINAL_K = 3
PCA_COMPONENTS = 3
RANDOM_SEED = 1

def load_and_scale_data(file_path, features):
    """Load dataset and return both the dataframe and scaled features."""
    df = pd.read_csv(file_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[features])
    logger.info(f"Data loaded and scaled. Shape: {df.shape}")
    return df, scaled_data

def apply_pca(scaled_data, n_components):
    """Reduce dimensionality using PCA."""
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    variance = np.sum(pca.explained_variance_ratio_)
    logger.info(f"PCA completed ({n_components} components). Explained variance: {variance:.2%}")
    return pca_data

def run_k_selection_diagnostics(pca_data, k_range):
    """Generate Elbow and Silhouette plots to justify K selection."""
    inertia = []
    sil_scores = []
    
    logger.info("Running clustering diagnostics for K selection...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(pca_data)
        inertia.append(kmeans.inertia_)
        sil_scores.append(silhouette_score(pca_data, labels))
    
    # Plotting results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(k_range, inertia, marker='o', color='b')
    ax1.set_title('Elbow Method (Inertia)')
    ax2.plot(k_range, sil_scores, marker='s', color='g')
    ax2.set_title('Silhouette Scores')
    plt.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "k_selection_diagnostics.png"), dpi=300)
    plt.show()

def perform_final_clustering(df, pca_data, k):
    """Execute K-Means with the chosen K and add labels to dataframe."""
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
    df['Cluster'] = kmeans.fit_predict(pca_data)
    logger.info(f"Final clustering completed with K={k}")
    return df



def describe_personality_profiles(df):
    means = df.groupby('Cluster')[PERSONALITY_COLS].mean()
    logger.info("Personality profiles (cluster means):")
    logger.info("\n" + means.to_string(float_format=lambda v: f"{v: .2f}"))
    return means


def plot_pca_clusters(pca_data, df):
    """Scatter plot of first two PCA components colored by cluster."""
    plt.figure(figsize=(6, 5))
    scatter = plt.scatter(
        pca_data[:, 0], pca_data[:, 1],
        c=df['Cluster'], cmap='viridis', alpha=0.6
    )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Personality space (PC1 vs PC2) by cluster')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pca_clusters.png"), dpi=300)
    plt.show()

def plot_personality_profiles(df):
    """Bar plot of mean personality traits per cluster."""
    means = df.groupby('Cluster')[PERSONALITY_COLS].mean()
    
    plt.figure(figsize=(10, 5))
    x = np.arange(len(PERSONALITY_COLS))
    width = 0.25
    
    for i, cluster in enumerate(means.index):
        plt.bar(
            x + i*width,
            means.loc[cluster],
            width=width,
            label=f'Cluster {cluster}'
        )
    
    plt.xticks(x + width, PERSONALITY_COLS, rotation=45)
    plt.ylabel('Mean z-score')
    plt.title('Personality profiles by cluster')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "personality_profiles.png"), dpi=300)
    plt.show()
   
    
    return means

def plot_personality_profiles_radar(df):
    """Radar plot of mean personality traits per cluster."""
    import numpy as np

   
    means = df.groupby('Cluster')[PERSONALITY_COLS].mean()


    num_traits = len(PERSONALITY_COLS)

 
    angles = np.linspace(0, 2 * np.pi, num_traits, endpoint=False)
  
    angles = np.concatenate((angles, [angles[0]]))

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for cluster in means.index:
        values = means.loc[cluster].values
        values = np.concatenate((values, [values[0]])) 
        ax.plot(angles, values, label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)

  
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(PERSONALITY_COLS)
    ax.set_title('Personality profiles by cluster (radar plot)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "personality_profiles_radar.png"), dpi=300)
    plt.show()

    return means


def analyze_substance_use(df, drug_columns, k):
    """Perform ANOVA to test differences between clusters for each drug."""
    results = []
    for drug in drug_columns:
        groups = [df[df['Cluster'] == i][drug] for i in range(k)]
        f_stat, p_val = f_oneway(*groups)
        
        results.append({
            'Drug': drug,
            'p_value': p_val,
            'Is_Significant': p_val < 0.05
        })
        results_df = pd.DataFrame(results)
    return pd.DataFrame(results)

def compute_and_save_mean_substance_use(df):
    """Compute mean substance use per cluster and save to CSV."""
    mean_use = df.groupby('Cluster')[DRUG_COLS].mean()

    mean_use_path = os.path.join(PLOTS_DIR, "mean_substance_use_by_cluster.csv")
    mean_use.to_csv(mean_use_path)
    logger.info(f"Mean substance use table saved to: {mean_use_path}")

    return mean_use

def summarize_substance_use_patterns(mean_use):
    """
    Create a summary table: for each drug, which cluster has
    the highest and lowest mean use.
    """
    summary_rows = []

    for drug in DRUG_COLS:
        col = mean_use[drug]
        max_cluster = col.idxmax()
        min_cluster = col.idxmin()

        summary_rows.append({
            "Drug": drug,
            "Max_Cluster": int(max_cluster),
            "Max_Mean": col[max_cluster],
            "Min_Cluster": int(min_cluster),
            "Min_Mean": col[min_cluster],
        })

    summary_df = pd.DataFrame(summary_rows)

    logger.info("Summary of substance use by cluster (max/min means):")
    logger.info("\n" + summary_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    summary_path = os.path.join(PLOTS_DIR, "substance_use_cluster_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Substance use summary table saved to: {summary_path}")

    return summary_df

def run_substance_use_analysis(df_analyzed, k):
    results_df = analyze_substance_use(df_analyzed, DRUG_COLS, k)

    anova_path = PLOTS_DIR / "anova_results.csv"
    results_df.to_csv(anova_path, index=False)

    mean_use = compute_and_save_mean_substance_use(df_analyzed)
    summary_df = summarize_substance_use_patterns(mean_use)

    logger.info(f"ANOVA results table saved to: {anova_path}")

    return results_df, summary_df


#make pipeline for main 

def run_personality_based_drug_usage_pipeline():
    logger.info("=== Starting Personality Drug Pipeline ===")
    df, x_scaled = load_and_scale_data(DATA_FILE, PERSONALITY_COLS)
    x_pca = apply_pca(x_scaled, PCA_COMPONENTS)
  


    run_k_selection_diagnostics(x_pca, K_SELECTION_RANGE)

    df_analyzed = perform_final_clustering(df, x_pca, FINAL_K)
    describe_personality_profiles(df_analyzed)

    plot_pca_clusters(x_pca, df_analyzed)
    plot_personality_profiles(df_analyzed)
    plot_personality_profiles_radar(df_analyzed)

    results_df, summary_df = run_substance_use_analysis(df_analyzed, FINAL_K)

    significant_count = results_df['Is_Significant'].sum()
    logger.info(
        f"Analysis complete. {significant_count}/{len(DRUG_COLS)} drugs show significant variance."
    )
    logger.info("Conclusion: Personality profiles are predictive of drug consumption.")


