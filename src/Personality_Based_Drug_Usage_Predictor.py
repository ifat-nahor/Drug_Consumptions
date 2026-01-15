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

# Configuration
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants (Project-wide settings)
DATA_FILE = r"C:\Users\ifatn\OneDrive\phyton_bit\final_proj\final_proj\data\processed\Drug_Consumption_Cleaned.csv"
PERSONALITY_COLS = ['Nscore', 'Escore', 'Oscore', 'AScore', 'Cscore', 'Impulsive', 'SS']
DRUG_COLS = [
    'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc',
    'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD',
    'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'
]
K_SELECTION_RANGE = range(2, 11)
FINAL_K = 3
PCA_COMPONENTS = 3
RANDOM_SEED = 42

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
    plt.savefig("k_selection_diagnostics.png", dpi=300)
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
    plt.savefig("pca_clusters.png", dpi=300)
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
    plt.savefig("personality_profiles.png", dpi=300)
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
    return pd.DataFrame(results)

