import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import logging
import sys
import warnings

# --- 1. CLEAN LOGGING & WARNINGS ---
# Silence technical noise from libraries
warnings.filterwarnings("ignore")
logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)
logger = logging.getLogger("GatewayAnalysis")

def run_gateway_analysis(df):
    """
    Identifies 'Gateway' correlations between substances for each cluster.
    """
    gateway_drugs = ['Alcohol', 'Cannabis', 'Nicotine']
    advanced_drugs = ['Coke', 'Ecstasy', 'LSD', 'Benzos']
    
    g_cols = [c for c in gateway_drugs if c in df.columns]
    a_cols = [c for c in advanced_drugs if c in df.columns]

    # Ensure numeric types to avoid the "categorical units" warning
    for col in g_cols + a_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for cluster_id in sorted(df['Cluster'].unique()):
        cluster_data = df[df['Cluster'] == cluster_id]
        
        # Calculate Correlation Matrix for the heatmap
        corr_matrix = cluster_data[g_cols + a_cols].corr()
        gateway_corr = corr_matrix.loc[g_cols, a_cols]
        
        # --- ANALYTICAL INSIGHTS ---
        # Find the strongest gateway connection in this cluster
        top_relation = gateway_corr.unstack().idxmax()
        max_val = gateway_corr.unstack().max()
        
        logger.info("-" * 45)
        logger.info(f"GATEWAY ANALYSIS: CLUSTER {cluster_id}")
        logger.info(f"-> Strongest Correlation: {top_relation[1]} is linked to {top_relation[0]}.")
        logger.info(f"-> Connection Strength: {max_val:.2f}")
        logger.info("-" * 45)

        # Visual 1: Interaction Heatmap
        plt.figure(figsize=(8, 5))
        sns.heatmap(gateway_corr, annot=True, cmap='Reds', fmt=".2f")
        plt.title(f"Gateway Correlation: Cluster {cluster_id}")
        plt.tight_layout()
        plt.show()

        # Visual 2: Gateway Network Map
        _plot_network(cluster_data, cluster_id, g_cols, a_cols)

def _plot_network(cluster_data, cluster_id, g_cols, a_cols):
    """Generates an intuitive graph where edge thickness represents correlation strength."""
    corr = cluster_data[g_cols + a_cols].corr()
    G = nx.Graph()
    
    threshold = 0.25
    for g in g_cols:
        for a in a_cols:
            weight = corr.loc[g, a]
            if weight > threshold:
                G.add_edge(g, a, weight=weight)

    if len(G.edges()) > 0:
        plt.figure(figsize=(10, 6))
        pos = nx.shell_layout(G)
        weights = [G[u][v]['weight'] * 8 for u, v in G.edges()]
        
        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='lavender', edgecolors='black')
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='tomato', alpha=0.6)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        plt.title(f"Visual Gateway Map: Cluster {cluster_id}")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    try:
        # Import Ifat's script to execute the clustering pipeline
        import Personality_Based_Drug_Usage_Predictor as predictor
        df_raw, scaled = predictor.load_and_scale_data(predictor.DATA_FILE, predictor.PERSONALITY_COLS)
        pca_results = predictor.apply_pca(scaled, predictor.PCA_COMPONENTS)
        df_final = predictor.perform_final_clustering(df_raw, pca_results, predictor.FINAL_K)
        
        run_gateway_analysis(df_final)
    except Exception as e:
        logger.error(f"Failed to run Gateway Analysis: {e}")