import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import logging
import warnings
import os

# --- 1. SETTINGS & LOGGING ---
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', force=True)
logger = logging.getLogger("GatewayAnalysis")

# Using the Absolute Path to ensure Windows finds the folder correctly
# Based on your previous screenshots:
output_folder = r"D:\Users\User\Documents\project git 2026\final_proj\plots_pics_for_behavioral_and_gateway"

# Double check if folder exists, if not, create it to prevent the Errno 2
if not os.path.exists(output_folder):
    try:
        os.makedirs(output_folder)
        logger.info(f"Created missing folder: {output_folder}")
    except Exception as e:
        logger.error(f"Could not create folder: {e}")

def run_gateway_analysis(df):
    gateway_drugs = ['Alcohol', 'Cannabis', 'Nicotine']
    advanced_drugs = ['Coke', 'Ecstasy', 'LSD', 'Benzos']
    
    # --- 2. DYNAMIC CLUSTER IDENTIFICATION ---
    stats = df.groupby('Cluster')[['Escore', 'Impulsive', 'SS', 'Nscore']].mean()
    
    impulsive_id = (stats['Impulsive'] + stats['SS']).idxmax()
    remaining_ids = [i for i in df['Cluster'].unique() if i != impulsive_id]
    extroverted_id = stats.loc[remaining_ids, 'Escore'].idxmax()
    stable_id = [i for i in remaining_ids if i != extroverted_id][0]

    cluster_mapping = {
        impulsive_id: "C1_Impulsive",
        extroverted_id: "C0_Extroverted",
        stable_id: "C2_Stable"
    }

    drug_cols = gateway_drugs + advanced_drugs
    df[drug_cols] = df[drug_cols].apply(pd.to_numeric, errors='coerce')

    for cluster_id, profile_name in cluster_mapping.items():
        cluster_data = df[df['Cluster'] == cluster_id]
        corr_matrix = cluster_data[drug_cols].corr()
        gateway_corr = corr_matrix.loc[gateway_drugs, advanced_drugs]
        
        logger.info(f"Generating plots for: {profile_name}")

        # --- 3. Heatmap ---
        plt.figure(figsize=(8, 5))
        sns.heatmap(gateway_corr, annot=True, cmap='YlOrRd', fmt=".2f", vmin=0.2, vmax=0.6)
        plt.title(f"Gateway Correlation Matrix - {profile_name}")
        plt.tight_layout()
        
        # Save using absolute path
        plt.savefig(os.path.join(output_folder, f"heatmap_{profile_name}.png"))
        plt.close() 

        # --- 4. Network Map ---
        _plot_gateway_network(cluster_data, profile_name, gateway_drugs, advanced_drugs)

def _plot_gateway_network(cluster_data, profile_name, g_cols, a_cols):
    corr = cluster_data[g_cols + a_cols].corr()
    G = nx.Graph()
    threshold = 0.25
    for g in g_cols:
        for a in a_cols:
            weight = corr.loc[g, a]
            if weight > threshold:
                G.add_edge(g, a, weight=weight)

    if len(G.edges()) > 0:
        plt.figure(figsize=(10, 7))
        pos = nx.circular_layout(G)
        weights = [G[u][v]['weight'] * 10 for u, v in G.edges()]
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lavender', edgecolors='black')
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='darkred', alpha=0.5)
        nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
        plt.title(f"Visual Gateway Map - {profile_name}")
        plt.axis('off')
        
        # Save using absolute path
        plt.savefig(os.path.join(output_folder, f"network_{profile_name}.png"))
        plt.close()

if __name__ == "__main__":
    try:
        import Personality_Based_Drug_Usage_Predictor as predictor
        df_raw, scaled = predictor.load_and_scale_data(predictor.DATA_FILE, predictor.PERSONALITY_COLS)
        pca_results = predictor.apply_pca(scaled, predictor.PCA_COMPONENTS)
        df_final = predictor.perform_final_clustering(df_raw, pca_results, predictor.FINAL_K)
        
        run_gateway_analysis(df_final)
        print(f"\nSuccess! Images are waiting for you in: {output_folder}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")