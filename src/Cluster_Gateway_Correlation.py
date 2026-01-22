import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import networkx as nx

# --- 1. LOGGING CONFIGURATION ---
# Setting up a professional logger to display analytical insights in the terminal
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 2. BYPASS PERMISSION ERRORS (Monkey Patching) ---
# Temporarily disabling directory creation to prevent crashes caused by 
# hardcoded paths in imported modules (e.g., paths belonging to other users)
original_makedirs = os.makedirs
os.makedirs = lambda *args, **kwargs: None 

try:
    # Identify the script's directory for relative path management
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    if SCRIPT_DIR not in sys.path:
        sys.path.append(SCRIPT_DIR)

    # Importing project dependencies while the bypass is active
    import Personality_Based_Drug_Usage_Predictor as predictor
    logger.info("Successfully linked to project dependencies and bypassed local path errors.")
finally:
    # Restore original system functionality
    os.makedirs = original_makedirs

# --- 3. PATH CONFIGURATION ---
# Defining the relative path to the pre-processed clustered dataset
DATA_PATH = os.path.join(SCRIPT_DIR, "..", "data", "processed", "Drug_Consumption_WITH_Clusters.csv")

def plot_gateway_network(cluster_data, cluster_id, gateway_drugs, advanced_drugs):
    """
    Creates an intuitive network visualization showing 'Gateway' pathways.
    - Nodes: Represent specific substances.
    - Edges (Lines): Represent the correlation strength between substances.
    - Thickness: Determined by the Pearson correlation coefficient.
    """
    # Calculate the correlation matrix for the specific cluster
    corr = cluster_data[gateway_drugs + advanced_drugs].corr()
    
    # Initialize a NetworkX graph object
    G = nx.Graph()
    
    # Add edges only for correlations exceeding a significance threshold (0.25)
    # This ensures the visual output focuses on meaningful patterns
    threshold = 0.25
    for g in gateway_drugs:
        for a in advanced_drugs:
            weight = corr.loc[g, a]
            if weight > threshold:
                G.add_edge(g, a, weight=weight)

    if len(G.edges()) == 0:
        logger.warning(f"No significant gateway patterns found for Cluster {cluster_id}")
        return

    # Visualization Setup
    plt.figure(figsize=(10, 7))
    # Using a circular layout to clearly separate gateway and target drugs
    pos = nx.shell_layout(G) 
    
    # Adjust edge thickness based on correlation strength (scaled for visibility)
    weights = [G[u][v]['weight'] * 8 for u, v in G.edges()]
    
    # Draw Graph Components
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='lavender', edgecolors='black')
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='tomato', alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold')
    
    plt.title(f"Visual Gateway Map: Cluster {cluster_id}\n(Line Thickness = Correlation Strength)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def analyze_gateway_patterns(df):
    """
    Core analysis function: Iterates through each cluster to generate 
    Heatmaps and Network Graphs for substance interaction analysis.
    """
    # Define drug categories for research hypothesis
    gateway_drugs = ['Alcohol', 'Cannabis', 'Nicotine']
    advanced_drugs = ['Coke', 'Ecstasy', 'LSD', 'Benzos']
    
    for cluster_id in sorted(df['Cluster'].unique()):
        # Isolate data for the current personality cluster
        cluster_data = df[df['Cluster'] == cluster_id]
        
        # --- PHASE 1: Targeted Heatmap ---
        # Provides precise statistical correlation values
        corr_matrix = cluster_data[gateway_drugs + advanced_drugs].corr()
        gateway_corr = corr_matrix.loc[gateway_drugs, advanced_drugs]
        
        logger.info(f"Generating Statistical Heatmap for Cluster {cluster_id}...")
        plt.figure(figsize=(8, 5))
        sns.heatmap(gateway_corr, annot=True, cmap='Reds', fmt=".2f")
        plt.title(f"Gateway Correlation Matrix: Cluster {cluster_id}")
        plt.show()

        # --- PHASE 2: Intuitive Network Map ---
        # Provides a behavioral 'pathway' visualization
        logger.info(f"Generating Behavioral Network Map for Cluster {cluster_id}...")
        plot_gateway_network(cluster_data, cluster_id, gateway_drugs, advanced_drugs)

        # --- PHASE 3: Automated Insight Generation ---
        # Automatically identifies and logs the strongest link for presentation notes
        top_relation = gateway_corr.unstack().idxmax()
        max_val = gateway_corr.unstack().max()
        logger.info(f"🏆 INSIGHT C{cluster_id}: Strongest Link is {top_relation[1]} -> {top_relation[0]} (R={max_val:.2f})")
        logger.info("="*65)

if __name__ == "__main__":
    # Check if the required data source is available
    if os.path.exists(DATA_PATH):
        logger.info(f"Clustered dataset found. Starting Analysis...")
        df_clustered = pd.read_csv(DATA_PATH)
        analyze_gateway_patterns(df_clustered)
    else:
        logger.error(f"Missing Source File: {DATA_PATH}")
        logger.warning("Ensure you have run the primary clustering script before executing this analysis.")