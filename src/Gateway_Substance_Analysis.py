import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import os
import warnings

warnings.filterwarnings("ignore")
OUTPUT_FOLDER = "plots_pics_for_behavioral_and_gateway"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def run_gateway_analysis(df):
    gateway_drugs = ['Alcohol', 'Cannabis', 'Nicotine']
    advanced_drugs = ['Coke', 'Ecstasy', 'LSD', 'Benzos']
    
    # 1. חישוב ממוצע אימפולסיביות לכל קלאסטר
    means = df.groupby('Cluster')['Impulsive'].mean()
    
    # 2. זיהוי חד-משמעי של הקבוצות לפי הערכים
    imp_idx = means.idxmax()  # הקבוצה הכי אימפולסיבית (ה-0.53 בקשר)
    sta_idx = means.idxmin()  # הקבוצה הכי יציבה (ה-0.40 בקשר)
    ext_idx = [i for i in means.index if i not in [imp_idx, sta_idx]][0]

    mapping = {
        imp_idx: "C0_Anxious_Impulsive",
        sta_idx: "C1_Stable_Conservative",
        ext_idx: "C2_Extroverted"
    }

    for cluster_id, profile_name in mapping.items():
        # ניקוי זיכרון הגרפים לפני כל סיבוב
        plt.close('all')
        plt.clf()
        
        cluster_data = df[df['Cluster'] == cluster_id]
        
        # חישוב קורלציה ספציפי לקבוצה הזו
        corr_matrix = cluster_data[gateway_drugs + advanced_drugs].apply(pd.to_numeric).corr()
        gateway_corr = corr_matrix.loc[gateway_drugs, advanced_drugs]

        # בדיקה לטרמינל: מה הקשר בין קנאביס לאקסטזי בקבוצה הזו?
        val = gateway_corr.loc['Cannabis', 'Ecstasy']
        print(f"DEBUG: {profile_name} | Cannabis-Ecstasy Correlation: {val:.2f}")

        # יצירת HEATMAP
        plt.figure(figsize=(8, 5))
        sns.heatmap(gateway_corr, annot=True, cmap='Reds', fmt=".2f", vmin=0.2, vmax=0.6)
        plt.title(f"Gateway Analysis: {profile_name.replace('_', ' ')}")
        plt.tight_layout()
        
        # שמירה עם שם קובץ ייחודי
        plt.savefig(f"{OUTPUT_FOLDER}/gateway_heatmap_{profile_name}.png")
        plt.close()

        # יצירת NETWORK
        _plot_network(cluster_data, profile_name, gateway_drugs, advanced_drugs)

def _plot_network(cluster_data, profile_name, g_cols, a_cols):
    plt.close('all')
    plt.clf()
    corr = cluster_data[g_cols + a_cols].apply(pd.to_numeric).corr()
    G = nx.Graph()
    for g in g_cols:
        for a in a_cols:
            if corr.loc[g, a] > 0.25:
                G.add_edge(g, a, weight=corr.loc[g, a])
    
    plt.figure(figsize=(10, 6))
    if len(G.edges()) > 0:
        pos = nx.spring_layout(G, k=0.5)
        nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lavender', 
                font_weight='bold', edge_color='tomato', width=2)
        
    plt.title(f"Network Map: {profile_name.replace('_', ' ')}")
    plt.axis('off')
    plt.savefig(f"{OUTPUT_FOLDER}/gateway_network_{profile_name}.png")
    plt.close()

if __name__ == "__main__":
    import Personality_Based_Drug_Usage_Predictor as predictor
    from sklearn.cluster import KMeans
    
    df_raw, scaled = predictor.load_and_scale_data(predictor.DATA_FILE, predictor.PERSONALITY_COLS)
    pca_res = predictor.apply_pca(scaled, predictor.PCA_COMPONENTS)
    
    kmeans = KMeans(n_clusters=3, random_state=1, n_init=10)
    df_raw['Cluster'] = kmeans.fit_predict(pca_res)
    
    run_gateway_analysis(df_raw)
    print("\nהגרפים נוצרו מחדש. אנא בדקי את הקבצים בתיקייה.")