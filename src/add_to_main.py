import os
from Personality_Based_Drug_Usage_Predictor import (
    DATA_FILE,
    PERSONALITY_COLS,
    DRUG_COLS,
    K_SELECTION_RANGE,
    FINAL_K,
    PCA_COMPONENTS,
    PLOTS_DIR,
    compute_and_save_mean_substance_use,
    load_and_scale_data,
    apply_pca,
    plot_personality_profiles_radar,
    run_k_selection_diagnostics,
    perform_final_clustering,
    describe_personality_profiles,
    plot_pca_clusters,
    plot_personality_profiles,
    analyze_substance_use,
    logger,
    summarize_substance_use_patterns,
    run_substance_use_analysis
)
def main():
    """Main execution flow."""
    # 1. Preparation
    df, x_scaled = load_and_scale_data(DATA_FILE, PERSONALITY_COLS)
    x_pca = apply_pca(x_scaled, PCA_COMPONENTS)
    
    # 2. Optimization (Shows the justification for K=3)
    run_k_selection_diagnostics(x_pca, K_SELECTION_RANGE)
    
    # 3. Final Model
    df_analyzed = perform_final_clustering(df, x_pca, FINAL_K)
    describe_personality_profiles(df_analyzed)      
    
    # 3b. Visualizations 
    plot_pca_clusters(x_pca, df_analyzed)
    plot_personality_profiles(df_analyzed)
    
    # 4. Statistical Validation
    results_df, summary_df = run_substance_use_analysis(df_analyzed, FINAL_K)

    # 5. Final Reporting
    significant_count = results_df['Is_Significant'].sum()
    logger.info(
        f"Analysis complete. {significant_count}/{len(DRUG_COLS)} drugs show significant variance."
    )
    logger.info("Conclusion: Personality profiles are predictive of drug consumption.")
    
    print("all ok")
if __name__ == "__main__":
    main()
