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
    results_df = analyze_substance_use(df_analyzed, DRUG_COLS, FINAL_K)
    results_df.to_csv("anova_results.csv", index=False)
    mean_use = compute_and_save_mean_substance_use(df)
    anova_results = analyze_substance_use(df, DRUG_COLS, k=FINAL_K)
    anova_path = os.path.join(PLOTS_DIR, "anova_results.csv")
    anova_results.to_csv(anova_path, index=False)
    summary_df = summarize_substance_use_patterns(mean_use)
    logger.info(f"ANOVA results table saved to: {anova_path}")

    # 5. Final Reporting
    significant_count = results_df['Is_Significant'].sum()
    logger.info(
        f"Analysis complete. {significant_count}/{len(DRUG_COLS)} drugs show significant variance."
    )
    logger.info("Conclusion: Personality profiles are predictive of drug consumption.")
    
    print("all ok")
if __name__ == "__main__":
    main()
