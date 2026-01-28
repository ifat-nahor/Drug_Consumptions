import runpy
from pathlib import Path
from src.data_cleaning import run_data_cleaning_pipeline
from src.Personality_Based_Drug_Usage_Predictor import run_personality_based_drug_usage_pipeline

def main():
    # Stage 1: Data Cleaning
    df_cleaned, cleaning_report = run_data_cleaning_pipeline()
    # Stage 2: Visualizations (ANOVA & Group Trends)
    # Locating the visualization scripts within the src/visiolazation directory
    vis_path = Path(__file__).parent / 'src' 
    runpy.run_path(str(vis_path / 'anova_per_drug_SS_IMP.py'))
    runpy.run_path(str(vis_path / 'grathbygroups.py'))
    print("all ok")
    # Stage 3: Prediction Model Pipeline
    run_personality_based_drug_usage_pipeline()
    

if __name__ == "__main__":
    main()