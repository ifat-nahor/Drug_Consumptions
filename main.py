# main.py
"""from src.data_cleaning import run_data_cleaning_pipeline

def main():
    df_cleaned, cleaning_report = run_data_cleaning_pipeline()


from src.Personality_Based_Drug_Usage_Predictor import (
    run_personality_based_drug_usage_pipeline
)

def main():
    run_personality_based_drug_usage_pipeline()
    print("all ok")

if __name__ == "__main__":
    main()"""""

from src.data_cleaning import run_data_cleaning_pipeline
from src.Personality_Based_Drug_Usage_Predictor import run_personality_based_drug_usage_pipeline

def main():
    df_cleaned, cleaning_report = run_data_cleaning_pipeline()
    run_personality_based_drug_usage_pipeline()
    print("all ok")

if __name__ == "__main__":
    main()
