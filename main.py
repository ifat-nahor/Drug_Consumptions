# main.py
from src.data_cleaning import run_data_cleaning_pipeline

def main():
    df_cleaned, cleaning_report = run_data_cleaning_pipeline()

if __name__ == "__main__":
    main()
