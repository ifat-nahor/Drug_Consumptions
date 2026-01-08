import pandas as pd
import os
from tests.data_cleaning import execute_complete_cleaning_pipeline

## Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, 'data', 'raw', 'Drug_Consumption.csv')

print(f"Loading data from {csv_path}...")
print(f"File exists: {os.path.exists(csv_path)}\n")

## Load raw data
df = pd.read_csv(csv_path)

print("="*80)
print("DATASET INFORMATION")
print("="*80)
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn Names:")
print(df.columns.tolist())

## AUTO-DETECT COLUMN NAMES
print("\n" + "="*80)
print("AUTO-DETECTING COLUMN NAMES")
print("="*80)

numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
print(f"Numeric columns found: {numeric_cols}")

if 'ID' in numeric_cols:
    numeric_cols.remove('ID')
    print("(Removed 'ID' from analysis)")

personality_cols = []
for col in numeric_cols:
    col_min = df[col].min()
    col_max = df[col].max()
    if col_min >= -4 and col_max <= 4:
        personality_cols.append(col)

print(f"\n✓ Detected {len(personality_cols)} personality trait columns:")
print(f"  {personality_cols}")

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns found: {categorical_cols}")

drug_columns = []
for col in categorical_cols:
    unique_values = df[col].unique()
    if any('CL' in str(val) for val in unique_values):
        drug_columns.append(col)

print(f"\n✓ Detected {len(drug_columns)} drug consumption columns:")
print(f"  {drug_columns}")

## VALIDATE
print("\n" + "="*80)
print("VALIDATION")
print("="*80)

if len(personality_cols) == 0:
    print("⚠ Warning: No personality columns detected!")
    personality_cols = numeric_cols

if len(drug_columns) == 0:
    print("⚠ Warning: No drug consumption columns detected!")

print(f"✓ Personality columns: {personality_cols}")
print(f"✓ Drug columns: {drug_columns}")

## RUN PIPELINE
print("\n" + "="*80)
print("RUNNING DATA CLEANING PIPELINE")
print("="*80)

df_cleaned, cleaning_report = execute_complete_cleaning_pipeline(
    df,
    personality_cols=personality_cols,
    drug_columns=drug_columns
)

## Save cleaned data
output_path = os.path.join(current_dir, 'data', 'processed', 'Drug_Consumption_Cleaned.csv')
df_cleaned.to_csv(output_path, index=False)
print(f"\n✓ Cleaned data saved to {output_path}")

print("\n" + "="*80)
print("PIPELINE COMPLETED SUCCESSFULLY")
print("="*80)
