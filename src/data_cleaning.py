import pandas as pd
import numpy as np
from scipy import stats
import warnings
import os
import logging

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print(" Warning: 'sklearn' library not found. Clustering will be skipped.")

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================
# Avoid raw numbers/strings
RANDOM_STATE = 1
KMEANS_CLUSTERS = 3
KMEANS_N_INIT = 10
DISCRETE_THRESHOLD = 10

# Drug Families Mapping
DRUG_FAMILIES = {
    'Stimulants': ['Amphet', 'Coke', 'Crack', 'Meth', 'Nicotine'],
    'Depressants': ['Alcohol', 'Benzos', 'Heroin'],
    'Hallucinogens': ['LSD', 'Mushrooms']
}

# Valid drug categories and columns
VALID_DRUG_CATEGORIES = ['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6']

CONSUMPTION_MAPPING = {
    'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3,
    'CL4': 4, 'CL5': 5, 'CL6': 6
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
# Use a logger instead of print statements
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_cleaning.log", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

"""
PROJECT STRUCTURE & FUNCTION MAP
==============================================================================
SECTION 1: Data Quality Assessment
├── assess_dataset_overview()          <-- Metrics & Shape
├── assess_missing_values()            <-- Null checks
└── detect_and_remove_duplicates()     <-- Deduplication

SECTION 2: Data Type Validation
└── validate_and_convert_numeric_columns()

SECTION 3: Outlier Detection
└── assess_outliers_personality_traits()

SECTION 4: Categorical Data Validation
├── validate_drug_consumption_encoding()
└── convert_drug_consumption_to_numeric()

SECTION 5: Demographic Validation
└── assess_demographic_variables()

SECTION 6: Statistical Assessment
├── calculate_descriptive_statistics()
└── calculate_primary_correlation()

SECTION 7: Normality Testing
└── assess_normality_and_distribution()

SECTION 8: Master Cleaning Pipeline
└── execute_complete_cleaning_pipeline() <-- Orchestrates Sections 1-7

SECTION 9: Advanced Feature Engineering (Post-Cleaning)
├── aggregate_drug_families()          <-- Creates Stimulants/Depressants scores
├── create_usage_groups_kmeans()       <-- ML Clustering (User Profiles)
└── execute_feature_engineering()      <-- Runs the engineering steps (Step 11)

SECTION 10: Execution Wrapper
├── detect_personality_and_drug_columns()
└── run_data_cleaning_pipeline()       <-- MAIN ENTRY POINT: Loads, Cleans, Saves
==============================================================================
"""

## Prints a standardized header to keep the code clean.
def log_step(title):
    logger.info(f"\n{'='*80}\n[{title}]\n{'='*80}")
    
## ============================================================================
## SECTION 1: DATA QUALITY ASSESSMENT FUNCTIONS
## ============================================================================

def assess_dataset_overview(df):
    """Prints basic dataset metrics (rows, columns, memory) for initial assessment."""
    log_step("STEP 1: DATASET OVERVIEW")
    
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    
    logger.info(f"Records: {df.shape[0]:,} | Features: {df.shape[1]}")
    logger.info(f"Memory Usage: {memory_mb:.2f} MB")
    
    return {'rows': df.shape[0], 'cols': df.shape[1], 'memory': memory_mb}


def assess_missing_values(df):
    """Identifies and prints columns containing missing data to ensure data completeness."""
    log_step("STEP 2: MISSING VALUES")
    
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    
    # quit early if no missing values
    if missing_counts.empty:
        logger.info(" No missing values detected - dataset is complete")
        return missing_counts
        
    missing_percentage = (missing_counts / len(df)) * 100
    
    report = pd.concat([missing_counts, missing_percentage], axis=1, keys=['Missing_Count', 'Missing_%'])
    
    logger.info(report.round(2))
    
    return report
    

def detect_and_remove_duplicates(df):
    """Detects and removes duplicate rows to ensure data integrity."""
    
    log_step("STEP 3: DUPLICATE DETECTION")
    
    initial_rows = df.shape[0]
    df = df.drop_duplicates(keep='first')
    duplicates_removed = initial_rows - df.shape[0]

    if duplicates_removed > 0:
        logger.info(f" Warning: Removed {duplicates_removed} duplicate rows")
        logger.info(f"New dataset size: {df.shape[0]:,} records")
    else:
        logger.info("No duplicate records found")

    return df, duplicates_removed

## ============================================================================
## SECTION 2: DATA TYPE VALIDATION FUNCTIONS
## ============================================================================

def validate_and_convert_numeric_columns(df, numeric_columns):
    """Converts specified columns to numeric types and tracks changes."""
    log_step("STEP 4: NUMERIC CONVERSION")
    
    cols_to_fix = [col for col in numeric_columns if col in df.columns]
    
    logger.info(f"Current Data Types (before conversion):\n{df[cols_to_fix].dtypes}\n{'-' * 20}")

    old_dtypes = df[cols_to_fix].dtypes
    nans_before = df[cols_to_fix].isna().sum().sum()

    ## Convert to numeric, any non-numeric values become NaN
    df[cols_to_fix] = df[cols_to_fix].apply(pd.to_numeric, errors='coerce')

    new_dtypes = df[cols_to_fix].dtypes
    errors_created = df[cols_to_fix].isna().sum().sum() - nans_before
    converted_cols = new_dtypes[new_dtypes != old_dtypes].index.tolist()
    
    conversion_report = {'converted_columns': converted_cols, 'errors': errors_created}

    logger.info(f" Successfully converted {len(cols_to_fix)} columns to numeric format.")
    if errors_created > 0:
        logger.info(f" Warning: {errors_created} values became NaN.")
    
    return df, conversion_report

## ============================================================================
## SECTION 3: OUTLIER DETECTION FUNCTIONS
## ============================================================================

def detect_outliers_iqr(data, column):
    """Calculates outlier count and boundaries using 3*IQR (Conservative)."""
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    outlier_count = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()

    return outlier_count, lower_bound, upper_bound


def assess_outliers_personality_traits(df, personality_cols):
    """Generates a summary report of outliers for personality traits."""
    log_step("STEP 5: OUTLIER DETECTION - PERSONALITY SCORES")
    
    outlier_summary = []
    
    ## Apply outlier detection to all personality variables
    for col in personality_cols:
        if col in df.columns:
            n_outliers, lower, upper = detect_outliers_iqr(df, col)
            
            outlier_summary.append({
                'Feature': col,
                'Outliers': n_outliers,
                'Outlier_Rate_%': n_outliers/len(df)*100,
                'Lower_Bound': lower,
                'Upper_Bound': upper,
                'Min_Value': df[col].min(),
                'Max_Value': df[col].max()
            })
    
    outlier_df = pd.DataFrame(outlier_summary)
    
    if not outlier_df.empty:
        logger.info(outlier_df.round(3).to_string(index=False))
    else:
        logger.info("No outliers found.")

    logger.info("\n Note: Personality scores are standardized (mean=0, SD=1)")
    logger.info("   Extreme outliers may indicate data entry errors or measurement issues")
    
    return outlier_df

## ============================================================================
## SECTION 4: CATEGORICAL DATA VALIDATION FUNCTIONS
## ============================================================================

def validate_drug_consumption_encoding(df, drug_columns):
    """Validates that drug columns only contain CL0-CL6 values."""
    log_step("STEP 6: DRUG CONSUMPTION VALIDATION")
        
    ## Check for invalid category labels
    validation_report = {
        'valid_columns': 0,
        'invalid_columns': 0,
        'invalid_entries': {}
    }
    
    for drug in drug_columns:
        if drug in df.columns:
            ## Find any values not in the valid set
            invalid = df[~df[drug].isin(VALID_DRUG_CATEGORIES)][drug].unique()
            
            if len(invalid) > 0:
                validation_report['invalid_columns'] += 1
                validation_report['invalid_entries'][drug] = invalid.tolist()
            else:
                validation_report['valid_columns'] += 1
    
    if validation_report['invalid_entries']:
        logger.info(f"\n Warning: {validation_report['invalid_columns']} columns with invalid categories:")
        for drug, values in validation_report['invalid_entries'].items():
            logger.info(f"  {drug}: {values}")
    else:
        logger.info(f"All {validation_report['valid_columns']} drug columns have valid encodings")
    
    return validation_report

def convert_drug_consumption_to_numeric(df, drug_columns):
    """Converts categorical values (CL0-CL6) to numeric (0-6)."""
    log_step("STEP 6.5: CONVERT DRUG CONSUMPTION TO NUMERIC")
    
    cols = [c for c in drug_columns if c in df.columns]
    
    df[cols] = df[cols].replace(CONSUMPTION_MAPPING)
    
    try:
        df[cols] = df[cols].astype(int)
        logger.info(f"Successfully converted {len(cols)} columns to numeric scale (0-6).")
    except ValueError as e:
        logger.error(f"Critical Error: Could not convert all values. Details: {e}")
    
    return df

## ============================================================================
## SECTION 5: DEMOGRAPHIC VALIDATION FUNCTIONS
## ============================================================================

def assess_demographic_variables(df):
    """Examine demographic distributions for data quality (Age, Gender, Education)."""
    log_step("STEP 7: DEMOGRAPHIC VARIABLES CONSISTENCY CHECK")
    
    demographic_report = {}
    
    ## Age distribution analysis
    if 'Age' in df.columns:
        logger.info("\n AGE DISTRIBUTION:")
        age_series = pd.to_numeric(df['Age'], errors='coerce')
        if not age_series.isna().all() and age_series.nunique() > DISCRETE_THRESHOLD:
            logger.info(f"   Unique Values: {age_series.nunique()}")
            logger.info(f"   Range: {age_series.min()} to {age_series.max()}")
            logger.info(f"   Mean: {age_series.mean():.2f} | SD: {age_series.std():.2f}")
            demographic_report['age'] = {
            'unique': age_series.nunique(),
            'range': (age_series.min(), age_series.max())
            }
            
        else:
            age_counts = df['Age'].value_counts().sort_index()
            logger.info(f"   Unique Levels: {len(age_counts)}")
            logger.info(f"   Distribution:\n{age_counts.to_string()}")
            demographic_report['age'] = age_counts.to_dict()
    
    ## Gender distribution analysis
    if 'Gender' in df.columns:
        logger.info("\n GENDER DISTRIBUTION:")
        gender_counts = df['Gender'].value_counts()
        for gender, count in gender_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"   {gender}: {count:,} ({pct:.1f}%)")
        demographic_report['gender'] = gender_counts.to_dict()
    
    ## Education level distribution analysis
    if 'Education' in df.columns:
        logger.info("\n EDUCATION DISTRIBUTION:")
        logger.info(f"   Unique Levels: {df['Education'].nunique()}")
        education_counts = df['Education'].value_counts()
        for edu, count in education_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"   {edu}: {count:,} ({pct:.1f}%)")
        demographic_report['education'] = education_counts.to_dict()
    
    ## Country/nationality distribution
    if 'Country' in df.columns:
        logger.info("\n COUNTRY DISTRIBUTION:")
        country_counts = df['Country'].value_counts().head(10)
        logger.info(f"   Total Countries: {df['Country'].nunique()}")
        logger.info("   Top 10 Countries:")
        for country, count in country_counts.items():
            pct = (count / len(df)) * 100
            logger.info(f"      {country}: {count:,} ({pct:.1f}%)")
        demographic_report['country'] = country_counts.to_dict()
    
    return demographic_report

## ============================================================================
## SECTION 6: STATISTICAL ASSESSMENT FUNCTIONS
## ============================================================================

def calculate_descriptive_statistics(df, numeric_cols):
    """Generate comprehensive descriptive statistics (mean, SD, skewness, etc.)."""
    log_step("STEP 8: DESCRIPTIVE STATISTICS - PERSONALITY TRAITS")
    
    ## Calculate comprehensive statistics
    stats_summary = df[numeric_cols].describe().T
    
    ## Add additional statistics
    stats_summary['Skewness'] = df[numeric_cols].skew()
    stats_summary['Kurtosis'] = df[numeric_cols].kurtosis()
    
    logger.info(stats_summary.round(3).to_string())
    
    return stats_summary

def calculate_primary_correlation(df, var1, var2):
    """Test correlation between two specific variables (e.g., Impulsivity & Sensation Seeking)."""
    log_step("STEP 9: PRIMARY CORRELATION ANALYSIS")
    logger.info(f"Hypothesis: Correlation between {var1} and {var2}")
    
    if var1 in df.columns and var2 in df.columns:
        ## Remove missing values before correlation calculation
        valid_data = df[[var1, var2]].dropna()
        valid_var1 = valid_data[var1]
        valid_var2 = valid_data[var2]
        
        ## Calculate Pearson correlation coefficient and statistical significance
        corr, p_value = stats.pearsonr(valid_var1, valid_var2)
        
        logger.info(f"\nPearson Correlation Coefficient: r = {corr:.4f}")
        logger.info(f"P-value: {p_value:.4e}")
        logger.info(f"N (valid pairs): {len(valid_var1)}")
        
        ## Interpret effect size (Cohen's guidelines)
        effect_size = abs(corr)
        if effect_size < 0.1: magnitude = "negligible"
        elif effect_size < 0.3: magnitude = "small"
        elif effect_size < 0.5: magnitude = "medium"
        else: magnitude = "large"
        
        logger.info(f"Effect Size: {magnitude.upper()} (|r| = {effect_size:.4f})")
        
        ## Interpret statistical significance
        if p_value < 0.001:
            logger.info("*** HIGHLY SIGNIFICANT (p < 0.001) ***")
        elif p_value < 0.01:
            logger.info("** SIGNIFICANT (p < 0.01) **")
        elif p_value < 0.05:
            logger.info("* SIGNIFICANT (p < 0.05) *")
        else:
            logger.info("NOT SIGNIFICANT (p ≥ 0.05)")
        
        return corr, p_value
    else:
        logger.error("Error: One or both columns not found")
        return None, None

## ============================================================================
## SECTION 7: NORMALITY TESTING FUNCTIONS
## ============================================================================

def assess_normality_and_distribution(df, numeric_cols):
    """Test normality assumption (Shapiro-Wilk) and assess distribution shape."""
    log_step("STEP 10: NORMALITY ASSESSMENT & DISTRIBUTION SHAPE")

    logger.info("Interpreting Normality:")
    logger.info("  Skewness: 0 = symmetric, ±0.5 = acceptable, >±1 = highly skewed")
    logger.info("  Kurtosis: 0 = normal, <3 = light tails, >3 = heavy tails")
    logger.info("  Shapiro-Wilk: p > 0.05 = normally distributed\n")

    normality_results = []
    
    for col in numeric_cols:
        if col in df.columns:
            data = df[col].dropna()
            if len(data) < 3:
                logger.info(f" Not enough data to assess normality for {col} (n={len(data)})")
                continue
            
            ## Calculate skewness & kurtosis
            skewness = data.skew()
            kurtosis = data.kurtosis()
            
            ## Shapiro-Wilk normality test (limited to 5000 samples for efficiency)
            sample_data = data.sample(min(5000, len(data)), random_state=RANDOM_STATE)
            stat, p_val = stats.shapiro(sample_data)
            
            ## Determine normality status based on all three tests
            is_normal = "Yes" if p_val > 0.05 else "No"
            
            normality_results.append({
                'Feature': col,
                'Skewness': f"{skewness:.3f}",
                'Kurtosis': f"{kurtosis:.3f}",
                'Shapiro_p': f"{p_val:.4f}",
                'Normal_Dist': is_normal,
                'N_Tested': len(sample_data)
            })
    
    normality_df = pd.DataFrame(normality_results)
    logger.info("\n" + normality_df.to_string(index=False))
    
    return normality_df

## ============================================================================
## SECTION 8: MASTER CLEANING FUNCTION
## ============================================================================

def execute_complete_cleaning_pipeline(df, personality_cols, drug_columns):
    """ Execute the complete data cleaning pipeline.
    Orchestrates all cleaning steps:
    1. Overview -> 2. Missing -> 3. Duplicates -> 4. Types -> 
    5. Outliers -> 6. Drugs Validation -> 7. Demographics -> 8. Statistics"""
    log_step("STARTING COMPLETE DATA QUALITY ASSESSMENT & CLEANING PIPELINE")
    
    cleaning_report = {}
    
    ## STEP 1: Dataset Overview
    cleaning_report['metrics'] = assess_dataset_overview(df)
    
    ## STEP 2: Missing Values
    cleaning_report['missing_values'] = assess_missing_values(df)
    
    ## STEP 3: Duplicates
    df, dup_count = detect_and_remove_duplicates(df)
    cleaning_report['duplicates_removed'] = dup_count
    
    ## STEP 4: Data Type Conversion
    df, conversion_report = validate_and_convert_numeric_columns(df, personality_cols)
    cleaning_report['conversions'] = conversion_report
    
    ## STEP 5: Outlier Detection
    outlier_summary = assess_outliers_personality_traits(df, personality_cols)
    cleaning_report['outliers'] = outlier_summary
    
    ## STEP 6: Drug Encoding Validation
    validation_report = validate_drug_consumption_encoding(df, drug_columns)
    cleaning_report['drug_validation'] = validation_report
    
    ## STEP 6.5: Convert Drug Consumption to Numeric
    df = convert_drug_consumption_to_numeric(df, drug_columns)
    cleaning_report['drug_conversion'] = 'Completed'
    
    ## STEP 7: Demographic Assessment
    demo_report = assess_demographic_variables(df)
    cleaning_report['demographics'] = demo_report
    
    ## STEP 8: Descriptive Statistics
    stats_summary = calculate_descriptive_statistics(df, personality_cols)
    cleaning_report['statistics'] = stats_summary
    
    ## STEP 9: Primary Correlation
    corr, p_val = calculate_primary_correlation(df, 'Impulsive', 'SS')
    cleaning_report['primary_correlation'] = {'correlation': corr, 'p_value': p_val}
    
    ## STEP 10: Normality Assessment
    normality = assess_normality_and_distribution(df, personality_cols)
    cleaning_report['normality'] = normality
    
    ## FINAL SUMMARY
    log_step("FINAL CLEANING SUMMARY")
    logger.info(f" Final Records: {df.shape[0]:,}")
    logger.info(f" Records Removed (Duplicates): {dup_count}")
    logger.info(f" Total Features: {df.shape[1]}")
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    logger.info(f" Data Completeness: {completeness:.2f}%")
    logger.info(" Dataset Status: READY FOR STATISTICAL ANALYSIS")
    logger.info("="*80)
    
    return df, cleaning_report

## ============================================================================
## SECTION 9: ADVANCED FEATURE ENGINEERING
## ============================================================================
## This section handles all creation of new variables:
## 1. Aggregating specific drugs into pharmacological families (Stimulants, etc.)
## 2. Using Machine Learning (K-Means) to create user clusters.

def aggregate_drug_families(df):
    """
    Part A: Group individual drugs into pharmacological families.
    Logic: Take the MAX usage score within the family.
    """
    log_step("STEP 11A: AGGREGATING DRUG FAMILIES")
    
    # Identify available columns
    available_cols = df.columns.tolist()
    
    for family_name, drugs_list in DRUG_FAMILIES.items():
        relevant_cols = [c for c in drugs_list if c in available_cols]
        if relevant_cols:
            col_name = f'Score_{family_name}'
            df[col_name] = df[relevant_cols].max(axis=1)
            logger.info(f" Created {col_name} from {len(relevant_cols)} columns")
                    
    return df

def create_usage_groups_kmeans(df, target_drug):
    """
    Part B: Apply K-Means Clustering to identify natural usage groups.
    """
    log_step(f"STEP 11B: GENERATING CLUSTERS (K-MEANS) FOR: {target_drug}")
    
    if not SKLEARN_AVAILABLE:
        logger.warning(" Skipping K-Means: scikit-learn library is not installed.")
        return df
    
    if target_drug not in df.columns:
        # Check if maybe the user wants to cluster on a family created in Part A
        logger.warning(f" Warning: Column '{target_drug}' not found. Skipping clustering.")
        return df
    
    # 1. Prepare Data
    original_count = len(df)
    if df[target_drug].isnull().sum() > 0:
        df = df.dropna(subset=[target_drug])
        logger.info(f" Note: Dropped {original_count - len(df)} rows due to missing '{target_drug}' data.")
        
    X = df[[target_drug]].values
    
    # 2. Fit K-Means
    try:
        kmeans = KMeans(n_clusters=KMEANS_CLUSTERS, random_state=RANDOM_STATE, n_init=KMEANS_N_INIT)
        kmeans.fit(X)
    
        temp_df = pd.DataFrame({'Score': df[target_drug], 'Cluster': kmeans.labels_})
        cluster_centers = temp_df.groupby('Cluster')['Score'].mean().sort_values()
        sorted_clusters = cluster_centers.index.tolist()
    
        name_mapping = {
            sorted_clusters[0]: 'Non-User',
            sorted_clusters[1]: 'Occasional',
            sorted_clusters[2]: 'Regular'
            }
    
        df['Usage_Group'] = temp_df['Cluster'].map(name_mapping)
    
        logger.info(f" Clusters created for '{target_drug}':")
        logger.info(df['Usage_Group'].value_counts().to_string())
    
    except Exception as e:
        logger.error(f" Error during K-Means clustering: {e}")
    return df

def execute_feature_engineering(df, clustering_target='Cannabis'):
    """ Master function for Section 9 that runs all feature engineering steps."""
    log_step("STARTING FEATURE ENGINEERING PIPELINE")
    
    # Run Part A: Families
    df = aggregate_drug_families(df)
    
    # Run Part B: Clustering
    df = create_usage_groups_kmeans(df, clustering_target)
    
    return df

## ============================================================================
## SECTION 10: PIPELINE WRAPPER
## ============================================================================

def detect_personality_and_drug_columns(df):
    """ Automatically detect personality trait columns and drug consumption columns."""
    
    # Select numeric columns (potential personality traits)
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    if "ID" in numeric_cols:
        # ID should not be treated as a numeric feature
        numeric_cols.remove("ID")

    personality_cols = []
    for col in numeric_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        # Personality scores in this dataset are standardized around [-4, 4]
        if col_min >= -5 and col_max <= 5:
            personality_cols.append(col)

    # Select categorical columns (potential drug variables)
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

    drug_columns = []
    for col in categorical_cols:
        unique_values = df[col].unique()
        # Drug columns contain codes like 'CL0', 'CL1', ..., 'CL6'
        if any("CL" in str(val) for val in unique_values):
            drug_columns.append(col)

    # Fallback: if no personality columns detected, use all numeric columns
    if not personality_cols:
        personality_cols = numeric_cols

    return personality_cols, drug_columns


def run_data_cleaning_pipeline():
    """
    High-level wrapper that:
    1. Loads the raw dataset from disk.
    2. Detects personality and drug columns.
    3. Runs the full cleaning pipeline.
    4. Saves the cleaned dataset to the processed folder.
    """
    # 1. Build path to the raw CSV file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "..", "data", "raw", "Drug_Consumption.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f" Error: File not found at {csv_path}")
        logger.info("Please check the path or put the CSV file in the 'data/raw' folder.")
        return None, None

    # 2. Load raw data
    logger.info(f" Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    # 3. Detect personality trait and drug consumption columns
    personality_cols, drug_columns = detect_personality_and_drug_columns(df)

    # 4. Run the existing full cleaning pipeline
    df_cleaned, cleaning_report = execute_complete_cleaning_pipeline(
        df,
        personality_cols=personality_cols,
        drug_columns=drug_columns,
    )
    
    # 5. Run Feature Engineering (Section 11)
    if 'execute_feature_engineering' in globals():
        df_final = execute_feature_engineering(df_cleaned)
    else:
        logger.warning(" Warning: Feature engineering function not found. Saving cleaned data only.")
        df_final = df_cleaned

    # 6. Save cleaned data to the processed folder
    output_path = os.path.join(
        current_dir, "..", "data", "processed", "Drug_Consumption_Cleaned.csv"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False)
    logger.info(f"\n Saved fully processed data to: {output_path}")
    return df_final, cleaning_report

# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
if __name__ == "__main__":
    logger.info(" Starting Data Cleaning Pipeline...")
    run_data_cleaning_pipeline()
