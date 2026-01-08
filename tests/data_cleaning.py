import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


## ============================================================================
## SECTION 1: DATA QUALITY ASSESSMENT FUNCTIONS
## ============================================================================

def assess_dataset_overview(df):
    """
    Generate initial dataset overview statistics.
    
    Purpose: Get complete understanding of dataset size and memory footprint.
    This helps identify if we're working with large data that needs optimization.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset to assess
    
    Returns:
    --------
    dict : Dictionary containing dataset metrics
    """
    
    print("\n" + "="*80)
    print("[STEP 1] DATASET OVERVIEW")
    print("="*80)
    
    metrics = {
        'total_records': df.shape[0],
        'total_features': df.shape[1],
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    print(f"Total Records: {metrics['total_records']:,}")
    print(f"Total Features: {metrics['total_features']}")
    print(f"Memory Usage: {metrics['memory_usage_mb']:.2f} MB")
    
    return metrics


def assess_missing_values(df):
    """
    Comprehensive missing data analysis.
    
    Purpose: Identify missing data patterns across all columns.
    Missing data can bias statistical analyses and must be documented thoroughly.
    We calculate both absolute count and percentage for better interpretation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to analyze
    
    Returns:
    --------
    pandas.DataFrame : Summary of missing values per column
    """
    
    print("\n" + "="*80)
    print("[STEP 2] MISSING VALUES ANALYSIS")
    print("="*80)
    
    missing_counts = df.isnull().sum()
    missing_percentage = (df.isnull().sum() / len(df)) * 100
    
    ## Create comprehensive missing values summary
    missing_df = pd.DataFrame({
        'Column': missing_counts.index,
        'Missing_Count': missing_counts.values,
        'Missing_Percentage': missing_percentage.values,
        'Data_Type': df.dtypes.values
    })
    
    ## Filter to show only columns with missing values and sort by severity
    missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
        'Missing_Count', ascending=False
    )
    
    if missing_df.empty:
        print("✓ No missing values detected - dataset is complete")
    else:
        print(missing_df.to_string(index=False))
    
    return missing_df


def detect_and_remove_duplicates(df):
    """
    Identify and remove duplicate entries.
    
    Purpose: Remove duplicate entries that could inflate sample size.
    Duplicates can occur due to data collection errors or participant re-entry.
    Removing them ensures each observation is truly independent.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to clean
    
    Returns:
    --------
    pandas.DataFrame : Dataset with duplicates removed
    int : Number of duplicates removed
    """
    
    print("\n" + "="*80)
    print("[STEP 3] DUPLICATE RECORDS DETECTION")
    print("="*80)
    
    duplicates_count = df.duplicated().sum()
    print(f"Total Duplicate Rows Found: {duplicates_count}")
    
    if duplicates_count > 0:
        print("⚠ Warning: Removing duplicate records...")
        df = df.drop_duplicates()
        print(f"✓ Removed {duplicates_count} duplicate rows")
        print(f"New dataset size: {df.shape[0]:,} records")
    else:
        print("✓ No duplicate records found")
    
    return df, duplicates_count


## ============================================================================
## SECTION 2: DATA TYPE VALIDATION FUNCTIONS
## ============================================================================

def validate_and_convert_numeric_columns(df, numeric_columns):
    """
    Ensure all numeric columns are properly typed for mathematical operations.
    
    Purpose: String-encoded numbers will cause errors in statistical functions.
    We use 'coerce' to convert invalid entries to NaN for later inspection.
    This step is critical before any numerical analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset to validate
    numeric_columns : list
        List of columns that should be numeric
    
    Returns:
    --------
    pandas.DataFrame : Dataset with validated numeric columns
    dict : Report of conversions made
    """
    
    print("\n" + "="*80)
    print("[STEP 4] DATA TYPE VALIDATION & CONVERSION")
    print("="*80)
    
    conversion_report = {
        'converted_columns': [],
        'errors': []
    }
    
    print("Current Data Types (before conversion):")
    print(df[numeric_columns].dtypes)
    
    ## Convert to numeric, any non-numeric values become NaN
    for col in numeric_columns:
        if col in df.columns:
            original_type = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors='coerce')
            new_type = df[col].dtype
            
            if original_type != new_type:
                conversion_report['converted_columns'].append(col)
    
    print("\n✓ Personality scores validated as numeric")
    print(f"✓ Converted {len(conversion_report['converted_columns'])} columns")
    
    return df, conversion_report


## ============================================================================
## SECTION 3: OUTLIER DETECTION FUNCTIONS
## ============================================================================

def detect_outliers_iqr(data, column):
    """
    Detect outliers using the Interquartile Range (IQR) method.
    
    Purpose: Detect extreme values that may represent data entry errors.
    Method: Interquartile Range (IQR) - robust to skewed distributions.
    We use 3×IQR threshold (more conservative than standard 1.5×IQR).
    This is critical for standardized scores where extreme values are suspicious.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Dataset containing the column
    column : str
        Column name to analyze for outliers
    
    Returns:
    --------
    tuple : (number_of_outliers, lower_bound, upper_bound)
    """
    
    ## First quartile (25th percentile)
    Q1 = data[column].quantile(0.25)
    
    ## Third quartile (75th percentile)
    Q3 = data[column].quantile(0.75)
    
    ## Interquartile range
    IQR = Q3 - Q1
    
    ## Define outlier boundaries (3×IQR is very conservative, reduces false positives)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    ## Identify outliers beyond these bounds
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    return len(outliers), lower_bound, upper_bound


def assess_outliers_personality_traits(df, personality_cols):
    """
    Comprehensive outlier detection for all personality traits.
    
    Purpose: Identify extreme values in personality measurements.
    Personality scores are standardized, so extreme outliers may indicate
    data entry errors or measurement problems that should be investigated.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with personality scores
    personality_cols : list
        List of personality score columns
    
    Returns:
    --------
    pandas.DataFrame : Summary of outliers per personality trait
    """
    
    print("\n" + "="*80)
    print("[STEP 5] OUTLIER DETECTION - PERSONALITY SCORES")
    print("="*80)
    
    outlier_summary = []
    
    ## Apply outlier detection to all personality variables
    for col in personality_cols:
        if col in df.columns:
            n_outliers, lower, upper = detect_outliers_iqr(df, col)
            
            outlier_summary.append({
                'Feature': col,
                'Outliers': n_outliers,
                'Outlier_Rate_%': f"{(n_outliers/len(df)*100):.2f}%",
                'Lower_Bound': f"{lower:.3f}",
                'Upper_Bound': f"{upper:.3f}",
                'Min_Value': f"{df[col].min():.3f}",
                'Max_Value': f"{df[col].max():.3f}"
            })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))
    print("\n📌 Note: Personality scores are standardized (mean=0, SD=1)")
    print("   Extreme outliers may indicate data entry errors or measurement issues")
    
    return outlier_df


## ============================================================================
## SECTION 4: CATEGORICAL DATA VALIDATION FUNCTIONS
## ============================================================================

def validate_drug_consumption_encoding(df, drug_columns):
    """
    Verify drug consumption values follow expected format.
    
    Purpose: Verify that all drug consumption values follow the expected format.
    Expected values: CL0 (never used) through CL6 (used in last day).
    Any deviations suggest data corruption or encoding errors that need correction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with drug consumption columns
    drug_columns : list
        List of drug consumption columns to validate
    
    Returns:
    --------
    dict : Report of validation results
    """
    
    print("\n" + "="*80)
    print("[STEP 6] DRUG CONSUMPTION ENCODING VALIDATION")
    print("="*80)
    
    print("Expected categories: CL0 (Never), CL1-CL6 (Progressive frequency)")
    print("CL0=Never | CL1=Decade Ago | CL2=Last Decade | CL3=Last Year")
    print("CL4=Last Month | CL5=Last Week | CL6=Last Day\n")
    
    ## Valid category labels as per data dictionary
    valid_categories = ['CL0', 'CL1', 'CL2', 'CL3', 'CL4', 'CL5', 'CL6']
    
    ## Check for invalid category labels
    validation_report = {
        'valid_columns': 0,
        'invalid_columns': 0,
        'invalid_entries': {}
    }
    
    print("Validating each drug column...")
    
    for drug in drug_columns:
        if drug in df.columns:
            ## Find any values not in the valid set
            invalid = df[~df[drug].isin(valid_categories)][drug].unique()
            
            if len(invalid) > 0:
                validation_report['invalid_columns'] += 1
                validation_report['invalid_entries'][drug] = invalid.tolist()
            else:
                validation_report['valid_columns'] += 1
    
    if validation_report['invalid_entries']:
        print(f"\n⚠ Warning: {validation_report['invalid_columns']} columns with invalid categories:")
        for drug, values in validation_report['invalid_entries'].items():
            print(f"  {drug}: {values}")
    else:
        print(f"✓ All {validation_report['valid_columns']} drug columns have valid encodings")
    
    return validation_report


def convert_drug_consumption_to_numeric(df, drug_columns):
    """
    Convert categorical drug consumption codes (CL0-CL6) to numeric values.
    
    Purpose: Transform categorical consumption ratings into numeric scale (0-6).
    This enables quantitative analysis such as correlation, regression, and clustering.
    The numeric mapping preserves the ordinal nature of the consumption frequency.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with categorical drug consumption values
    drug_columns : list
        List of drug consumption columns to convert
    
    Returns:
    --------
    pandas.DataFrame : Dataset with numeric drug consumption values
    """
    
    print("\n" + "="*80)
    print("[STEP 6.5] CONVERTING DRUG CONSUMPTION TO NUMERIC")
    print("="*80)
    
    ## Mapping: CL0 → 0, CL1 → 1, ..., CL6 → 6
    consumption_mapping = {
        'CL0': 0,  ## Never Used
        'CL1': 1,  ## Used over a Decade Ago
        'CL2': 2,  ## Used in Last Decade
        'CL3': 3,  ## Used in Last Year
        'CL4': 4,  ## Used in Last Month
        'CL5': 5,  ## Used in Last Week
        'CL6': 6   ## Used in Last Day
    }
    
    print("Mapping: CL0→0 (Never), CL1→1 (Decade Ago), ..., CL6→6 (Last Day)")
    
    ## Apply mapping to all drug columns
    for drug in drug_columns:
        if drug in df.columns:
            df[drug] = df[drug].map(consumption_mapping)
    
    print(f"✓ Converted {len(drug_columns)} drug consumption columns to numeric scale (0-6)")
    
    return df


## ============================================================================
## SECTION 5: DEMOGRAPHIC VALIDATION FUNCTIONS
## ============================================================================

def assess_demographic_variables(df):
    """
    Examine demographic distributions for data quality and representativeness.
    
    Purpose: Check demographic distributions for data quality issues.
    This identifies impossible values (e.g., age < 0) or severely imbalanced groups.
    Also provides descriptive statistics for the Methods section of reports.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with demographic variables
    
    Returns:
    --------
    dict : Summary of demographic distributions
    """
    
    print("\n" + "="*80)
    print("[STEP 7] DEMOGRAPHIC VARIABLES CONSISTENCY CHECK")
    print("="*80)
    
    demographic_report = {}
    def assess_demographic_variables(df):
   
    
     print("\n" + "="*80)
    print("[STEP 7] DEMOGRAPHIC VARIABLES CONSISTENCY CHECK")
    print("="*80)
    
    demographic_report = {}
    
    ## Age distribution analysis
    if 'Age' in df.columns:
        print(f"\n📊 AGE DISTRIBUTION:")
        age_series = pd.to_numeric(df['Age'], errors='coerce')
        print(f"   Unique Values: {age_series.nunique()}")
        print(f"   Range: {age_series.min()} to {age_series.max()}")
        print(f"   Mean: {age_series.mean():.2f} | SD: {age_series.std():.2f}")
        demographic_report['age'] = {
            'unique': age_series.nunique(),
            'range': (age_series.min(), age_series.max())
        }
    
    ## Gender distribution analysis
    if 'Gender' in df.columns:
        print(f"\n📊 GENDER DISTRIBUTION:")
        gender_counts = df['Gender'].value_counts()
        for gender, count in gender_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {gender}: {count:,} ({pct:.1f}%)")
        demographic_report['gender'] = gender_counts.to_dict()
    
    ## Education level distribution analysis
    if 'Education' in df.columns:
        print(f"\n📊 EDUCATION DISTRIBUTION:")
        print(f"   Unique Levels: {df['Education'].nunique()}")
        education_counts = df['Education'].value_counts()
        for edu, count in education_counts.items():
            pct = (count / len(df)) * 100
            print(f"   {edu}: {count:,} ({pct:.1f}%)")
        demographic_report['education'] = education_counts.to_dict()
    
    ## Country/nationality distribution
    if 'Country' in df.columns:
        print(f"\n📊 COUNTRY DISTRIBUTION:")
        country_counts = df['Country'].value_counts().head(10)
        print(f"   Total Countries: {df['Country'].nunique()}")
        print(f"   Top 10 Countries:")
        for country, count in country_counts.items():
            pct = (count / len(df)) * 100
            print(f"      {country}: {count:,} ({pct:.1f}%)")
        demographic_report['country'] = country_counts.to_dict()
    
    return demographic_report


    
    
   


## ============================================================================
## SECTION 6: STATISTICAL ASSESSMENT FUNCTIONS
## ============================================================================

def calculate_descriptive_statistics(df, numeric_cols):
    """
    Generate comprehensive descriptive statistics.
    
    Purpose: Generate descriptive statistics (mean, SD, min, max, quartiles).
    These values are essential for the Results section and data reporting.
    They provide a baseline understanding of the data distribution.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with numeric variables
    numeric_cols : list
        List of numeric columns to summarize
    
    Returns:
    --------
    pandas.DataFrame : Comprehensive statistical summary
    """
    
    print("\n" + "="*80)
    print("[STEP 8] DESCRIPTIVE STATISTICS - PERSONALITY TRAITS")
    print("="*80)
    
    ## Calculate comprehensive statistics
    stats_summary = df[numeric_cols].describe().T
    
    ## Add additional statistics
    stats_summary['Skewness'] = df[numeric_cols].skew()
    stats_summary['Kurtosis'] = df[numeric_cols].kurtosis()
    
    print(stats_summary.round(3))
    
    return stats_summary


def calculate_primary_correlation(df, var1, var2):
    """
    Test the primary research hypothesis: correlation between impulsivity and sensation seeking.
    
    Purpose: Test the core research question of the analysis.
    Pearson's r measures linear relationship strength (0=no correlation, ±1=perfect).
    p-value determines if correlation is statistically significant (p < 0.05).
    This is the CORE analysis for your research question.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset containing both variables
    var1 : str
        First variable (e.g., 'Impulsive')
    var2 : str
        Second variable (e.g., 'SS')
    
    Returns:
    --------
    tuple : (correlation_coefficient, p_value)
    """
    
    print("\n" + "="*80)
    print("[STEP 9] PRIMARY CORRELATION ANALYSIS")
    print("="*80)
    print(f"Hypothesis: Correlation between {var1} and {var2}")
    
    if var1 in df.columns and var2 in df.columns:
        ## Remove missing values before correlation calculation
        valid_var1 = df[var1].dropna()
        valid_var2 = df[var2].dropna()
        
        ## Calculate Pearson correlation coefficient and statistical significance
        corr, p_value = stats.pearsonr(valid_var1, valid_var2)
        
        print(f"\nPearson Correlation Coefficient: r = {corr:.4f}")
        print(f"P-value: {p_value:.4e}")
        print(f"N (valid pairs): {len(valid_var1)}")
        
        ## Interpret effect size (Cohen's guidelines)
        effect_size = abs(corr)
        if effect_size < 0.1:
            magnitude = "negligible"
        elif effect_size < 0.3:
            magnitude = "small"
        elif effect_size < 0.5:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        print(f"Effect Size: {magnitude.upper()} (|r| = {effect_size:.4f})")
        
        ## Interpret statistical significance
        if p_value < 0.001:
            print("*** HIGHLY SIGNIFICANT (p < 0.001) ***")
        elif p_value < 0.01:
            print("** SIGNIFICANT (p < 0.01) **")
        elif p_value < 0.05:
            print("* SIGNIFICANT (p < 0.05) *")
        else:
            print("NOT SIGNIFICANT (p ≥ 0.05)")
        
        return corr, p_value
    else:
        print(f"⚠ Error: One or both columns not found")
        return None, None


## ============================================================================
## SECTION 7: NORMALITY TESTING FUNCTIONS
## ============================================================================

def assess_normality_and_distribution(df, numeric_cols):
    """
    Test normality assumption and assess distribution characteristics.
    
    Purpose: Test if data follows normal distribution (assumption for many tests).
    Skewness: measures asymmetry (0 = perfectly symmetric, ±0.5 acceptable).
    Kurtosis: measures tail heaviness (0 = normal, >3 = heavy tails).
    Shapiro-Wilk: formal normality test (p > 0.05 = normal distribution).
    This is critical for choosing parametric vs non-parametric statistical tests.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset with numeric variables
    numeric_cols : list
        List of columns to test
    
    Returns:
    --------
    pandas.DataFrame : Summary of normality assessments
    """
    
    print("\n" + "="*80)
    print("[STEP 10] NORMALITY ASSESSMENT & DISTRIBUTION SHAPE")
    print("="*80)
    print("Interpreting Normality:")
    print("  Skewness: 0 = symmetric, ±0.5 = acceptable, >±1 = highly skewed")
    print("  Kurtosis: 0 = normal, <3 = light tails, >3 = heavy tails")
    print("  Shapiro-Wilk: p > 0.05 = normally distributed\n")
    
    normality_results = []
    
    for col in numeric_cols:
        if col in df.columns:
            data = df[col].dropna()
            
            ## Calculate skewness (asymmetry measure)
            skewness = data.skew()
            
            ## Calculate kurtosis (tail weight measure)
            kurtosis = data.kurtosis()
            
            ## Shapiro-Wilk normality test (limited to 5000 samples for efficiency)
            sample_data = data.sample(min(5000, len(data)), random_state=42)
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
    print(normality_df.to_string(index=False))
    
    return normality_df


## ============================================================================
## SECTION 8: MASTER CLEANING FUNCTION
## ============================================================================

def execute_complete_cleaning_pipeline(df, personality_cols, drug_columns):
    """
    Execute the complete data cleaning pipeline.
    
    Purpose: Run all cleaning and validation steps in proper sequence.
    This master function orchestrates all cleaning operations and returns
    a fully cleaned, validated dataset ready for statistical analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw dataset to clean
    personality_cols : list
        List of personality score columns
    drug_columns : list
        List of drug consumption columns
    
    Returns:
    --------
    pandas.DataFrame : Fully cleaned and validated dataset
    dict : Comprehensive cleaning report
    """
    
    print("\n" + "="*80)
    print("COMPLETE DATA QUALITY ASSESSMENT & CLEANING PIPELINE")
    print("="*80)
    
    cleaning_report = {}
    
    ## STEP 1: Dataset Overview
    metrics = assess_dataset_overview(df)
    cleaning_report['metrics'] = metrics
    
    ## STEP 2: Missing Values
    missing_df = assess_missing_values(df)
    cleaning_report['missing_values'] = missing_df
    
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
    cleaning_report['primary_correlation'] = {
        'correlation': corr,
        'p_value': p_val
    }
    
    ## STEP 10: Normality Assessment
    normality = assess_normality_and_distribution(df, personality_cols)
    cleaning_report['normality'] = normality
    
    ## FINAL SUMMARY
    print("\n" + "="*80)
    print("[FINAL] CLEANING PIPELINE SUMMARY")
    print("="*80)
    print(f"✓ Initial Records: {metrics['total_records']:,}")
    print(f"✓ Final Records: {df.shape[0]:,}")
    print(f"✓ Records Removed (Duplicates): {dup_count}")
    print(f"✓ Total Features: {df.shape[1]}")
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()
    completeness = ((total_cells - missing_cells) / total_cells) * 100
    print(f"✓ Data Completeness: {completeness:.2f}%")
    print(f"✓ Dataset Status: READY FOR STATISTICAL ANALYSIS ✓")
    print("="*80)
    
    return df, cleaning_report
  





'''SECTION 1: Data Quality Assessment
├── assess_dataset_overview()
├── assess_missing_values()
└── detect_and_remove_duplicates()

SECTION 2: Data Type Validation
└── validate_and_convert_numeric_columns()

SECTION 3: Outlier Detection
├── detect_outliers_iqr()
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

SECTION 8: Master Function
└── execute_complete_cleaning_pipeline() ← זה מריץ הכל!
'''
