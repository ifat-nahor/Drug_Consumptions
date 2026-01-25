import sys
import os
import pytest
import pandas as pd
import numpy as np

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
# This block ensures the test file can locate the 'src' directory,
# even if they are in sibling folders.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import functions from the source module
from src.data_cleaning import (
    validate_and_convert_numeric_columns,
    detect_outliers_iqr,
    validate_drug_consumption_encoding,
    assess_demographic_variables,
    aggregate_drug_families,
    convert_drug_consumption_to_numeric
)

# ============================================================================
# 1. TEST FIXTURES (MOCK DATA)
# ============================================================================

@pytest.fixture
def sample_df():
    """
    Creates a small mock dataset representing the structure of the real data.
    Includes intentional errors to test validation logic.
    """
    data = {
        'ID': [1, 2, 3, 4, 5],
        # Numeric column with one invalid string entry
        'Nscore': [0.5, -0.5, 3.5, 0.1, 'Invalid_Text'],  
        # Categorical age ranges (should trigger 'distribution' logic, not mean)
        'Age': ['18-24', '25-34', '18-24', '35-44', '18-24'], 
        # Drug column with one invalid category code (CL99)
        'Cannabis': ['CL0', 'CL1', 'CL2', 'CL6', 'CL99'], 
        'Alcohol': ['CL0', 'CL5', 'CL5', 'CL5', 'CL5']
    }
    return pd.DataFrame(data)

@pytest.fixture
def numeric_df():
    """
    Creates a simple dataset with a clear outlier for statistical testing.
    """
    data = {
        'Trait_A': [0.1, 0.2, 0.15, 0.18, 100.0]  # 100.0 is the outlier
    }
    return pd.DataFrame(data)

# ============================================================================
# 2. UNIT TESTS
# ============================================================================

def test_numeric_conversion(sample_df):
    """
    Test Step 4: Validate that non-numeric strings are converted to NaN.
    """
    cols = ['Nscore']
    # Pass a copy to avoid modifying the fixture for other tests
    df_cleaned, report = validate_and_convert_numeric_columns(sample_df.copy(), cols)
    
    # Assert: The string 'Invalid_Text' should become NaN (Not a Number)
    assert pd.isna(df_cleaned.loc[4, 'Nscore'])
    # Assert: Valid numbers should remain unchanged
    assert df_cleaned.loc[0, 'Nscore'] == 0.5
    # Assert: The report should track exactly 1 error
    assert report['errors'] == 1

def test_outlier_detection(numeric_df):
    """
    Test Step 5: Validate IQR-based outlier detection.
    """
    count, lower, upper = detect_outliers_iqr(numeric_df, 'Trait_A')
    
    # Assert: Should detect exactly 1 outlier (the value 100.0)
    assert count == 1
    # Assert: The calculated upper bound should be significantly less than 100
    assert upper < 100

def test_drug_validation_logic(sample_df):
    """
    Test Step 6: Validate that invalid category codes (e.g., CL99) are caught.
    """
    drug_cols = ['Cannabis']
    report = validate_drug_consumption_encoding(sample_df, drug_cols)
    
    # Assert: One column should be flagged as invalid
    assert report['invalid_columns'] == 1
    # Assert: The specific invalid value 'CL99' must be in the report
    assert report['invalid_entries']['Cannabis'] == ['CL99']

def test_smart_demographic_logic(sample_df):
    """
    Test Step 7: Validate that categorical Age data returns a count distribution,
    not a statistical mean (which would cause errors).
    """
    report = assess_demographic_variables(sample_df)
    
    # Assert: 'age' report should contain specific keys for categories (e.g., '18-24')
    # If it tried to calculate a mean, this structure would be different or empty.
    assert '18-24' in report['age']
    # Assert: Correct count verification (there are 3 entries of '18-24' in the fixture)
    assert report['age']['18-24'] == 3

def test_drug_families_creation(sample_df):
    """
    Test Step 11A: Validate that drug family aggregation runs and returns a DataFrame.
    """
    # Note: This function relies on the global DRUG_FAMILIES dictionary in the source code.
    df_result = aggregate_drug_families(sample_df.copy())
    
    # Assert: Result is a DataFrame and preserves original columns
    assert isinstance(df_result, pd.DataFrame)
    assert 'ID' in df_result.columns
    # (Since Cannabis isn't in a default family in the mapping, we mainly ensure no crash)

def test_drug_to_numeric_conversion(sample_df):
    """
    Test Step 6.5: Validate conversion of 'CLx' strings to integers.
    """
    # Pre-cleaning: Remove the row with 'CL99' so conversion succeeds
    df = sample_df.copy()
    df = df[df['Cannabis'] != 'CL99'] 
    
    drug_cols = ['Cannabis']
    df_numeric = convert_drug_consumption_to_numeric(df, drug_cols)
    
    # Assert: 'CL0' becomes integer 0
    assert df_numeric.loc[0, 'Cannabis'] == 0
    # Assert: 'CL2' becomes integer 2
    assert df_numeric.loc[2, 'Cannabis'] == 2
    # Assert: The column data type is now integer
    assert pd.api.types.is_integer_dtype(df_numeric['Cannabis'])