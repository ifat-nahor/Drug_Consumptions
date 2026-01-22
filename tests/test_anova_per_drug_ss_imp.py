import pytest
import runpy
import pandas as pd
from pathlib import Path
from unittest.mock import patch

def test_anova_real_data_execution():
    """
     Ensures the ANOVA script processes the real 1885-row dataset
    and produces valid statistical outputs for the neuroscience project.
    """
    # Identify the absolute path of the test file's directory
    current_dir = Path(__file__).resolve().parent
    
    # Define path to the target script located directly in the 'src' folder
    script_path = current_dir.parent / 'src' / 'anova_per_drug_SS_IMP.py'
    
    # Safety Check: Verify if the script exists before execution
    assert script_path.exists(), f"Critical Error: Script not found at {script_path}. Check if the file is inside the 'src' directory."

    # Execution: Run the script while mocking GUI outputs (plots) to avoid interruptions
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
        # Execute the script and capture all global variables in a dictionary
        script_globals = runpy.run_path(str(script_path))
    
    # 1. Data Verification: Check if the main DataFrame 'df' was created and loaded
    df = script_globals.get('df')
    assert isinstance(df, pd.DataFrame), "Test Failed: DataFrame 'df' was not found in the script."
    assert len(df) > 1800, "Test Failed: The dataset appears incomplete (expected ~1885 rows)"

    # 2. Factor Verification: Confirm that Stage 2 (Median Split) was performed
    assert 'Impulsive_Bin' in df.columns, "Test Failed: 'Impulsive_Bin' factor was not created."
    assert set(df['Impulsive_Bin'].unique()) == {'Low', 'High'}, "Test Failed: Binning did not create correct 'Low'/'High' labels."

    # 3. Results Verification: Confirm the statistical results table 'results_df' exists
    results_df = script_globals.get('results_df')
    assert results_df is not None, "Test Failed: 'results_df' table was not found."
    assert not results_df.empty, "Test Failed: The statistical results table is empty."

    print("ANOVA Test Passed: Real data processed successfully.")