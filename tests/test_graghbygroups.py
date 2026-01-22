import pytest
import runpy
import pandas as pd
from pathlib import Path
from unittest.mock import patch

def test_drug_grouping_logic_on_real_data():
    """
    Validation: Confirms that neuro-pharmacological grouping correctly identifies
    peak usage frequencies within the actual dataset.
    """
    # Path navigation: from tests folder to src/grathbygroups.py
    current_dir = Path(__file__).resolve().parent
    script_path = current_dir.parent / 'src' / 'grathbygroups.py'

    assert script_path.exists(), f"Critical Error: Grouping script not found at {script_path}"

    # Run the grouping script and intercept visualizations
    with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.savefig'):
        script_globals = runpy.run_path(str(script_path))
    
    df = script_globals.get('df')
    
    # Column Verification: Check for aggregated category columns
    expected_categories = ['Hard Drugs_Max_Freq', 'Psychedelics_Max_Freq']
    for category in expected_categories:
        assert category in df.columns, f"Test Failed: Missing category column {category}"

    # Range Verification: Ensure frequency values are within the 0-6 scale
    assert df['Hard Drugs_Max_Freq'].max() <= 6, "Test Failed: Frequency value exceeds the allowed scale (0-6)."
    assert df['Hard Drugs_Max_Freq'].min() >= 0, "Test Failed: Negative frequency value detected."

    print("Grouping Test Passed: Real data aggregation verified.")