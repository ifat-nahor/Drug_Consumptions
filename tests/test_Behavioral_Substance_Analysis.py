import pytest
import pandas as pd
import numpy as np
import os
import sys
import logging
# English: Import stats models for ANOVA logic
try:
    from scipy import stats
except ImportError:
    logger.error("English: Scipy is missing. Run 'pip install scipy'.")
# English: Add the root directory to sys.path so the test can find the 'src' folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src._Behavioral_Substance_Analysis import run_behavioral_analysis

# English: Create a mock dataset for testing purposes
@pytest.fixture
def mock_df():
    data = {
        'Cluster': [0, 1, 2, 0],
        'Alcohol': [1, 0, 1, 1],      
        'Cannabis': [1, 0, 0.2, 0.8], 
        'Nicotine': [0, 0, 0, 1],     
        'Coke': [0, 0, 0, 0], 'Ecstasy': [0, 0, 0, 0], 'LSD': [0, 0, 0, 0], 
        'Mushrooms': [0, 0, 0, 0], 'Benzos': [0, 0, 0, 0],
        'Escore': [1.0, -1.0, 0.5, 1.2],
        'Impulsive': [0.8, -0.5, 0.2, 0.9],
        'SS': [0.7, -0.8, 0.1, 0.8],
        'Nscore': [0.5, -0.2, 0.1, 0.4]
    }
    return pd.DataFrame(data)

def test_usage_diversity_logic(mock_df):
    """English: Verify that poly-drug diversity counts substances > 0.5 correctly."""
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg') # English: Silence GUI plots during test
    
    run_behavioral_analysis(mock_df)
    
    # English: Row 0 has Alcohol=1 and Cannabis=1 -> Expected 2
    assert mock_df.loc[0, 'Usage_Diversity'] == 2
    # English: Row 1 has all zeros -> Expected 0
    assert mock_df.loc[1, 'Usage_Diversity'] == 0

def test_folder_creation():
    """English: Check if the plot directory is identified correctly."""
    output_dir = r"D:\Users\User\Documents\project git 2026\final_proj\plots_pics_for_behavioral_and_gateway"
    assert os.path.exists(output_dir)