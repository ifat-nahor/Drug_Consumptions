import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from Personality_Based_Drug_Usage_Predictor import (
    perform_final_clustering, describe_personality_profiles
)

def test_clustering():
    df_dummy = pd.DataFrame({'PC1': [1,2], 'PC2': [3,4], 'Cluster': [0,1]})
    result = perform_final_clustering(df_dummy, df_dummy[['PC1','PC2']].values, 2)
    assert len(result) == 2
    print("Clustering PASSED ✓")

def test_profiles():
    df_dummy = pd.DataFrame({
        'PC1': [1,2], 'PC2': [3,4], 'Cluster': [0,1],
        'Nscore': [0.1, 0.2], 'Escore': [0.3, 0.4], 
        'Oscore': [0.5, 0.6], 'AScore': [0.7, 0.8], 
        'Cscore': [0.9, 1.0], 'Impulsive': [1.1, 1.2], 'SS': [1.3, 1.4]
    })
    profiles = describe_personality_profiles(df_dummy)
    assert len(profiles) >= 1
    print("Profiles PASSED ✓")

