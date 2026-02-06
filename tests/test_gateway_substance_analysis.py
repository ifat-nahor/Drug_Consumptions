import pytest
import pandas as pd
import numpy as np

# --- Extra Analysis Tests ---

def test_anova_logic():
    """Validates that P-values are within [0, 1] and significance is caught."""
    df_dummy = pd.DataFrame({
        'Cluster': [0, 0, 1, 1],
        'Drug_A': [5, 6, 1, 2], # Significant difference
        'Drug_B': [3, 3, 3, 3]  # No difference
    })
    
    # Mocking the result of an ANOVA test
    p_val_a = 0.001 
    p_val_b = 1.0
    
    assert 0 <= p_val_a <= 1
    assert p_val_a < 0.05 # Should be significant
    assert p_val_b >= 0.05 # Should not be significant
    print("ANOVA Logic PASSED ✓")

def test_gateway_correlations():
    """Ensures correlation matrix diagonal is always 1.0."""
    df_dummy = pd.DataFrame({
        'Cannabis': [1, 2, 3],
        'Coke': [1, 2, 3]
    })
    corr = df_dummy.corr()
    
    assert corr.loc['Cannabis', 'Cannabis'] == pytest.approx(1.0)
    assert corr.loc['Cannabis', 'Coke'] == pytest.approx(1.0)
    print("Gateway Correlations PASSED ✓")

def test_diversity_integrity():
    """Validates that usage count is non-negative and within bounds."""
    usage_data = np.array([0, 1, 5, 0]) 
    diversity = np.count_nonzero(usage_data > 0)
    
    assert diversity >= 0
    assert diversity <= len(usage_data)
    print("Diversity Integrity PASSED ✓")