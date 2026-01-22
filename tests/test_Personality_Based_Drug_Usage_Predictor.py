# tests/test_personality_pipeline.py
import pytest
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from Personality_Based_Drug_Usage_Predictor import (
    load_and_scale_data, apply_pca, perform_final_clustering,
    describe_personality_profiles
)

@pytest.fixture
def df_sample():
    return pd.read_csv('tests/test_sample.csv')

def test_load_scale(df_sample):
    logger.info("Testing data loading and scaling")
    df, scaled = load_and_scale_data(df_sample, ['Nscore'])
    assert scaled.shape == (len(df_sample), 1)
    assert np.abs(scaled.mean()) < 1e-6
    logger.info("Load/scale passed")

def test_pca(df_sample):
    logger.info("Testing PCA reduction")
    _, scaled = load_and_scale_data(df_sample, ['Nscore'])
    pca = apply_pca(scaled, 1)
    assert pca.shape == (len(df_sample), 1)
    logger.info("PCA passed")

def test_clustering(df_sample):
    logger.info("Testing KMeans clustering")
    _, scaled = load_and_scale_data(df_sample, ['Nscore'])
    pca = apply_pca(scaled, 1)
    df_clust = perform_final_clustering(df_sample, pca, 3)
    assert df_clust['Cluster'].nunique() == 3
    logger.info("Clustering passed")

def test_profiles(df_sample):
    logger.info("Testing personality profiles")
    _, scaled = load_and_scale_data(df_sample, ['Nscore'])
    pca = apply_pca(scaled, 1)
    df_clust = perform_final_clustering(df_sample, pca, 3)
    profiles = describe_personality_profiles(df_clust)
    assert profiles.shape == (3, 1)
    logger.info("Profiles passed")


