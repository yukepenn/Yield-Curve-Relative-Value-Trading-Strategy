"""
Unit tests for the feature engineering module.

This module contains tests for the FeatureEngineer class and its methods.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.feature_engineering import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='B')
    treasury_data = pd.DataFrame(
        np.random.randn(len(dates), 15),
        index=dates,
        columns=[f'Y{i}' for i in range(15)]
    )
    macro_data = pd.DataFrame(
        np.random.randn(len(dates), 5),
        index=dates,
        columns=[f'M{i}' for i in range(5)]
    )
    return treasury_data, macro_data

def test_feature_engineer_initialization(sample_data):
    """Test FeatureEngineer initialization."""
    treasury_data, macro_data = sample_data
    feature_engineer = FeatureEngineer(treasury_data, macro_data)
    assert feature_engineer.treasury_data.equals(treasury_data)
    assert feature_engineer.macro_data.equals(macro_data)

def test_calendar_features(sample_data):
    """Test calendar feature computation."""
    treasury_data, macro_data = sample_data
    feature_engineer = FeatureEngineer(treasury_data, macro_data)
    calendar_features = feature_engineer._compute_calendar_features(treasury_data.index)
    
    assert isinstance(calendar_features, pd.DataFrame)
    assert len(calendar_features) == len(treasury_data)
    assert 'day_of_week' in calendar_features.columns
    assert 'is_holiday' in calendar_features.columns

def test_trend_features(sample_data):
    """Test trend feature computation."""
    treasury_data, macro_data = sample_data
    feature_engineer = FeatureEngineer(treasury_data, macro_data)
    trend_features = feature_engineer._compute_trend_features()
    
    assert isinstance(trend_features, pd.DataFrame)
    assert len(trend_features) == len(treasury_data)
    assert 'trend_5d' in trend_features.columns
    assert 'momentum_20d' in trend_features.columns

def test_yield_curve_features(sample_data):
    """Test yield curve feature computation."""
    treasury_data, macro_data = sample_data
    feature_engineer = FeatureEngineer(treasury_data, macro_data)
    curve_features = feature_engineer._compute_yield_curve_features()
    
    assert isinstance(curve_features, pd.DataFrame)
    assert len(curve_features) == len(treasury_data)
    assert 'level' in curve_features.columns
    assert 'slope' in curve_features.columns
    assert 'curvature' in curve_features.columns

def test_carry_features(sample_data):
    """Test carry feature computation."""
    treasury_data, macro_data = sample_data
    feature_engineer = FeatureEngineer(treasury_data, macro_data)
    carry_features = feature_engineer._compute_carry_features()
    
    assert isinstance(carry_features, pd.DataFrame)
    assert len(carry_features) == len(treasury_data)
    assert 'carry_2s10s' in carry_features.columns
    assert 'rolldown_5s30s' in carry_features.columns

def test_target_computation(sample_data):
    """Test target computation."""
    treasury_data, macro_data = sample_data
    feature_engineer = FeatureEngineer(treasury_data, macro_data)
    targets = feature_engineer._compute_targets()
    
    assert isinstance(targets, pd.DataFrame)
    assert len(targets) == len(treasury_data)
    assert 'spread_change_2s10s' in targets.columns
    assert 'direction_5s30s' in targets.columns

def test_full_feature_creation(sample_data):
    """Test full feature creation process."""
    treasury_data, macro_data = sample_data
    feature_engineer = FeatureEngineer(treasury_data, macro_data)
    features, targets = feature_engineer.create_features()
    
    assert isinstance(features, pd.DataFrame)
    assert isinstance(targets, pd.DataFrame)
    assert len(features) == len(treasury_data)
    assert len(targets) == len(treasury_data)
    assert not features.isna().any().any()
    assert not targets.isna().any().any() 