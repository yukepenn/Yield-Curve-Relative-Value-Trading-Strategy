"""
Feature Engineering Module for Yield Curve Trading Strategy.

This module computes features and targets for predicting yield curve spread movements.
All features are computed using only historical data to prevent forward-looking bias.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from sklearn.decomposition import PCA
from datetime import datetime, timedelta
import holidays

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BUSINESS_DAYS_PER_YEAR = 252
SPREADS = {
    '2s10s': ('2-Year', '10-Year'),
    '5s30s': ('5-Year', '30-Year'),
    '2s5s': ('2-Year', '5-Year'),
    '10s30s': ('10-Year', '30-Year'),
    '3m10y': ('3-Month', '10-Year')
}

LOOKBACK_PERIODS = [1, 5, 10, 21, 63, 126, 252]  # 1d, 1w, 2w, 1m, 3m, 6m, 1y
US_HOLIDAYS = holidays.US()

class FeatureEngineer:
    def __init__(self, treasury_data: pd.DataFrame, macro_data: pd.DataFrame):
        """
        Initialize feature engineering with raw data.
        
        Parameters
        ----------
        treasury_data : pd.DataFrame
            Daily Treasury yield data
        macro_data : pd.DataFrame
            Macro indicators data
        """
        self.treasury_data = treasury_data
        self.macro_data = macro_data
        self.features = pd.DataFrame()
        
    def _compute_calendar_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """
        Compute calendar-based features.
        
        Args:
            dates (pd.DatetimeIndex): DateTimeIndex of the data
            
        Returns:
            pd.DataFrame: Calendar features
        """
        logger.info("Computing calendar features...")
        
        # Ensure dates have frequency set
        dates = pd.DatetimeIndex(dates).to_period('D').to_timestamp('D')
        
        calendar_features = pd.DataFrame(index=dates)
        calendar_features['day_of_week'] = dates.dayofweek
        calendar_features['day_of_month'] = dates.day
        calendar_features['week_of_year'] = dates.isocalendar().week
        calendar_features['month'] = dates.month
        calendar_features['quarter'] = dates.quarter
        calendar_features['year'] = dates.year
        
        # Compute end-of-period indicators
        calendar_features['is_month_end'] = dates.month != dates.shift(1).month
        calendar_features['is_quarter_end'] = dates.quarter != dates.shift(1).quarter
        calendar_features['is_year_end'] = dates.year != dates.shift(1).year
        
        # Add holiday features
        us_holidays = holidays.US()
        calendar_features['is_holiday'] = [date in us_holidays for date in dates]
        calendar_features['is_day_before_holiday'] = [date + pd.Timedelta(days=1) in us_holidays for date in dates]
        calendar_features['is_day_after_holiday'] = [date - pd.Timedelta(days=1) in us_holidays for date in dates]
        
        return calendar_features

    def _compute_trend_features(self, data: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """Compute trend and momentum features for a given series."""
        features = pd.DataFrame(index=data.index)
        
        # Returns over different horizons
        for period in LOOKBACK_PERIODS:
            # Level changes
            features[f'{col_name}_change_{period}d'] = data[col_name].diff(period)
            
            # Percentage changes
            features[f'{col_name}_pct_change_{period}d'] = data[col_name].pct_change(period)
            
            # Moving averages
            features[f'{col_name}_ma_{period}d'] = data[col_name].rolling(window=period).mean()
            
            # Distance from moving average
            features[f'{col_name}_dist_ma_{period}d'] = data[col_name] - features[f'{col_name}_ma_{period}d']
            
            # Volatility
            features[f'{col_name}_vol_{period}d'] = data[col_name].rolling(window=period).std()
            
            # Z-score
            features[f'{col_name}_zscore_{period}d'] = (
                (data[col_name] - features[f'{col_name}_ma_{period}d']) / 
                features[f'{col_name}_vol_{period}d']
            )
            
            # Momentum (rate of change)
            features[f'{col_name}_mom_{period}d'] = (
                data[col_name].diff(period) / period
            )
        
        # RSI
        delta = data[col_name].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features[f'{col_name}_rsi_14d'] = 100 - (100 / (1 + rs))
        
        return features

    def _compute_yield_curve_features(self) -> pd.DataFrame:
        """Compute yield curve shape features using PCA."""
        # Select key tenors for PCA
        key_tenors = ['2-Year', '5-Year', '10-Year', '30-Year']
        yields_for_pca = self.treasury_data[key_tenors]
        
        # Standardize yields
        yields_std = (yields_for_pca - yields_for_pca.mean()) / yields_for_pca.std()
        
        # Compute PCA
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(yields_std)
        
        # Create DataFrame with PCA components
        pca_df = pd.DataFrame(
            pca_features,
            index=yields_for_pca.index,
            columns=['yield_pc1_level', 'yield_pc2_slope', 'yield_pc3_curvature']
        )
        
        # Add explained variance as features
        for i, ratio in enumerate(pca.explained_variance_ratio_):
            pca_df[f'yield_pc{i+1}_var_ratio'] = ratio
            
        return pca_df

    def _compute_carry_features(self) -> pd.DataFrame:
        """Compute carry and roll-down features."""
        features = pd.DataFrame(index=self.treasury_data.index)
        
        for spread_name, (short_tenor, long_tenor) in SPREADS.items():
            # Current spread level
            spread = self.treasury_data[long_tenor] - self.treasury_data[short_tenor]
            features[f'{spread_name}_level'] = spread
            
            # Carry (approximated by current spread)
            features[f'{spread_name}_carry'] = spread / BUSINESS_DAYS_PER_YEAR
            
            # Historical carry performance
            for period in [5, 21, 63]:  # 1w, 1m, 3m
                features[f'{spread_name}_carry_return_{period}d'] = (
                    features[f'{spread_name}_carry'].rolling(period).sum()
                )
        
        return features

    def _compute_targets(self) -> pd.DataFrame:
        """Compute regression and classification targets."""
        targets = pd.DataFrame(index=self.treasury_data.index)
        
        for spread_name, (short_tenor, long_tenor) in SPREADS.items():
            # Current spread
            spread = self.treasury_data[long_tenor] - self.treasury_data[short_tenor]
            
            # Regression target: next day spread change in basis points
            targets[f'y_{spread_name}_next_day'] = spread.shift(-1) - spread
            
            # Classification targets
            # Binary direction (+1 steepener, -1 flattener)
            targets[f'y_{spread_name}_direction'] = np.sign(targets[f'y_{spread_name}_next_day'])
            
            # Ternary with noise threshold (±1 bp)
            targets[f'y_{spread_name}_ternary'] = pd.cut(
                targets[f'y_{spread_name}_next_day'],
                bins=[-np.inf, -1, 1, np.inf],
                labels=[-1, 0, 1]
            )
        
        return targets

    def create_features(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create all features and target variables.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Features and targets DataFrames
        """
        logger.info("Starting feature engineering process...")
        
        # Initialize features DataFrame with index from treasury data
        features = pd.DataFrame(index=self.treasury_data.index)
        
        # Add different types of features
        features = pd.concat([features, self._compute_calendar_features(self.treasury_data.index)], axis=1)
        features = pd.concat([features, self._compute_trend_features(self.treasury_data, '2-Year')], axis=1)
        features = pd.concat([features, self._compute_yield_curve_features()], axis=1)
        features = pd.concat([features, self._compute_carry_features()], axis=1)
        features = pd.concat([features, self.macro_data], axis=1)
        
        # Compute target variables
        targets = self._compute_targets()
        
        # Handle missing values
        features = features.ffill().fillna(0)  # Forward fill then fill remaining NaNs with 0
        targets = targets.ffill().fillna(0)    # Same for targets
        
        logger.info(f"Feature engineering complete. Created {features.shape[1]} features.")
        
        return features, targets

def create_train_val_test_splits(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    train_end: str = '2015-12-31',
    val_end: str = '2018-12-31'
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create time-based train/validation/test splits.
    
    Parameters
    ----------
    features : pd.DataFrame
        Feature DataFrame
    targets : pd.DataFrame
        Target DataFrame
    train_end : str
        End date for training data
    val_end : str
        End date for validation data
        
    Returns
    -------
    Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]
        Dictionary containing train, validation, and test splits
    """
    # Create splits
    train_features = features[features.index <= train_end]
    train_targets = targets[targets.index <= train_end]
    
    val_features = features[(features.index > train_end) & (features.index <= val_end)]
    val_targets = targets[(targets.index > train_end) & (targets.index <= val_end)]
    
    test_features = features[features.index > val_end]
    test_targets = targets[targets.index > val_end]
    
    return {
        'train': (train_features, train_targets),
        'validation': (val_features, val_targets),
        'test': (test_features, test_targets)
    } 