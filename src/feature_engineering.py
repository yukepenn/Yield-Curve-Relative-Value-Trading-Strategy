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
from pathlib import Path

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
        
    def _compute_calendar_features(self) -> None:
        """
        Compute calendar-based features and add them to self.features.
        """
        logger.info("Computing calendar features...")
        
        # Ensure dates have frequency set
        dates = pd.DatetimeIndex(self.treasury_data.index).to_period('D').to_timestamp('D')
        
        # Add calendar features
        self.features['day_of_week'] = dates.dayofweek
        self.features['day_of_month'] = dates.day
        self.features['week_of_year'] = dates.isocalendar().week
        self.features['month'] = dates.month
        self.features['quarter'] = dates.quarter
        self.features['year'] = dates.year
        
        # Compute end-of-period indicators
        self.features['is_month_end'] = dates.month != dates.shift(1).month
        self.features['is_quarter_end'] = dates.quarter != dates.shift(1).quarter
        self.features['is_year_end'] = dates.year != dates.shift(1).year
        
        # Add holiday features
        us_holidays = holidays.US()
        self.features['is_holiday'] = [date in us_holidays for date in dates]
        self.features['is_day_before_holiday'] = [date + pd.Timedelta(days=1) in us_holidays for date in dates]
        self.features['is_day_after_holiday'] = [date - pd.Timedelta(days=1) in us_holidays for date in dates]

    def _compute_trend_features(self, col: str) -> pd.DataFrame:
        """Compute trend features for a given column.
        
        Args:
            col: Column name to compute features for
            
        Returns:
            DataFrame containing trend features
        """
        try:
            if col not in self.treasury_data.columns:
                logger.warning(f"Column {col} not found in treasury data")
                return pd.DataFrame()
            
            # Get the series and ensure it's numeric
            series = pd.to_numeric(self.treasury_data[col], errors='coerce')
            
            # Handle NaN and infinite values
            series = series.replace([np.inf, -np.inf], np.nan)
            series = series.ffill(limit=5).bfill(limit=5)
            
            # Initialize features DataFrame
            features = pd.DataFrame(index=series.index)
            
            # Compute features for each lookback period
            for lookback in LOOKBACK_PERIODS:
                # Level change
                features[f'{col}_change_{lookback}d'] = series - series.shift(lookback)
                
                # Percentage change (with clipping to handle extreme values)
                pct_change = series.pct_change(lookback)
                features[f'{col}_pct_change_{lookback}d'] = pct_change.clip(-1, 1)
                
                # Moving average
                ma = series.rolling(lookback).mean()
                features[f'{col}_ma_{lookback}d'] = ma
                
                # Distance from moving average
                features[f'{col}_dist_ma_{lookback}d'] = series - ma
                
                # Volatility
                features[f'{col}_vol_{lookback}d'] = series.rolling(lookback).std()
                
                # Z-score
                zscore = (series - ma) / features[f'{col}_vol_{lookback}d']
                features[f'{col}_zscore_{lookback}d'] = zscore.replace([np.inf, -np.inf], np.nan)
                
                # Momentum
                features[f'{col}_momentum_{lookback}d'] = series.diff(lookback)
                
                # RSI
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(lookback).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(lookback).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                features[f'{col}_rsi_{lookback}d'] = rsi.replace([np.inf, -np.inf], 50)
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing trend features for {col}: {str(e)}")
            return pd.DataFrame()

    def _compute_yield_curve_features(self) -> pd.DataFrame:
        """Compute yield curve shape features using PCA.
        
        Returns:
            pd.DataFrame: PCA features for the yield curve
        """
        try:
            # Select key tenors for PCA
            key_tenors = ['2-Year', '5-Year', '10-Year', '30-Year']
            yields_for_pca = self.treasury_data[key_tenors].copy()
            
            # Handle NaN values
            yields_for_pca = yields_for_pca.replace([np.inf, -np.inf], np.nan)
            yields_for_pca = yields_for_pca.ffill(limit=5).bfill(limit=5)
            
            # Drop any remaining NaN rows
            yields_for_pca = yields_for_pca.dropna()
            
            if len(yields_for_pca) == 0:
                logger.warning("No valid data for PCA after cleaning")
                return pd.DataFrame(index=self.treasury_data.index)
            
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
            
            # Reindex to match original data
            pca_df = pca_df.reindex(self.treasury_data.index)
            
            # Forward fill and backward fill NaN values
            pca_df = pca_df.ffill(limit=5).bfill(limit=5)
            
            return pca_df
            
        except Exception as e:
            logger.error(f"Error computing yield curve features: {str(e)}")
            return pd.DataFrame(index=self.treasury_data.index)

    def _compute_carry_features(self) -> pd.DataFrame:
        """Compute carry and roll-down features.
        
        Returns:
            pd.DataFrame: Carry and roll-down features
        """
        try:
            features = pd.DataFrame(index=self.treasury_data.index)
            
            for spread_name, (short_tenor, long_tenor) in SPREADS.items():
                try:
                    # Get clean yield data
                    short_yield = self.treasury_data[short_tenor].replace([np.inf, -np.inf], np.nan)
                    long_yield = self.treasury_data[long_tenor].replace([np.inf, -np.inf], np.nan)
                    
                    # Forward and backward fill NaN values
                    short_yield = short_yield.ffill(limit=5).bfill(limit=5)
                    long_yield = long_yield.ffill(limit=5).bfill(limit=5)
                    
                    # Current spread level
                    spread = long_yield - short_yield
                    features[f'{spread_name}_level'] = spread
                    
                    # Carry (approximated by current spread)
                    carry = spread / BUSINESS_DAYS_PER_YEAR
                    features[f'{spread_name}_carry'] = carry
                    
                    # Historical carry performance
                    for period in [5, 21, 63]:  # 1w, 1m, 3m
                        carry_return = carry.rolling(period).sum()
                        features[f'{spread_name}_carry_return_{period}d'] = carry_return
                        
                except Exception as e:
                    logger.error(f"Error computing carry features for {spread_name}: {str(e)}")
                    continue
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing carry features: {str(e)}")
            return pd.DataFrame(index=self.treasury_data.index)

    def _compute_macro_features(self) -> pd.DataFrame:
        """Compute features from macro indicators.
        
        Returns:
            pd.DataFrame: Macro features
        """
        try:
            features = pd.DataFrame(index=self.macro_data.index)
            
            # Resample macro data to daily frequency and forward fill
            macro_daily = self.macro_data.resample('D').ffill()
            
            # Compute features for each macro indicator
            for col in macro_daily.columns:
                # Level
                features[f'macro_{col}_level'] = macro_daily[col]
                
                # Percentage change
                pct_change = macro_daily[col].pct_change()
                features[f'macro_{col}_pct_change'] = pct_change.clip(-1, 1)
                
                # Z-score
                mean = macro_daily[col].rolling(252).mean()  # 1-year rolling mean
                std = macro_daily[col].rolling(252).std()    # 1-year rolling std
                zscore = (macro_daily[col] - mean) / std
                features[f'macro_{col}_zscore'] = zscore.replace([np.inf, -np.inf], np.nan)
                
                # Distance from mean
                features[f'macro_{col}_dist_mean'] = macro_daily[col] - mean
            
            # Reindex to match treasury data index
            features = features.reindex(self.treasury_data.index)
            
            # Forward fill and backward fill NaN values
            features = features.ffill(limit=5).bfill(limit=5)
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing macro features: {str(e)}")
            return pd.DataFrame(index=self.treasury_data.index)

    def _compute_targets(self) -> pd.DataFrame:
        """
        Compute regression and classification targets.
        
        Returns:
            pd.DataFrame: Target variables
        """
        try:
            targets = pd.DataFrame(index=self.treasury_data.index)
            
            for spread_name, (short_tenor, long_tenor) in SPREADS.items():
                # Current spread
                spread = self.treasury_data[long_tenor] - self.treasury_data[short_tenor]
                spread = spread.replace([np.inf, -np.inf], np.nan).ffill(limit=5)
                
                # Next day spread change in basis points
                next_day_change = spread.shift(-1) - spread
                targets[f'y_{spread_name}_next_day'] = next_day_change * 100  # Convert to basis points
                
                # Binary classification target (up/down)
                targets[f'y_{spread_name}_direction'] = (next_day_change > 0).astype(int)
                
                # Ternary classification target (up/flat/down)
                threshold = 0.5  # basis points
                ternary = pd.Series(1, index=spread.index)  # Default to flat (1)
                ternary[next_day_change * 100 > threshold] = 2  # Up
                ternary[next_day_change * 100 < -threshold] = 0  # Down
                targets[f'y_{spread_name}_ternary'] = ternary
                
                # Log class distribution for classification targets
                logger.info(f"Class distribution for y_{spread_name}_direction:")
                direction_dist = targets[f'y_{spread_name}_direction'].value_counts(normalize=True)
                for cls, pct in direction_dist.items():
                    n_samples = (targets[f'y_{spread_name}_direction'] == cls).sum()
                    logger.info(f"  Class {cls}: {n_samples} samples ({pct:.1%})")
                    
                logger.info(f"Class distribution for y_{spread_name}_ternary:")
                ternary_dist = targets[f'y_{spread_name}_ternary'].value_counts(normalize=True)
                for cls, pct in ternary_dist.items():
                    n_samples = (targets[f'y_{spread_name}_ternary'] == cls).sum()
                    logger.info(f"  Class {cls}: {n_samples} samples ({pct:.1%})")
            
            return targets
            
        except Exception as e:
            logger.error(f"Error computing targets: {str(e)}")
            return pd.DataFrame()

    def create_features(self):
        """Create features and targets from raw data.
        
        Returns:
            tuple: (features, targets) DataFrames
        """
        try:
            logger.info("Generating features and targets...")
            
            # Initialize features DataFrame with treasury data index
            self.features = pd.DataFrame(index=self.treasury_data.index)
            logger.info(f"Initialized features with shape: {self.features.shape}")
            
            # Compute calendar features
            self._compute_calendar_features()
            logger.info(f"After calendar features: {self.features.shape}")
            
            # Compute macro features
            macro_features = self._compute_macro_features()
            if not macro_features.empty:
                self.features = pd.concat([self.features, macro_features], axis=1)
                logger.info(f"After macro features: {self.features.shape}")
            else:
                logger.warning("No macro features generated")
            
            # Compute trend features for each treasury series
            for col in self.treasury_data.columns:
                trend_features = self._compute_trend_features(col)
                if not trend_features.empty:
                    self.features = pd.concat([self.features, trend_features], axis=1)
                    logger.info(f"After trend features for {col}: {self.features.shape}")
                else:
                    logger.warning(f"No trend features generated for {col}")
            
            # Compute yield curve features
            pca_features = self._compute_yield_curve_features()
            if not pca_features.empty:
                self.features = pd.concat([self.features, pca_features], axis=1)
                logger.info(f"After PCA features: {self.features.shape}")
            else:
                logger.warning("No PCA features generated")
            
            # Compute carry features
            carry_features = self._compute_carry_features()
            if not carry_features.empty:
                self.features = pd.concat([self.features, carry_features], axis=1)
                logger.info(f"After carry features: {self.features.shape}")
            else:
                logger.warning("No carry features generated")
            
            # Compute targets
            self.targets = self._compute_targets()
            logger.info(f"After computing targets: {self.targets.shape}")
            
            # Validate and preprocess data
            logger.info("Validating and preprocessing data...")
            
            # Replace infinite values with NaN
            self.features = self.features.replace([np.inf, -np.inf], np.nan)
            self.targets = self.targets.replace([np.inf, -np.inf], np.nan)
            logger.info(f"After replacing infinities: {self.features.shape}")
            
            # Forward fill then backward fill NaN values
            self.features = self.features.ffill(limit=5).bfill(limit=5)
            self.targets = self.targets.ffill(limit=5).bfill(limit=5)
            logger.info(f"After filling NaNs: {self.features.shape}")
            
            # Ensure all features are numeric
            for col in self.features.columns:
                self.features[col] = pd.to_numeric(self.features[col], errors='coerce')
            
            # Cap extreme values at 5 standard deviations
            for col in self.features.columns:
                if self.features[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    mean = self.features[col].mean()
                    std = self.features[col].std()
                    if std > 0:  # Only cap if there's variation
                        self.features[col] = self.features[col].clip(
                            lower=mean - 5*std,
                            upper=mean + 5*std
                        )
            logger.info(f"After capping extremes: {self.features.shape}")
            
            # Check for remaining NaN values
            nan_cols = self.features.isna().sum()
            if nan_cols.any():
                logger.warning("Columns with NaN values:")
                for col in nan_cols[nan_cols > 0].index:
                    logger.warning(f"  {col}: {nan_cols[col]} NaN values")
                    
                # Drop columns with too many NaN values (>50%)
                threshold = len(self.features) * 0.5
                cols_to_drop = nan_cols[nan_cols > threshold].index
                if len(cols_to_drop) > 0:
                    logger.warning(f"Dropping {len(cols_to_drop)} columns with >50% NaN values")
                    self.features = self.features.drop(columns=cols_to_drop)
                
                # Fill remaining NaN values with column means
                self.features = self.features.fillna(self.features.mean())
            
            # Log final shapes
            # Forward fill then backward fill NaN values
            self.features = self.features.ffill(limit=5).bfill(limit=5)
            self.targets = self.targets.ffill(limit=5).bfill(limit=5)
            logger.info(f"After filling NaNs: {self.features.shape}")
            
            # Drop any remaining NaN values
            valid_idx = ~self.features.isna().any(axis=1)
            self.features = self.features[valid_idx]
            self.targets = self.targets[valid_idx]
            
            # Log final shapes
            logger.info(f"Final feature shape: {self.features.shape}")
            logger.info(f"Final target shape: {self.targets.shape}")
            
            if len(self.features) > 0:
                logger.info(f"Features date range: {self.features.index[0]} to {self.features.index[-1]}")
                logger.info(f"Number of features: {self.features.shape[1]}")
                # Log sample of feature names
                logger.info(f"Sample feature names: {list(self.features.columns[:5])}")
            else:
                logger.warning("No features generated after preprocessing")
            
            return self.features, self.targets
            
        except Exception as e:
            logger.error(f"Error in create_features: {str(e)}")
            raise

    def save_processed_data(self) -> None:
        """Save processed data to CSV files."""
        try:
            # Create processed directory if it doesn't exist
            processed_dir = Path('data/processed')
            processed_dir.mkdir(exist_ok=True)
            
            # Save features and targets
            self.features.to_csv(processed_dir / 'features.csv')
            self.targets.to_csv(processed_dir / 'targets.csv')
            
            # Save feature statistics
            self._save_feature_statistics(processed_dir)
            
            logger.info("Processed data saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {str(e)}")
            raise

    def _save_feature_statistics(self, processed_dir: Path) -> None:
        """Save feature statistics to a text file."""
        try:
            # Compute statistics
            feature_stats = pd.DataFrame({
                'mean': self.features.mean(),
                'std': self.features.std(),
                'min': self.features.min(),
                'max': self.features.max()
            }).round(4)
            
            # Save to file
            feature_stats.to_csv(processed_dir / 'feature_stats.txt')
            logger.info("Feature statistics saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving feature statistics: {str(e)}")
            raise

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

def main():
    """Run the feature engineering pipeline."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set up paths
    data_dir = Path('data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    # Load raw data
    logger.info("Loading raw data...")
    treasury_data = pd.read_csv(raw_dir / 'treasury_yields.csv', index_col=0, parse_dates=True)
    macro_data = pd.read_csv(raw_dir / 'macro_indicators.csv', index_col=0, parse_dates=True)
    
    # Clean raw data
    logger.info("Cleaning raw data...")
    
    # Step 1: Replace NaN with 0
    treasury_data = treasury_data.fillna(0)
    macro_data = macro_data.fillna(0)
    
    # Step 2: Forward fill to handle missing values
    treasury_data = treasury_data.ffill()
    macro_data = macro_data.ffill()
    
    # Step 3: For any columns that start with 0, backfill to next available value
    for col in treasury_data.columns:
        if treasury_data[col].iloc[0] == 0:
            treasury_data[col] = treasury_data[col].bfill()
    
    for col in macro_data.columns:
        if macro_data[col].iloc[0] == 0:
            macro_data[col] = macro_data[col].bfill()
    
    # Step 4: Replace any remaining 0s with NaN and forward fill
    treasury_data = treasury_data.replace(0, np.nan).ffill()
    macro_data = macro_data.replace(0, np.nan).ffill()
    
    logger.info(f"Loaded data with shapes: Treasury={treasury_data.shape}, Macro={macro_data.shape}")
    
    # Initialize feature engineering
    logger.info("Initializing feature engineering...")
    feature_engineer = FeatureEngineer(treasury_data, macro_data)
    
    # Generate features and targets
    features, targets = feature_engineer.create_features()
    
    # Save processed data
    logger.info("Saving processed data...")
    feature_engineer.save_processed_data()
    
    logger.info("Feature engineering pipeline complete. Check data/processed/ for results.")

if __name__ == "__main__":
    main() 