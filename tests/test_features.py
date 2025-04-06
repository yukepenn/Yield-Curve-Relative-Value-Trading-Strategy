"""
Test suite for feature engineering module.
"""

import sys
import os
import pandas as pd
import logging
from pathlib import Path

# Add src to path for imports
src_path = str(Path(__file__).parent.parent / 'src')
sys.path.insert(0, src_path)

from feature_engineering import FeatureEngineer, create_train_val_test_splits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_feature_engineering():
    """Test feature engineering process and validate results."""
    logger.info("Loading raw data...")
    
    # Load treasury and macro data
    try:
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        treasury_data = pd.read_csv(data_dir / 'treasury_yields.csv', index_col=0, parse_dates=True)
        macro_data = pd.read_csv(data_dir / 'macro_indicators.csv', index_col=0, parse_dates=True)
        logger.info(f"Loaded data with shapes: Treasury={treasury_data.shape}, Macro={macro_data.shape}")
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        logger.error("Please run data_ingestion.py first to generate the required data files.")
        return False
    
    # Initialize feature engineering
    logger.info("Initializing feature engineering...")
    feature_engineer = FeatureEngineer(treasury_data, macro_data)
    
    # Generate features and targets
    logger.info("Generating features and targets...")
    features, targets = feature_engineer.create_features()
    
    # Log feature statistics
    logger.info(f"Generated {len(features.columns)} features and {len(targets.columns)} targets")
    logger.info(f"Date range: {features.index.min()} to {features.index.max()}")
    
    # Create train/val/test splits
    logger.info("Creating data splits...")
    splits = create_train_val_test_splits(features, targets)
    
    # Log split sizes
    for split_name, (split_features, split_targets) in splits.items():
        logger.info(f"{split_name.capitalize()} set size: {len(split_features)} samples")
        
    # Save processed features and targets
    logger.info("Saving processed data...")
    processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
    processed_dir.mkdir(exist_ok=True)
    
    features.to_csv(processed_dir / 'features.csv')
    targets.to_csv(processed_dir / 'targets.csv')
    
    # Save feature statistics
    with open(processed_dir / 'feature_stats.txt', 'w') as f:
        f.write("Feature Engineering Statistics\n")
        f.write("===========================\n\n")
        f.write(f"Total features: {len(features.columns)}\n")
        f.write(f"Total targets: {len(targets.columns)}\n")
        f.write(f"Date range: {features.index.min()} to {features.index.max()}\n\n")
        
        f.write("Feature Categories:\n")
        f.write("------------------\n")
        calendar_features = [col for col in features.columns if any(x in col for x in ['is_', 'days_to_'])]
        trend_features = [col for col in features.columns if any(x in col for x in ['_change_', '_ma_', '_vol_', '_zscore_', '_mom_', '_rsi_'])]
        pca_features = [col for col in features.columns if 'yield_pc' in col]
        carry_features = [col for col in features.columns if '_carry' in col]
        macro_features = [col for col in features.columns if col in macro_data.columns]
        
        f.write(f"Calendar features: {len(calendar_features)}\n")
        f.write(f"Trend features: {len(trend_features)}\n")
        f.write(f"PCA features: {len(pca_features)}\n")
        f.write(f"Carry features: {len(carry_features)}\n")
        f.write(f"Macro features: {len(macro_features)}\n")
        
        f.write("\nTarget Types:\n")
        f.write("------------\n")
        regression_targets = [col for col in targets.columns if 'next_day' in col]
        binary_targets = [col for col in targets.columns if 'direction' in col]
        ternary_targets = [col for col in targets.columns if 'ternary' in col]
        
        f.write(f"Regression targets: {len(regression_targets)}\n")
        f.write(f"Binary classification targets: {len(binary_targets)}\n")
        f.write(f"Ternary classification targets: {len(ternary_targets)}\n")
    
    logger.info("Feature engineering test complete. Check data/processed/ for results.")
    return True

if __name__ == "__main__":
    test_feature_engineering() 