"""
Test script for model training with selected features.
"""

from src.model_training import ModelTrainer
import logging
import pandas as pd
from pathlib import Path
import numpy as np

# Configure logging
Path('results/logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/test_model_training.log'),
        logging.StreamHandler()
    ]
)

def check_data_files():
    """Check data files existence and print sample info."""
    features_path = Path('data/processed/features.csv')
    targets_path = Path('data/processed/targets.csv')
    selected_features_path = Path('results/feature_analysis/y_2s10s_next_day/selected_features.csv')
    
    # Check features file
    if features_path.exists():
        features = pd.read_csv(features_path, nrows=5)
        logging.info(f"Features file exists. Sample columns: {', '.join(features.columns[:5])}")
    else:
        logging.error(f"Features file not found at {features_path}")
        return False
    
    # Check targets file
    if targets_path.exists():
        targets = pd.read_csv(targets_path, nrows=5)
        logging.info(f"Targets file exists. Sample columns: {', '.join(targets.columns[:5])}")
    else:
        logging.error(f"Targets file not found at {targets_path}")
        return False
    
    # Check selected features file
    if selected_features_path.exists():
        selected = pd.read_csv(selected_features_path)
        logging.info(f"Selected features file exists. Sample features: {', '.join(selected[selected.columns[1]][:5])}")
    else:
        logging.error(f"Selected features file not found at {selected_features_path}")
        return False
    
    return True

def main():
    # Check data files first
    if not check_data_files():
        logging.error("Data file check failed. Exiting.")
        return
    
    # Initialize trainer for 2s10s spread with next-day prediction
    trainer = ModelTrainer(
        spread='2s10s',
        prediction_type='next_day',
        model_type='rf'  # Using Random Forest for initial test
    )
    
    # Load data
    features, target = trainer.load_data()
    if features is None or target is None:
        logging.error("Failed to load data. Exiting.")
        return
    
    logging.info(f"Successfully loaded data:")
    logging.info(f"Features shape: {features.shape}")
    logging.info(f"Target shape: {target.shape}")
    logging.info(f"Date range: {features.index.min()} to {features.index.max()}")
    
    # Create and train model
    trainer.create_model()
    results = trainer.walk_forward_validation()
    
    if results is None:
        logging.error("Walk-forward validation failed. Exiting.")
        return
    
    # Save results
    trainer.save_results(results)
    trainer.save_model()
    
    # Log summary
    logging.info(f"Test training completed for 2s10s spread with next-day prediction")
    logging.info(f"Average MSE: {np.mean(results['mse']):.6f}")
    logging.info(f"Standard Deviation MSE: {np.std(results['mse']):.6f}")
    
    if hasattr(trainer.model, 'feature_importances_'):
        logging.info(f"Top 5 features by importance:")
        feature_importance = pd.Series(
            trainer.model.feature_importances_,
            index=trainer.features.columns
        ).sort_values(ascending=False)
        for feature, importance in feature_importance.head().items():
            logging.info(f"{feature}: {importance:.6f}")

if __name__ == "__main__":
    main() 