"""
Test script for training models across all spreads and prediction types.
"""

from src.model_training import ModelTrainer
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import json

# Configure logging
Path('results/logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/test_all_models.log'),
        logging.StreamHandler()
    ]
)

# Define spreads and prediction types
SPREADS = ['2s10s', '5s30s', '2s5s', '10s30s', '3m10y']
PREDICTION_TYPES = ['next_day', 'direction', 'ternary']
MODEL_TYPES = ['rf', 'xgb']  # We'll use Random Forest and XGBoost for testing

def check_data_files(spread: str, prediction_type: str) -> bool:
    """Check data files existence for a specific spread and prediction type."""
    features_path = Path('data/processed/features.csv')
    targets_path = Path('data/processed/targets.csv')
    selected_features_path = Path(f'results/feature_analysis/y_{spread}_{prediction_type}/selected_features.csv')
    
    # Check features file
    if not features_path.exists():
        logging.error(f"Features file not found at {features_path}")
        return False
    
    # Check targets file
    if not targets_path.exists():
        logging.error(f"Targets file not found at {targets_path}")
        return False
    
    # Check selected features file
    if not selected_features_path.exists():
        logging.error(f"Selected features file not found at {selected_features_path}")
        return False
    
    return True

def train_model(spread: str, prediction_type: str, model_type: str) -> Dict:
    """Train a model for a specific spread, prediction type, and model type."""
    logging.info(f"\n{'='*80}")
    logging.info(f"Training {model_type} model for {spread} {prediction_type}")
    logging.info(f"{'='*80}")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(
            spread=spread,
            prediction_type=prediction_type,
            model_type=model_type
        )
        
        # Train model using unified train method
        results = trainer.train()
        
        if results is None:
            logging.error(f"Training failed for {spread} {prediction_type}")
            return None
        
        # Prepare summary
        summary = {
            'spread': spread,
            'prediction_type': prediction_type,
            'model_type': model_type,
            'num_features': len(trainer.features.columns),
            'num_samples': len(trainer.features),
            'date_range': {
                'start': trainer.features.index.min().strftime('%Y-%m-%d'),
                'end': trainer.features.index.max().strftime('%Y-%m-%d')
            }
        }
        
        # Add metrics based on prediction type
        if prediction_type == 'next_day':
            summary['mse'] = results['mse']
            summary['train_loss'] = results.get('train_loss')
            summary['val_loss'] = results.get('val_loss')
        else:
            summary['accuracy'] = results.get('accuracy')
            summary['f1'] = results.get('f1')
            if prediction_type == 'direction':
                summary['roc_auc'] = results.get('roc_auc')
        
        # Add top features if available
        if hasattr(trainer.model, 'feature_importances_'):
            feature_importance = pd.Series(
                trainer.model.feature_importances_,
                index=trainer.features.columns
            ).sort_values(ascending=False)
            summary['top_features'] = feature_importance.head(5).to_dict()
        
        return summary
        
    except Exception as e:
        logging.error(f"Error in train_model for {spread} {prediction_type} {model_type}: {str(e)}")
        return None

def main():
    # Initialize results storage
    all_results = []
    
    # Test each combination
    for spread in SPREADS:
        for prediction_type in PREDICTION_TYPES:
            # Skip 10s30s ternary as it has no meaningful classification
            if spread == '10s30s' and prediction_type == 'ternary':
                logging.info(f"Skipping {spread} {prediction_type} as it has no meaningful classification")
                continue
            
            # Check data files
            if not check_data_files(spread, prediction_type):
                logging.error(f"Skipping {spread} {prediction_type} due to missing data files")
                continue
            
            # Train models
            for model_type in MODEL_TYPES:
                try:
                    results = train_model(spread, prediction_type, model_type)
                    if results:
                        all_results.append(results)
                        # Save intermediate results
                        with open('results/model_training/all_models_summary.json', 'w') as f:
                            json.dump(all_results, f, indent=4)
                except Exception as e:
                    logging.error(f"Error training {model_type} model for {spread} {prediction_type}: {str(e)}")
                    continue
    
    logging.info(f"\n{'='*80}")
    logging.info("All model training completed")
    logging.info(f"Results saved to results/model_training/all_models_summary.json")
    logging.info(f"{'='*80}")

if __name__ == "__main__":
    main() 