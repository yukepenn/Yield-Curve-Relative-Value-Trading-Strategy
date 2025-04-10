"""
Test script for training models across all prediction types and model types for 2s10s spread.
"""

from src.model_training import ModelTrainer
import logging
import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import json
import traceback

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

# Define model configurations
SPREAD = '2s10s'  # Focus on 2s10s spread
PREDICTION_TYPES = ['next_day']  # Only next_day prediction
MODEL_TYPES = {
    'next_day': ['mlp', 'lstm']
}

def check_data_files() -> bool:
    """Check data files existence."""
    features_path = Path('data/processed/features.csv')
    targets_path = Path('data/processed/targets.csv')
    
    # Check features file
    if not features_path.exists():
        logging.error(f"Features file not found at {features_path}")
        return False
    
    # Check targets file
    if not targets_path.exists():
        logging.error(f"Targets file not found at {targets_path}")
        return False
    
    return True

def train_model(prediction_type: str, model_type: str) -> Dict:
    """Train a model for a specific prediction type and model type."""
    logging.info(f"\n{'='*80}")
    logging.info(f"Training {model_type} model for {SPREAD} {prediction_type}")
    logging.info(f"{'='*80}")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer(
            spread=SPREAD,
            prediction_type=prediction_type,
            model_type=model_type,
            epochs=100  # Add epochs parameter
        )
        
        # Load data first
        features, target = trainer.load_data()
        if features is None or target is None:
            logging.error(f"Failed to load data for {SPREAD} {prediction_type}")
            return None
        
        # Ensure proper data types for classification tasks
        if prediction_type in ['direction', 'ternary']:
            # Convert target to integer type for classification
            target = target.astype(np.int64)
            # Ensure features are numeric
            features = features.astype(np.float32)
        
        # Train model using unified train method
        results = trainer.train()
        
        if results is None:
            logging.error(f"Training failed for {SPREAD} {prediction_type}")
            return None
        
        # Prepare summary
        summary = {
            'spread': SPREAD,
            'prediction_type': prediction_type,
            'model_type': model_type,
            'num_features': len(features.columns),
            'num_samples': len(features),
            'date_range': {
                'start': features.index.min().strftime('%Y-%m-%d'),
                'end': features.index.max().strftime('%Y-%m-%d')
            },
            'hyperparameters': results.get('best_params', {}),
            'training_time': results.get('training_time', None)
        }
        
        # Add metrics based on prediction type
        if prediction_type == 'next_day':
            summary['mse'] = results['mse']
            summary['rmse'] = np.sqrt(results['mse'])
            summary['mae'] = results.get('mae')
            summary['train_loss'] = results.get('train_loss')
            summary['val_loss'] = results.get('val_loss')
        else:
            summary['accuracy'] = results.get('accuracy')
            summary['f1'] = results.get('f1')
            if prediction_type == 'direction':
                summary['roc_auc'] = results.get('roc_auc')
        
        # Add feature importance if available
        if hasattr(trainer.model, 'feature_importances_'):
            feature_importance = pd.Series(
                trainer.model.feature_importances_,
                index=features.columns
            ).sort_values(ascending=False)
            summary['top_features'] = feature_importance.head(10).to_dict()
        
        # Save individual model results
        results_dir = Path(f'results/model_training/{SPREAD}_{prediction_type}')
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / f'{model_type}_results.json', 'w') as f:
            json.dump(summary, f, indent=4)
        
        return summary
        
    except Exception as e:
        logging.error(f"Error in train_model for {SPREAD} {prediction_type} {model_type}:")
        logging.error(traceback.format_exc())
        return None

def main():
    # Check data files first
    if not check_data_files():
        logging.error("Required data files not found. Exiting.")
        return
    
    # Initialize results storage
    all_results = []
    
    # Test each prediction type with its supported models
    for prediction_type in PREDICTION_TYPES:
        logging.info(f"\nTesting models for {prediction_type} prediction")
        
        # Get supported models for this prediction type
        supported_models = MODEL_TYPES[prediction_type]
        
        # Train each supported model
        for model_type in supported_models:
            try:
                results = train_model(prediction_type, model_type)
                if results:
                    all_results.append(results)
                    # Save intermediate results
                    with open('results/model_training/2s10s_all_models_summary.json', 'w') as f:
                        json.dump(all_results, f, indent=4)
            except Exception as e:
                logging.error(f"Error training {model_type} model for {prediction_type}:")
                logging.error(traceback.format_exc())
                continue
    
    # Log final summary
    logging.info(f"\n{'='*80}")
    logging.info("All model training completed for 2s10s spread")
    logging.info(f"Total models trained: {len(all_results)}")
    logging.info(f"Results saved to results/model_training/2s10s_all_models_summary.json")
    
    # Print performance summary
    logging.info("\nPerformance Summary:")
    for result in all_results:
        model_info = f"{result['model_type']} - {result['prediction_type']}"
        if result['prediction_type'] == 'next_day':
            logging.info(f"{model_info}: RMSE = {result.get('rmse', 'N/A'):.4f}")
        else:
            logging.info(f"{model_info}: Accuracy = {result.get('accuracy', 'N/A')*100:.2f}%")
    
    logging.info(f"{'='*80}")

if __name__ == "__main__":
    main() 