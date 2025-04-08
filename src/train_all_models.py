"""
Train all models with hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
from model_training import ModelTrainer
import torch
import joblib

# Configure logging
Path('results/logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/train_all_models.log'),
        logging.StreamHandler()
    ]
)

def train_all_models():
    """Train all models with hyperparameter tuning."""
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Define model configurations
    spreads = ['2s10s', '5s30s', '2s5s', '10s30s', '3m10y']
    prediction_types = ['next_day', 'direction', 'ternary']
    model_types = ['rf', 'xgb', 'lstm']  # Removed linear models as they don't need tuning
    
    # Initialize results dictionary
    all_results = []
    
    # Train all combinations
    for spread in spreads:
        for pred_type in prediction_types:
            # Skip ternary for 10s30s as discussed before
            if spread == '10s30s' and pred_type == 'ternary':
                continue
                
            for model_type in model_types:
                try:
                    logging.info(f"Training {model_type} for {spread} {pred_type}")
                    
                    # Initialize model trainer
                    trainer = ModelTrainer(spread, pred_type, model_type)
                    
                    # Load data
                    features, target = trainer.load_data()
                    if features is None or target is None:
                        logging.error(f"Failed to load data for {spread} {pred_type}")
                        continue
                    
                    # Create model
                    trainer.create_model()
                    
                    # Perform walk-forward validation with hyperparameter tuning
                    results = trainer.walk_forward_validation(n_splits=3)  # Reduced splits for faster tuning
                    
                    if results is None:
                        logging.error(f"Walk-forward validation failed for {spread} {pred_type} {model_type}")
                        continue
                    
                    # Calculate average metrics
                    avg_results = {
                        'spread': spread,
                        'prediction_type': pred_type,
                        'model_type': model_type,
                        'num_features': len(features.columns),
                        'num_samples': len(features),
                        'date_range': {
                            'start': features.index.min().strftime('%Y-%m-%d'),
                            'end': features.index.max().strftime('%Y-%m-%d')
                        }
                    }
                    
                    # Add metrics based on prediction type
                    if pred_type == 'next_day':
                        avg_results['mse'] = np.mean(results['mse'])
                        avg_results['mse_std'] = np.std(results['mse'])
                    else:
                        avg_results['accuracy'] = np.mean(results['accuracy'])
                        avg_results['f1'] = np.mean(results['f1'])
                        if pred_type == 'direction':
                            avg_results['roc_auc'] = np.mean(results['roc_auc'])
                    
                    # Add best hyperparameters
                    avg_results['best_params'] = results['best_params']
                    
                    # Add top features for tree-based models
                    if model_type in ['rf', 'xgb'] and results['feature_importance']:
                        # Average feature importance across folds
                        all_importances = {}
                        for fold_importance in results['feature_importance']:
                            for feature, importance in fold_importance.items():
                                all_importances.setdefault(feature, []).append(importance)
                        
                        avg_importance = {
                            feature: np.mean(importances)
                            for feature, importances in all_importances.items()
                        }
                        
                        # Get top 5 features
                        top_features = dict(
                            sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                        )
                        avg_results['top_features'] = top_features
                    
                    # Save individual model results
                    model_dir = Path(f'results/model_training/{spread}_{pred_type}')
                    model_dir.mkdir(parents=True, exist_ok=True)
                    
                    with open(model_dir / f'{model_type}_results.json', 'w') as f:
                        json.dump(results, f, indent=4)
                    
                    # Save model if it's the best performing one
                    if model_type == 'lstm':
                        torch.save(trainer.model.state_dict(), 
                                 f'results/model_pickles/{spread}_{pred_type}_{model_type}.pt')
                    else:
                        joblib.dump(trainer.model, 
                                  f'results/model_pickles/{spread}_{pred_type}_{model_type}.pkl')
                    
                    all_results.append(avg_results)
                    logging.info(f"Completed training {model_type} for {spread} {pred_type}")
                    
                except Exception as e:
                    logging.error(f"Error training {model_type} for {spread} {pred_type}: {e}")
                    continue
    
    # Save all results
    with open('results/model_training/all_models_summary.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logging.info("Completed training all models")

if __name__ == '__main__':
    train_all_models() 