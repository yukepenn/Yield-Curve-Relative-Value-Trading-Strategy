"""
Model training module for yield curve spread prediction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Union
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import joblib
import logging
from datetime import datetime

# Configure logging
Path('results/logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/model_training.log'),
        logging.StreamHandler()
    ]
)

class ModelTrainer:
    """Base class for model training with walk-forward validation."""
    
    def __init__(self, spread: str, prediction_type: str, model_type: str):
        """
        Initialize ModelTrainer.
        
        Args:
            spread: Name of the spread ('2s10s', '5s30s', '2s5s', '10s30s', '3m10y')
            prediction_type: Type of prediction ('next_day', 'direction', 'ternary')
            model_type: Type of model ('ridge', 'lasso', 'rf', 'xgb')
        """
        self.spread = spread
        self.prediction_type = prediction_type
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()  # Initialize scaler
        self.features = None
        self.target = None
        self.results = {}
        
        # Create directories for saving models and results
        self.models_dir = Path('results/model_pickles')
        self.results_dir = Path('results/model_training')
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initialized ModelTrainer for {spread} {prediction_type} with {model_type}")
    
    def load_data(self):
        """Load and prepare the data."""
        try:
            # Load features and targets
            features_path = Path('data/processed/features.csv')
            targets_path = Path('data/processed/targets.csv')
            
            if not features_path.exists():
                logging.error(f"Features file not found at {features_path}")
                return None, None
            if not targets_path.exists():
                logging.error(f"Targets file not found at {targets_path}")
                return None, None
            
            # Load data with datetime index
            features = pd.read_csv(features_path)
            targets = pd.read_csv(targets_path)
            
            # Convert index to datetime
            features.index = pd.to_datetime(features.iloc[:, 0])
            targets.index = pd.to_datetime(targets.iloc[:, 0])
            
            # Drop the date column since it's now in the index
            features = features.drop(features.columns[0], axis=1)
            targets = targets.drop(targets.columns[0], axis=1)
            
            # Log data shapes
            logging.info(f"Loaded features shape: {features.shape}")
            logging.info(f"Loaded targets shape: {targets.shape}")
            
            # Align features and targets
            common_index = features.index.intersection(targets.index)
            if len(common_index) == 0:
                logging.error("No common dates between features and targets")
                return None, None
            
            features = features.loc[common_index]
            targets = targets.loc[common_index]
            
            # Log aligned shapes
            logging.info(f"Aligned features shape: {features.shape}")
            logging.info(f"Aligned targets shape: {targets.shape}")
            
            # Get target column
            target_col = f'y_{self.spread}_{self.prediction_type}'
            if target_col not in targets.columns:
                logging.error(f"Target column {target_col} not found")
                return None, None
            
            # Store features and target as instance variables
            self.features = features
            self.target = targets[target_col]
            
            # Scale features for linear models
            if self.model_type in ['ridge', 'lasso']:
                self.features = pd.DataFrame(
                    self.scaler.fit_transform(self.features),
                    index=self.features.index,
                    columns=self.features.columns
                )
            
            logging.info(f"Loaded {len(self.features)} samples with {self.features.shape[1]} features")
            logging.info(f"Date range: {self.features.index.min()} to {self.features.index.max()}")
            return self.features, self.target
            
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            logging.error(f"Error details: {e.__class__.__name__}")
            return None, None
    
    def create_model(self) -> None:
        """Create model instance based on model_type."""
        if self.prediction_type == 'next_day':
            if self.model_type == 'ridge':
                self.model = Ridge(alpha=1.0)
            elif self.model_type == 'lasso':
                self.model = Lasso(alpha=1.0)
            elif self.model_type == 'rf':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'xgb':
                self.model = xgb.XGBRegressor(random_state=42)
        
        elif self.prediction_type in ['direction', 'ternary']:
            # Handle class imbalance
            class_weights = compute_class_weight(
                'balanced',
                classes=np.unique(self.target),
                y=self.target
            )
            class_weight_dict = dict(zip(np.unique(self.target), class_weights))
            
            if self.model_type == 'rf':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    class_weight=class_weight_dict,
                    random_state=42
                )
            elif self.model_type == 'xgb':
                if self.prediction_type == 'ternary':
                    self.model = xgb.XGBClassifier(
                        objective='multi:softmax',
                        num_class=3,
                        scale_pos_weight=class_weights[1:].mean()/class_weights[0],
                        random_state=42
                    )
                else:
                    self.model = xgb.XGBClassifier(
                        scale_pos_weight=class_weights[1]/class_weights[0],
                        random_state=42
                    )
        
        else:
            raise ValueError(f"Unknown target type: {self.prediction_type}")
        
        logging.info(f"Created {self.model_type} model for {self.prediction_type} prediction")
    
    def walk_forward_validation(self, n_splits: int = 5) -> Dict:
        """
        Perform walk-forward validation.
        
        Args:
            n_splits: Number of splits for time series cross-validation
            
        Returns:
            Dictionary containing validation results
        """
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            results = {
                'mse': [],
                'accuracy': [],
                'f1': [],
                'roc_auc': [],
                'feature_importance': [],
                'predictions': [],
                'actuals': []
            }
            
            # Convert to numpy arrays for sklearn compatibility
            X = self.features.values
            y = self.target.values
            
            for train_idx, val_idx in tscv.split(X):
                # Split data
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                # Train model
                self.model.fit(X_train, y_train)
                
                # Make predictions
                if self.prediction_type == 'next_day':
                    y_pred = self.model.predict(X_val)
                    results['mse'].append(mean_squared_error(y_val, y_pred))
                else:
                    y_pred = self.model.predict(X_val)
                    y_prob = self.model.predict_proba(X_val)
                    results['accuracy'].append(accuracy_score(y_val, y_pred))
                    results['f1'].append(f1_score(y_val, y_pred, average='weighted'))
                    if self.prediction_type == 'direction':
                        results['roc_auc'].append(roc_auc_score(y_val, y_prob[:, 1]))
                
                # Store predictions and actuals
                results['predictions'].extend(y_pred)
                results['actuals'].extend(y_val)
                
                # Store feature importance if available
                if hasattr(self.model, 'feature_importances_'):
                    results['feature_importance'].append(
                        pd.Series(self.model.feature_importances_, index=self.features.columns)
                    )
            
            # Calculate average metrics
            self.results = {
                'mse': np.mean(results['mse']) if results['mse'] else None,
                'accuracy': np.mean(results['accuracy']) if results['accuracy'] else None,
                'f1': np.mean(results['f1']) if results['f1'] else None,
                'roc_auc': np.mean(results['roc_auc']) if results['roc_auc'] else None,
                'feature_importance': pd.concat(results['feature_importance'], axis=1).mean(axis=1) 
                    if results['feature_importance'] else None,
                'predictions': results['predictions'],
                'actuals': results['actuals']
            }
            
            logging.info(f"Completed walk-forward validation for {self.spread} {self.prediction_type}")
            return self.results
            
        except Exception as e:
            logging.error(f"Error in walk-forward validation: {str(e)}")
            return None
    
    def save_model(self) -> None:
        """Save trained model and results."""
        # Save model
        model_path = self.models_dir / f"{self.spread}_{self.prediction_type}_{self.model_type}.joblib"
        joblib.dump(self.model, model_path)
        
        # Save scaler if used
        if self.model_type in ['ridge', 'lasso']:
            scaler_path = self.models_dir / f"{self.spread}_{self.prediction_type}_{self.model_type}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
        
        # Save results
        results_path = self.results_dir / f"{self.spread}_{self.prediction_type}_{self.model_type}_results.csv"
        pd.DataFrame(self.results).to_csv(results_path)
        
        # Save feature importance if available
        if self.results['feature_importance'] is not None:
            importance_path = self.results_dir / f"{self.spread}_{self.prediction_type}_{self.model_type}_importance.csv"
            self.results['feature_importance'].to_csv(importance_path)
        
        logging.info(f"Saved model and results to {model_path} and {results_path}")

    @staticmethod
    def run_batch_training():
        """Run training for all spreads and prediction types."""
        # Define spreads and prediction types
        spreads = ['2s10s', '5s30s', '2s5s', '10s30s', '3m10y']
        prediction_types = ['next_day', 'direction', 'ternary']
        model_types = ['ridge', 'lasso', 'rf', 'xgb']
        
        # Create results directories if they don't exist
        Path('results/model_training').mkdir(parents=True, exist_ok=True)
        Path('results/model_pickles').mkdir(parents=True, exist_ok=True)
        
        # Initialize results DataFrame
        results_summary = []
        
        # Run training for each combination
        for spread in spreads:
            for pred_type in prediction_types:
                # Skip ternary for 10s30s as discussed
                if spread == '10s30s' and pred_type == 'ternary':
                    logging.info(f"Skipping {spread} {pred_type} as previously discussed")
                    continue
                    
                for model_type in model_types:
                    try:
                        logging.info(f"Training {spread} {pred_type} with {model_type}")
                        
                        # Initialize trainer
                        trainer = ModelTrainer(
                            spread=spread,
                            prediction_type=pred_type,
                            model_type=model_type
                        )
                        
                        # Load data and train model
                        results = trainer.train()
                        
                        if results is not None:
                            # Add to summary
                            results_summary.append({
                                'spread': spread,
                                'prediction_type': pred_type,
                                'model_type': model_type,
                                'mse': results.get('mse', None),
                                'accuracy': results.get('accuracy', None),
                                'f1': results.get('f1', None),
                                'roc_auc': results.get('roc_auc', None)
                            })
                            logging.info(f"Completed {spread} {pred_type} with {model_type}")
                        else:
                            logging.error(f"Training failed for {spread} {pred_type} with {model_type}")
                        
                    except Exception as e:
                        logging.error(f"Error training {spread} {pred_type} with {model_type}: {str(e)}")
        
        # Save summary results
        if results_summary:
            summary_df = pd.DataFrame(results_summary)
            summary_df.to_csv('results/model_training/summary_results.csv', index=False)
            logging.info("Training completed. Summary saved to results/model_training/summary_results.csv")
        else:
            logging.error("No successful training runs to summarize")

    def train(self):
        """Train the model and save results."""
        try:
            # Load data
            X, y = self.load_data()
            if X is None or y is None:
                raise ValueError("Failed to load data")
            
            # Create model
            self.create_model()
            
            # Perform walk-forward validation
            results = self.walk_forward_validation()
            
            # Save model and results
            self.save_model()
            
            logging.info(f"Successfully trained {self.spread} {self.prediction_type} with {self.model_type}")
            return results
            
        except Exception as e:
            logging.error(f"Error training {self.spread} {self.prediction_type} with {self.model_type}: {str(e)}")
            return None

if __name__ == "__main__":
    ModelTrainer.run_batch_training() 