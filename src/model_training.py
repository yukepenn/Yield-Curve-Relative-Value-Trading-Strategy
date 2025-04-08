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
import json
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
    
    def load_selected_features(self) -> List[str]:
        """Load selected features from feature analysis results."""
        feature_file = Path(f'results/feature_analysis/y_{self.spread}_{self.prediction_type}/selected_features.csv')
        if not feature_file.exists():
            logging.error(f"Selected features file not found at {feature_file}")
            return None
        selected_features = pd.read_csv(feature_file)['0'].tolist()  # Use the second column with header '0'
        logging.info(f"Loaded {len(selected_features)} selected features")
        return selected_features
    
    def load_data(self):
        """Load and prepare the data with selected features."""
        try:
            # Load selected features
            selected_features = self.load_selected_features()
            if selected_features is None:
                return None, None
            
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
            features = pd.read_csv(features_path, index_col=0)  # Use first column as index
            targets = pd.read_csv(targets_path, index_col=0)  # Use first column as index
            
            # Convert index to datetime
            features.index = pd.to_datetime(features.index)
            targets.index = pd.to_datetime(targets.index)
            
            # Filter features to only selected ones
            try:
                features = features[selected_features]
            except KeyError as e:
                logging.error(f"Some selected features not found in features data: {e}")
                return None, None
            
            # Get target column name
            target_col = f'y_{self.spread}_{self.prediction_type}'
            if target_col not in targets.columns:
                logging.error(f"Target column {target_col} not found in targets data")
                return None, None
            
            # Align features and targets
            common_index = features.index.intersection(targets.index)
            features = features.loc[common_index]
            target = targets.loc[common_index, target_col]
            
            # Scale features if using linear models
            if self.model_type in ['ridge', 'lasso']:
                features = pd.DataFrame(
                    self.scaler.fit_transform(features),
                    index=features.index,
                    columns=features.columns
                )
            
            self.features = features
            self.target = target
            
            logging.info(f"Loaded {len(features)} samples with {len(features.columns)} features")
            logging.info(f"Date range: {features.index.min()} to {features.index.max()}")
            
            return features, target
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
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
                        random_state=42
                    )
                else:
                    self.model = xgb.XGBClassifier(
                        objective='binary:logistic',
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
                'mse': results['mse'] if results['mse'] else None,  # Store all MSE values
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
        """Save trained model to disk."""
        model_file = self.models_dir / f"{self.spread}_{self.prediction_type}_{self.model_type}.pkl"
        joblib.dump(self.model, model_file)
        logging.info(f"Saved model to {model_file}")
    
    def save_results(self, results: Dict) -> None:
        """Save training results with feature information."""
        results_dir = self.results_dir / f"{self.spread}_{self.prediction_type}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics = {
            'spread': self.spread,
            'prediction_type': self.prediction_type,
            'model_type': self.model_type,
            'num_features': len(self.features.columns),
            'selected_features': list(self.features.columns),
            'mse': results.get('mse'),
            'accuracy': results.get('accuracy'),
            'f1': results.get('f1'),
            'roc_auc': results.get('roc_auc'),
            'feature_importance': results.get('feature_importance', {}).to_dict() if isinstance(results.get('feature_importance'), pd.Series) else {}
        }
        
        # Save to JSON
        results_file = results_dir / f"{self.model_type}_results.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        logging.info(f"Saved results to {results_file}")
    
    def train(self):
        """Train model with selected features."""
        # Load data with selected features
        features, target = self.load_data()
        if features is None or target is None:
            return
        
        # Create model
        self.create_model()
        
        # Perform walk-forward validation
        results = self.walk_forward_validation()
        
        # Save model and results
        self.save_model()
        self.save_results(results)

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

if __name__ == "__main__":
    ModelTrainer.run_batch_training() 