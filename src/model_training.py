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
    
    def __init__(self, spread_name: str, target_type: str, model_type: str):
        """
        Initialize model trainer.
        
        Args:
            spread_name: Name of the yield spread (e.g., '2s10s', '5s30s')
            target_type: Type of target ('next_day', 'direction', 'ternary')
            model_type: Type of model ('ridge', 'lasso', 'rf', 'xgb')
        """
        self.spread_name = spread_name
        self.target_type = target_type
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.features = None
        self.target = None
        self.results = {}
        
        # Set up paths
        self.results_dir = Path('results/model_training')
        self.models_dir = Path('results/model_pickles')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initialized ModelTrainer for {spread_name} {target_type} with {model_type}")
    
    def load_data(self) -> None:
        """Load and prepare features and target data."""
        # Load features
        features_path = Path('data/processed/features.csv')
        self.features = pd.read_csv(features_path, index_col=0)
        
        # Load targets
        targets_path = Path('data/processed/targets.csv')
        targets = pd.read_csv(targets_path, index_col=0)
        
        # Select target column
        target_col = f'y_{self.spread_name}_{self.target_type}'
        if target_col not in targets.columns:
            raise ValueError(f"Target column {target_col} not found")
        
        self.target = targets[target_col]
        
        # Align features and target
        common_dates = self.features.index.intersection(self.target.index)
        self.features = self.features.loc[common_dates]
        self.target = self.target.loc[common_dates]
        
        # Scale features for linear models
        if self.model_type in ['ridge', 'lasso']:
            self.features = pd.DataFrame(
                self.scaler.fit_transform(self.features),
                index=self.features.index,
                columns=self.features.columns
            )
        
        logging.info(f"Loaded {len(self.features)} samples with {len(self.features.columns)} features")
    
    def create_model(self) -> None:
        """Create model instance based on model_type."""
        if self.target_type == 'next_day':
            if self.model_type == 'ridge':
                self.model = Ridge(alpha=1.0)
            elif self.model_type == 'lasso':
                self.model = Lasso(alpha=1.0)
            elif self.model_type == 'rf':
                self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif self.model_type == 'xgb':
                self.model = xgb.XGBRegressor(random_state=42)
        
        elif self.target_type in ['direction', 'ternary']:
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
                if self.target_type == 'ternary':
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
            raise ValueError(f"Unknown target type: {self.target_type}")
        
        logging.info(f"Created {self.model_type} model for {self.target_type} prediction")
    
    def walk_forward_validation(self, n_splits: int = 5) -> Dict:
        """
        Perform walk-forward validation.
        
        Args:
            n_splits: Number of splits for time series cross-validation
            
        Returns:
            Dictionary containing validation results
        """
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
        
        for train_idx, val_idx in tscv.split(self.features):
            # Split data
            X_train = self.features.iloc[train_idx]
            y_train = self.target.iloc[train_idx]
            X_val = self.features.iloc[val_idx]
            y_val = self.target.iloc[val_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            if self.target_type == 'next_day':
                y_pred = self.model.predict(X_val)
                results['mse'].append(mean_squared_error(y_val, y_pred))
            else:
                y_pred = self.model.predict(X_val)
                y_prob = self.model.predict_proba(X_val)
                results['accuracy'].append(accuracy_score(y_val, y_pred))
                results['f1'].append(f1_score(y_val, y_pred, average='weighted'))
                if self.target_type == 'direction':
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
        
        logging.info(f"Completed walk-forward validation for {self.spread_name} {self.target_type}")
        return self.results
    
    def save_model(self) -> None:
        """Save trained model and results."""
        # Save model
        model_path = self.models_dir / f"{self.spread_name}_{self.target_type}_{self.model_type}.joblib"
        joblib.dump(self.model, model_path)
        
        # Save scaler if used
        if self.model_type in ['ridge', 'lasso']:
            scaler_path = self.models_dir / f"{self.spread_name}_{self.target_type}_{self.model_type}_scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
        
        # Save results
        results_path = self.results_dir / f"{self.spread_name}_{self.target_type}_{self.model_type}_results.csv"
        pd.DataFrame(self.results).to_csv(results_path)
        
        # Save feature importance if available
        if self.results['feature_importance'] is not None:
            importance_path = self.results_dir / f"{self.spread_name}_{self.target_type}_{self.model_type}_importance.csv"
            self.results['feature_importance'].to_csv(importance_path)
        
        logging.info(f"Saved model and results to {model_path} and {results_path}")

def train_all_models():
    """Train models for all spreads and target types."""
    spreads = ['2s10s', '5s30s', '2s5s', '10s30s', '3m10y']
    target_types = ['next_day', 'direction', 'ternary']
    model_types = ['ridge', 'lasso', 'rf', 'xgb']
    
    for spread in spreads:
        for target_type in target_types:
            # Skip 10s30s ternary classification
            if spread == '10s30s' and target_type == 'ternary':
                continue
                
            for model_type in model_types:
                try:
                    trainer = ModelTrainer(spread, target_type, model_type)
                    trainer.load_data()
                    trainer.create_model()
                    trainer.walk_forward_validation()
                    trainer.save_model()
                except Exception as e:
                    logging.error(f"Error training {spread} {target_type} with {model_type}: {str(e)}")

if __name__ == "__main__":
    train_all_models() 