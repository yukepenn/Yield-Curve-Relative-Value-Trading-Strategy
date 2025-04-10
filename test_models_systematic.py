"""
Systematic testing of all model and prediction type combinations.
This script tests each combination one by one with comprehensive error handling and data validation.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import json
import traceback
from src.model_training import ModelTrainer

# Configure logging
Path('results/logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/systematic_testing.log'),
        logging.StreamHandler()
    ]
)

class SystematicTester:
    """Class to handle systematic testing of all model and prediction type combinations."""
    
    def __init__(self, spread: str = '2s10s'):
        """
        Initialize SystematicTester.
        
        Args:
            spread: Name of the spread to test ('2s10s', '5s30s', '2s5s', '10s30s', '3m10y')
        """
        self.spread = spread
        
        # Define model types for each prediction type
        self.model_config = {
            'next_day': {
                'models': ['ridge', 'lasso', 'rf', 'xgb', 'mlp', 'lstm', 'arima'],
                'is_regression': True
            },
            'direction': {
                'models': ['rf', 'xgb', 'mlp', 'lstm'],
                'is_regression': False
            },
            'ternary': {
                'models': ['rf', 'xgb', 'mlp', 'lstm'],
                'is_regression': False
            }
        }
        
        self.results = []
        self.error_log = []
        
        # Create results directories
        self.results_dir = Path('results/systematic_testing')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Initialized SystematicTester for {spread}")
    
    def validate_data(self, features: pd.DataFrame, target: pd.Series, prediction_type: str) -> bool:
        """
        Validate data before training.
        
        Args:
            features: Feature DataFrame
            target: Target Series
            prediction_type: Type of prediction
            
        Returns:
            bool: True if data is valid, False otherwise
        """
        try:
            # Check for missing values
            if features.isnull().any().any():
                logging.warning(f"Features contain missing values: {features.isnull().sum().sum()}")
                return False
            
            if target.isnull().any():
                logging.warning(f"Target contains missing values: {target.isnull().sum()}")
                return False
            
            # Check data types
            if not all(features.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
                logging.warning("Features contain non-numeric columns")
                return False
            
            # Check for infinite values
            if np.isinf(features.values).any():
                logging.warning("Features contain infinite values")
                return False
            
            if np.isinf(target.values).any():
                logging.warning("Target contains infinite values")
                return False
            
            # Check for sufficient data
            if len(features) < 100:
                logging.warning(f"Insufficient data points: {len(features)}")
                return False
            
            # Validate target values based on prediction type
            if prediction_type == 'next_day':
                if not np.issubdtype(target.dtype, np.number):
                    logging.warning("Target must be numeric for next_day prediction")
                    return False
            elif prediction_type == 'direction':
                unique_values = np.unique(target)
                if not (set(unique_values) <= {0, 1}):
                    logging.warning("Target must be binary (0,1) for direction prediction")
                    return False
            elif prediction_type == 'ternary':
                unique_values = np.unique(target)
                if not (set(unique_values) <= {0, 1, 2}):
                    logging.warning("Target must be ternary (0,1,2) for ternary prediction")
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"Error in data validation: {str(e)}")
            return False
    
    def test_model(self, prediction_type: str, model_type: str) -> Optional[Dict]:
        """
        Test a single model and prediction type combination.
        
        Args:
            prediction_type: Type of prediction ('next_day', 'direction', 'ternary')
            model_type: Type of model
            
        Returns:
            Optional[Dict]: Results dictionary if successful, None if failed
        """
        logging.info(f"\n{'='*80}")
        logging.info(f"Testing {self.spread} {prediction_type} with {model_type}")
        logging.info(f"{'='*80}")
        
        try:
            # Initialize trainer
            trainer = ModelTrainer(
                spread=self.spread,
                prediction_type=prediction_type,
                model_type=model_type,
                tune_hyperparameters=True
            )
            
            # Load and validate data
            features, target = trainer.load_data()
            if features is None or target is None:
                logging.error("Failed to load data")
                return None
            
            if not self.validate_data(features, target, prediction_type):
                logging.error("Data validation failed")
                return None
            
            # Ensure proper data types
            if prediction_type in ['direction', 'ternary']:
                target = target.astype(np.int64)
                features = features.astype(np.float32)
            
            # Train model
            start_time = datetime.now()
            results = trainer.train()
            training_time = (datetime.now() - start_time).total_seconds()
            
            if results is None:
                logging.error("Training failed")
                return None
            
            # Prepare results summary
            summary = {
                'spread': self.spread,
                'prediction_type': prediction_type,
                'model_type': model_type,
                'num_features': len(features.columns),
                'num_samples': len(features),
                'date_range': {
                    'start': features.index.min().strftime('%Y-%m-%d'),
                    'end': features.index.max().strftime('%Y-%m-%d')
                },
                'hyperparameters': results.get('hyperparameters', {}),
                'training_time': training_time,
                'metrics': {
                    'mse': results.get('mse'),
                    'accuracy': results.get('accuracy'),
                    'f1': results.get('f1'),
                    'roc_auc': results.get('roc_auc'),
                    'train_loss': results.get('train_loss'),
                    'val_loss': results.get('val_loss')
                },
                'status': 'success'
            }
            
            logging.info(f"Successfully tested {self.spread} {prediction_type} with {model_type}")
            return summary
            
        except Exception as e:
            error_msg = f"Error testing {self.spread} {prediction_type} with {model_type}: {str(e)}"
            logging.error(error_msg)
            logging.error(traceback.format_exc())
            
            self.error_log.append({
                'spread': self.spread,
                'prediction_type': prediction_type,
                'model_type': model_type,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return {
                'spread': self.spread,
                'prediction_type': prediction_type,
                'model_type': model_type,
                'status': 'error',
                'error': str(e)
            }
    
    def run_tests(self):
        """Run systematic testing of all model and prediction type combinations."""
        logging.info("Starting systematic testing")
        
        for prediction_type, config in self.model_config.items():
            # Skip ternary for 10s30s
            if self.spread == '10s30s' and prediction_type == 'ternary':
                logging.info(f"Skipping {self.spread} {prediction_type} as previously discussed")
                continue
            
            for model_type in config['models']:
                result = self.test_model(prediction_type, model_type)
                if result:
                    self.results.append(result)
                
                # Save intermediate results
                self.save_results()
        
        logging.info("Completed systematic testing")
    
    def save_results(self):
        """Save current results and error log."""
        # Save results
        results_file = self.results_dir / f"{self.spread}_systematic_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=4)
        
        # Save error log
        if self.error_log:
            error_file = self.results_dir / f"{self.spread}_systematic_test_errors.json"
            with open(error_file, 'w') as f:
                json.dump(self.error_log, f, indent=4)
        
        logging.info(f"Saved results to {results_file}")

def main():
    """Main function to run systematic testing."""
    # Test for 2s10s spread
    tester = SystematicTester(spread='2s10s')
    tester.run_tests()
    
    # You can add more spreads here
    # tester = SystematicTester(spread='5s30s')
    # tester.run_tests()

if __name__ == "__main__":
    main() 