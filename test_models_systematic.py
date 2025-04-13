"""
Systematic testing of models for yield spread prediction.
This script tests models with comprehensive error handling and data validation.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import traceback
from src.model_training import ModelTrainer
import os
from enum import Enum

class ErrorCategory(Enum):
    """Categories of errors that can occur during testing."""
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    MODEL_TRAINING = "model_training"
    UNEXPECTED = "unexpected"

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

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, (dict, list)):
            # Recursively handle nested dictionaries and lists
            if isinstance(obj, dict):
                return {k: self.default(v) for k, v in obj.items()}
            else:
                return [self.default(v) for v in obj]
        return super(NumpyEncoder, self).default(obj)

def ensure_json_serializable(obj):
    """Helper function to ensure an object is JSON serializable."""
    if obj is None:
        return None
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, (dict, list)):
        if isinstance(obj, dict):
            return {k: ensure_json_serializable(v) for k, v in obj.items()}
        else:
            return [ensure_json_serializable(v) for v in obj]
    return obj

class SystematicTester:
    """Class to handle systematic testing of models."""
    
    def __init__(self, spreads: Optional[List[str]] = None):
        """
        Initialize SystematicTester.
        
        Args:
            spreads: List of spreads to test. If None, tests only 2s10s and 5s30s spreads
        """
        self.spreads = spreads if spreads else ['2s10s', '5s30s']
        self.results = {}
        self.errors = {}
        
        # Set paths
        self.data_dir = Path('data/processed')
        self.results_dir = Path('results/model_training')
        self.error_logs_dir = Path('results/logs')
        
        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.error_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Define prediction types and models to test
        self.prediction_types = ['next_day', 'direction', 'ternary']
        self.model_types = {
            'next_day': ['arima', 'mlp', 'lstm', 'xgb', 'rf', 'lasso', 'ridge'],
            'direction': ['mlp', 'lstm', 'xgb', 'rf', 'lasso', 'ridge'],
            'ternary': ['mlp', 'lstm', 'xgb', 'rf', 'lasso', 'ridge']
        }
        
        # Initialize data paths
        self.data_paths = {
            'features': Path('data/processed/features.csv'),
            'targets': Path('data/processed/targets.csv')
        }
        
        # Timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize results and errors for each spread
        for spread in self.spreads:
            # Set file paths for this spread
            results_file = f"results/systematic_testing/{spread}_systematic_test_results.json"
            errors_file = f"results/systematic_testing/{spread}_systematic_test_errors.json"
            
            # Create results directories
            Path('results/systematic_testing').mkdir(parents=True, exist_ok=True)
            
            # Load existing results if any
            if os.path.exists(results_file):
                try:
                    with open(results_file, 'r') as f:
                        loaded_results = json.load(f)
                    if isinstance(loaded_results, dict):
                        self.results[spread] = loaded_results
                    else:
                        logging.warning(f"Loaded results file for {spread} is not a dictionary. Reinitializing as empty dictionary.")
                        self.results[spread] = {}
                except Exception as e:
                    logging.error(f"Error loading results file for {spread}: {e}")
                    self.results[spread] = {}
            else:
                self.results[spread] = {}
                    
            # Load existing errors if any
            if os.path.exists(errors_file):
                try:
                    with open(errors_file, 'r') as f:
                        loaded_errors = json.load(f)
                    if isinstance(loaded_errors, dict):
                        self.errors[spread] = loaded_errors
                    else:
                        logging.warning(f"Loaded errors file for {spread} is not a dictionary. Reinitializing as empty dictionary.")
                        self.errors[spread] = {}
                except Exception as e:
                    logging.error(f"Error loading errors file for {spread}: {e}")
                    self.errors[spread] = {}
            else:
                self.errors[spread] = {}
        
        logging.info(f"Initialized SystematicTester for spreads: {', '.join(self.spreads)}")

    def validate_data(self, spread: str, prediction_type: str) -> Tuple[bool, List[str]]:
        """
        Validate data for the given prediction type.
        
        Args:
            spread: Name of the spread to test ('2s10s', '5s30s', '2s5s', '10s30s', '3m10y')
            prediction_type: Type of prediction ('next_day', 'direction', 'ternary')
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            # Check if required files exist
            for file_type, path in self.data_paths.items():
                if not path.exists():
                    issues.append(f"Missing required file: {path}")
            
            if issues:
                return False, issues
            
            # Load and validate data
            try:
                features = pd.read_csv(self.data_paths['features'], index_col=0)
                targets = pd.read_csv(self.data_paths['targets'], index_col=0)
                logging.info(f"Loaded features shape: {features.shape}, targets shape: {targets.shape}")
            except Exception as e:
                issues.append(f"Error loading data files: {str(e)}")
                return False, issues
            
            # Check for missing values
            feature_nulls = features.isnull().sum()
            if feature_nulls.any():
                null_cols = feature_nulls[feature_nulls > 0]
                issues.append(f"Features contain missing values in columns: {null_cols.index.tolist()}")
            
            target_nulls = targets.isnull().sum()
            if target_nulls.any():
                null_cols = target_nulls[target_nulls > 0]
                issues.append(f"Targets contain missing values in columns: {null_cols.index.tolist()}")
            
            # Check data types and try to convert
            non_numeric_features = []
            for col in features.columns:
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except Exception:
                    non_numeric_features.append(col)
            
            if non_numeric_features:
                issues.append(f"Features contain non-numeric data in columns: {non_numeric_features}")
            
            non_numeric_targets = []
            for col in targets.columns:
                try:
                    targets[col] = pd.to_numeric(targets[col], errors='coerce')
                except Exception:
                    non_numeric_targets.append(col)
            
            if non_numeric_targets:
                issues.append(f"Targets contain non-numeric data in columns: {non_numeric_targets}")
            
            # Check for sufficient data
            if len(features) < 100:
                issues.append(f"Insufficient data points: {len(features)} (minimum required: 100)")
            
            # Check target column exists
            target_col = f'y_{spread}_{prediction_type}'
            if target_col not in targets.columns:
                issues.append(f"Target column {target_col} not found in targets data")
            
            # Log data info
            if not issues:
                logging.info(f"Data validation successful - Features: {features.shape}, Targets: {targets.shape}")
            
            return len(issues) == 0, issues
            
        except Exception as e:
            issues.append(f"Error during validation: {str(e)}")
            logging.error(f"Validation error: {str(e)}\n{traceback.format_exc()}")
            return False, issues

    def log_error(self, spread: str, test_key: str, error_msg: str, category: ErrorCategory, stack_trace: Optional[str] = None) -> None:
        """Log an error that occurred during testing."""
        # Create error logs directory
        error_logs_dir = Path('results/logs')
        error_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare error data
        error_data = {
            'spread': spread,
            'test_key': test_key,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'message': error_msg,
            'category': category.value,
            'stack_trace': stack_trace
        }
        
        # Save error to JSON
        error_file = error_logs_dir / f"{spread}_{test_key}_error.json"
        try:
            with open(error_file, 'w') as f:
                json.dump(ensure_json_serializable(error_data), f, indent=4)
            logging.error(f"Error logged to {error_file}")
        except Exception as e:
            logging.error(f"Error saving error log: {e}")
            logging.error(traceback.format_exc())

    def save_results(self, spread: str, test_key: str, results: Dict, model_type: str, prediction_type: str) -> None:
        """Save test results to file."""
        # Create results directory structure
        results_dir = Path(f"results/model_training/{spread}_{prediction_type}")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare results data with metadata
        results_data = {
            'spread': spread,
            'prediction_type': prediction_type,
            'model_type': model_type,
            'test_key': test_key,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'results': ensure_json_serializable(results)
        }
        
        # Save results to JSON
        results_file = results_dir / f"{model_type}_results.json"
        try:
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=4)
            logging.info(f"Results saved to {results_file}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            logging.error(traceback.format_exc())
            
        # Save predictions to CSV if available
        if 'predictions' in results:
            predictions_file = results_dir / f"{model_type}_predictions.csv"
            try:
                predictions_df = pd.DataFrame({
                    'date': results['predictions'].keys(),
                    'prediction': results['predictions'].values()
                })
                predictions_df.to_csv(predictions_file, index=False)
                logging.info(f"Predictions saved to {predictions_file}")
            except Exception as e:
                logging.error(f"Error saving predictions: {e}")
                logging.error(traceback.format_exc())

    def run_tests(self) -> None:
        """Run systematic tests for all spreads, prediction types, and models."""
        try:
            for spread in self.spreads:
                logging.info(f"\nStarting tests for spread: {spread}")
                
                # Get prediction types for this spread
                spread_prediction_types = self.prediction_types.copy()
                if spread == '10s30s':
                    spread_prediction_types.remove('ternary')
                    logging.info("Skipping ternary classification for 10s30s spread")
                
                # Test each prediction type and model combination
                for prediction_type in spread_prediction_types:
                    for model_type in self.model_types[prediction_type]:
                        test_key = f"{model_type}_{prediction_type}"
                        logging.info(f"Starting test for {spread} - {test_key}")
                        
                        try:
                            # Validate data
                            is_valid, issues = self.validate_data(spread, prediction_type)
                            if not is_valid:
                                self.log_error(
                                    spread,
                                    test_key,
                                    f"Data validation failed: {', '.join(issues)}",
                                    ErrorCategory.DATA_VALIDATION
                                )
                                continue
                            
                            # Initialize model trainer
                            trainer = ModelTrainer(
                                spread=spread,
                                prediction_type=prediction_type,
                                model_type=model_type,
                                tune_hyperparameters=True
                            )
                            
                            # Train model
                            try:
                                results = trainer.train()
                                if results:
                                    self.save_results(spread, test_key, results, model_type, prediction_type)
                                    logging.info(f"Successfully completed test for {spread} - {test_key}")
                                else:
                                    self.log_error(
                                        spread,
                                        test_key,
                                        "Training returned no results",
                                        ErrorCategory.MODEL_TRAINING
                                    )
                            except Exception as e:
                                self.log_error(
                                    spread,
                                    test_key,
                                    f"Error during training: {str(e)}",
                                    ErrorCategory.MODEL_TRAINING,
                                    traceback.format_exc()
                                )
                                
                        except Exception as e:
                            self.log_error(
                                spread,
                                test_key,
                                f"Unexpected error: {str(e)}",
                                ErrorCategory.UNEXPECTED,
                                traceback.format_exc()
                            )
                
                logging.info(f"Completed all tests for spread: {spread}")
            
            logging.info("\nCompleted all systematic tests")
            
        except Exception as e:
            logging.error(f"Fatal error in run_tests: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    tester = SystematicTester()
    tester.run_tests() 