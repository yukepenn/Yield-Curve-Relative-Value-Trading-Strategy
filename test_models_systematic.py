"""
Systematic testing of LSTM and MLP models for yield spread prediction.
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
        return super(NumpyEncoder, self).default(obj)

class SystematicTester:
    """Class to handle systematic testing of LSTM and MLP models."""
    
    def __init__(self, spread: str = '2s10s'):
        """
        Initialize SystematicTester.
        
        Args:
            spread: Name of the spread to test ('2s10s')
        """
        self.spread = spread
        self.results = {}
        self.errors = {}
        
        # Set paths
        self.data_dir = Path('data/processed')
        self.results_dir = Path('results/model_training')
        self.error_logs_dir = Path('results/logs')
        
        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.error_logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Set file paths
        self.results_file = f"results/systematic_testing/{spread}_systematic_test_results.json"
        self.errors_file = f"results/systematic_testing/{spread}_systematic_test_errors.json"
        
        # Create results directories
        Path('results/systematic_testing').mkdir(parents=True, exist_ok=True)
        
        # Define prediction types and models to test
        self.prediction_types = ['next_day', 'direction', 'ternary']
        self.model_types = ['lstm', 'mlp']
        
        # Initialize data paths
        self.data_paths = {
            'features': Path('data/processed/features.csv'),
            'targets': Path('data/processed/targets.csv')
        }
        
        # Timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load existing results if any
        if os.path.exists(self.results_file):
            try:
                with open(self.results_file, 'r') as f:
                    loaded_results = json.load(f)
                if isinstance(loaded_results, dict):
                    self.results = loaded_results
                else:
                    logging.warning("Loaded results file is not a dictionary. Reinitializing as empty dictionary.")
                    self.results = {}
            except Exception as e:
                logging.error(f"Error loading results file: {e}")
                self.results = {}
                
        # Load existing errors if any
        if os.path.exists(self.errors_file):
            try:
                with open(self.errors_file, 'r') as f:
                    loaded_errors = json.load(f)
                if isinstance(loaded_errors, dict):
                    self.errors = loaded_errors
                else:
                    logging.warning("Loaded errors file is not a dictionary. Reinitializing as empty dictionary.")
                    self.errors = {}
            except Exception as e:
                logging.error(f"Error loading errors file: {e}")
                self.errors = {}
        
        logging.info(f"Initialized SystematicTester for {spread} with LSTM and MLP models")

    def validate_data(self, prediction_type: str) -> Tuple[bool, List[str]]:
        """
        Validate data for the given prediction type.
        
        Args:
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
            target_col = f'y_{self.spread}_{prediction_type}'
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

    def log_error(self, test_key: str, error_msg: str, category: ErrorCategory, stack_trace: Optional[str] = None) -> None:
        """Log error with timestamp and save to file."""
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'category': category.value,
            'message': error_msg,
            'stack_trace': stack_trace
        }
        
        # Add to errors dictionary
        if test_key not in self.errors:
            self.errors[test_key] = []
        self.errors[test_key].append(error_entry)
        
        # Log to console
        logging.error(f"{category.value.upper()}: {error_msg}")
        
        # Save to file
        try:
            with open(self.errors_file, 'w') as f:
                json.dump(self.errors, f, indent=4, cls=NumpyEncoder)
        except Exception as e:
            logging.error(f"Error saving errors to file: {e}")

    def save_results(self, test_key: str, results: Dict, model_type: str, prediction_type: str) -> None:
        """Save test results to file."""
        try:
            self.results[test_key] = {
                'timestamp': datetime.now().isoformat(),
                'metrics': results,
                'status': 'success',
                'model_config': {
                    'spread': self.spread,
                    'prediction_type': prediction_type,
                    'model_type': model_type
                },
                'training_info': {
                    'num_samples': len(results['predictions']),
                    'mse': results.get('mse'),
                    'train_loss': results.get('train_loss'),
                    'val_loss': results.get('val_loss'),
                    'accuracy': results.get('accuracy'),
                    'f1_score': results.get('f1_score')
                }
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=4, cls=NumpyEncoder)
            logging.info(f"Results saved to {self.results_file}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            logging.error(traceback.format_exc())

    def run_tests(self) -> None:
        """Run systematic tests for all prediction types and models."""
        try:
            # Skip ternary classification for 10s30s spread
            if self.spread == '10s30s':
                self.prediction_types.remove('ternary')
                logging.info("Skipping ternary classification for 10s30s spread")
            
            # Test each prediction type and model combination
            for prediction_type in self.prediction_types:
                for model_type in self.model_types:
                    test_key = f"{model_type}_{prediction_type}"
                    logging.info(f"Starting test for {test_key}")
                    
                    try:
                        # Validate data
                        is_valid, issues = self.validate_data(prediction_type)
                        if not is_valid:
                            self.log_error(
                                test_key,
                                f"Data validation failed: {', '.join(issues)}",
                                ErrorCategory.DATA_VALIDATION
                            )
                            continue
                        
                        # Initialize model trainer
                        trainer = ModelTrainer(
                            spread=self.spread,
                            prediction_type=prediction_type,
                            model_type=model_type,
                            tune_hyperparameters=True
                        )
                        
                        # Train model
                        try:
                            results = trainer.train()
                            if results:
                                self.save_results(test_key, results, model_type, prediction_type)
                                logging.info(f"Successfully completed test for {test_key}")
                            else:
                                self.log_error(
                                    test_key,
                                    "Training returned no results",
                                    ErrorCategory.MODEL_TRAINING
                                )
                        except Exception as e:
                            self.log_error(
                                test_key,
                                f"Error during training: {str(e)}",
                                ErrorCategory.MODEL_TRAINING,
                                traceback.format_exc()
                            )
                            
                    except Exception as e:
                        self.log_error(
                            test_key,
                            f"Unexpected error: {str(e)}",
                            ErrorCategory.UNEXPECTED,
                            traceback.format_exc()
                        )
            
            logging.info("Completed all systematic tests")
            
        except Exception as e:
            logging.error(f"Fatal error in run_tests: {str(e)}\n{traceback.format_exc()}")

if __name__ == "__main__":
    tester = SystematicTester()
    tester.run_tests() 