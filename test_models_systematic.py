"""
Systematic testing of LSTM model for next_day prediction.
This script tests LSTM model with comprehensive error handling and data validation.
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
    """Class to handle systematic testing of LSTM model for next_day prediction."""
    
    def __init__(self, spread: str = '2s10s'):
        """
        Initialize SystematicTester.
        
        Args:
            spread: Name of the spread to test ('2s10s')
        """
        self.spread = spread
        self.results = {}
        self.errors = {}
        
        # Set file paths
        self.results_file = f"results/systematic_testing/{spread}_systematic_test_results.json"
        self.errors_file = f"results/systematic_testing/{spread}_systematic_test_errors.json"
        
        # Create results directories
        Path('results/systematic_testing').mkdir(parents=True, exist_ok=True)
        
        # Initialize data paths
        self.data_paths = {
            'features': Path('data/processed/features.csv'),
            'targets': Path('data/processed/targets.csv'),
            'selected_features': Path(f'results/feature_analysis/y_{spread}_next_day/selected_features.csv')
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
        
        logging.info(f"Initialized SystematicTester for {spread} with next_day LSTM model")

    def validate_data(self, prediction_type: str) -> Tuple[bool, List[str]]:
        """
        Validate data for next_day prediction.
        
        Args:
            prediction_type: Type of prediction ('next_day')
            
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
        if stack_trace:
            logging.error(f"Stack trace: {stack_trace}")
        
        # Save to file
        try:
            with open(self.errors_file, 'w') as f:
                json.dump(self.errors, f, indent=4, cls=NumpyEncoder)
        except Exception as e:
            logging.error(f"Error saving errors to file: {e}")

    def save_results(self, test_key: str, results: Dict) -> None:
        """Save test results to file."""
        try:
            self.results[test_key] = {
                'timestamp': datetime.now().isoformat(),
                'metrics': results,
                'status': 'success',
                'model_config': {
                    'spread': self.spread,
                    'prediction_type': 'next_day',
                    'model_type': 'lstm'
                },
                'training_info': {
                    'num_samples': len(results['predictions']),
                    'mse': results['mse'],
                    'train_loss': results['train_loss'],
                    'val_loss': results['val_loss']
                }
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=4, cls=NumpyEncoder)
            logging.info(f"Results saved to {self.results_file}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            logging.error(traceback.format_exc())

    def run_tests(self):
        """
        Run focused test for LSTM model on next_day prediction.
        Logs progress, errors, and results throughout the testing process.
        """
        if self.spread != '2s10s':
            logging.warning("This test is configured for 2s10s spread only")
            return None
            
        logging.info(f"Starting focused testing for {self.spread} (LSTM only)")
        
        test_key = f"{self.spread}_next_day_lstm"
        logging.info(f"Starting test {test_key}")
        
        try:
            # Validate data first
            is_valid, issues = self.validate_data('next_day')
            if not is_valid:
                error_msg = f"Data validation failed: {', '.join(issues)}"
                self.log_error(
                    test_key,
                    error_msg,
                    ErrorCategory.DATA_VALIDATION,
                    None
                )
                return None
            
            # Train and evaluate model
            try:
                # Initialize model trainer with detailed logging
                logging.info("Initializing ModelTrainer with LSTM configuration")
                model = ModelTrainer(
                    spread=self.spread,
                    prediction_type='next_day',
                    model_type='lstm'
                )
                
                # Log model configuration
                logging.info(f"Model configuration: {model.__dict__}")
                
                # Train model with error handling
                logging.info("Starting model training")
                results = model.train()
                
                if results is None:
                    error_msg = "Model training returned no results"
                    self.log_error(
                        test_key,
                        error_msg,
                        ErrorCategory.MODEL_TRAINING,
                        None
                    )
                    return None
                
                # Validate training results
                required_metrics = ['mse', 'train_loss', 'val_loss', 'predictions', 'actuals']
                missing_metrics = [metric for metric in required_metrics if metric not in results]
                if missing_metrics:
                    error_msg = f"Missing required metrics in results: {missing_metrics}"
                    self.log_error(
                        test_key,
                        error_msg,
                        ErrorCategory.MODEL_TRAINING,
                        None
                    )
                    return None
                
                # Store results with detailed information
                self.save_results(test_key, results)
                
                # Log success metrics
                logging.info(f"Training completed successfully with MSE: {results['mse']:.4f}")
                logging.info(f"Train loss: {results['train_loss']:.4f}, Val loss: {results['val_loss']:.4f}")
                
            except Exception as e:
                error_msg = f"Error during model training: {str(e)}"
                self.log_error(
                    test_key,
                    error_msg,
                    ErrorCategory.MODEL_TRAINING,
                    traceback.format_exc()
                )
                return None
            
        except Exception as e:
            error_msg = f"Unexpected error during testing: {str(e)}"
            self.log_error(
                test_key,
                error_msg,
                ErrorCategory.UNEXPECTED,
                traceback.format_exc()
            )
            return None
        
        return self.results

def main():
    """
    Main function to run focused testing for 2s10s spread with LSTM model.
    """
    logging.info("Starting focused testing for 2s10s spread")
    
    try:
        # Initialize tester for 2s10s spread
        tester = SystematicTester(spread='2s10s')
        
        # Run tests
        results = tester.run_tests()
        
        if results is None:
            logging.error("Testing failed - invalid spread configuration")
            return
        
        logging.info("Testing completed successfully")
        
    except Exception as e:
        logging.error(f"Testing failed with error: {str(e)}")
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main() 