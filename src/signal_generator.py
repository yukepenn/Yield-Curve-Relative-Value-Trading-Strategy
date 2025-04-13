"""
Signal generation and aggregation module for yield curve trading strategy.
"""

import pandas as pd
import numpy as np
import logging
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Union
import json
import yaml
import glob

# Configure logging for debug and error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: Union[str, Path] = 'config.yaml') -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)    

class SignalGenerator:
    """Generate and aggregate trading signals from multiple models."""
    
    def __init__(self, config_path: Union[str, Path] = 'config.yaml'):
        """
        Initialize SignalGenerator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        self.thresholds = self.config['signal_thresholds']
        self.ensemble_weights = self.config['ensemble']['weights']
        self.confidence_scaling = self.config['ensemble']['confidence_scaling']
        self.min_agreement = self.thresholds['ensemble']['min_agreement']
    
    def process_next_day_signal(self, prediction: float) -> Tuple[int, float]:
        """
        Process regression (next_day) model prediction.
        
        Args:
            prediction: Predicted change in spread
            
        Returns:
            tuple: (signal direction, confidence score)
                signal: 1 for steepener, -1 for flattener, 0 for neutral
                confidence: float between 0 and 1
        """
        min_change = self.thresholds['next_day']['min_change_bp']
        neutral_zone = self.thresholds['next_day']['neutral_zone']
        
        if prediction > min_change:
            signal = 1  # steepener
            confidence = min(abs(prediction) / (2 * min_change), 1.0)
        elif prediction < -min_change:
            signal = -1  # flattener
            confidence = min(abs(prediction) / (2 * min_change), 1.0)
        elif neutral_zone[0] <= prediction <= neutral_zone[1]:
            signal = 0  # neutral
            confidence = 1.0
        else:
            signal = 0  # weak signal treated as neutral
            confidence = 0.5
            
        return signal, confidence
    
    def process_direction_signal(self, prediction: int) -> Tuple[int, float]:
        """
        Process binary classification (direction) model prediction.
        
        Args:
            prediction: Binary prediction (0 for flatten, 1 for steepen)
            
        Returns:
            tuple: (signal direction, confidence score)
                signal: 1 for steepener, -1 for flattener
                confidence: fixed confidence value since these are binary predictions
        """
        # For binary predictions, we use a fixed confidence since we don't have probabilities
        BINARY_CONFIDENCE = 1.0  # Could be configured in config.yaml if needed
        
        if prediction == 1:
            signal = 1  # steepener
            confidence = BINARY_CONFIDENCE
        elif prediction == 0:
            signal = -1  # flattener
            confidence = BINARY_CONFIDENCE
        else:
            # Invalid prediction value
            logging.warning(f"Invalid binary prediction value: {prediction}, expected 0 or 1")
            signal = 0  # neutral
            confidence = 0.0
            
        return signal, confidence
    
    def process_ternary_signal(self, prediction: Union[Dict[str, float], float, int, tuple]) -> Tuple[int, float]:
        """
        Process ternary classification model prediction.
        
        Args:
            prediction: Can be:
                - Dict with class probabilities (e.g. {"steepen": 0.6, "neutral": 0.2, "flatten": 0.2})
                - Integer class index (0: flatten, 1: neutral, 2: steepen)
                - Float probability (0-1 range, where >0.5 indicates steepen, <0.5 indicates flatten)
                - Tuple (in which case the first element is used)
            
        Returns:
            Tuple[int, float]: (signal direction, confidence score)
                signal: 1 for steepener, -1 for flattener, 0 for neutral
                confidence: float between 0 and 1
        """
        try:
            # Handle tuple input by taking the first element
            if isinstance(prediction, tuple):
                prediction = prediction[0]
            
            # First try to convert to numeric if it's a string
            if isinstance(prediction, str):
                try:
                    prediction = ast.literal_eval(prediction)
                except (ValueError, SyntaxError):
                    prediction = pd.to_numeric(prediction, errors='coerce')
            
            if isinstance(prediction, dict):
                # Class probability dictionary
                max_class = max(prediction.items(), key=lambda x: x[1])[0]
                max_prob = prediction[max_class]
                
                if max_prob < self.thresholds['ternary']['probability']:
                    logging.debug(f"Low probability {max_prob} for class {max_class}, treating as neutral")
                    return 0, max_prob

                signal_map = {'steepen': 1, 'neutral': 0, 'flatten': -1}
                signal = signal_map.get(max_class, 0)
                confidence = max_prob

            elif isinstance(prediction, (int, float, np.integer, np.floating)):
                # Handle numeric predictions
                pred_value = pd.to_numeric(prediction, errors='coerce')
                if pd.isna(pred_value):
                    logging.warning(f"Invalid prediction value: {prediction}")
                    return 0, 0.0
                
                # If prediction is between 0 and 1, treat it as a probability
                if 0 <= pred_value <= 1:
                    if pred_value > 0.5:
                        signal = 1  # steepen
                        confidence = pred_value
                    elif pred_value < 0.5:
                        signal = -1  # flatten
                        confidence = 1 - pred_value
                    else:
                        signal = 0  # neutral
                        confidence = 0.5
                else:
                    # Treat as class index
                    pred_class = int(round(pred_value))
                    if pred_class not in [0, 1, 2]:
                        logging.warning(f"Invalid class index {pred_class} in ternary prediction")
                        return 0, 0.0
                    
                    # Map class indices to signals
                    signal_map = {2: 1, 1: 0, 0: -1}
                    signal = signal_map.get(pred_class, 0)
                    
                    # For numeric predictions, we don't have probabilities
                    # Use a default confidence based on the prediction type
                    if pred_class == 1:  # neutral
                        confidence = 0.8  # Slightly lower confidence for neutral predictions
                    else:
                        confidence = 1.0  # High confidence for directional predictions

            else:
                logging.warning(f"Unsupported prediction type: {type(prediction)}")
                return 0, 0.0

            logging.debug(f"Processed ternary prediction: {prediction} -> signal={signal}, confidence={confidence}")
            return signal, confidence
            
        except Exception as e:
            logging.error(f"Error processing ternary prediction {prediction}: {str(e)}")
            return 0, 0.0
    
    def aggregate_signals(self, predictions: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Tuple[int, float]]:
        """
        Aggregate signals from multiple models into a final trading decision.
        
        Args:
            predictions: Dictionary containing predictions from each model type and model
                {
                    'next_day': {
                        'arima': {date: prediction},
                        'lstm': {date: prediction},
                        ...
                    },
                    'direction': {
                        'xgb': {date: prediction},
                        'rf': {date: prediction},
                        ...
                    },
                    'ternary': {
                        'xgb': {date: prediction},
                        'rf': {date: prediction},
                        ...
                    }
                }
            
        Returns:
            dict: Date-indexed results with (final signal, confidence score)
        """
        # Add debug logging
        logging.debug(f"Received predictions structure: {type(predictions)}")
        if isinstance(predictions, dict):
            for pred_type, model_preds in predictions.items():
                logging.debug(f"Prediction type: {pred_type}, Type: {type(model_preds)}")
                if isinstance(model_preds, dict):
                    for model, pred_dict in model_preds.items():
                        logging.debug(f"Model: {model}, Type: {type(pred_dict)}")
                        if isinstance(pred_dict, dict):
                            logging.debug(f"First prediction value: {next(iter(pred_dict.values())) if pred_dict else 'Empty'}")
        
        signals = {}
        confidences = {}
        
        try:
            # Process each prediction type
            for pred_type, model_preds in predictions.items():
                if not isinstance(model_preds, dict):
                    logging.warning(f"Invalid model predictions format for {pred_type}")
                    continue
                    
                if pred_type not in signals:
                    signals[pred_type] = {}
                    confidences[pred_type] = {}
                
                # Process each model's prediction
                for model, pred_dict in model_preds.items():
                    if not isinstance(pred_dict, dict):
                        logging.warning(f"Invalid prediction format for {model} in {pred_type}")
                        continue
                        
                    for date, pred in pred_dict.items():
                        try:
                            if pred_type == 'next_day':
                                signal, confidence = self.process_next_day_signal(pred)
                            elif pred_type == 'direction':
                                signal, confidence = self.process_direction_signal(pred)
                            else:  # ternary
                                signal, confidence = self.process_ternary_signal(pred)
                            
                            if date not in signals[pred_type]:
                                signals[pred_type][date] = {}
                                confidences[pred_type][date] = {}
                            
                            signals[pred_type][date][model] = signal
                            confidences[pred_type][date][model] = confidence
                            
                        except Exception as e:
                            logging.error(f"Error processing prediction for {date} from {model}: {str(e)}")
                            continue
            
            # Calculate weighted signal for each date
            final_signals = {}
            final_confidences = {}
            
            for pred_type, date_signals in signals.items():
                for date, model_signals in date_signals.items():
                    if date not in final_signals:
                        final_signals[date] = []
                        final_confidences[date] = []
                    
                    weight = self.ensemble_weights.get(pred_type, 1.0)
                    for model, signal in model_signals.items():
                        confidence = confidences[pred_type][date][model]
                        
                        if self.confidence_scaling:
                            final_signals[date].append(signal * weight * confidence)
                            final_confidences[date].append(weight * confidence)
                        else:
                            final_signals[date].append(signal * weight)
                            final_confidences[date].append(weight)
            
            # Compute final signals for each date
            results = {}
            for date in final_signals:
                total_weight = sum(final_confidences[date])
                if total_weight > 0:
                    ensemble_signal = sum(final_signals[date]) / total_weight
                else:
                    ensemble_signal = 0
                
                # Convert to discrete signal
                if ensemble_signal > 0.5 / self.min_agreement:
                    final_signal = 1
                elif ensemble_signal < -0.5 / self.min_agreement:
                    final_signal = -1
                else:
                    final_signal = 0
                
                results[date] = (final_signal, total_weight)
            
            return results
            
        except Exception as e:
            logging.error(f"Error in aggregate_signals: {str(e)}")
            return {}
    
    def generate_signals(self, spread: str, date: pd.Timestamp,
                        predictions: Dict[str, Dict[str, Dict[str, float]]]) -> Dict:
        """
        Generate final trading signal for a spread on a given date.
        
        Args:
            spread: Name of the spread (e.g., '2s10s', '5s30s')
            date: Trading date
            predictions: Dictionary of model predictions by type and model
            
        Returns:
            dict: Trading signal information
        """
        # Process individual model signals
        model_signals = {}
        model_confidences = {}
        
        # Process each prediction type
        for pred_type, model_preds in predictions.items():
            if pred_type not in model_signals:
                model_signals[pred_type] = {}
                model_confidences[pred_type] = {}
            
            # Process each model's prediction
            for model, pred_dict in model_preds.items():
                # Get prediction for this date
                date_str = date.strftime('%Y-%m-%d')
                if date_str not in pred_dict:
                    logging.warning(f"No prediction found for {model} ({pred_type}) on {date_str}")
                    continue
                    
                pred = pred_dict[date_str]
                
                try:
                    if pred_type == 'next_day':
                        signal, confidence = self.process_next_day_signal(pred)
                    elif pred_type == 'direction':
                        signal, confidence = self.process_direction_signal(pred)
                    else:  # ternary
                        signal, confidence = self.process_ternary_signal(pred)
                    
                    model_signals[pred_type][model] = signal
                    model_confidences[pred_type][model] = confidence
                except Exception as e:
                    logging.error(f"Error processing {model} ({pred_type}) prediction: {str(e)}")
                    continue
        
        # Aggregate signals
        try:
            results = self.aggregate_signals(predictions)
            date_str = date.strftime('%Y-%m-%d')
            if date_str in results:
                final_signal, final_confidence = results[date_str]
            else:
                logging.warning(f"No aggregated signal found for {date_str}")
                final_signal = 0
                final_confidence = 0.0
        except Exception as e:
            logging.error(f"Error aggregating signals: {str(e)}")
            final_signal = 0
            final_confidence = 0.0
        
        return {
            'spread': spread,
            'date': date,
            'signal': final_signal,
            'confidence': final_confidence,
            'model_signals': model_signals,
            'model_confidences': model_confidences
        }

def load_predictions(prediction_dir: Path) -> dict:
    """Load predictions from model-specific CSV files in directory structure."""
    predictions = {
        'next_day': {},
        'direction': {},
        'ternary': {}
    }
    
    # Load predictions for each spread and prediction type
    for pred_type in ['next_day', 'direction', 'ternary']:
        pred_dirs = glob.glob(str(prediction_dir / f'*_{pred_type}'))
        for pred_dir in pred_dirs:
            spread = Path(pred_dir).name.split('_')[0]
            
            # Find all prediction CSV files
            pred_files = glob.glob(str(Path(pred_dir) / '*_predictions.csv'))
            
            for pred_file in pred_files:
                try:
                    model_name = Path(pred_file).name.split('_')[0]  # Extract model name from filename
                    df = pd.read_csv(pred_file)
                    
                    if 'date' in df.columns and 'prediction' in df.columns:
                        if spread not in predictions[pred_type]:
                            predictions[pred_type][spread] = {}
                            
                        # Convert predictions to the expected format
                        if pred_type == 'next_day':
                            # Next day predictions are already in the right format (float)
                            pred_dict = dict(zip(df['date'], df['prediction']))
                        elif pred_type == 'direction':
                            # Direction predictions are binary (0 or 1), keep as is
                            pred_dict = dict(zip(df['date'], df['prediction'].astype(int)))
                        else:  # ternary
                            # For ternary, predictions should already be class indices (0: flatten, 1: neutral, 2: steepen)
                            pred_dict = dict(zip(df['date'], df['prediction'].astype(int)))
                            
                        predictions[pred_type][spread][model_name] = pred_dict
                        logging.info(f"Loaded {len(pred_dict)} predictions for {spread} {pred_type} - {model_name}")
                except Exception as e:
                    logging.error(f"Error loading predictions from {pred_file}: {str(e)}")
    
    return predictions

def save_signals(signals: List[Dict], spread: str, output_dir: Path = Path('results/signals')):
    """Save generated signals to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{spread}_signals.json"
    
    # Convert datetime to string for JSON serialization
    signals_json = []
    for signal in signals:
        signal_dict = signal.copy()
        signal_dict['date'] = signal_dict['date'].strftime('%Y-%m-%d')
        signals_json.append(signal_dict)
    
    with open(output_file, 'w') as f:
        json.dump(signals_json, f, indent=4)
    
    logging.info(f"Saved signals for {spread} to {output_file}")

def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize signal generator
    generator = SignalGenerator()
    
    # Process both spreads
    spreads = ['2s10s', '5s30s']
    for spread in spreads:
        logging.info(f"Processing signals for {spread} spread")
        
        # Load predictions
        predictions = load_predictions(Path('results/model_training'))  # Fixed path
        if not predictions:
            logging.warning(f"No predictions found for {spread}")
            continue
        
        # Generate signals for each date
        signals = []
        
        # Get all unique dates from all prediction types and models
        all_dates = set()
        for pred_type in predictions:
            if spread in predictions[pred_type]:
                for model in predictions[pred_type][spread]:
                    all_dates.update(predictions[pred_type][spread][model].keys())
        
        # Sort dates
        dates = sorted(list(all_dates))
        logging.info(f"Found {len(dates)} unique dates for {spread}")
        
        for date_str in dates:
            try:
                date = pd.to_datetime(date_str)
                
                # Get predictions for this date from all models
                date_predictions = {}
                for pred_type in predictions:
                    if spread in predictions[pred_type]:
                        date_predictions[pred_type] = {}
                        for model in predictions[pred_type][spread]:
                            if date_str in predictions[pred_type][spread][model]:
                                date_predictions[pred_type][model] = {
                                    date_str: predictions[pred_type][spread][model][date_str]
                                }
                
                # Check if we have any valid predictions
                has_predictions = False
                for pred_type in date_predictions:
                    if date_predictions[pred_type]:  # Check if any models have predictions
                        has_predictions = True
                        break
                
                if not has_predictions:
                    logging.warning(f"No valid predictions found for {date_str}")
                    continue
                    
                signal = generator.generate_signals(spread, date, date_predictions)
                signals.append(signal)
                
            except Exception as e:
                logging.error(f"Error processing date {date_str}: {str(e)}")
                continue
        
        # Save signals
        if signals:
            save_signals(signals, spread)
            logging.info(f"Generated {len(signals)} signals for {spread}")
            
            # Print summary
            steepener_count = sum(1 for s in signals if s['signal'] == 1)
            flattener_count = sum(1 for s in signals if s['signal'] == -1)
            neutral_count = sum(1 for s in signals if s['signal'] == 0)
            
            logging.info(f"Signal distribution for {spread}:")
            logging.info(f"  Steepener signals: {steepener_count}")
            logging.info(f"  Flattener signals: {flattener_count}")
            logging.info(f"  Neutral signals: {neutral_count}")
        else:
            logging.warning(f"No signals generated for {spread}")

if __name__ == '__main__':
    main() 