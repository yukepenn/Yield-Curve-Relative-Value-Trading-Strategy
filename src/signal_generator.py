"""
Signal generation and aggregation module for yield curve trading strategy.
"""

import pandas as pd
import numpy as np
import logging
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional, Any   
import json
import yaml
import glob
from dataclasses import dataclass
from enum import Enum
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils import (
    DurationCalculator,
    DV01Calculator,
    RiskMetricsCalculator,
    ConfigLoader,
    DataProcessor,
    SPREADS,
    SignalProcessor
)

# Configure logging for debug and error tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SignalType(Enum):
    """Enum for signal types to ensure type safety."""
    STEEPER = 1
    NEUTRAL = 0
    FLATTENER = -1

@dataclass
class Signal:
    """Data class to hold signal information."""
    value: SignalType
    confidence: float
    model_type: str
    model_name: str
    timestamp: pd.Timestamp

class EnsembleSignal:
    """Class to handle signal aggregation with weights and agreement thresholds."""
    
    def __init__(self, weights: Dict[str, float], min_agreement: int, confidence_scaling: bool, neutral_threshold: float):
        """
        Initialize EnsembleSignal.
        
        Args:
            weights: Dictionary of model weights
            min_agreement: Minimum number of models that must agree
            confidence_scaling: Whether to scale votes by confidence
            neutral_threshold: Minimum vote required for directional signal
        """
        self.weights = weights
        self.min_agreement = min_agreement
        self.confidence_scaling = confidence_scaling
        self.neutral_threshold = neutral_threshold
        self.signals: List[Signal] = []
    
    def add_signal(self, signal: Signal) -> None:
        """Add a signal to the ensemble."""
        self.signals.append(signal)
    
    def get_agreement_count(self, signal_type: SignalType) -> int:
        """Count how many models agree on a particular signal type."""
        return sum(1 for s in self.signals if s.value == signal_type)
    
    def get_weighted_vote(self) -> Tuple[SignalType, float]:
        """
        Calculate weighted vote considering both model weights and confidence.
        
        Returns:
            Tuple[SignalType, float]: (final signal, confidence)
        """
        if not self.signals:
            return SignalType.NEUTRAL, 0.0
            
        # Calculate weighted votes for each signal type
        votes = {SignalType.STEEPER: 0.0, SignalType.NEUTRAL: 0.0, SignalType.FLATTENER: 0.0}
        total_weight = 0.0
        
        for signal in self.signals:
            weight = self.weights.get(signal.model_type, 1.0)
            confidence = signal.confidence if self.confidence_scaling else 1.0
            votes[signal.value] += weight * confidence
            total_weight += weight
            
        if total_weight == 0:
            return SignalType.NEUTRAL, 0.0
            
        # Normalize votes
        for signal_type in votes:
            votes[signal_type] /= total_weight
            
        # Find the signal with highest vote
        max_signal = max(votes.items(), key=lambda x: x[1])[0]
        max_vote = votes[max_signal]
        
        # Check agreement threshold and neutral threshold
        if max_signal != SignalType.NEUTRAL:
            agreement_count = self.get_agreement_count(max_signal)
            if agreement_count < self.min_agreement or max_vote < self.neutral_threshold:
                return SignalType.NEUTRAL, max_vote
                
        return max_signal, max_vote

class SignalGenerator:
    """Generate and aggregate trading signals from multiple models."""
    
    def __init__(self, config_path: Union[str, Path] = 'config.yaml'):
        """
        Initialize SignalGenerator.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader.load_config(config_path)
        self.thresholds = self.config['signal_thresholds']
        self.ensemble_weights = self.config['ensemble']['weights']
        self.confidence_scaling = self.config['ensemble']['confidence_scaling']
        self.min_agreement = self.thresholds['ensemble']['min_agreement']
        self.neutral_threshold = self.config['ensemble']['neutral_threshold']
    
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
            
        Returns:
            dict: Date-indexed results with (final signal, confidence score)
        """
        results = {}
        
        try:
            # Get all unique dates
            dates = self._get_all_dates(predictions)
            
            for date in dates:
                ensemble = EnsembleSignal(
                    weights=self.ensemble_weights,
                    min_agreement=self.min_agreement,
                    confidence_scaling=self.confidence_scaling,
                    neutral_threshold=self.neutral_threshold
                )
                
                # Process each prediction type
                for pred_type, model_preds in predictions.items():
                    if not isinstance(model_preds, dict):
                        continue
                        
                    # Process each model's prediction
                    for model, pred_dict in model_preds.items():
                        if not isinstance(pred_dict, dict) or date not in pred_dict:
                            continue
                            
                        pred = pred_dict[date]
                        try:
                            if pred_type == 'next_day':
                                signal, confidence = self.process_next_day_signal(pred)
                            elif pred_type == 'direction':
                                signal, confidence = self.process_direction_signal(pred)
                            else:  # ternary
                                signal, confidence = self.process_ternary_signal(pred)
                                
                            # Create Signal object
                            signal_obj = Signal(
                                value=SignalType(signal),
                                confidence=confidence,
                                model_type=pred_type,
                                model_name=model,
                                timestamp=pd.Timestamp(date)
                            )
                            
                            ensemble.add_signal(signal_obj)
                            
                        except Exception as e:
                            logging.error(f"Error processing prediction for {model} on {date}: {str(e)}")
                            continue
                
                # Get final signal from ensemble
                final_signal, final_confidence = ensemble.get_weighted_vote()
                results[date] = (final_signal.value, final_confidence)
                
        except Exception as e:
            logging.error(f"Error in aggregate_signals: {str(e)}")
            return {}
        
        return results
    
    def _get_all_dates(self, predictions: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
        """Get all unique dates from predictions."""
        dates = set()
        for model_preds in predictions.values():
            for pred_dict in model_preds.values():
                dates.update(pred_dict.keys())
        return sorted(list(dates))
    
    def generate_signals(self, spread: str, date: pd.Timestamp,
                     predictions: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[str, Any]:
        """
        Generate trading signals by combining model predictions.
        
        Args:
            spread: Name of the spread
            date: Current date
            predictions: Dictionary containing predictions from each model type and model
            
        Returns:
            dict: Dictionary containing signal type and confidence
        """
        try:
            # Get thresholds from config
            next_day_threshold = self.config['signal_thresholds']['next_day']['min_change_bp']
            direction_threshold = self.config['signal_thresholds']['direction']['probability']
            ternary_threshold = self.config['signal_thresholds']['ternary']['probability']
            neutral_zone = self.config['signal_thresholds']['next_day']['neutral_zone']
            min_agreement = self.config['signal_thresholds']['ensemble']['min_agreement']
            
            # Initialize vote counters
            steepener_votes = 0
            flattener_votes = 0
            
            # Process next_day predictions
            if 'next_day' in predictions and predictions['next_day']:
                for model, preds in predictions['next_day'].items():
                    for date_str, pred in preds.items():
                        if pred > next_day_threshold:
                            steepener_votes += 1
                        elif pred < -next_day_threshold:
                            flattener_votes += 1
            
            # Process direction predictions
            if 'direction' in predictions and predictions['direction']:
                for model, preds in predictions['direction'].items():
                    for date_str, pred in preds.items():
                        if pred == 1:
                            steepener_votes += 1
                        elif pred == 0:
                            flattener_votes += 1
            
            # Process ternary predictions
            if 'ternary' in predictions and predictions['ternary']:
                for model, preds in predictions['ternary'].items():
                    for date_str, pred in preds.items():
                        if pred == 2:  # Steepener
                            steepener_votes += 1
                        elif pred == 0:  # Flattener
                            flattener_votes += 1
            
            # Determine final signal and confidence
            if max(steepener_votes, flattener_votes) >= min_agreement:
                if steepener_votes > flattener_votes:
                    signal_type = SignalType.STEEPER
                    confidence = steepener_votes / 3.0  # Normalize to [0,1]
                else:
                    signal_type = SignalType.FLATTENER
                    confidence = flattener_votes / 3.0  # Normalize to [0,1]
            else:
                signal_type = SignalType.NEUTRAL
                confidence = 0.0
            
            # Return simplified signal format
            return {
                'date': date,
                'signal': signal_type.value,  # Store enum value for JSON serialization
                'confidence': float(confidence)
            }
            
        except Exception as e:
            logging.error(f"Error generating signals: {str(e)}")
            return {
                'date': date,
                'signal': SignalType.NEUTRAL.value,
                'confidence': 0.0
            }  # Safe default

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
    
    # Get absolute path to project root
    root_dir = Path(__file__).parent.parent
    
    # Initialize signal generator with absolute path to config
    generator = SignalGenerator(root_dir / 'config.yaml')
    
    # Process both spreads
    spreads = ['2s10s', '5s30s']
    for spread in spreads:
        logging.info(f"Processing signals for {spread} spread")
        
        # Load predictions using absolute path
        predictions = load_predictions(root_dir / 'results' / 'model_training')
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
            save_signals(signals, spread, root_dir / 'results' / 'signals')
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