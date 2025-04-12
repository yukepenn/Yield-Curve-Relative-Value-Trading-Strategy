"""
Signal generation and aggregation module for yield curve trading strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from pathlib import Path
import logging
from .utils import ConfigLoader

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
    
    def process_direction_signal(self, probability: float) -> Tuple[int, float]:
        """
        Process binary classification (direction) model prediction.
        
        Args:
            probability: Probability of steepener (assumes binary: steepen vs flatten)
            
        Returns:
            tuple: (signal direction, confidence score)
        """
        threshold = self.thresholds['direction']['probability']
        
        if probability > threshold:
            signal = 1  # steepener
            confidence = probability
        elif probability < (1 - threshold):
            signal = -1  # flattener
            confidence = 1 - probability
        else:
            signal = 0  # neutral
            confidence = 0.5
            
        return signal, confidence
    
    def process_ternary_signal(self, probabilities: Dict[str, float]) -> Tuple[int, float]:
        """
        Process ternary classification model prediction.
        
        Args:
            probabilities: Dictionary with probabilities for each class
                {'steepen': p1, 'neutral': p2, 'flatten': p3}
            
        Returns:
            tuple: (signal direction, confidence score)
        """
        threshold = self.thresholds['ternary']['probability']
        
        max_prob = max(probabilities.values())
        max_class = max(probabilities, key=probabilities.get)
        
        if max_prob > threshold:
            if max_class == 'steepen':
                signal = 1
            elif max_class == 'flatten':
                signal = -1
            else:
                signal = 0
            confidence = max_prob
        else:
            signal = 0
            confidence = max_prob
            
        return signal, confidence
    
    def aggregate_signals(self, predictions: Dict[str, Union[float, Dict[str, float]]]) -> Tuple[int, float]:
        """
        Aggregate signals from multiple models into a final trading decision.
        
        Args:
            predictions: Dictionary containing predictions from each model type
                {
                    'next_day': float,  # Predicted change in bps
                    'direction': float,  # Probability of steepener
                    'ternary': {'steepen': p1, 'neutral': p2, 'flatten': p3}
                }
            
        Returns:
            tuple: (final signal, confidence score)
                signal: 1 for steepener, -1 for flattener, 0 for neutral
                confidence: Weighted average confidence
        """
        signals = {}
        confidences = {}
        
        # Process each model type
        if 'next_day' in predictions:
            signals['next_day'], confidences['next_day'] = self.process_next_day_signal(
                predictions['next_day'])
            
        if 'direction' in predictions:
            signals['direction'], confidences['direction'] = self.process_direction_signal(
                predictions['direction'])
            
        if 'ternary' in predictions:
            signals['ternary'], confidences['ternary'] = self.process_ternary_signal(
                predictions['ternary'])
        
        # Calculate weighted signal
        weighted_signals = []
        weighted_confidences = []
        
        for model_type in signals:
            weight = self.ensemble_weights[model_type]
            signal = signals[model_type]
            confidence = confidences[model_type]
            
            if self.confidence_scaling:
                weighted_signals.append(signal * weight * confidence)
                weighted_confidences.append(weight * confidence)
            else:
                weighted_signals.append(signal * weight)
                weighted_confidences.append(weight)
        
        # Compute final signal
        total_weight = sum(weighted_confidences)
        if total_weight > 0:
            ensemble_signal = sum(weighted_signals) / total_weight
        else:
            ensemble_signal = 0
        
        # Convert to discrete signal
        if ensemble_signal > 0.5 / self.min_agreement:
            final_signal = 1
        elif ensemble_signal < -0.5 / self.min_agreement:
            final_signal = -1
        else:
            final_signal = 0
        
        # Calculate final confidence
        final_confidence = abs(ensemble_signal) * 2  # Scale to [0,1]
        final_confidence = min(final_confidence, 1.0)
        
        return final_signal, final_confidence
    
    def generate_signals(self, spread: str, date: pd.Timestamp,
                        predictions: Dict[str, Union[float, Dict[str, float]]]) -> Dict:
        """
        Generate final trading signal for a spread on a given date.
        
        Args:
            spread: Name of the spread (e.g., '2s10s', '5s30s')
            date: Trading date
            predictions: Dictionary of model predictions
            
        Returns:
            dict: Trading signal information
                {
                    'spread': str,
                    'date': pd.Timestamp,
                    'signal': int,  # 1 (steepener), -1 (flattener), 0 (neutral)
                    'confidence': float,
                    'model_signals': dict,  # Individual model signals
                    'model_confidences': dict  # Individual model confidences
                }
        """
        # Process individual model signals
        model_signals = {}
        model_confidences = {}
        
        if 'next_day' in predictions:
            model_signals['next_day'], model_confidences['next_day'] = self.process_next_day_signal(
                predictions['next_day'])
            
        if 'direction' in predictions:
            model_signals['direction'], model_confidences['direction'] = self.process_direction_signal(
                predictions['direction'])
            
        if 'ternary' in predictions:
            model_signals['ternary'], model_confidences['ternary'] = self.process_ternary_signal(
                predictions['ternary'])
        
        # Aggregate signals
        final_signal, final_confidence = self.aggregate_signals(predictions)
        
        return {
            'spread': spread,
            'date': date,
            'signal': final_signal,
            'confidence': final_confidence,
            'model_signals': model_signals,
            'model_confidences': model_confidences
        } 