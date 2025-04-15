"""
Utility classes for yield curve trading strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import yaml
import logging
from dataclasses import dataclass
from enum import Enum

class SignalType(Enum):
    """Enum for signal types."""
    STEEPER = 1
    NEUTRAL = 0
    FLATTENER = -1

@dataclass
class Signal:
    """Data class for signal information."""
    value: SignalType
    confidence: float
    model_type: str
    model_name: str
    timestamp: pd.Timestamp

class SignalProcessor:
    """Utility class for processing signals."""
    
    def __init__(self, config_path: Union[str, Path] = 'config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.thresholds = self.config['signal_thresholds']
        self.ensemble_config = self.config['ensemble']
    
    @staticmethod
    def validate_signal(signal: Union[int, float, Dict]) -> Tuple[SignalType, float]:
        """
        Validate and convert a signal to standard format.
        
        Args:
            signal: Input signal (int, float, or dict)
            
        Returns:
            tuple: (SignalType, confidence)
        """
        if isinstance(signal, dict):
            value = signal.get('signal', 0)
            confidence = signal.get('confidence', 0.0)
        else:
            value = signal
            confidence = 1.0
            
        # Convert to SignalType
        try:
            signal_type = SignalType(value)
        except ValueError:
            logging.warning(f"Invalid signal value: {value}")
            signal_type = SignalType.NEUTRAL
            
        # Validate confidence
        if not 0 <= confidence <= 1:
            logging.warning(f"Invalid confidence value: {confidence}")
            confidence = 0.0
            
        return signal_type, confidence
    
    def smooth_signals(self, signals: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Apply rolling window smoothing to signals.
        
        Args:
            signals: DataFrame with signals
            window: Rolling window size
            
        Returns:
            DataFrame: Smoothed signals
        """
        smoothed = signals.copy()
        smoothed['signal'] = signals['signal'].rolling(window, min_periods=1).mean()
        smoothed['confidence'] = signals['confidence'].rolling(window, min_periods=1).mean()
        return smoothed
    
    def aggregate_signals(self, signals: List[Signal]) -> Signal:
        """
        Aggregate multiple signals into a single signal.
        
        Args:
            signals: List of signals to aggregate
            
        Returns:
            Signal: Aggregated signal
        """
        if not signals:
            return Signal(SignalType.NEUTRAL, 0.0, 'ensemble', 'none', pd.Timestamp.now())
            
        # Get weights from config
        weights = self.ensemble_config['weights']
        min_agreement = self.thresholds['ensemble']['min_agreement']
        neutral_threshold = self.ensemble_config['neutral_threshold']
        confidence_scaling = self.ensemble_config['confidence_scaling']
        
        # Count weighted votes for each signal type
        votes = {SignalType.STEEPER: 0.0, SignalType.NEUTRAL: 0.0, SignalType.FLATTENER: 0.0}
        total_weight = 0.0
        
        for signal in signals:
            weight = weights.get(signal.model_type, 1.0)
            confidence = signal.confidence if confidence_scaling else 1.0
            votes[signal.value] += weight * confidence
            total_weight += weight
            
        if total_weight == 0:
            return Signal(SignalType.NEUTRAL, 0.0, 'ensemble', 'none', pd.Timestamp.now())
            
        # Normalize votes
        for signal_type in votes:
            votes[signal_type] /= total_weight
            
        # Find the signal with highest vote
        max_signal = max(votes.items(), key=lambda x: x[1])[0]
        max_vote = votes[max_signal]
        
        # Check agreement threshold and neutral threshold
        if max_signal != SignalType.NEUTRAL:
            agreement_count = sum(1 for s in signals if s.value == max_signal)
            if agreement_count < min_agreement or max_vote < neutral_threshold:
                return Signal(SignalType.NEUTRAL, max_vote, 'ensemble', 'none', pd.Timestamp.now())
                
        return Signal(max_signal, max_vote, 'ensemble', 'none', pd.Timestamp.now())

class DurationCalculator:
    """Calculate bond durations and convexity."""
    
    def calculate_modified_duration(self, maturity: float, yield_value: float, 
                                 coupon_rate: float = 0.0) -> float:
        """
        Calculate modified duration for a bond.
        
        Args:
            maturity: Time to maturity in years
            yield_value: Current yield
            coupon_rate: Annual coupon rate (default: 0 for zero-coupon)
            
        Returns:
            float: Modified duration
        """
        if coupon_rate == 0:  # Zero-coupon bond
            return maturity / (1 + yield_value)
            
        # For coupon bonds
        periods = int(maturity * 2)  # Semi-annual periods
        cash_flows = np.array([coupon_rate/2] * periods)
        cash_flows[-1] += 1  # Add principal
        times = np.arange(1, periods + 1) / 2
        discount_factors = 1 / (1 + yield_value/2) ** times
        macaulay_duration = np.sum(times * cash_flows * discount_factors) / np.sum(cash_flows * discount_factors)
        return macaulay_duration / (1 + yield_value/2)

class DV01Calculator:
    """Calculate DV01 (dollar value of 1 basis point) for positions."""
    
    def __init__(self, config_path: Union[str, Path] = 'config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.duration_calculator = DurationCalculator()
        self.dv01_target = self.config['dv01']['target']
        self.dv01_ratios = self.config['dv01']['ratios']
    
    def calculate_dv01(self, position_size: float, yield_value: float, 
                      maturity: float, coupon_rate: float = 0.0) -> float:
        """
        Calculate DV01 for a position.
        
        Args:
            position_size: Size of the position in notional terms
            yield_value: Current yield of the bond
            maturity: Time to maturity in years
            coupon_rate: Annual coupon rate (default: 0 for zero-coupon)
            
        Returns:
            float: DV01 value
        """
        modified_duration = self.duration_calculator.calculate_modified_duration(
            maturity, yield_value, coupon_rate)
        return position_size * modified_duration * 0.0001
    
    def calculate_dv01_ratio(self, spread: str, short_maturity: float, long_maturity: float,
                           short_yield: float, long_yield: float) -> float:
        """
        Calculate DV01 ratio between two bonds for spread trading.
        
        Args:
            spread: Name of the spread (e.g., '2s10s')
            short_maturity: Maturity of short bond in years
            long_maturity: Maturity of long bond in years
            short_yield: Yield of short bond
            long_yield: Yield of long bond
            
        Returns:
            float: DV01 ratio (long/short)
        """
        # Use fixed ratio from config if available
        if spread in self.dv01_ratios:
            return self.dv01_ratios[spread]
            
        # Otherwise calculate dynamically
        short_dv01 = self.calculate_dv01(1.0, short_yield, short_maturity)
        long_dv01 = self.calculate_dv01(1.0, long_yield, long_maturity)
        return long_dv01 / short_dv01

class RiskMetricsCalculator:
    """Calculate various risk metrics."""
    
    def __init__(self, config_path: Union[str, Path] = 'config.yaml'):
        """Initialize with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.dv01_calculator = DV01Calculator(config_path)
        self.risk_config = self.config['risk']
        self.portfolio_config = self.config['portfolio']
    
    def calculate_total_dv01(self, positions: Dict[str, Dict[str, Dict[str, float]]]) -> float:
        """Calculate total portfolio DV01."""
        total_dv01 = 0.0
        for spread, position in positions.items():
            for leg in ['short_term', 'long_term']:
                if position[leg]['size'] != 0:
                    total_dv01 += abs(self.dv01_calculator.calculate_dv01(
                        position[leg]['size'],
                        position[leg]['yield'],
                        self._get_maturity(spread, leg)
                    ))
        return total_dv01
    
    def calculate_var(self, returns: pd.Series, confidence: float = None) -> float:
        """Calculate Value at Risk."""
        if confidence is None:
            confidence = self.risk_config['var_confidence']
        return -np.percentile(returns, (1 - confidence) * 100)
    
    def check_position_limits(self, positions: Dict[str, Dict[str, Dict[str, float]]]) -> bool:
        """Check if position limits are violated."""
        total_dv01 = self.calculate_total_dv01(positions)
        if total_dv01 > self.config['position']['max_position']['total_portfolio']:
            return False
            
        for spread, position in positions.items():
            spread_dv01 = 0
            for leg in ['short_term', 'long_term']:
                if position[leg]['size'] != 0:
                    spread_dv01 += abs(self.dv01_calculator.calculate_dv01(
                        position[leg]['size'],
                        position[leg]['yield'],
                        self._get_maturity(spread, leg)
                    ))
            if spread_dv01 > self.config['position']['max_position']['dv01_per_spread']:
                return False
                
        return True
    
    def _get_maturity(self, spread: str, leg: str) -> float:
        """Get maturity for a spread leg."""
        if spread == '2s10s':
            return 2 if leg == 'short_term' else 10
        elif spread == '5s30s':
            return 5 if leg == 'short_term' else 30
        else:
            raise ValueError(f"Unsupported spread: {spread}")

class ConfigLoader:
    """Load and validate configuration."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path] = 'config.yaml') -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config

class DataProcessor:
    """Process and validate input data."""
    
    def __init__(self, config_path: Union[str, Path] = 'config.yaml'):
        """
        Initialize DataProcessor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader.load_config(config_path)
        
    def load_data(self, data_type: str) -> pd.DataFrame:
        """
        Load data from specified path.
        
        Args:
            data_type: Type of data to load ('yields', 'features', 'targets')
            
        Returns:
            DataFrame: Loaded data
        """
        if data_type == 'yields':
            data_path = Path('data/raw/treasury_yields.csv')
        elif data_type == 'features':
            data_path = Path('data/processed/features.csv')
        elif data_type == 'targets':
            data_path = Path('data/processed/targets.csv')
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Load data with appropriate settings
        if data_type == 'yields':
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            data = pd.read_csv(data_path)
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
        
        return data
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate data has required columns and no missing values."""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        if df.isnull().any().any():
            raise ValueError("Data contains missing values")
        return True

# Define spreads
SPREADS = {
    '2s10s': ('2-Year', '10-Year'),
    '5s30s': ('5-Year', '30-Year')
} 