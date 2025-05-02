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
        self.dv01_ratios = self.config['dv01'].get('ratios', {})
    
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
    
    def calculate_dv01_ratio(self,
                           spread: str,
                           short_maturity: float,
                           long_maturity: float,
                           short_yield: float,
                           long_yield: float) -> float:
        """
        Calculate DV01 ratio between two bonds. Uses static config ratio only if non-null.
        
        Args:
            spread: The spread name (e.g., '2s10s', '5s30s')
            short_maturity: Maturity of the short leg in years
            long_maturity: Maturity of the long leg in years
            short_yield: Yield of the short leg
            long_yield: Yield of the long leg
            
        Returns:
            float: The DV01 ratio
            
        Raises:
            ValueError: If short leg DV01 is zero
        """
        cfg = self.dv01_ratios.get(spread, None)
        # Only use the config ratio if it's set to a positive number
        if cfg is not None and isinstance(cfg, (int, float)):
            return float(cfg)

        # Otherwise, compute dynamically:
        short_dv01 = self.calculate_dv01(1.0, short_yield, short_maturity)
        long_dv01 = self.calculate_dv01(1.0, long_yield, long_maturity)
        
        # Avoid division by zero
        if short_dv01 == 0:
            raise ValueError(f"Short leg DV01 is zero for {spread}. Cannot compute ratio.")
            
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
        self.position_limits = self.risk_config['position_limits']
        self.pnl_history = {spread: [] for spread in SPREADS.keys()}
        self.correlation_window = 60  # Rolling window for correlation calculation
    
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
    
    def check_concentration_limit(self, positions: Dict[str, Dict[str, Dict[str, float]]]) -> bool:
        """Check if any spread exceeds the maximum concentration limit."""
        total_dv01 = self.calculate_total_dv01(positions)
        if total_dv01 == 0:
            return True  # No positions, so no concentration issues

        # Count how many spreads actually have a non-zero position
        active_spreads = sum(
            1 for pos in positions.values()
            if abs(pos['short_term']['size']) + abs(pos['long_term']['size']) > 0
        )

        # Only enforce concentration when there are 2+ spreads on
        if active_spreads < 2:
            return True

        # Now do the usual per-spread DV01 / total_DV01 check
        max_conc = self.position_limits['max_concentration']
        for spread, pos in positions.items():
            spread_dv01 = sum(
                abs(self.dv01_calculator.calculate_dv01(
                    pos[leg]['size'],
                    pos[leg]['yield'],
                    self._get_maturity(spread, leg)
                ))
                for leg in ('short_term', 'long_term')
            )
            if spread_dv01 / total_dv01 > max_conc:
                logging.warning(f"Concentration limit exceeded for {spread}: {spread_dv01/total_dv01:.2%}")
                return False

        return True
    
    def update_pnl_history(self, spread: str, pnl: float) -> None:
        """Update PnL history for correlation calculation."""
        self.pnl_history[spread].append(pnl)
        if len(self.pnl_history[spread]) > self.correlation_window:
            self.pnl_history[spread].pop(0)
    
    def check_correlation_limit(self) -> bool:
        """Check if any spread pair exceeds the correlation threshold."""
        # Need minimum data points for correlation
        min_required = 20
        spreads = list(self.pnl_history.keys())
        
        for i in range(len(spreads)):
            for j in range(i + 1, len(spreads)):
                spread1, spread2 = spreads[i], spreads[j]
                if len(self.pnl_history[spread1]) >= min_required and len(self.pnl_history[spread2]) >= min_required:
                    corr = np.corrcoef(
                        self.pnl_history[spread1][-min_required:],
                        self.pnl_history[spread2][-min_required:]
                    )[0, 1]
                    if abs(corr) > self.position_limits['correlation_threshold']:
                        logging.warning(f"Correlation limit exceeded between {spread1} and {spread2}: {corr:.2f}")
                        return False
        return True
    
    def check_position_limits(self, positions: Dict[str, Dict[str, Dict[str, float]]]) -> bool:
        """Check if position limits are violated."""
        # Check total DV01 limit
        total_dv01 = self.calculate_total_dv01(positions)
        if total_dv01 > self.config['position']['max_position']['total_portfolio']:
            logging.warning(f"Total DV01 limit exceeded: {total_dv01}")
            return False
            
        # Check per-spread DV01 limit
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
                logging.warning(f"Per-spread DV01 limit exceeded for {spread}: {spread_dv01}")
                return False
        
        # Check concentration limit
        if not self.check_concentration_limit(positions):
            return False
            
        # Check correlation limit
        if not self.check_correlation_limit():
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
    def load_config(config_path: Union[str, Path] = None) -> Dict:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        # Get absolute path to project root if config_path is not provided
        if config_path is None:
            root_dir = Path(__file__).parent.parent
            config_path = root_dir / 'config.yaml'
            
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config

class DataProcessor:
    """Process and validate data for the trading strategy."""
    
    def __init__(self, config_path: Union[str, Path] = None):
        """Initialize with configuration."""
        # Get absolute path to project root if config_path is not provided
        if config_path is None:
            root_dir = Path(__file__).parent.parent
            config_path = root_dir / 'config.yaml'
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get absolute paths
        self.root_dir = Path(__file__).parent.parent
        self.data_dir = self.root_dir / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
    
    def load_data(self, data_type: str) -> pd.DataFrame:
        """
        Load data from CSV files.
        
        Args:
            data_type: Type of data to load ('yields', 'macro')
            
        Returns:
            DataFrame: Loaded data
        """
        try:
            if data_type == 'yields':
                file_path = self.raw_dir / 'treasury_yields.csv'
                if not file_path.exists():
                    logging.error(f"Yield data file not found: {file_path}")
                    return pd.DataFrame()
                    
                df = pd.read_csv(file_path, index_col=0)
                df.index = pd.to_datetime(df.index)
                
                # Rename columns to match expected format
                df = df.rename(columns={
                    '2-Year': '2y',
                    '5-Year': '5y',
                    '10-Year': '10y',
                    '30-Year': '30y'
                })
                
                # Validate required columns
                required_columns = ['2y', '5y', '10y', '30y']
                if not self.validate_data(df, required_columns):
                    return pd.DataFrame()
                    
                return df
                
            elif data_type == 'macro':
                file_path = self.raw_dir / 'macro_indicators.csv'
                if not file_path.exists():
                    logging.error(f"Macro data file not found: {file_path}")
                    return pd.DataFrame()
                    
                df = pd.read_csv(file_path)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
                
            else:
                logging.error(f"Unknown data type: {data_type}")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error loading {data_type} data: {str(e)}")
            return pd.DataFrame()
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Validate that DataFrame has required columns and no missing values.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            bool: True if validation passes
        """
        # Check required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logging.error(f"Missing required columns: {missing_cols}")
            return False
            
        # Check for missing values
        missing_values = df[required_columns].isnull().sum()
        if missing_values.any():
            logging.warning(f"Found missing values:\n{missing_values[missing_values > 0]}")
            
        return True

# Define spreads
SPREADS = {
    '2s10s': ('2y', '10y'),
    '5s30s': ('5y', '30y')
} 