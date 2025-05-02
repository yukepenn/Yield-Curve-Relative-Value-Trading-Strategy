"""
Trade execution simulation module for yield curve trading strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json
import sys
import copy

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
from src.signal_generator import SignalGenerator, SignalType

class BacktestEngine:
    """Simulate trading based on model signals."""
    
    def __init__(self, config_path: Union[str, Path] = None):
        """
        Initialize BacktestEngine.
        
        Args:
            config_path: Path to configuration file
        """
        # Get absolute path to project root if config_path is not provided
        if config_path is None:
            root_dir = Path(__file__).parent.parent
            config_path = root_dir / 'config.yaml'
            
        self.config = ConfigLoader.load_config(config_path)
        self.signal_generator = SignalGenerator(config_path)
        self.signal_processor = SignalProcessor(config_path)
        self.dv01_calculator = DV01Calculator(config_path)
        self.risk_calculator = RiskMetricsCalculator(config_path)
        self.data_processor = DataProcessor(config_path)
        
        # Initialize state variables
        self.positions = {spread: {'short_term': {'size': 0, 'yield': 0}, 
                                 'long_term': {'size': 0, 'yield': 0}} 
                         for spread in SPREADS.keys()}
        self.trade_history = []
        self.daily_pnl = pd.DataFrame()
        self.portfolio_history = pd.DataFrame()
        
        # Load configuration parameters
        self.dv01_target = self.config['dv01']['target']
        self.transaction_costs = self.config['transaction_costs']
        self.rebalancing_freq = self.config['position']['rebalancing']['frequency']
        self.rebalancing_threshold = self.config['position']['rebalancing']['threshold_pct']
        self.max_dv01_per_spread = self.config['position']['max_position']['dv01_per_spread']
        self.max_portfolio_dv01 = self.config['position']['max_position']['total_portfolio']
        
        # Set up paths
        self.results_dir = Path(self.config['paths']['results']) / 'backtest'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Add new state variables for risk monitoring
        self.initial_portfolio_value = 1_000_000  # Starting with $1M
        self.equity_curve = [self.initial_portfolio_value]
        self.max_drawdown_limit = self.config['risk']['max_drawdown'] * self.initial_portfolio_value
    
    def load_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Load signals from JSON files and convert to DataFrame format.
        
        Returns:
            dict: Dictionary of DataFrames with signals for each spread
        """
        signals = {}
        for spread in SPREADS.keys():
            signal_path = Path(self.config['paths']['results']) / 'signals' / f'{spread}_signals.json'
            if signal_path.exists():
                with open(signal_path, 'r') as f:
                    signal_data = json.load(f)
                
                # Convert to DataFrame
                df = pd.DataFrame(signal_data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                signals[spread] = df
                logging.info(f"Loaded {len(df)} signals for {spread}")
            else:
                logging.error(f"Signal file not found: {signal_path}")
                return {}
        
        return signals
    
    def process_signal(self, signal_data: Dict) -> Tuple[SignalType, float]:
        """
        Process a signal from the signal generator.
        
        Args:
            signal_data: Dictionary containing signal information
            
        Returns:
            tuple: (signal direction, confidence)
        """
        return self.signal_processor.validate_signal(signal_data)
    
    def compute_position_size(self, spread: str, signal: SignalType, 
                            yields: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Compute DV01-neutral position sizes for a spread trade.
        
        Args:
            spread: Name of the spread (e.g., '2s10s')
            signal: Trading signal
            yields: Dictionary of current yields
            
        Returns:
            dict: Position sizes for each leg
        """
        if signal == SignalType.NEUTRAL:
            return {'short_term': {'size': 0, 'yield': 0}, 'long_term': {'size': 0, 'yield': 0}}
        
        # Get maturities and yields based on spread
        short_maturity, long_maturity = self._get_maturities(spread)
        short_yield = yields[SPREADS[spread][0]]
        long_yield = yields[SPREADS[spread][1]]
        
        # Calculate DV01 ratio using utility class
        ratio = self.dv01_calculator.calculate_dv01_ratio(
            spread, short_maturity, long_maturity, short_yield, long_yield)
        
        # Calculate maximum allowed DV01 based on configuration limits
        max_concentration = self.config['risk']['position_limits']['max_concentration']
        max_dv01 = min(
            self.max_dv01_per_spread,  # Per-spread limit
            self.max_portfolio_dv01 * max_concentration  # Concentration limit
        )
        
        # Calculate base notional for long leg to achieve target DV01, capped by max_dv01
        long_duration = self.dv01_calculator.duration_calculator.calculate_modified_duration(
            long_maturity, long_yield)
        base_notional = min(
            abs(self.dv01_target / (long_duration * 0.0001)),
            abs(max_dv01 / (long_duration * 0.0001))
        )
        
        # Calculate notionals for both legs
        if signal == SignalType.STEEPER:
            short_size = ratio * base_notional  # long the short end
            long_size = -base_notional  # short the long end
        else:  # FLATTENER
            short_size = -ratio * base_notional  # short the short end
            long_size = base_notional  # long the long end
        
        return {
            'short_term': {'size': short_size, 'yield': short_yield},
            'long_term': {'size': long_size, 'yield': long_yield}
        }
    
    def calculate_pnl(self, spread: str, position: Dict[str, Dict[str, float]],
                     prev_yields: Dict[str, float], curr_yields: Dict[str, float]) -> float:
        """
        Calculate daily P&L for a spread position.
        
        Args:
            spread: Name of the spread
            position: Current position sizes and yields
            prev_yields: Previous day's yields
            curr_yields: Current day's yields
            
        Returns:
            float: Daily P&L
        """
        if not position['short_term']['size'] and not position['long_term']['size']:
            return 0.0
        
        # Get maturities and yields
        short_maturity, long_maturity = self._get_maturities(spread)
        short_prev = prev_yields[SPREADS[spread][0]]
        short_curr = curr_yields[SPREADS[spread][0]]
        long_prev = prev_yields[SPREADS[spread][1]]
        long_curr = curr_yields[SPREADS[spread][1]]
        
        # Calculate P&L for each leg
        short_dv01 = self.dv01_calculator.calculate_dv01(
            abs(position['short_term']['size']),
            position['short_term']['yield'],
            short_maturity
        )
        long_dv01 = self.dv01_calculator.calculate_dv01(
            abs(position['long_term']['size']),
            position['long_term']['yield'],
            long_maturity
        )
        
        # P&L = DV01 * yield change (in bp) * position sign
        short_pnl = short_dv01 * (short_curr - short_prev) * 100 * np.sign(position['short_term']['size'])
        long_pnl = long_dv01 * (long_curr - long_prev) * 100 * np.sign(position['long_term']['size'])
        
        return short_pnl + long_pnl
    
    def calculate_carry(self, spread: str, position: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate daily carry for a spread position.
        
        Args:
            spread: Name of the spread
            position: Current position sizes and yields
            
        Returns:
            float: Daily carry
        """
        if not position['short_term']['size'] and not position['long_term']['size']:
            return 0.0
        
        # Calculate carry for each leg (assuming ACT/360)
        short_carry = position['short_term']['size'] * position['short_term']['yield'] / 360
        long_carry = position['long_term']['size'] * position['long_term']['yield'] / 360
        
        return short_carry + long_carry
    
    def calculate_transaction_cost(self, spread: str, old_position: Dict[str, Dict[str, float]],
                                 new_position: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate transaction costs for position changes.
        
        Args:
            spread: Name of the spread
            old_position: Previous position
            new_position: New position
            
        Returns:
            float: Transaction cost
        """
        cost = 0.0
        
        # Calculate cost for each leg
        for leg in ['short_term', 'long_term']:
            size_change = abs(new_position[leg]['size'] - old_position[leg]['size'])
            if size_change > 0:
                # Cost in basis points
                cost += size_change * self.transaction_costs['spread_bp'] / 10000
                # Cost in DV01 terms
                maturity = self._get_maturities(spread)[0 if leg == 'short_term' else 1]
                dv01 = self.dv01_calculator.calculate_dv01(
                    size_change,
                    new_position[leg]['yield'],
                    maturity
                )
                cost += dv01 * self.transaction_costs['dv01_cost']
        
        return cost
    
    def _get_maturities(self, spread: str) -> Tuple[float, float]:
        """Get maturities for a spread."""
        if spread == '2s10s':
            return 2, 10
        elif spread == '5s30s':
            return 5, 30
        else:
            raise ValueError(f"Unsupported spread: {spread}. Only 2s10s and 5s30s are supported.")
    
    def safe_yield_lookup(self, date: pd.Timestamp, yield_data: pd.DataFrame) -> Optional[Dict]:
        """
        Safely look up yield data for a given date, handling missing dates.
        
        Args:
            date: Date to look up yield data for
            yield_data: DataFrame containing yield data
            
        Returns:
            dict: Yield data for the date if available, None otherwise
        """
        if date not in yield_data.index:
            # Try to find the most recent available date
            available_dates = yield_data.index[yield_data.index <= date]
            if len(available_dates) == 0:
                return None
            most_recent = available_dates[-1]
            logging.info(f"Using yield data from {most_recent} for {date}")
            return yield_data.loc[most_recent].to_dict()
        return yield_data.loc[date].to_dict()
    
    def run_backtest(self, start_date: Union[str, pd.Timestamp], 
                    end_date: Union[str, pd.Timestamp]) -> Dict:
        """Run backtest simulation."""
        # Load signals and data
        signals = self.load_signals()
        if not signals:
            raise ValueError("No signals found")
            
        # Load yield data
        yield_data = self.data_processor.load_data('yields')
        
        # Initialize results
        results = {
            'daily_pnl': [],
            'positions': [],  # Will store daily position snapshots
            'trades': []
        }
        
        # Run simulation
        current_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Initialize daily tracking variables
        daily_pnl = 0.0
        
        while current_date <= end_date:
            # Reset daily PnL
            daily_pnl = 0.0
            
            # Get current yields safely
            current_yields = self.safe_yield_lookup(current_date, yield_data)
            if current_yields is None:
                logging.warning(f"No yield data available for or before {current_date}, skipping.")
                current_date += timedelta(days=1)
                continue
            
            # Process each spread
            for spread in SPREADS.keys():
                if current_date not in signals[spread].index:
                    continue
                
                # Get signal
                signal_data = signals[spread].loc[current_date].to_dict()
                signal, confidence = self.process_signal(signal_data)
                
                # Calculate potential new position
                new_position = self.compute_position_size(spread, signal, current_yields)
                
                # Check if rebalancing is needed based on DV01 deviation
                current_dv01 = self.risk_calculator.calculate_total_dv01({spread: self.positions[spread]})
                target_dv01 = self.dv01_target if signal != SignalType.NEUTRAL else 0
                
                # Only rebalance if DV01 deviation exceeds threshold or we're closing position (signal is NEUTRAL)
                if signal == SignalType.NEUTRAL or (target_dv01 > 0 and abs(current_dv01 - target_dv01) / target_dv01 >= self.rebalancing_threshold):
                    # Create a test portfolio with the new position
                    test_portfolio = self.positions.copy()
                    test_portfolio[spread] = new_position
                    
                    # Check all position limits including concentration and correlation
                    if not self.risk_calculator.check_position_limits(test_portfolio):
                        logging.warning(f"Position limits exceeded for {spread} on {current_date}")
                        new_position = self.positions[spread]  # Keep current position instead of zeroing
                else:
                    # Skip rebalancing, maintain current position
                    logging.info(f"Skipping rebalance for {spread} on {current_date} - DV01 deviation {abs(current_dv01 - target_dv01) / target_dv01:.2%} within threshold {self.rebalancing_threshold:.2%}")
                    new_position = self.positions[spread]
                
                # Calculate P&L and carry
                if current_date > pd.to_datetime(start_date):
                    prev_date = current_date - timedelta(days=1)
                    prev_yields = self.safe_yield_lookup(prev_date, yield_data)
                    if prev_yields is not None:
                        pnl = self.calculate_pnl(spread, self.positions[spread], prev_yields, current_yields)
                        carry = self.calculate_carry(spread, self.positions[spread])
                        
                        # Update PnL history for correlation tracking
                        self.risk_calculator.update_pnl_history(spread, pnl + carry)
                    else:
                        logging.warning(f"No yield data available for or before {prev_date}, skipping P&L calculation.")
                        pnl = 0.0
                        carry = 0.0
                else:
                    pnl = 0.0
                    carry = 0.0
                
                # Calculate transaction costs
                if new_position != self.positions[spread]:
                    cost = self.calculate_transaction_cost(spread, self.positions[spread], new_position)
                else:
                    cost = 0.0
                
                # Update daily PnL
                daily_pnl += pnl + carry - cost
                
                # Update position and record results
                old_position = self.positions[spread]
                self.positions[spread] = new_position
                results['daily_pnl'].append({
                    'date': current_date,
                    'spread': spread,
                    'pnl': pnl,
                    'carry': carry,
                    'cost': cost,
                    'total': pnl + carry - cost
                })
                
                # Record trade if position changed
                if (old_position['short_term']['size'] != new_position['short_term']['size'] or 
                    old_position['long_term']['size'] != new_position['long_term']['size']):
                    trade = {
                        'date': current_date,
                        'spread': spread,
                        'signal': signal,
                        'confidence': confidence,
                        'old_position': old_position,
                        'new_position': new_position,
                        'cost': cost
                    }
                    results['trades'].append(trade)
            
            # Update equity curve and check drawdown limit
            current_equity = self.equity_curve[-1] + daily_pnl
            self.equity_curve.append(current_equity)
            
            # Calculate current drawdown
            peak = max(self.equity_curve)
            drawdown = peak - current_equity
            
            # Check if max drawdown limit is breached
            if drawdown > self.max_drawdown_limit:
                logging.warning(f"Max drawdown limit breached on {current_date}. Closing all positions.")
                # Close all positions
                for spread in SPREADS.keys():
                    self.positions[spread] = {
                        'short_term': {'size': 0, 'yield': 0},
                        'long_term': {'size': 0, 'yield': 0}
                    }
            
            # Record end-of-day position snapshot
            results['positions'].append({
                'date': current_date,
                'positions': copy.deepcopy(self.positions)
            })
            
            current_date += timedelta(days=1)
        
        # Add equity curve to results
        results['equity_curve'] = pd.Series(self.equity_curve, index=[pd.to_datetime(start_date)] + [
            pd.to_datetime(start_date) + timedelta(days=i) for i in range(len(self.equity_curve)-1)
        ])
        
        # Convert results to DataFrames
        results['daily_pnl'] = pd.DataFrame(results['daily_pnl'])
        
        # Convert trades list to DataFrame with proper handling of dates
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            results['trades'] = trades_df.reset_index(drop=True)
        else:
            results['trades'] = pd.DataFrame(columns=['date', 'spread', 'signal', 'confidence', 'old_position', 'new_position', 'cost'])
        
        # Validate required columns
        required_columns = ['date', 'spread', 'signal', 'confidence', 'old_position', 'new_position', 'cost']
        missing_columns = [col for col in required_columns if col not in results['trades'].columns]
        if missing_columns:
            error_msg = f"Missing required columns in trades DataFrame: {missing_columns}"
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        return results

    def save_results(self, results: Dict) -> None:
        """
        Save backtest results to disk.
        
        Args:
            results: Dictionary containing backtest results
        """
        # Save daily P&L
        pnl_df = results['daily_pnl']
        for spread in SPREADS.keys():
            spread_pnl = pnl_df[pnl_df['spread'] == spread]
            spread_pnl.to_csv(self.results_dir / f'{spread}_daily_pnl.csv', index=True)
        
        # Save trades
        trades_df = results['trades']
        trades_df.to_csv(self.results_dir / 'trades.csv', index=True)
        
        logging.info(f"Results saved to {self.results_dir}")
    
    def get_backtest_summary(self, results: Dict) -> Dict:
        """
        Generate summary statistics for the backtest.
        
        Args:
            results: Dictionary containing backtest results
            
        Returns:
            dict: Summary statistics
        """
        pnl_df = results['daily_pnl']
        trades_df = results['trades']
        
        # Calculate portfolio-level metrics
        daily_total = pnl_df.groupby('date')['total'].sum()
        cumulative_pnl = daily_total.cumsum()
        
        # Annualization factor (252 trading days)
        ann_factor = np.sqrt(252)
        
        # Calculate Sharpe ratio with zero standard deviation guard
        std = daily_total.std()
        sharpe = float(ann_factor * daily_total.mean() / std) if std > 0 else 0.0
        
        summary = {
            'total_pnl': float(cumulative_pnl.iloc[-1]),
            'sharpe_ratio': sharpe,
            'max_drawdown': float(self._calculate_max_drawdown(cumulative_pnl)),
            'win_rate': float((daily_total > 0).mean()),
            'avg_daily_pnl': float(daily_total.mean()),
            'total_trades': len(trades_df),
            'avg_trade_cost': float(trades_df['cost'].mean()) if len(trades_df) > 0 else 0.0,
            'total_costs': float(pnl_df['cost'].sum())
        }
        
        # Add per-spread metrics
        for spread in SPREADS.keys():
            spread_pnl = pnl_df[pnl_df['spread'] == spread]
            spread_daily = spread_pnl.groupby('date')['total'].sum()
            spread_trades = trades_df[trades_df['spread'] == spread]
            
            summary[f'{spread}_total_pnl'] = float(spread_daily.sum())
            spread_std = spread_daily.std()
            summary[f'{spread}_sharpe'] = float(ann_factor * spread_daily.mean() / spread_std) if spread_std > 0 else 0.0
            summary[f'{spread}_trades'] = len(spread_trades)
        
        return summary
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from peak."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve - rolling_max
        return abs(float(drawdowns.min()))

def main():
    """Run backtest with generated signals."""
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize backtest engine
        backtest = BacktestEngine()
        
        # Load signals
        signals = backtest.load_signals()
        if not signals:
            logging.error("Failed to load signals")
            return
        
        # Load yield data
        yields_path = Path('data/raw/treasury_yields.csv')
        if yields_path.exists():
            yields_data = pd.read_csv(yields_path, index_col=0, parse_dates=True)
            logging.info("Loaded yield data")
        else:
            logging.error(f"Yield data file not found: {yields_path}")
            return
        
        # Get date range from signals
        start_date = min(df.index[0] for df in signals.values())
        end_date = max(df.index[-1] for df in signals.values())
        
        # Run backtest
        logging.info(f"Running backtest from {start_date} to {end_date}")
        results = backtest.run_backtest(
            start_date=start_date,
            end_date=end_date
        )
        
        # Save results
        backtest.save_results(results)
        logging.info("Backtest completed and results saved")
        
        # Print summary
        summary = backtest.get_backtest_summary(results)
        logging.info("\nBacktest Summary:")
        for key, value in summary.items():
            logging.info(f"{key}: {value}")
            
    except Exception as e:
        logging.error(f"Error running backtest: {str(e)}")
        raise

if __name__ == "__main__":
    main() 