"""
Trade execution simulation module for yield curve trading strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

from .utils import (
    DurationCalculator,
    DV01Calculator,
    RiskMetricsCalculator,
    ConfigLoader,
    DataProcessor
)
from .signal_generator import SignalGenerator

class BacktestEngine:
    """Simulate trading based on model signals."""
    
    def __init__(self, config_path: Union[str, Path] = 'config.yaml'):
        """
        Initialize BacktestEngine.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader.load_config(config_path)
        self.signal_generator = SignalGenerator(config_path)
        self.dv01_calculator = DV01Calculator()
        self.risk_calculator = RiskMetricsCalculator()
        
        # Initialize state variables
        self.positions = {}  # Current positions for each spread
        self.trade_history = []  # List of all trades
        self.daily_pnl = pd.DataFrame()  # Daily P&L for each spread
        self.portfolio_history = pd.DataFrame()  # Portfolio value history
        
        # Load configuration parameters
        self.dv01_target = self.config['dv01']['target']
        self.transaction_costs = self.config['transaction_costs']
        self.rebalancing_freq = self.config['position']['rebalancing']['frequency']
        self.rebalancing_threshold = self.config['position']['rebalancing']['threshold_pct']
        
        # Set up paths
        self.results_dir = Path(self.config['paths']['results']) / 'backtest'
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_position_size(self, spread: str, signal: int, 
                            yields: Dict[str, float]) -> Dict[str, float]:
        """
        Compute DV01-neutral position sizes for a spread trade.
        
        Args:
            spread: Name of the spread (e.g., '2s10s')
            signal: Trading signal (1 for steepener, -1 for flattener, 0 for neutral)
            yields: Dictionary of current yields
            
        Returns:
            dict: Position sizes for each leg
                {
                    'short_term': {'size': float, 'yield': float},
                    'long_term': {'size': float, 'yield': float}
                }
        """
        if signal == 0:
            return {'short_term': {'size': 0, 'yield': 0}, 'long_term': {'size': 0, 'yield': 0}}
        
        # Get maturities based on spread
        if spread == '2s10s':
            short_maturity, long_maturity = 2, 10
            short_yield = yields['2y']
            long_yield = yields['10y']
        elif spread == '5s30s':
            short_maturity, long_maturity = 5, 30
            short_yield = yields['5y']
            long_yield = yields['30y']
        else:
            raise ValueError(f"Unsupported spread: {spread}")
        
        # Calculate DV01 ratio using utility class
        ratio = self.dv01_calculator.calculate_dv01_ratio(
            short_maturity, long_maturity, short_yield, long_yield)
        
        # Base notional for long leg to achieve target DV01
        long_duration = self.dv01_calculator.duration_calculator.calculate_modified_duration(
            long_maturity, long_yield)
        base_notional = abs(self.dv01_target / (long_duration * 0.0001))
        
        # Calculate notionals for both legs
        if signal == 1:  # steepener
            short_size = ratio * base_notional  # long the short end
            long_size = -base_notional  # short the long end
        else:  # flattener
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
        
        # Get maturities
        if spread == '2s10s':
            short_maturity, long_maturity = 2, 10
            short_prev = prev_yields['2y']
            short_curr = curr_yields['2y']
            long_prev = prev_yields['10y']
            long_curr = curr_yields['10y']
        elif spread == '5s30s':
            short_maturity, long_maturity = 5, 30
            short_prev = prev_yields['5y']
            short_curr = curr_yields['5y']
            long_prev = prev_yields['30y']
            long_curr = curr_yields['30y']
        
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
        # Calculate change in position size for each leg
        short_change = abs(new_position['short_term']['size'] - old_position['short_term']['size'])
        long_change = abs(new_position['long_term']['size'] - old_position['long_term']['size'])
        
        # Apply transaction cost in basis points
        cost_bp = self.transaction_costs['spread_bp']
        short_cost = short_change * cost_bp * 0.0001  # Convert bp to decimal
        long_cost = long_change * cost_bp * 0.0001
        
        return short_cost + long_cost
    
    def needs_rebalancing(self, spread: str, position: Dict[str, Dict[str, float]],
                         curr_yields: Dict[str, float]) -> bool:
        """
        Check if position needs rebalancing based on DV01 neutrality.
        
        Args:
            spread: Name of the spread
            position: Current position
            curr_yields: Current yields
            
        Returns:
            bool: Whether rebalancing is needed
        """
        if not position['short_term']['size'] and not position['long_term']['size']:
            return False
        
        # Get maturities
        if spread == '2s10s':
            short_maturity, long_maturity = 2, 10
        elif spread == '5s30s':
            short_maturity, long_maturity = 5, 30
        
        # Calculate current DV01s
        short_dv01 = abs(self.dv01_calculator.calculate_dv01(
            position['short_term']['size'],
            curr_yields['2y' if spread == '2s10s' else '5y'],
            short_maturity
        ))
        long_dv01 = abs(self.dv01_calculator.calculate_dv01(
            position['long_term']['size'],
            curr_yields['10y' if spread == '2s10s' else '30y'],
            long_maturity
        ))
        
        # Check if DV01 difference exceeds threshold
        dv01_diff = abs(short_dv01 - long_dv01)
        return dv01_diff > (self.dv01_target * self.rebalancing_threshold)
    
    def run_backtest(self, start_date: pd.Timestamp, end_date: pd.Timestamp,
                    yields_data: pd.DataFrame, signals: Dict[str, pd.DataFrame]) -> Dict:
        """
        Run backtest simulation.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            yields_data: DataFrame with daily yields
            signals: Dictionary of signal DataFrames for each spread
            
        Returns:
            dict: Backtest results
        """
        # Initialize results storage
        self.positions = {
            '2s10s': {'short_term': {'size': 0, 'yield': 0}, 'long_term': {'size': 0, 'yield': 0}},
            '5s30s': {'short_term': {'size': 0, 'yield': 0}, 'long_term': {'size': 0, 'yield': 0}}
        }
        self.daily_pnl = pd.DataFrame(index=pd.date_range(start_date, end_date))
        self.portfolio_history = pd.DataFrame(index=pd.date_range(start_date, end_date))
        
        # Run daily simulation
        prev_date = None
        for date in pd.date_range(start_date, end_date):
            if date not in yields_data.index:
                continue
            
            curr_yields = yields_data.loc[date]
            
            # Process each spread
            for spread in ['2s10s', '5s30s']:
                if spread not in signals:
                    continue
                
                # Get signal for current day
                if date not in signals[spread].index:
                    continue
                    
                signal = signals[spread].loc[date, 'signal']
                
                # Calculate P&L and carry if we have a position
                if prev_date is not None:
                    prev_yields = yields_data.loc[prev_date]
                    pnl = self.calculate_pnl(spread, self.positions[spread], prev_yields, curr_yields)
                    carry = self.calculate_carry(spread, self.positions[spread])
                    self.daily_pnl.loc[date, f'{spread}_pnl'] = pnl
                    self.daily_pnl.loc[date, f'{spread}_carry'] = carry
                
                # Check if we need to update position
                new_position = self.compute_position_size(spread, signal, curr_yields)
                
                # Calculate transaction costs if position changes
                costs = self.calculate_transaction_cost(spread, self.positions[spread], new_position)
                self.daily_pnl.loc[date, f'{spread}_costs'] = costs
                
                # Update position
                self.positions[spread] = new_position
                
                # Store position information
                self.portfolio_history.loc[date, f'{spread}_short_size'] = new_position['short_term']['size']
                self.portfolio_history.loc[date, f'{spread}_long_size'] = new_position['long_term']['size']
            
            prev_date = date
        
        # Calculate portfolio-level metrics
        self.daily_pnl['total_pnl'] = self.daily_pnl.filter(like='_pnl').sum(axis=1)
        self.daily_pnl['total_carry'] = self.daily_pnl.filter(like='_carry').sum(axis=1)
        self.daily_pnl['total_costs'] = self.daily_pnl.filter(like='_costs').sum(axis=1)
        self.daily_pnl['net_pnl'] = self.daily_pnl['total_pnl'] + self.daily_pnl['total_carry'] - self.daily_pnl['total_costs']
        
        # Calculate cumulative P&L
        self.portfolio_history['cumulative_pnl'] = self.daily_pnl['net_pnl'].cumsum()
        
        # Save results
        self.save_results()
        
        return self.get_backtest_summary()
    
    def get_backtest_summary(self) -> Dict:
        """
        Generate summary statistics for the backtest.
        
        Returns:
            dict: Summary statistics
        """
        daily_returns = self.daily_pnl['net_pnl'] / self.dv01_target
        
        summary = {
            'total_return': float(self.portfolio_history['cumulative_pnl'].iloc[-1]),
            'sharpe_ratio': float(np.sqrt(252) * daily_returns.mean() / daily_returns.std()),
            'max_drawdown': float(self.calculate_max_drawdown(self.portfolio_history['cumulative_pnl'])),
            'hit_rate': float((daily_returns > 0).mean()),
            'avg_daily_pnl': float(self.daily_pnl['net_pnl'].mean()),
            'avg_daily_carry': float(self.daily_pnl['total_carry'].mean()),
            'total_transaction_costs': float(self.daily_pnl['total_costs'].sum()),
            'var_95': float(self.risk_calculator.calculate_var(daily_returns, 0.95)),
            'expected_shortfall_95': float(self.risk_calculator.calculate_expected_shortfall(daily_returns, 0.95))
        }
        
        return summary
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown from peak."""
        rolling_max = equity_curve.expanding().max()
        drawdowns = equity_curve - rolling_max
        return abs(float(drawdowns.min()))
    
    def save_results(self) -> None:
        """Save backtest results to disk."""
        # Save daily P&L
        self.daily_pnl.to_csv(self.results_dir / 'daily_pnl.csv')
        
        # Save portfolio history
        self.portfolio_history.to_csv(self.results_dir / 'portfolio_history.csv')
        
        # Save trade history
        pd.DataFrame(self.trade_history).to_csv(self.results_dir / 'trade_history.csv', index=False)
        
        # Save summary statistics
        summary = self.get_backtest_summary()
        with open(self.results_dir / 'backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=4) 