"""
Portfolio management and risk metrics module for yield curve trading strategy.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union
from pathlib import Path
import logging
import json
from datetime import datetime, timedelta

from .utils import RiskMetricsCalculator, ConfigLoader, SPREADS

class PortfolioManager:
    """Manage portfolio of spread trading strategies."""
    
    def __init__(self, config_path: Union[str, Path] = 'config.yaml'):
        """
        Initialize PortfolioManager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader.load_config(config_path)
        self.risk_calculator = RiskMetricsCalculator(config_path)
        
        # Load configuration parameters
        self.weighting_scheme = self.config['portfolio']['weighting_scheme']
        self.rebalancing_frequency = self.config['portfolio']['rebalancing_frequency']
        self.vol_target = self.config['portfolio']['vol_target']
        self.lookback_window = self.config['portfolio']['lookback_window']
        
        # Set up paths
        self.results_dir = Path(self.config['paths']['results']) / 'portfolio'
        self.backtest_dir = Path(self.config['paths']['results']) / 'backtest'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize state variables
        self.weights = {}
        self.portfolio_returns = pd.Series()
        self.strategy_returns = pd.DataFrame()
    
    def load_backtest_results(self) -> Dict[str, pd.Series]:
        """
        Load backtest results for each spread.
        
        Returns:
            dict: Dictionary of strategy returns
        """
        strategy_returns = {}
        
        for spread in SPREADS.keys():
            # Load daily P&L
            pnl_file = self.backtest_dir / f'{spread}_daily_pnl.csv'
            if pnl_file.exists():
                pnl_data = pd.read_csv(pnl_file)
                pnl_data['date'] = pd.to_datetime(pnl_data['date'])
                pnl_data.set_index('date', inplace=True)
                
                # Calculate returns
                strategy_returns[spread] = pnl_data['total'] / self.config['dv01']['target']
            else:
                logging.warning(f"Backtest results not found for {spread}")
        
        return strategy_returns
    
    def run_analysis(self) -> Dict:
        """
        Run complete portfolio analysis.
        
        Returns:
            dict: Analysis results
        """
        # Load backtest results
        strategy_returns = self.load_backtest_results()
        
        if not strategy_returns:
            raise ValueError("No backtest results found")
        
        # Run analysis
        analysis = self.analyze_portfolio(strategy_returns)
        
        # Save results
        self.save_results(analysis)
        
        return analysis
    
    def compute_portfolio_weights(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Compute portfolio weights based on selected scheme.
        
        Args:
            returns: DataFrame of strategy returns
            
        Returns:
            dict: Strategy weights
        """
        if self.weighting_scheme == 'equal_dv01':
            # Equal DV01 means equal weights since each strategy is DV01-neutral
            n_strategies = len(returns.columns)
            weights = {col: 1.0 / n_strategies for col in returns.columns}
            
        elif self.weighting_scheme == 'risk_adjusted':
            # Scale weights by inverse volatility
            lookback = min(self.lookback_window, len(returns))
            vols = returns.tail(lookback).std()
            inv_vols = 1.0 / vols
            weights = (inv_vols / inv_vols.sum()).to_dict()
            
        else:
            raise ValueError(f"Unknown weighting scheme: {self.weighting_scheme}")
        
        return weights
    
    def calculate_portfolio_metrics(self, returns: pd.Series) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Args:
            returns: Series of portfolio returns
            
        Returns:
            dict: Performance metrics
        """
        # Annualization factor
        ann_factor = np.sqrt(252)  # Assuming daily returns
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(returns)) - 1
        ann_vol = returns.std() * ann_factor
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # Drawdown analysis
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Risk metrics
        var_95 = self.risk_calculator.calculate_var(returns)
        es_95 = self.risk_calculator.calculate_expected_shortfall(returns)
        
        # Return statistics
        pos_months = returns.groupby(returns.index.to_period('M')).sum()
        win_rate = (pos_months > 0).mean()
        
        return {
            'total_return': float(total_return),
            'annualized_return': float(ann_return),
            'annualized_volatility': float(ann_vol),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'var_95': float(var_95),
            'expected_shortfall_95': float(es_95),
            'win_rate_monthly': float(win_rate),
            'number_of_trades': len(returns)
        }
    
    def calculate_risk_contributions(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate risk contribution of each strategy.
        
        Args:
            returns: DataFrame of strategy returns
            weights: Strategy weights
            
        Returns:
            dict: Risk contributions
        """
        # Convert weights to array
        w = np.array([weights[col] for col in returns.columns])
        
        # Calculate covariance matrix
        cov = returns.cov().values
        
        # Portfolio volatility
        port_vol = np.sqrt(w.dot(cov).dot(w))
        
        # Marginal risk contributions
        mrc = cov.dot(w)
        
        # Component risk contributions
        crc = w * mrc / port_vol if port_vol > 0 else w * 0
        
        return dict(zip(returns.columns, crc))
    
    def combine_strategy_returns(self, strategy_returns: Dict[str, pd.Series]) -> pd.Series:
        """
        Combine individual strategy returns into portfolio returns.
        
        Args:
            strategy_returns: Dictionary of strategy return series
            
        Returns:
            Series: Portfolio returns
        """
        # Convert to DataFrame
        returns_df = pd.DataFrame(strategy_returns)
        self.strategy_returns = returns_df
        
        # Compute weights
        self.weights = self.compute_portfolio_weights(returns_df)
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0, index=returns_df.index)
        for strategy in self.weights:
            portfolio_returns += returns_df[strategy] * self.weights[strategy]
        
        self.portfolio_returns = portfolio_returns
        return portfolio_returns
    
    def analyze_portfolio(self, strategy_returns: Dict[str, pd.Series]) -> Dict:
        """
        Analyze portfolio performance and risk.
        
        Args:
            strategy_returns: Dictionary of strategy return series
            
        Returns:
            dict: Portfolio analysis results
        """
        # Combine strategy returns
        portfolio_returns = self.combine_strategy_returns(strategy_returns)
        
        # Calculate portfolio metrics
        portfolio_metrics = self.calculate_portfolio_metrics(portfolio_returns)
        
        # Calculate risk contributions
        risk_contrib = self.calculate_risk_contributions(self.strategy_returns, self.weights)
        
        # Calculate correlation matrix
        correlation = self.risk_calculator.calculate_correlation_matrix(self.strategy_returns)
        
        # Calculate concentration
        concentration = self.risk_calculator.calculate_hhi(np.array(list(self.weights.values())))
        
        analysis = {
            'portfolio_metrics': portfolio_metrics,
            'weights': self.weights,
            'risk_contributions': risk_contrib,
            'correlation_matrix': correlation.to_dict(),
            'concentration': float(concentration)
        }
        
        return analysis
    
    def save_results(self, analysis: Dict) -> None:
        """
        Save portfolio analysis results.
        
        Args:
            analysis: Dictionary of analysis results
        """
        # Save portfolio returns
        self.portfolio_returns.to_csv(self.results_dir / 'portfolio_returns.csv')
        
        # Save strategy returns
        self.strategy_returns.to_csv(self.results_dir / 'strategy_returns.csv')
        
        # Save analysis results
        with open(self.results_dir / 'portfolio_analysis.json', 'w') as f:
            json.dump(analysis, f, indent=4, cls=NumpyJSONEncoder)
        
        # Save correlation matrix plot
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                self.strategy_returns.corr(),
                annot=True,
                cmap='RdYlBu',
                center=0,
                vmin=-1,
                vmax=1
            )
            plt.title('Strategy Correlation Matrix')
            plt.tight_layout()
            plt.savefig(self.results_dir / 'correlation_matrix.png')
            plt.close()
        except ImportError:
            logging.warning("Seaborn not installed. Skipping correlation plot.")

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        return super().default(obj)

def main():
    """Run portfolio analysis."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Initialize portfolio manager
        portfolio = PortfolioManager()
        
        # Run analysis
        logging.info("Starting portfolio analysis...")
        results = portfolio.run_analysis()
        
        # Print summary
        logging.info("\nPortfolio Analysis Summary:")
        for key, value in results['portfolio_metrics'].items():
            logging.info(f"{key}: {value:.4f}")
        
        logging.info("\nStrategy Weights:")
        for strategy, weight in results['weights'].items():
            logging.info(f"{strategy}: {weight:.4f}")
        
        logging.info("\nRisk Contributions:")
        for strategy, contrib in results['risk_contributions'].items():
            logging.info(f"{strategy}: {contrib:.4f}")
        
        logging.info(f"\nConcentration: {results['concentration']:.4f}")
        
    except Exception as e:
        logging.error(f"Error in portfolio analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 