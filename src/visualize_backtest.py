import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestVisualizer:
    def __init__(self):
        self.results_dir = Path("results/backtest")
        
        # Load PnL data
        self.pnl_2s10s = pd.read_csv(self.results_dir / "2s10s_daily_pnl.csv")
        self.pnl_5s30s = pd.read_csv(self.results_dir / "5s30s_daily_pnl.csv")
        
        # Convert date columns
        self.pnl_2s10s['date'] = pd.to_datetime(self.pnl_2s10s['date'])
        self.pnl_5s30s['date'] = pd.to_datetime(self.pnl_5s30s['date'])
        
        # Load trades data
        self.trades = pd.read_csv(self.results_dir / "trades.csv")
        self.trades['date'] = pd.to_datetime(self.trades['date'])
        
        # Set date as index for PnL data
        self.pnl_2s10s.set_index('date', inplace=True)
        self.pnl_5s30s.set_index('date', inplace=True)

    def plot_equity_curves(self):
        plt.figure(figsize=(12, 6))
        
        # Calculate cumulative PnL
        cum_pnl_2s10s = self.pnl_2s10s['total'].cumsum()
        cum_pnl_5s30s = self.pnl_5s30s['total'].cumsum()
        combined_pnl = cum_pnl_2s10s + cum_pnl_5s30s
        
        plt.plot(cum_pnl_2s10s.index, cum_pnl_2s10s.values, label='2s10s')
        plt.plot(cum_pnl_5s30s.index, cum_pnl_5s30s.values, label='5s30s')
        plt.plot(combined_pnl.index, combined_pnl.values, label='Combined')
        
        plt.title('Cumulative PnL Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative PnL')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'equity_curves.png')
        plt.close()

    def plot_drawdowns(self):
        plt.figure(figsize=(12, 6))
        
        def calculate_drawdown(equity_curve):
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = equity_curve - running_max
            return drawdown
        
        # Calculate drawdowns
        dd_2s10s = calculate_drawdown(self.pnl_2s10s['total'].cumsum())
        dd_5s30s = calculate_drawdown(self.pnl_5s30s['total'].cumsum())
        combined_pnl = self.pnl_2s10s['total'].cumsum() + self.pnl_5s30s['total'].cumsum()
        dd_combined = calculate_drawdown(combined_pnl)
        
        plt.plot(dd_2s10s.index, dd_2s10s.values, label='2s10s')
        plt.plot(dd_5s30s.index, dd_5s30s.values, label='5s30s')
        plt.plot(dd_combined.index, dd_combined.values, label='Combined')
        
        plt.title('Drawdown Analysis')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'drawdowns.png')
        plt.close()

    def plot_monthly_returns(self):
        plt.figure(figsize=(15, 6))
        
        # Calculate monthly returns
        monthly_2s10s = self.pnl_2s10s['total'].resample('M').sum().to_frame()
        monthly_5s30s = self.pnl_5s30s['total'].resample('M').sum().to_frame()
        
        # Create a subplot for each spread
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot heatmaps
        monthly_2s10s_pivot = monthly_2s10s.pivot_table(
            index=monthly_2s10s.index.year,
            columns=monthly_2s10s.index.month,
            values='total'
        )
        
        monthly_5s30s_pivot = monthly_5s30s.pivot_table(
            index=monthly_5s30s.index.year,
            columns=monthly_5s30s.index.month,
            values='total'
        )
        
        sns.heatmap(monthly_2s10s_pivot, cmap='RdYlGn', center=0, ax=ax1)
        ax1.set_title('2s10s Monthly Returns')
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Year')
        
        sns.heatmap(monthly_5s30s_pivot, cmap='RdYlGn', center=0, ax=ax2)
        ax2.set_title('5s30s Monthly Returns')
        ax2.set_xlabel('Month')
        ax2.set_ylabel('Year')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'monthly_returns.png')
        plt.close()

    def plot_trade_analysis(self):
        plt.figure(figsize=(15, 10))
        
        # Separate trades by spread
        trades_2s10s = self.trades[self.trades['spread'] == '2s10s']
        trades_5s30s = self.trades[self.trades['spread'] == '5s30s']
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot trade signals distribution
        trades_2s10s['signal'].value_counts().plot(kind='bar', ax=ax1, title='2s10s Trade Signals')
        trades_5s30s['signal'].value_counts().plot(kind='bar', ax=ax2, title='5s30s Trade Signals')
        
        # Plot confidence distribution
        sns.histplot(data=trades_2s10s, x='confidence', ax=ax3, bins=20)
        ax3.set_title('2s10s Confidence Distribution')
        
        sns.histplot(data=trades_5s30s, x='confidence', ax=ax4, bins=20)
        ax4.set_title('5s30s Confidence Distribution')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'trade_analysis.png')
        plt.close()

    def plot_correlation_analysis(self):
        plt.figure(figsize=(12, 6))
        
        # Calculate rolling correlation between spreads
        combined_df = pd.DataFrame({
            '2s10s': self.pnl_2s10s['total'],
            '5s30s': self.pnl_5s30s['total']
        })
        
        rolling_corr = combined_df['2s10s'].rolling(window=30).corr(combined_df['5s30s'])
        
        plt.plot(rolling_corr.index, rolling_corr.values)
        plt.title('30-Day Rolling Correlation between 2s10s and 5s30s Returns')
        plt.xlabel('Date')
        plt.ylabel('Correlation')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'correlation_analysis.png')
        plt.close()

    def generate_all_plots(self):
        """Generate all visualization plots."""
        try:
            logger.info("Generating equity curves plot...")
            self.plot_equity_curves()
            
            logger.info("Generating drawdown analysis plot...")
            self.plot_drawdowns()
            
            logger.info("Generating monthly returns heatmap...")
            self.plot_monthly_returns()
            
            logger.info("Generating trade analysis plots...")
            self.plot_trade_analysis()
            
            logger.info("Generating correlation analysis plot...")
            self.plot_correlation_analysis()
            
            logger.info("All plots generated successfully!")
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            raise

if __name__ == "__main__":
    visualizer = BacktestVisualizer()
    visualizer.generate_all_plots() 