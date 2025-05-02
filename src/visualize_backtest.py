import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import xlsxwriter
import yaml
import json
import ast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BacktestVisualizer:
    def __init__(self):
        self.results_dir = Path("results/backtest")
        
        # Load configuration
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
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
        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                                gridspec_kw={'height_ratios': [3, 2, 2, 2]})
        
        # Calculate cumulative PnL
        cum_pnl_2s10s = self.pnl_2s10s['total'].cumsum()
        cum_pnl_5s30s = self.pnl_5s30s['total'].cumsum()
        combined_pnl = cum_pnl_2s10s + cum_pnl_5s30s
        
        # Plot 2s10s strategy
        axes[0].plot(cum_pnl_2s10s.index, cum_pnl_2s10s.values, label='2s10s', color='C0')
        trade_dates = self.trades[self.trades['spread'] == '2s10s']['date']
        axes[0].scatter(trade_dates, cum_pnl_2s10s.loc[trade_dates], 
                       color='C0', marker='D', s=50, alpha=0.5,
                       label='2s10s trades')
        axes[0].set_title('2s10s Strategy')
        axes[0].set_ylabel('Cumulative PnL')
        axes[0].grid(True)
        axes[0].legend()
        
        # Plot 5s30s strategy
        axes[1].plot(cum_pnl_5s30s.index, cum_pnl_5s30s.values, label='5s30s', color='C1')
        trade_dates = self.trades[self.trades['spread'] == '5s30s']['date']
        axes[1].scatter(trade_dates, cum_pnl_5s30s.loc[trade_dates], 
                       color='C1', marker='D', s=50, alpha=0.5,
                       label='5s30s trades')
        axes[1].set_title('5s30s Strategy')
        axes[1].set_ylabel('Cumulative PnL')
        axes[1].grid(True)
        axes[1].legend()
        
        # Plot combined strategy
        axes[2].plot(combined_pnl.index, combined_pnl.values, label='Combined', color='C2')
        axes[2].set_title('Combined Strategy')
        axes[2].set_ylabel('Cumulative PnL')
        axes[2].grid(True)
        axes[2].legend()
        
        # Plot drawdown
        def calculate_drawdown(equity_curve):
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = equity_curve - running_max
            return drawdown
        
        # Calculate drawdowns as percentages
        dd_2s10s = calculate_drawdown(cum_pnl_2s10s) / cum_pnl_2s10s.max() * 100
        dd_5s30s = calculate_drawdown(cum_pnl_5s30s) / cum_pnl_5s30s.max() * 100
        dd_combined = calculate_drawdown(combined_pnl) / combined_pnl.max() * 100
        
        # Plot combined drawdown
        axes[3].plot(dd_combined.index, dd_combined.values, color='C3', label='Drawdown')
        
        # Add max drawdown limit line
        max_drawdown_limit = -20  # 20% drawdown limit
        axes[3].axhline(y=max_drawdown_limit, color='r', linestyle='--', 
                       label=f'Max Drawdown Limit ({max_drawdown_limit}%)')
        
        axes[3].set_title('Combined Strategy Drawdown')
        axes[3].set_ylabel('% Drawdown')
        axes[3].set_xlabel('Date')
        axes[3].grid(True)
        axes[3].legend()
        
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
        # Calculate monthly returns
        monthly_2s10s = self.pnl_2s10s['total'].resample('M').sum().to_frame()
        monthly_5s30s = self.pnl_5s30s['total'].resample('M').sum().to_frame()
        monthly_combined = (self.pnl_2s10s['total'] + self.pnl_5s30s['total']).resample('M').sum().to_frame()
        
        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Function to create and annotate heatmap
        def create_heatmap(data, ax, title):
            pivot = data.pivot_table(
                index=data.index.year,
                columns=data.index.month,
                values='total'
            )
            
            # Create heatmap
            sns.heatmap(pivot, cmap='RdYlGn', center=0, ax=ax, annot=True, fmt='.0f')
            ax.set_title(title)
            ax.set_xlabel('Month')
            ax.set_ylabel('Year')
            
            # Find best and worst months
            best_month = pivot.max().max()
            worst_month = pivot.min().min()
            
            # Get coordinates of best and worst months
            best_coords = np.where(pivot == best_month)
            worst_coords = np.where(pivot == worst_month)
            
            # Annotate best month
            if best_month > 0:
                ax.text(best_coords[1][0] + 0.5, best_coords[0][0] + 0.5, 
                       f'Best: {best_month:.0f}', 
                       ha='center', va='center', color='green', fontweight='bold')
            
            # Annotate worst month
            if worst_month < 0:
                ax.text(worst_coords[1][0] + 0.5, worst_coords[0][0] + 0.5, 
                       f'Worst: {worst_month:.0f}', 
                       ha='center', va='center', color='red', fontweight='bold')
        
        # Create heatmaps
        create_heatmap(monthly_2s10s, ax1, '2s10s Monthly Returns')
        create_heatmap(monthly_5s30s, ax2, '5s30s Monthly Returns')
        create_heatmap(monthly_combined, ax3, 'Combined Monthly Returns')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'monthly_returns.png')
        plt.close()

    def plot_trade_analysis(self):
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # Trade signals distribution
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Trade-level PnL & Carry analysis
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        
        # Plot trade signals distribution
        trades_2s10s = self.trades[self.trades['spread'] == '2s10s']
        trades_5s30s = self.trades[self.trades['spread'] == '5s30s']
        
        trades_2s10s['signal'].value_counts().plot(kind='bar', ax=ax1)
        ax1.set_title('2s10s Trade Signals')
        ax1.set_xlabel('Signal')
        ax1.set_ylabel('Count')
        
        trades_5s30s['signal'].value_counts().plot(kind='bar', ax=ax2)
        ax2.set_title('5s30s Trade Signals')
        ax2.set_xlabel('Signal')
        ax2.set_ylabel('Count')
        
        # Plot trade-level PnL & Carry analysis
        sns.histplot(self.pnl_2s10s['total'], bins=20, ax=ax3)
        ax3.set_title('2s10s Daily PnL')
        ax3.set_xlabel('PnL')
        ax3.set_ylabel('Frequency')
        
        sns.histplot(self.pnl_5s30s['total'], bins=20, ax=ax4)
        ax4.set_title('5s30s Daily PnL')
        ax4.set_xlabel('PnL')
        ax4.set_ylabel('Frequency')
        
        sns.histplot(self.pnl_2s10s['carry'], bins=20, ax=ax5)
        ax5.set_title('2s10s Daily Carry')
        ax5.set_xlabel('Carry')
        ax5.set_ylabel('Frequency')
        
        sns.histplot(self.pnl_5s30s['carry'], bins=20, ax=ax6)
        ax6.set_title('5s30s Daily Carry')
        ax6.set_xlabel('Carry')
        ax6.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'trade_analysis.png')
        plt.close()
        
        # Create duration vs. PnL scatter plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for spread, ax in [('2s10s', ax1), ('5s30s', ax2)]:
            df = self.trades[self.trades['spread'] == spread].sort_values('date')
            durations = []
            pnls = []
            
            for i in range(1, len(df)):
                # Calculate duration in days
                duration = (df.iloc[i].date - df.iloc[i-1].date).days
                if duration > 0:  # Only include valid durations
                    # Calculate PnL for the trade
                    start_date = df.iloc[i-1].date
                    end_date = df.iloc[i].date
                    trade_pnl = self.pnl_2s10s if spread == '2s10s' else self.pnl_5s30s
                    trade_pnl = trade_pnl.loc[start_date:end_date, 'total'].sum()
                    
                    durations.append(duration)
                    pnls.append(trade_pnl)
            
            ax.scatter(durations, pnls, alpha=0.5)
            ax.set_title(f'{spread} Duration vs. PnL')
            ax.set_xlabel('Days Held')
            ax.set_ylabel('Trade PnL')
            ax.grid(True)
            
            # Add regression line
            if durations and pnls:
                z = np.polyfit(durations, pnls, 1)
                p = np.poly1d(z)
                ax.plot(durations, p(durations), "r--", alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'duration_vs_pnl.png')
        plt.close()

    def plot_correlation_analysis(self):
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Calculate rolling correlation between spreads
        combined_df = pd.DataFrame({
            '2s10s': self.pnl_2s10s['total'],
            '5s30s': self.pnl_5s30s['total']
        })
        
        # Calculate rolling correlation with 30-day window
        rolling_corr = combined_df['2s10s'].rolling(window=30).corr(combined_df['5s30s'])
        
        # Plot rolling correlation
        ax1.plot(rolling_corr.index, rolling_corr.values, label='30-Day Rolling Correlation')
        ax1.set_title('30-Day Rolling Correlation between 2s10s and 5s30s Returns')
        ax1.set_ylabel('Correlation')
        ax1.grid(True)
        ax1.legend()
        
        # Add horizontal band for correlation threshold
        correlation_threshold = self.config['risk']['position_limits']['correlation_threshold']
        ax1.fill_between(rolling_corr.index,
                        correlation_threshold,
                        -correlation_threshold,
                        color='lightgray', alpha=0.5,
                        label=f'Correlation Threshold (±{correlation_threshold})')
        
        # Plot individual strategy returns
        ax2.plot(combined_df.index, combined_df['2s10s'], label='2s10s Returns', alpha=0.7)
        ax2.plot(combined_df.index, combined_df['5s30s'], label='5s30s Returns', alpha=0.7)
        ax2.set_title('Daily Returns')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Returns')
        ax2.grid(True)
        ax2.legend()
        
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
            
            # Export timeline data to Excel
            logger.info("Exporting timeline data to Excel...")

            # Build timeline DataFrame
            daily = pd.concat([
                self.pnl_2s10s.assign(spread="2s10s"),
                self.pnl_5s30s.assign(spread="5s30s")
            ]).reset_index()
            daily = daily[['date','spread','pnl','carry','cost','total']]

            # Merge with trades
            trades_flat = self.trades.copy()
            
            # Parse position strings to dictionaries using ast.literal_eval
            trades_flat['old_position'] = trades_flat['old_position'].apply(ast.literal_eval)
            trades_flat['new_position'] = trades_flat['new_position'].apply(ast.literal_eval)
            
            # Calculate total position sizes
            trades_flat['old_size'] = trades_flat['old_position'].apply(
                lambda p: p['short_term']['size'] + p['long_term']['size']
            )
            trades_flat['new_size'] = trades_flat['new_position'].apply(
                lambda p: p['short_term']['size'] + p['long_term']['size']
            )

            with pd.ExcelWriter(self.results_dir/"backtest_timeline.xlsx", engine="xlsxwriter") as writer:
                daily.to_excel(writer, sheet_name="DailyPNL", index=False)
                trades_flat.to_excel(writer, sheet_name="Trades", index=False)
            logger.info("Wrote backtest_timeline.xlsx")
            
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
            raise

if __name__ == "__main__":
    visualizer = BacktestVisualizer()
    visualizer.generate_all_plots() 