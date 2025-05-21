import pandas as pd
import numpy as np
from pathlib import Path

def calculate_performance_metrics(pnl_file, initial_capital=1_000_000):
    # Read the CSV file
    df = pd.read_csv(pnl_file)
    
    # Filter out weekends (where total PnL is 0)
    df = df[df['total'] != 0]
    
    # Calculate metrics based on absolute PnL values
    daily_pnl = df['total']
    
    # Calculate annualized metrics
    trading_days = 252  # Standard number of trading days in a year
    total_days = len(daily_pnl)
    
    # Calculate total PnL and average daily PnL
    total_pnl = daily_pnl.sum()
    avg_daily_pnl = daily_pnl.mean()
    
    # Calculate annualized return
    annualized_return = (avg_daily_pnl * trading_days) / initial_capital
    
    # Calculate volatility
    daily_vol = daily_pnl.std()
    annualized_volatility = daily_vol * np.sqrt(trading_days)
    
    # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
    risk_free_rate = 0.02
    sharpe_ratio = (annualized_return - risk_free_rate) / (annualized_volatility / initial_capital)
    
    # Calculate drawdown metrics
    cumulative_pnl = daily_pnl.cumsum()
    rolling_max = cumulative_pnl.expanding().max()
    drawdowns = (cumulative_pnl - rolling_max)
    max_drawdown = drawdowns.min()
    max_drawdown_pct = (max_drawdown / initial_capital) * 100
    
    # Calculate win rate
    win_rate = len(daily_pnl[daily_pnl > 0]) / len(daily_pnl)
    
    # Calculate profit factor
    gross_profit = daily_pnl[daily_pnl > 0].sum()
    gross_loss = abs(daily_pnl[daily_pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Calculate average win/loss
    avg_win = daily_pnl[daily_pnl > 0].mean()
    avg_loss = daily_pnl[daily_pnl < 0].mean()
    
    # Calculate total return percentage
    total_return_pct = (total_pnl / initial_capital) * 100
    
    return {
        'Initial Capital': f'${initial_capital:,.2f}',
        'Total PnL': f'${total_pnl:,.2f} ({total_return_pct:.2f}%)',
        'Average Daily PnL': f'${avg_daily_pnl:,.2f}',
        'Annualized Return': f'{annualized_return:.2%}',
        'Annualized Volatility': f'${annualized_volatility:,.2f}',
        'Sharpe Ratio': f'{sharpe_ratio:.2f}',
        'Max Drawdown': f'${abs(max_drawdown):,.2f} ({abs(max_drawdown_pct):.2f}%)',
        'Win Rate': f'{win_rate:.2%}',
        'Profit Factor': f'{profit_factor:.2f}',
        'Average Win': f'${avg_win:,.2f}',
        'Average Loss': f'${avg_loss:,.2f}',
        'Number of Trading Days': total_days
    }

if __name__ == '__main__':
    # Calculate metrics for 2s10s strategy
    pnl_file_2s10s = Path('results/backtest/archive/2s10s_daily_pnl.csv')
    metrics_2s10s = calculate_performance_metrics(pnl_file_2s10s)
    
    # Calculate metrics for 5s30s strategy
    pnl_file_5s30s = Path('results/backtest/archive/5s30s_daily_pnl.csv')
    metrics_5s30s = calculate_performance_metrics(pnl_file_5s30s)
    
    print("\n2s10s Strategy Performance Metrics:")
    print("-" * 50)
    for metric, value in metrics_2s10s.items():
        print(f"{metric}: {value}")
        
    print("\n5s30s Strategy Performance Metrics:")
    print("-" * 50)
    for metric, value in metrics_5s30s.items():
        print(f"{metric}: {value}") 