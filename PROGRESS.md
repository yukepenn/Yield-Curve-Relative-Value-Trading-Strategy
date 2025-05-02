# Project Progress

## Latest Updates
### 2024-03-21
- Initial backtest results show significant underperformance
- Total PnL: -$4.4M with Sharpe ratio of -16.78
- Win rate of only 21.6% across both strategies
- Need to investigate and address performance issues
- Fixed data shape issue in model training by flattening predictions and targets before saving to CSV
- Successfully trained LSTM model for 2s10s spread with next_day prediction type
- Achieved MSE of 1.83 on validation set
- Improved model training with better probability handling and metric calculation
- Successfully trained both MLP and LSTM models for direction prediction
- MLP achieved accuracy of 0.543 and ROC-AUC of 0.537
- LSTM achieved accuracy of 0.556 and ROC-AUC of 0.523
- Successfully trained models for ternary classification with proper multi-class metrics
- MLP achieved accuracy of 0.427 and macro-F1 of 0.355
- LSTM achieved accuracy of 0.434 and macro-F1 of 0.319

## Completed Tasks
- [x] Initial project setup
- [x] Data collection module implementation
- [x] Basic yield curve analysis
- [x] Trading strategy development
- [x] Backtesting framework implementation
- [x] Strategy optimization
- [x] Initial backtest run

## In Progress
- [ ] Strategy performance analysis
- [ ] Risk management implementation
- [ ] Transaction cost modeling refinement
- [ ] Position sizing optimization
- [ ] Investigation of poor backtest results

## Next Steps
1. Analyze poor backtest performance:
   - Review entry/exit signals
   - Check position sizing logic
   - Validate risk management rules
   - Examine transaction cost assumptions

2. Implement risk management improvements:
   - Add stop-loss mechanisms
   - Implement position limits
   - Enhance drawdown controls
   - Add volatility filters

3. Refine strategy parameters:
   - Recalibrate signal thresholds
   - Adjust position sizing rules
   - Optimize entry/exit timing
   - Fine-tune risk parameters

4. Enhance transaction cost model:
   - Update cost assumptions
   - Add slippage modeling
   - Implement market impact
   - Consider liquidity constraints

5. Analyze model performance metrics
- Compare results across different spreads and prediction types
- Implement backtesting with trained models

## Current Status
The project has completed initial development and testing phases. However, the latest backtest results show significant underperformance, with both strategies showing negative PnL and poor risk-adjusted returns. Immediate focus is needed on strategy improvement and risk management.

- Model training pipeline is now working correctly
- LSTM model successfully trained and saved
- Results and predictions saved to appropriate directories

## Timeline
- Week 1-2: Project setup and data collection
- Week 3-4: Strategy development and backtesting
- Week 5: Initial optimization
- Week 6: Performance analysis and improvements

### Backtesting Results
- Completed initial backtest of the trading strategy
- Results saved to results/backtest directory
- Overall performance metrics:
  - Total PnL: -$602,358
  - Sharpe Ratio: -1.28
  - Max Drawdown: $2,656,859
  - Win Rate: 56.8%
  - Average Daily PnL: -$181.71
  - Total Trades: 4,150
- Strategy breakdown:
  - 2s10s Spread:
    - Total PnL: $2,568,362
    - Sharpe Ratio: 8.15
    - Number of Trades: 2,084
  - 5s30s Spread:
    - Total PnL: -$3,170,720
    - Sharpe Ratio: -15.22
    - Number of Trades: 2,066

## 2025-04-30
- Ran initial backtest with updated SPREADS dictionary
- Results show:
  - 2s10s strategy performed well (Sharpe 8.15)
  - 5s30s strategy performed poorly (Sharpe -15.22)
  - Overall negative performance due to 5s30s drag
- Next steps:
  - Analyze 5s30s strategy parameters
  - Consider adjusting or removing 5s30s strategy
  - Further optimize 2s10s strategy

## 2025-04-30 (Update)
- Generated visualization plots for backtest results:
  - Equity curves showing performance of both strategies
  - Drawdown analysis for risk assessment
  - Monthly returns heatmap for seasonality analysis
  - Trade analysis plots for strategy evaluation
  - Correlation analysis between spreads
- Next steps:
  - Analyze visualizations for strategy insights
  - Identify potential improvements based on patterns
  - Review correlation between spreads for portfolio optimization

## 2025-04-30 (Signal Generation)
- Regenerated signals for both spreads:
  - 2s10s Spread:
    * Total signals: 3,315
    * Strong flattener bias (73.5% flattener signals)
    * Balanced steepener/neutral signals (~13% each)
  - 5s30s Spread:
    * Total signals: 3,315
    * More balanced distribution
    * Slight flattener bias (52.5% flattener signals)
    * Strong steepener presence (41.4%)
    * Few neutral signals (6.1%)
- Next steps:
  - Analyze signal bias in 2s10s spread
  - Investigate model agreement levels
  - Review signal confidence distributions

## 2025-04-30 (Backtest Results)
- Ran backtest with updated signals:
  - Overall Performance:
    * Total PnL: -$602,358.04
    * Sharpe Ratio: -1.28
    * Max Drawdown: $2.66M
    * Win Rate: 56.80%
    * Total Trades: 4,150
  - 2s10s Strategy:
    * Strong performance with $2.57M PnL
    * Excellent Sharpe ratio of 8.15
    * 2,084 trades executed
  - 5s30s Strategy:
    * Poor performance with -$3.17M PnL
    * Very low Sharpe ratio of -15.22
    * 2,066 trades executed
- Next steps:
  - Investigate 5s30s strategy issues
  - Consider reducing 5s30s exposure
  - Optimize position sizing
  - Review risk management parameters

## 2025-04-30 (Model Training Refactoring)
- Implemented DRY principles in model training code:
  - Created separate walk-forward validation functions for sklearn and PyTorch models
  - Fixed model factory return value issues
  - Improved data preparation interface consistency
  - Enhanced GPU memory management
  - Better error handling and logging
- Key improvements:
  - Clear separation of concerns between model types
  - Type-safe data handling
  - Proper model state management
  - Consistent metric calculation
  - Better memory efficiency
- Next steps:
  - Test refactored code with all model types
  - Add comprehensive unit tests
  - Document new API structure
  - Optimize batch processing

## 2025-04-30 (Model Training Improvements)
- Enhanced model evaluation and results handling:
  - Added ROC-AUC metric for classification tasks
  - Improved probability prediction handling
  - Standardized naming conventions
  - Better metrics organization
- Key improvements:
  - Proper probability handling for ROC-AUC
  - One-vs-rest ROC-AUC for ternary classification
  - Consistent naming across codebase
  - Enhanced results storage format
- Next steps:
  - Test ROC-AUC calculation
  - Validate probability predictions
  - Review saved results format
  - Update visualization code for new metrics

## 2025-04-30 (Code Polish)
- Enhanced model training code quality:
  - Made sklearn model factory stateless
  - Improved ROC-AUC calculation with built-in multi-class support
  - Removed unnecessary probability storage
  - Cleaned up unused imports
- Key improvements:
  - Better state management
  - More efficient memory usage
  - Cleaner code organization
  - Improved maintainability
- Next steps:
  - Consider refactoring hyperparameter tuning
  - Add comprehensive unit tests
  - Review memory usage patterns
  - Optimize data handling

## May 1, 2024
- ✅ Completed backtest visualization
  - Generated equity curves plot
  - Generated drawdown analysis plot
  - Generated monthly returns heatmap
  - Generated trade analysis plots
  - Generated correlation analysis plot