# Yield Curve Relative Value Trading Strategy

This project implements a systematic trading strategy focused on exploiting relative value opportunities in the yield curve through spread trading, utilizing machine learning models and ensuring DV01-neutral positions.

## Project Structure

```
├── data/
│   ├── raw/         # Original FRED data
│   │   ├── treasury_yields.csv
│   │   └── macro_indicators.csv
│   ├── processed/   # Cleaned and processed data
│   │   ├── features.csv
│   │   ├── targets.csv
│   │   ├── train_features.csv
│   │   ├── train_targets.csv
│   │   ├── val_features.csv
│   │   ├── val_targets.csv
│   │   ├── test_features.csv
│   │   ├── test_targets.csv
│   │   └── feature_stats.txt
│   └── external/    # Additional data sources
├── notebooks/       # Analysis notebooks
│   └── 2_feature_analysis.ipynb
├── src/
│   ├── data_ingestion.py    # FRED API integration
│   └── feature_engineering.py # Feature generation
├── results/
│   ├── model_pickles/       # Saved models
│   ├── plots/              # Generated plots
│   ├── performance_reports/ # Backtest results
│   └── logs/               # Log files
└── tests/                  # Unit tests (to be implemented)
```

## Project Rules

### 1. Documentation Updates
- Update PROGRESS.md for all changes, findings, fixes, and bugs
- Update CHANGES.md for all changes (forward-only, no deletions)
- Include timestamps and summaries in all updates

### 2. Commit Messages
Format: "Type(component): description"
- Feat(component): add new component
- Fix(component): fix api error
- Docs(readme): update readme
- Refactor(utils): refactor utils
- Style(tailwind): add new tailwind class
- Test(unit): add unit test
- Chore(deps): update dependencies

### 3. Coding Standards
- Python 3.8+ and PEP 8 style guidelines
- Meaningful variable and function names
- Comprehensive docstrings
- Modular code structure
- Proper error handling
- Reproducibility with random seeds

### 4. Data Processing
- All numeric values rounded to 4 decimal places
- Proper handling of missing values
- Clear separation of raw and processed data
- Time-based data splitting

## Current Status

### Completed
1. Data Ingestion
   - FRED API integration
   - Treasury yields (3M to 30Y)
   - Macro indicators
   - Data cleaning and alignment

2. Feature Engineering
   - 166 features generated
   - 15 targets (regression and classification)
   - Calendar features
   - Trend features
   - Yield curve features
   - Carry features
   - Data splits (40% train, 20% val, 40% test)

### Next Steps
1. Feature Analysis
   - Feature importance
   - Correlation analysis
   - Distribution analysis
   - Feature-target relationships

2. Model Development
   - Model architecture design
   - Cross-validation implementation
   - Performance metrics

3. Backtesting System
   - Position sizing logic
   - Transaction cost modeling
   - Risk management rules

## Recent Updates

### Model Training Framework
- Implemented comprehensive model training framework supporting multiple model types:
  - Traditional ML: Random Forest, XGBoost, Ridge, Lasso
  - Deep Learning: LSTM, Feed-Forward MLP
  - Statistical: ARIMA
- Added unified training interface through `ModelTrainer` class
- Implemented walk-forward validation for all models
- Added proper error handling and logging
- Integrated GPU support for deep learning models

### Model Types and Features
- Feed-Forward MLP:
  - Flexible architecture with configurable hidden layers [512, 256, 128]
  - BatchNormalization and Dropout for regularization
  - Early stopping and model persistence
  - Support for regression and classification tasks
- LSTM Model:
  - Sequence modeling capabilities
  - Configurable hidden layers and sequence length
- ARIMA Model:
  - Statistical time series modeling
  - Automatic parameter selection

### Testing Framework
- Comprehensive testing suite in `test_all_models.py`:
  - Tests all combinations of spreads and prediction types
  - Supports all model architectures
  - Saves detailed results and metrics
- Individual model testing in `test_model_training.py`:
  - Quick testing of specific model configurations
  - Detailed feature importance analysis
  - Performance metrics logging

### Results and Analysis
- All results saved in structured format:
  - Model pickles in `results/model_pickles/`
  - Training results in `results/model_training/`
  - Detailed logs in `results/logs/`
- Performance metrics tracked:
  - MSE for regression tasks
  - Accuracy, F1-score for classification
  - Feature importance analysis
  - Training/validation loss curves

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Set up .env file with FRED API key

## Usage

1. Data Ingestion:
   ```bash
   python src/data_ingestion.py
   ```

2. Feature Engineering:
   ```bash
   python src/feature_engineering.py
   ```

3. Feature Analysis:
   Run notebooks/2_feature_analysis.ipynb

4. Model Training:
   ```python
   # Initialize trainer
   trainer = ModelTrainer(
       spread='2s10s',
       prediction_type='next_day',
       model_type='mlp'  # or 'lstm', 'rf', 'xgb', 'arima'
   )

   # Train model
   results = trainer.train()

   # Results include metrics and predictions
   print(f"MSE: {results['mse']}")
   print(f"Training Loss: {results['train_loss']}")
   print(f"Validation Loss: {results['val_loss']}")
   ```

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
