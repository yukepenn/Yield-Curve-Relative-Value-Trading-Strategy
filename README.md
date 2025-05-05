# Yield Curve Relative Value Trading Strategy

A sophisticated systematic trading strategy focused on exploiting relative value opportunities in the yield curve through spread trading, utilizing machine learning models and ensuring DV01-neutral positions.

## Project Structure

```
├── data/
│   ├── raw/         # Original FRED data
│   ├── processed/   # Cleaned and processed data
│   └── external/    # Additional data sources
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py      # FRED API integration
│   ├── feature_engineering.py # Feature generation
│   ├── feature_analysis.py    # Feature analysis and selection
│   ├── model_training.py      # Model training framework
│   ├── backtest.py           # Backtesting system
│   ├── portfolio.py          # Portfolio management
│   ├── signal_generator.py   # Trading signal generation
│   ├── utils.py             # Utility functions
│   └── visualize_backtest.py # Performance visualization
├── models/                   # Saved model files
├── results/                  # Analysis results and logs
├── config.yaml              # Configuration settings
├── requirements.txt         # Project dependencies
├── CHANGES.md              # Change log
└── PROGRESS.md             # Project progress tracking
```

## Core Components

### 1. Data Processing Pipeline
- **Data Ingestion** (`data_ingestion.py`)
  - FRED API integration for treasury yields
  - Macro indicators data collection
  - Automated data cleaning and alignment
  - Missing value handling

- **Feature Engineering** (`feature_engineering.py`)
  - Calendar features (day of week, holidays)
  - Trend features (momentum, RSI, moving averages)
  - Yield curve features (PCA decomposition)
  - Carry and roll-down features
  - Macro indicators integration

- **Feature Analysis** (`feature_analysis.py`)
  - Feature importance analysis
  - Correlation analysis
  - Distribution analysis
  - Feature-target relationships

### 2. Model Training Framework (`model_training.py`)
- **Supported Models**
  - Traditional ML:
    - Random Forest
    - XGBoost
    - Ridge Regression
    - Lasso Regression
  - Deep Learning:
    - LSTM with configurable architecture
    - Feed-Forward MLP
  - Statistical:
    - ARIMA with automatic parameter selection

- **Training Features**
  - Walk-forward validation
  - Hyperparameter tuning
  - GPU support
  - Model persistence
  - Comprehensive logging
  - Early stopping
  - Cross-validation

### 3. Trading System
- **Signal Generation** (`signal_generator.py`)
  - Model prediction processing
  - Signal thresholding
  - Position sizing logic

- **Portfolio Management** (`portfolio.py`)
  - Position tracking
  - Risk management
  - Performance monitoring

- **Backtesting** (`backtest.py`)
  - Historical performance analysis
  - Transaction cost modeling
  - Risk metrics calculation

- **Visualization** (`visualize_backtest.py`)
  - Performance charts
  - Risk metrics visualization
  - Trade analysis plots

## Technical Implementation

### Feature Engineering
```python
# Example from feature_engineering.py
def _compute_trend_features(self, col: str) -> pd.DataFrame:
    features = pd.DataFrame(index=series.index)
    for lookback in LOOKBACK_PERIODS:
        # Level change
        features[f'{col}_change_{lookback}d'] = series - series.shift(lookback)
        # Percentage change
        features[f'{col}_pct_change_{lookback}d'] = series.pct_change(lookback)
        # Moving average
        features[f'{col}_ma_{lookback}d'] = series.rolling(lookback).mean()
        # Volatility
        features[f'{col}_vol_{lookback}d'] = series.rolling(lookback).std()
```

### Model Training
```python
# Example from model_training.py
class ModelTrainer:
    def __init__(self, spread: str, prediction_type: str, model_type: str):
        self.spread = spread
        self.prediction_type = prediction_type
        self.model_type = model_type
        self.hyperparameters = self.set_default_hyperparameters()

    def train(self):
        if self.model_type == 'lstm':
            return self.train_lstm()
        elif self.model_type == 'mlp':
            return self.train_mlp()
        elif self.model_type == 'arima':
            return self.train_arima()
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/yield-curve-trading.git
cd yield-curve-trading
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure settings:
- Copy `config.yaml.example` to `config.yaml`
- Update configuration parameters
- Set up FRED API key in environment variables

## Usage

1. Data Processing:
```bash
# Data ingestion
python src/data_ingestion.py

# Feature engineering
python src/feature_engineering.py

# Feature analysis
python src/feature_analysis.py
```

2. Model Training:
```python
from src.model_training import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    spread='2s10s',
    prediction_type='next_day',
    model_type='mlp'
)

# Train model
results = trainer.train()
```

3. Backtesting:
```python
from src.backtest import Backtest

backtest = Backtest(
    model_path='models/model.pkl',
    start_date='2020-01-01',
    end_date='2021-12-31'
)

results = backtest.run()
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

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Feat(component): add amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FRED API for data access
- Contributors and maintainers
- Open source community
