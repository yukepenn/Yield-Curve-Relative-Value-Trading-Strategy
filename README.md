# Yield Curve Relative Value Trading Strategy

This project implements a systematic trading strategy focused on exploiting relative value opportunities in the yield curve through spread trading, utilizing machine learning models and ensuring DV01-neutral positions.

## Project Structure

```
├── data/
│   ├── raw/         # Original FRED data
│   ├── processed/   # Cleaned datasets
│   └── external/    # Additional data sources
├── notebooks/       # Analysis notebooks
├── src/
│   ├── data_ingestion.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── backtest.py
│   ├── portfolio.py
│   ├── metrics.py
│   └── shap_analysis.py
├── results/
│   ├── model_pickles/
│   ├── plots/
│   ├── performance_reports/
│   └── logs/
└── tests/           # Unit tests
```

## Features

- Data collection from FRED
- Feature engineering with yield spreads and macro indicators
- Machine learning model development
- DV01-neutral position sizing
- Comprehensive backtesting framework
- Performance analytics and risk metrics
- SHAP-based model interpretability

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

## Usage

Detailed usage instructions will be added as the project develops.

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 