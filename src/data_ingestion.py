"""
Data ingestion module for fetching Treasury yields and macroeconomic indicators from FRED.

This module provides functions to download and update data from the Federal Reserve
Economic Data (FRED) database, including Treasury yields and key economic indicators.
"""

import os
from datetime import datetime, timedelta
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('results/logs/data_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Treasury yield curve tickers (3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y)
TREASURY_TICKERS = {
    'DTB3': '3-Month',
    'DTB6': '6-Month',
    'DGS1': '1-Year',
    'DGS2': '2-Year',
    'DGS3': '3-Year',
    'DGS5': '5-Year',
    'DGS7': '7-Year',
    'DGS10': '10-Year',
    'DGS20': '20-Year',
    'DGS30': '30-Year'
}

# Macroeconomic indicators from FRED
MACRO_TICKERS = {
    # Interest Rates and Monetary Policy
    'FEDFUNDS': 'Federal Funds Rate',
    'DFEDTARU': 'Fed Funds Target Upper',
    'DFEDTARL': 'Fed Funds Target Lower',
    'IORB': 'Interest on Reserve Balances',
    
    # Inflation Metrics
    'CPIAUCSL': 'CPI All Items',
    'CPILFESL': 'Core CPI',
    'PCEPI': 'PCE Price Index',
    'PCEPILFE': 'Core PCE Price Index',
    'T5YIE': '5Y Breakeven Inflation',
    'T10YIE': '10Y Breakeven Inflation',
    
    # Credit Spreads
    'BAA10Y': 'Baa Corporate vs 10Y Treasury',
    'AAA10Y': 'Aaa Corporate vs 10Y Treasury',
    'BAMLH0A0HYM2': 'ICE BofA High Yield Spread',
    
    # Economic Indicators
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Nonfarm Payrolls',
    'ICSA': 'Initial Jobless Claims',
    'GDPC1': 'Real GDP',
    'INDPRO': 'Industrial Production',
    'UMCSENT': 'Consumer Sentiment',
    'HOUST': 'Housing Starts',
    'PERMIT': 'Building Permits',
    'RSAFS': 'Retail Sales',
    
    # Money Supply and Bank Credit
    'M2SL': 'M2 Money Supply',
    'TOTCI': 'Commercial and Industrial Loans',
    
    # Market Indicators
    'SP500': 'S&P 500',
    'VIXCLS': 'CBOE VIX',
    'DTWEXB': 'Dollar Index',
    'DCOILWTICO': 'WTI Crude Oil Price',
    'GOLDAMGBD228NLBM': 'Gold Price',
    
    # Business Cycle
    'USREC': 'NBER Recession Indicator',
    'USSLIND': 'Leading Index',
    'NAPMNMI': 'ISM Services PMI',
    'NAPM': 'ISM Manufacturing PMI'
}

def initialize_fred():
    """Initialize FRED API connection with API key."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv('FRED_API_KEY', 'df2de59d691115cec25d648d66e1f40c')
        
        # Initialize FRED
        fred = Fred(api_key=api_key)
        logger.info("Successfully initialized FRED API connection")
        return fred
    except Exception as e:
        logger.error(f"Failed to initialize FRED API: {str(e)}")
        raise

def fetch_treasury_data(fred, start_date=None, end_date=None):
    """
    Fetch Treasury yield data from FRED.

    Parameters
    ----------
    fred : fredapi.Fred
        Initialized FRED API connection
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, defaults to 10 years ago.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, defaults to today.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Treasury yield data
    """
    try:
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365*10)  # 10 years of data

        # Initialize empty DataFrame
        treasury_data = pd.DataFrame()

        # Fetch data for each Treasury tenor
        for ticker, name in TREASURY_TICKERS.items():
            logger.info(f"Fetching {name} Treasury yield data")
            series = fred.get_series(ticker, start_date, end_date)
            treasury_data[name] = series

        # Calculate key spreads
        if all(x in treasury_data.columns for x in ['2-Year', '10-Year']):
            treasury_data['2s10s Spread'] = treasury_data['10-Year'] - treasury_data['2-Year']
        
        if all(x in treasury_data.columns for x in ['5-Year', '30-Year']):
            treasury_data['5s30s Spread'] = treasury_data['30-Year'] - treasury_data['5-Year']
        
        if all(x in treasury_data.columns for x in ['3-Month', '10-Year']):
            treasury_data['3m10y Spread'] = treasury_data['10-Year'] - treasury_data['3-Month']
            
        if all(x in treasury_data.columns for x in ['2-Year', '5-Year']):
            treasury_data['2s5s Spread'] = treasury_data['5-Year'] - treasury_data['2-Year']
            
        if all(x in treasury_data.columns for x in ['10-Year', '30-Year']):
            treasury_data['10s30s Spread'] = treasury_data['30-Year'] - treasury_data['10-Year']

        logger.info("Successfully fetched Treasury yield data")
        return treasury_data

    except Exception as e:
        logger.error(f"Failed to fetch Treasury data: {str(e)}")
        raise

def fetch_macro_data(fred, start_date=None, end_date=None):
    """
    Fetch macroeconomic indicator data from FRED.

    Parameters
    ----------
    fred : fredapi.Fred
        Initialized FRED API connection
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, defaults to 10 years ago.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format. If None, defaults to today.

    Returns
    -------
    pd.DataFrame
        DataFrame containing macroeconomic indicator data
    """
    try:
        # Set default dates if not provided
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=365*10)  # 10 years of data

        # Initialize empty DataFrame
        macro_data = pd.DataFrame()

        # Fetch data for each macro indicator
        for ticker, name in MACRO_TICKERS.items():
            logger.info(f"Fetching {name} data")
            series = fred.get_series(ticker, start_date, end_date)
            macro_data[name] = series

        logger.info("Successfully fetched macroeconomic data")
        return macro_data

    except Exception as e:
        logger.error(f"Failed to fetch macro data: {str(e)}")
        raise

def save_data(data, filename, directory='data/raw'):
    """
    Save data to CSV file.

    Parameters
    ----------
    data : pd.DataFrame
        Data to save
    filename : str
        Name of the file (without .csv extension)
    directory : str, optional
        Directory to save the file in, by default 'data/raw'
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Construct full file path
        filepath = os.path.join(directory, f"{filename}.csv")
        
        # Save data
        data.to_csv(filepath)
        logger.info(f"Successfully saved data to {filepath}")

    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")
        raise

def update_all_data():
    """
    Update all data from FRED and save to files.
    """
    try:
        # Initialize FRED
        fred = initialize_fred()
        
        # Fetch Treasury data
        treasury_data = fetch_treasury_data(fred)
        save_data(treasury_data, 'treasury_yields')
        
        # Fetch macro data
        macro_data = fetch_macro_data(fred)
        save_data(macro_data, 'macro_indicators')
        
        logger.info("Successfully updated all data")
        
    except Exception as e:
        logger.error(f"Failed to update all data: {str(e)}")
        raise

if __name__ == '__main__':
    # When run as a script, update all data
    update_all_data() 