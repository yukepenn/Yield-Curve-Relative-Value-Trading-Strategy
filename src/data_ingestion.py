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
    'RRPONTSYD': 'Reverse Repo Rate',
    'WALCL': 'Fed Balance Sheet',
    'TOTCI': 'Commercial and Industrial Loans',
    
    # Inflation Metrics
    'CPIAUCSL': 'CPI All Items',
    'CPILFESL': 'Core CPI',
    'PCEPI': 'PCE Price Index',
    'PCEPILFE': 'Core PCE Price Index',
    'T5YIE': '5Y Breakeven Inflation',
    'T10YIE': '10Y Breakeven Inflation',
    'T5YIFR': '5Y Forward Inflation',
    'DFII10': '10Y TIPS Rate',
    'PPIFIS': 'PPI Final Demand',
    'PPIFGS': 'PPI Finished Goods',
    
    # Credit Spreads
    'BAA10Y': 'Baa Corporate vs 10Y Treasury',
    'AAA10Y': 'Aaa Corporate vs 10Y Treasury',
    'BAMLH0A0HYM2': 'ICE BofA High Yield Spread',
    'BAMLC0A4CBBB': 'BBB Corporate Spread',
    'BAMLC0A1CAAAEY': 'AA Corporate Spread',
    'BAMLC0A2CAA': 'A Corporate Spread',
    'DRTSCIS': 'CMBS Spread',
    
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
    'DGORDER': 'Durable Goods Orders',
    'IPMAN': 'Manufacturing Production',
    'IPMANSICS': 'Manufacturing Capacity Utilization',
    'TCU': 'Total Capacity Utilization',
    'RETAILIMSA': 'Retail Inventories',
    'WHLSLRIMSA': 'Wholesale Inventories',
    'CP': 'Corporate Profits',
    
    # Money Supply and Bank Credit
    'M2SL': 'M2 Money Supply',
    'M1SL': 'M1 Money Supply',
    'MZM': 'MZM Money Supply',
    'BOGMBASE': 'Monetary Base',
    'TOTBKCR': 'Total Bank Credit',
    
    # Market Indicators
    'SP500': 'S&P 500',
    'VIXCLS': 'CBOE VIX',
    'DTWEXB': 'Dollar Index',
    'DCOILWTICO': 'WTI Crude Oil Price',
    'DEXUSEU': 'USD/EUR Exchange Rate',
    'DEXJPUS': 'JPY/USD Exchange Rate',
    'DEXCHUS': 'CNY/USD Exchange Rate',
    
    # Volatility and Risk Measures
    'GVZCLS': 'Gold VIX',
    'OVXCLS': 'Oil VIX',
    'VXTYN': 'Treasury VIX',
    'VXDCLS': 'DJIA Volatility Index',
    'VXNCLS': 'NASDAQ Volatility Index',
    'VXEEMCLS': 'Emerging Markets VIX',
    'VXEWZCLS': 'Brazil VIX',
    
    # Bond Market Indicators
    'T10Y3M': '10Y-3M Treasury Spread',
    'T10Y2Y': '10Y-2Y Treasury Spread',
    'T10YFF': '10Y-Fed Funds Spread',
    
    # Market Liquidity Measures
    'TEDRATE': 'TED Spread',
    
    # Business Cycle
    'USREC': 'NBER Recession Indicator',
    'USSLIND': 'Leading Index',
    'CSCICP03USM665S': 'Consumer Confidence',
    'MICH': 'Consumer Expectations',
    'RRSFS': 'Real Retail Sales',
    'IPB50001SQ': 'Business Equipment Production',
    'IPCONGD': 'Consumer Goods Production',
    'IPDMAT': 'Durable Materials Production',
    'IPNMAT': 'Nondurable Materials Production',
    'IPFINAL': 'Final Products Production'
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

def clean_and_align_data(df, is_trading_data=True):
    """
    Clean and align time series data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with datetime index
    is_trading_data : bool
        If True, keep only business days
        If False, keep all days
        
    Returns
    -------
    pd.DataFrame
        Cleaned and aligned DataFrame
    """
    if df.empty:
        return df
        
    # Convert index to datetime if not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort index
    df = df.sort_index()
    
    # Get the initial number of missing values
    initial_missing = df.isna().sum()
    if initial_missing.any():
        logger.info("Initial missing values:")
        for col in df.columns:
            missing_count = initial_missing[col]
            if missing_count > 0:
                logger.info(f"- {col}: {missing_count} missing values")
    
    # Handle missing values for each column separately to prevent forward bias
    for col in df.columns:
        series = df[col]
        
        # Find the first non-null value index
        first_valid_idx = series.first_valid_index()
        
        if first_valid_idx is not None:
            # Fill missing values before first valid value with 0
            mask_before_first = (series.index < first_valid_idx)
            df.loc[mask_before_first, col] = 0
            
            # Forward fill after first valid value until next valid value
            df[col] = df[col].ffill()
    
    # For trading data, only keep business days
    if is_trading_data:
        df = df[df.index.dayofweek < 5]
    
    # Log data quality statistics
    logger.info("Data quality statistics:")
    logger.info(f"- Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"- Number of observations: {len(df)}")
    for col in df.columns:
        unique_vals = df[col].nunique()
        logger.info(f"- {col}: {unique_vals} unique values")
        first_nonzero = df[df[col] != 0][col].first_valid_index()
        if first_nonzero:
            logger.info(f"- {col}: First non-zero value at {first_nonzero}")
    
    return df

def fetch_treasury_data(fred, start_date=None, end_date=None):
    """
    Fetch Treasury yield data from FRED.
    
    Parameters
    ----------
    fred : fredapi.Fred
        Initialized FRED API connection
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format. If None, defaults to 2010-01-01.
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
            start_date = datetime(2010, 1, 1)  # Default to January 1, 2010

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

        # Clean and align data
        treasury_data = clean_and_align_data(treasury_data, is_trading_data=True)

        # Round all values to 4 decimal places
        treasury_data = treasury_data.round(4)

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
        Start date in 'YYYY-MM-DD' format. If None, defaults to 2010-01-01.
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
            start_date = datetime(2010, 1, 1)  # Default to January 1, 2010

        # Initialize empty DataFrame
        macro_data = pd.DataFrame()
        problematic_series = []

        # Fetch data for each macro indicator
        for ticker, name in MACRO_TICKERS.items():
            try:
                logger.info(f"Fetching {name} data")
                series = fred.get_series(ticker, start_date, end_date)
                if series is not None and not series.empty:
                    macro_data[name] = series
                else:
                    problematic_series.append((ticker, name, "Empty series"))
            except Exception as e:
                problematic_series.append((ticker, name, str(e)))
                logger.warning(f"Failed to fetch {name} ({ticker}): {str(e)}")
                continue

        if problematic_series:
            logger.warning("The following series had issues:")
            for ticker, name, error in problematic_series:
                logger.warning(f"- {name} ({ticker}): {error}")

        if macro_data.empty:
            raise ValueError("No macro data was successfully fetched")

        # Clean and align data - most macro data is not daily trading data
        macro_data = clean_and_align_data(macro_data, is_trading_data=False)

        # Round all values to 4 decimal places
        macro_data = macro_data.round(4)

        logger.info("Successfully fetched macroeconomic data")
        return macro_data

    except Exception as e:
        logger.error(f"Failed to fetch macro data: {str(e)}")
        raise

def save_data(df, filename):
    """
    Save DataFrame to CSV file.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filename : str
        Target filename in data/raw directory
    """
    try:
        # Create data directory if it doesn't exist
        os.makedirs('data/raw', exist_ok=True)
        
        # Save to CSV
        filepath = os.path.join('data/raw', filename)
        df.to_csv(filepath)
        logger.info(f"Successfully saved data to {filepath}")
        
    except Exception as e:
        logger.error(f"Failed to save data to {filename}: {str(e)}")
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
        save_data(treasury_data, 'treasury_yields.csv')
        
        # Fetch macro data
        macro_data = fetch_macro_data(fred)
        save_data(macro_data, 'macro_indicators.csv')
        
        logger.info("Successfully updated all data")
        
    except Exception as e:
        logger.error(f"Failed to update all data: {str(e)}")
        raise

if __name__ == '__main__':
    # When run as a script, update all data
    update_all_data() 