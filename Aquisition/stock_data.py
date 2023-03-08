"""This module contains functions for retrieving and saving historical
stock data from Yahoo Finance.

The module provides functions for retrieving historical stock data for
Apple Inc., the S&P 500 Index, and the NASDAQ-100 Index from Yahoo Finance.
It also includes a function for saving a Pandas DataFrame as a CSV file
to a specified directory.

Functions:
- get_apple_data: retrieves historical stock data for Apple Inc.
- get_snp_data: retrieves historical stock data for the S&P 500 Index.
- get_nasdaq_data: retrieves historical stock data for the NASDAQ-100 Index.
- save_locally: saves a Pandas DataFrame locally as a CSV file.
"""
import os
import yfinance as yf


def get_apple_data():
    """
    Retrieves historical stock data for Apple Inc. from Yahoo Finance.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the historical stock data for Apple Inc.

    """
    # create a Ticker object for Apple Inc.
    apple = yf.Ticker("AAPL")

    # get the historical stock data for Apple Inc.
    aapl_stock = apple.history(start="2017-04-01", end="2022-05-31", interval="1d")

    # reset the index of the DataFrame
    aapl_stock.reset_index(inplace=True)

    # return the DataFrame containing the stock data
    return aapl_stock


def get_snp_data():
    """
    Retrieves historical stock data for the S&P 500 Index from Yahoo Finance.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the historical stock data for the S&P 500 Index.

    """
    # create a Ticker object for the S&P 500 Index
    snp = yf.Ticker("SPY")

    # get the historical stock data for the S&P 500 Index
    snp_stock = snp.history(start="2017-04-01", end="2022-05-31", interval="1d")

    # reset the index of the DataFrame
    snp_stock.reset_index(inplace=True)

    # return the DataFrame containing the stock data
    return snp_stock


def get_nasdaq_data():
    """
    Retrieves historical stock data for the NASDAQ-100 Index from Yahoo Finance.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the historical stock data for the NASDAQ-100 Index.

    """
    # create a Ticker object for the NASDAQ-100 Index
    nasdaq = yf.Ticker("QQQ")

    # get the historical stock data for the NASDAQ-100 Index
    nasdaq_stock = nasdaq.history(start="2017-04-01", end="2022-05-31", interval="1d")

    # reset the index of the DataFrame
    nasdaq_stock.reset_index(inplace=True)

    # return the DataFrame containing the stock data
    return nasdaq_stock


def save_locally(data, directory):
    """
    Saves a Pandas DataFrame locally as a CSV file.

    Args:
        data: A Pandas DataFrame to be saved.
        dir: The directory where the CSV file should be saved.

    Returns:
        None

    Example Usage:
        stock_data = pd.read_csv('stock_data.csv')
        save_locally(stock_data, 'C:/Users/MyUser/Documents/stock_data.csv')
    """
    if not os.path.exists(directory):
        data.to_csv(directory)
        print("Data saved locally to: " + directory)
    else:
        print("Data already exists locally at: " + directory)
