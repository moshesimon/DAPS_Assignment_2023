"""This module contains functions for processing and saving stock data.

The module provides functions for processing historical stock data for Apple Inc.,
the S&P 500 Index, and the NASDAQ-100 Index. The functions remove unnecessary columns,
calculate daily returns, and convert the datetime column to just the date. The module
also includes a function for saving a Pandas DataFrame as a CSV file to a specified directory.

Functions:
- process_apple_stock: processes historical stock data for Apple Inc.
- process_snp_stock: processes historical stock data for the S&P 500 Index.
- process_nasdaq_stock: processes historical stock data for the NASDAQ-100 Index.
- save_locally: saves a Pandas DataFrame locally as a CSV file.
"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from config import figures_dir


def process_apple_stock(apple):
    """
    Processes Apple stock data by removing unnecessary columns, calculating daily returns, and
    converting the datetime column to just the date. Also draws a heatmap correlation matrix
    between Apple stock features and saves the image.

    Args:
        apple: A Pandas DataFrame containing Apple stock data.

    Returns:
        A Pandas DataFrame containing the processed Apple stock data.

    Example Usage:
        apple_data = pd.read_csv('apple_stock_data.csv')
        processed_apple_data = process_apple_stock(apple_data)
    """
    print("Processing apple stock data...")
    # draw heatmap corrolation matrix between apple stock features and save image
    sns.heatmap(
        apple.corr(numeric_only=True),
        annot=True,
    )
    plt.savefig(os.path.join(figures_dir, "apple_corr_heatmap.png"))
    plt.close()
    # remove unnecessary columns
    apple = apple.drop(["Dividends", "Stock Splits", "Open", "High", "Low"], axis=1)
    # add daily return column
    apple["Daily Return"] = apple["Close"].pct_change()
    # multiply daily return by 100 to get percentage
    apple["Daily Return"] = apple["Daily Return"].apply(lambda x: x * 100)
    # fill NaN values with 0
    apple["Daily Return"] = apple["Daily Return"].fillna(0)
    # convert datetime column to just date
    apple["Date"] = apple["Date"].apply(lambda x: x.date())

    print(apple.head())
    print(apple.info())
    return apple


def process_snp_stock(snp):
    """
    Processes S&P stock data by removing unnecessary columns, calculating daily returns,
    and converting the datetime column to just the date.

    Args:
        snp: A Pandas DataFrame containing S&P stock data.

    Returns:
        A Pandas DataFrame containing the processed S&P stock data.

    Example Usage:
        snp_data = pd.read_csv('snp_stock_data.csv')
        processed_snp_data = process_snp_stock(snp_data)
    """
    print("Processing S&P stock data...")
    # remove unnecessary columns
    snp = snp.drop(
        ["Dividends", "Capital Gains", "Stock Splits", "Open", "High", "Low"], axis=1
    )
    # add daily return column
    snp["Daily Return"] = snp["Close"].pct_change()
    # multiply daily return by 100 to get percentage
    snp["Daily Return"] = snp["Daily Return"].apply(lambda x: x * 100)
    # fill NaN values with 0
    snp["Daily Return"] = snp["Daily Return"].fillna(0)
    # convert datetime column to just date
    snp["Date"] = snp["Date"].apply(lambda x: x.date())

    print(snp.head())
    print(snp.info())
    return snp


def process_nasdaq_stock(nasdaq):
    """
    Processes Nasdaq stock data by removing unnecessary columns, calculating daily
    returns, and converting the datetime column to just the date.

    Args:
        nasdaq: A Pandas DataFrame containing Nasdaq stock data.

    Returns:
        A Pandas DataFrame containing the processed Nasdaq stock data.

    Example Usage:
        nasdaq_data = pd.read_csv('nasdaq_stock_data.csv')
        processed_nasdaq_data = process_nasdaq_stock(nasdaq_data)
    """
    print("Processing Nasdaq stock data...")
    # remove unnecessary columns
    nasdaq = nasdaq.drop(
        ["Dividends", "Capital Gains", "Stock Splits", "Open", "High", "Low"], axis=1
    )
    # add daily return column
    nasdaq["Daily Return"] = nasdaq["Close"].pct_change()
    # multiply daily return by 100 to get percentage
    nasdaq["Daily Return"] = nasdaq["Daily Return"].apply(lambda x: x * 100)
    # fill NaN values with 0
    nasdaq["Daily Return"] = nasdaq["Daily Return"].fillna(0)
    # convert datetime column to just date
    nasdaq["Date"] = nasdaq["Date"].apply(lambda x: x.date())

    print(nasdaq.head())
    print(nasdaq.info())
    return nasdaq


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
