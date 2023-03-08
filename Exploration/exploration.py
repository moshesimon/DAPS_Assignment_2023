"""
This module contains functions for exploring the relationship between Apple stock
data and sentiment data.

Functions:
- explore_relationship_between_apple_and_indexes: Plots the 30-day moving average
of daily returns for Apple, S&P 500, and Nasdaq indexes.
- explore_relationship_between_apple_and_sentiment: Plots the relationship between
daily returns of Apple and sentiment scores.
- explore_trend_and_seasonality_of_sentiment: Performs seasonal decomposition of the
sentiment data and plots the resulting trend, seasonal, and residual components.
- explore_trend_and_seasonality_of_apple_close: Explores the seasonality of the Apple
stock data and looks for any seasonal patterns, such as weekly or monthly cycles, that
may be present in the data.
- explore_correlation_between_sentiment_and_apple_close: Plots the correlation between
tweet data (sentiment, engagement, and weighted sentiment) and Apple close price in a
scatter plot. It also plots the heatmap of the correlation between the tweet data and
Apple close price.
- test_monthly_seasonality: Tests for monthly seasonality in the Apple stock price data.
- test_yearly_seasonality: Tests for yearly seasonality in the Apple stock price data.
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import YearLocator, DateFormatter
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from config import figures_dir


def explore_trend_and_seasonality_of_sentiment(data):
    """
    This function performs seasonal decomposition of the sentiment data
    and plots the resulting trend, seasonal, and residual components.
    Additionally, it plots the autocorrelation and partial autocorrelation
    functions of the seasonal component to identify significant seasonal cycles.

    Args:
        data (pandas.DataFrame): A DataFrame containing the sentiment data
        with date as the index and a 'weighted_sentiment' column.

    Returns:
        None.
    """

    sentiment_data = data["weighted_sentiment"]

    # Decompose the data
    decomposition = STL(sentiment_data, period=52).fit()
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot the trend and seasonal components
    _, ax = plt.subplots(figsize=[12, 7])
    ax.plot(data["Date"], trend, c="r")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trend")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title("Trend of Tweet Sentiment")
    plt.grid()
    # save plot
    plt.savefig(os.path.join(figures_dir, "Trend of Tweet Sentiment.png"))
    plt.close()

    # plot seasonal component
    _, ax = plt.subplots(figsize=[12, 7])
    ax.plot(data["Date"], seasonal, c="r", label="AAPL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Seasonality")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title("Seasonality of Tweet Sentiment")
    plt.grid()
    # save plot
    plt.savefig(os.path.join(figures_dir, "Seasonality of Tweet Sentiment.png"))
    plt.close()

    # plot residual component
    _, ax = plt.subplots(figsize=[12, 7])
    ax.plot(data["Date"], residual, c="r", label="AAPL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title("Residual of Tweet Sentiment")
    plt.grid()
    # save plot
    plt.savefig(os.path.join(figures_dir, "Residual of Tweet Sentiment.png"))
    plt.close()

    # Plot ACF and PACF to identify significant seasonal cycles
    plot_acf(seasonal)
    plt.title("Autocorrelation Function of Seasonal Component of Sentiment")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    # save plot
    plt.savefig(
        os.path.join(
            figures_dir,
            "Autocorrelation Function of Seasonal Component of Sentiment.png",
        )
    )

    plot_pacf(seasonal)
    plt.title("Partial Autocorrelation Function of Seasonal Component of Sentiment")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    # save plot
    plt.savefig(
        os.path.join(
            figures_dir,
            "Partial Autocorrelation Function of Seasonal Component of Sentiment.png",
        )
    )


def explore_trend_and_seasonality_of_apple_close(data):
    """
    This function will explore the seasonality of the stock data and look for
    any seasonal patterns, such as weekly or monthly cycles, that may be present in the data.

    Args:
        data (pandas.DataFrame): A DataFrame containing the Apple stock data
        with date as the index and a 'Close apple' column.

    Returns:
        None.
    """

    apple_data = data["Close apple"]

    # Decompose the data
    decomposition = STL(apple_data, period=52).fit()
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    # Plot the trend and seasonal components
    _, ax = plt.subplots(figsize=[12, 7])
    ax.plot(data["Date"], trend, c="r", label="AAPL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Trend")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title("Trend of Apple Close Price")
    plt.grid()
    # save plot
    plt.savefig(os.path.join(figures_dir, "Trend of Apple Close Price.png"))
    plt.close()

    # plot seasonal component
    _, ax = plt.subplots(figsize=[12, 7])
    ax.plot(data["Date"], seasonal, c="r", label="AAPL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Seasonality")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title("Seasonality of Apple Close Price")
    plt.grid()
    # save plot
    plt.savefig(os.path.join(figures_dir, "Seasonality of Apple Close Price.png"))
    plt.close()

    # plot residual component
    _, ax = plt.subplots(figsize=[12, 7])
    ax.plot(data["Date"], residual, c="r", label="AAPL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Residual")
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%Y"))
    plt.title("Residual of Apple Close Price")
    plt.grid()
    # save plot
    plt.savefig(os.path.join(figures_dir, "Residual of Apple Close Price.png"))
    plt.close()

    # Plot ACF and PACF to identify significant seasonal cycles
    plot_acf(seasonal)
    plt.title("Autocorrelation Function of Seasonal Component of Apple Close Price")
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    # save plot
    plt.savefig(
        os.path.join(
            figures_dir,
            "Autocorrelation Function of Seasonal Component of Apple Close Price.png",
        )
    )

    plot_pacf(seasonal)
    plt.title(
        "Partial Autocorrelation Function of Seasonal Component of Apple Close Price"
    )
    plt.xlabel("Lag")
    plt.ylabel("Correlation")
    # save plot
    plt.savefig(
        os.path.join(
            figures_dir,
            "Partial Autocorrelation Function of Seasonal Component of Apple Close Price.png",
        )
    )


def explore_correlation_between_sentiment_and_apple_close(data):
    """
    This function plots the correlation between tweet data (sentiment, engagement,
    and weighted sentiment) and Apple close price in a scatter plot. It also plots
    the heatmap of the correlation between the tweet data and Apple close price.

    Args:
    - data: a pandas DataFrame that contains tweet data and Apple stock data

    Returns:
    - None
    """

    # plot the correlation between sentiment and Apple close price
    plt.scatter(data["Close apple"], data["sentiment"])
    plt.title("Correlation between Sentiment and Apple Close Price")
    plt.xlabel("Apple Close Price")
    plt.ylabel("Sentiment")
    # save plot
    plt.savefig(
        os.path.join(
            figures_dir, "Correlation between Sentiment and Apple Close Price.png"
        )
    )
    plt.close()

    # plot the correlation between engagement and Apple close price
    plt.scatter(data["Close apple"], data["engagement"])
    plt.title("Correlation between Engagement and Apple Close Price")
    plt.xlabel("Apple Close Price")
    plt.ylabel("Engagement")
    # save plot
    plt.savefig(
        os.path.join(
            figures_dir, "Correlation between Engagement and Apple Close Price.png"
        )
    )
    plt.close()

    # plot the correlation between weighted sentiment and Apple close price
    plt.scatter(data["Close apple"], data["weighted_sentiment"])
    plt.title("Correlation between Weighted Sentiment and Apple Close Price")
    plt.xlabel("Apple Close Price")
    plt.ylabel("Weighted Sentiment")
    # save plot
    plt.savefig(
        os.path.join(
            figures_dir,
            "Correlation between Weighted Sentiment and Apple Close Price.png",
        )
    )
    plt.close()

    # plot heatmap of correlation between sentiment and Apple close price
    plt.figure(figsize=(10, 10))
    sns.heatmap(data.corr(), annot=True, fmt=".2f")
    plt.tight_layout()
    plt.title("Correlation between Tweet Data and Close Price Data")
    # save plot
    plt.savefig(
        os.path.join(
            figures_dir, "Correlation between Tweet Data and Close Price Data.png"
        )
    )
    plt.close()


def test_monthly_seasonality(data):
    """
    Test for monthly seasonality in the Apple stock price data.

    Args:
        data (pandas.DataFrame): A dataframe containing the Apple stock price data.

    Returns:
        None

    Prints:
        str: A message indicating whether there is evidence of monthly seasonality
        in the Apple close price data or not.
    """

    # Extract the month from the date column
    data["month"] = data["Date"].dt.month

    # rename column Close_apple
    data.rename(columns={"Close apple": "Close_apple"}, inplace=True)


    # Define the model formula
    formula = "Close_apple ~ 1 + C(month)"

    # Fit a linear regression model using the formula
    model = smf.ols(formula=formula, data=data).fit()

    

    # Calculate the ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Analyze the results
    if np.any(anova_table["PR(>F)"] < 0.05):
        print("There is evidence of monthly seasonality in the Apple close price data")
    else:
        print(
            "There is no evidence of monthly seasonality in the Apple close price data"
        )
    # rename column Close_apple
    data.rename(columns={"Close_apple": "Close apple"}, inplace=True)


# test if there is evidence of yearly seasonality in the Apple close price data
def test_yearly_seasonality(data):
    """
    Test for yearly seasonality in the Apple stock price data

    Args:
        data (pandas.DataFrame): A dataframe containing the Apple stock price data.

    Returns:
        None

    Prints:
        str: A message indicating whether there is evidence of yearly seasonality in the Apple close price data or not.
    """


    # Extract the year from the date column
    data["year"] = data["Date"].dt.year

    # rename column Close_apple
    data.rename(columns={"Close apple": "Close_apple"}, inplace=True)

    # Define the model formula
    formula = "Close_apple ~ 1 + C(year)"

    # Fit a linear regression model using the formula
    model = smf.ols(formula=formula, data=data).fit()

    # Calculate the ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Analyze the results
    if np.any(anova_table["PR(>F)"] < 0.05):
        print("There is evidence of yearly seasonality in the Apple close price data")
    else:
        print(
            "There is no evidence of yearly seasonality in the Apple close price data"
        )
    # rename column Close_apple
    data.rename(columns={"Close_apple": "Close apple"}, inplace=True)


# test if there is evidence of weekly seasonality in the Apple close price data
def test_weekly_seasonality(data):
    """
    Test for weekly seasonality in the Apple stock price data

    Args:
        data (pandas.DataFrame): A dataframe containing the Apple stock price data.

    Returns:
        None

    Prints:
        str: A message indicating whether there is evidence of weekly seasonality in the Apple close price data or not.
    """


    # Extract the weekday from the date column
    data["weekday"] = data["Date"].dt.weekday

    # rename column Close_apple
    data.rename(columns={"Close apple": "Close_apple"}, inplace=True)

    # Define the model formula
    formula = "Close_apple ~ 1 + C(weekday)"

    # Fit a linear regression model using the formula
    model = smf.ols(formula=formula, data=data).fit()

    # Calculate the ANOVA table
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Analyze the results
    if np.any(anova_table["PR(>F)"] < 0.05):
        print("There is evidence of weekly seasonality in the Apple close price data")
    else:
        print(
            "There is no evidence of weekly seasonality in the Apple close price data"
        )
    # rename column Close_apple
    data.rename(columns={"Close_apple": "Close apple"}, inplace=True)
