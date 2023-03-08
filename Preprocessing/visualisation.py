"""A module for plotting various financial data visualizations using Matplotlib,
Pandas, and Seaborn.

The module contains several functions to plot different types of financial data visualizations,
including histograms, boxplots, and scatter plots, for Apple stock data and Apple tweet engagement
and sentiment data. It requires the following packages to be installed: Matplotlib,
Pandas, and Seaborn.

Functions:

plot_daily_returns: Plot the daily returns of the Apple, S&P and Nasdaq stock data.
plot_apple_daily_return_histogram: Plot the daily returns of the Apple stock data as a histogram.
plot_apple_daily_return_density: Plot the density of daily returns of the Apple stock data.
plot_snp_histogram: Plot a histogram of the S&P stock data.
plot_nasdaq_histogram: Plot a histogram of the Nasdaq stock data.
plot_apple_boxplot: Plot a boxplot of the Apple stock data.
plot_tweet_engament_scatter: Plot a scatter plot of the Apple tweet engagement data.
plot_tweet_engament_histogram: Plot a histogram of the Apple tweet engagement data.
plot_tweet_engament_boxplot: Plot a boxplot of the Apple tweet engagement data.
plot_apple_close_scatter: Plot a scatter plot of the Apple stock data.
plot_apple_close_histogram: Plot a histogram of the Apple stock data.
plot_apple_close_boxplot: Plot a boxplot of the Apple stock data.
plot_apple_volume_scatter: Plot a scatter plot of the Apple stock data.
plot_apple_volume_histogram: Plot a histogram of the Apple stock volume.
plot_apple_volume_boxplot: Plot a boxplot of the Apple stock volume.
plot_tweet_sentiment_scatter: Plot a scatter plot of the Apple tweet sentiment data.
plot_tweet_sentiment_histogram: Plot a histogram of the Apple tweet sentiment data.
plot_tweet_sentiment_boxplot: Plot a boxplot of the Apple tweet sentiment data.
plot_all: Plot all the available visualizations.
"""
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from config import figures_dir


def plot_daily_returns(apple, snp, nasdaq):
    """
    Plot the daily returns of the Apple, S&P and Nasdaq stock data.

    Parameters:
    apple (DataFrame): Apple stock data
    snp (DataFrame): S&P stock data
    nasdaq (DataFrame): Nasdaq stock data

    Returns:
    None
    """
    # plot stock data
    _, ax = plt.subplots(ncols=1, nrows=3, figsize=(15, 10), sharex=True)
    apple["Daily Return"].rolling(30).mean().plot(ax=ax[0], c="r")
    snp["Daily Return"].rolling(30).mean().plot(ax=ax[1], c="g")
    nasdaq["Daily Return"].rolling(30).mean().plot(ax=ax[2], c="b")
    ax[0].set_title("Apple Daily Returns")
    ax[1].set_title("S&P Daily Returns")
    ax[2].set_title("Nasdaq Daily Returns")
    ax[0].set_ylabel("Daily Return")
    ax[1].set_ylabel("Daily Return")
    ax[2].set_ylabel("Daily Return")
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    # save plot
    plt.savefig(os.path.join(figures_dir, "apple vs snp vs nasdaq Daily Return.png"))
    # close plot
    plt.close()


def plot_apple_daily_return_histogram(apple):
    """
    Plot the daily returns of the Apple stock data.

    Parameters:
    apple (DataFrame): Apple stock data

    Returns:
    None

    """
    apple["Daily Return"].hist(bins=100)
    # set title
    plt.title("Apple Daily Returns")
    # set x label
    plt.xlabel("Daily Return")
    # set y label
    plt.ylabel("Frequency")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Daily Return histogram.png"))
    # clear plot
    plt.close()


def plot_apple_daily_return_density(apple):
    """
    Plot the density of daily returns of the Apple stock data.

    Parameters:
    apple (DataFrame): Apple stock data

    Returns:
    None

    """

    apple["Daily Return"].plot(kind="kde")
    # set title
    plt.title("Apple Daily Returns")
    # set x label
    plt.xlabel("Daily Return")
    # set y label
    plt.ylabel("Frequency")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Daily Return density.png"))
    # close plot
    plt.close()


def plot_snp_histogram(snp):
    """
    Plot a histogram of the S&P stock data.

    Parameters:
    snp (DataFrame): S&P stock data

    Returns:
    None
    """
    snp["Daily Return"].hist(bins=100)
    # set title
    plt.title("S&P Daily Returns")
    # set x label
    plt.xlabel("Daily Return")
    # set y label
    plt.ylabel("Frequency")
    # save plot
    plt.savefig(os.path.join(figures_dir, "S&P Daily Return histogram.png"))
    # close plot
    plt.close()


def plot_nasdaq_histogram(nasdaq):
    """
    Plot a histogram of the Nasdaq stock data.

    Parameters:
    nasdaq (DataFrame): Nasdaq stock data

    Returns:
    None
    """

    nasdaq["Daily Return"].hist(bins=100)
    # set title
    plt.title("Nasdaq Daily Returns")
    # set x label
    plt.xlabel("Daily Return")
    # set y label
    plt.ylabel("Frequency")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Nasdaq Daily Return histogram.png"))
    # close plot
    plt.close()


def plot_apple_boxplot(apple):
    """
    Plot a boxplot of the Apple stock data.

    Parameters:
    apple (DataFrame): Apple stock data

    Returns:
    None
    """
    sns.boxplot(apple["Daily Return"])
    # set title
    plt.title("Apple Daily Returns")
    # set y label
    plt.ylabel("Daily Return")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Daily Return boxplot.png"))
    # close plot
    plt.close()


def plot_tweet_engament_scatter(tweets):
    """
    Plot a scatter plot of the Apple tweet engagement data.

    Parameters:
    tweets (DataFrame): Apple tweet engagement data

    Returns:
    None
    """
    # plot scatter
    tweets.index = pd.to_datetime(tweets.index)
    plt.figure(figsize=(15, 10))
    plt.scatter(tweets.index, tweets["engagement"])
    # set title
    plt.title("Apple Tweet Engagement Scatter")
    # set x label
    plt.xlabel("Date")
    # set y label
    plt.ylabel("Engagement")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Tweet Engagement Scatter.png"))
    # close plot
    plt.close()


def plot_tweet_engament_histogram(tweets):
    """
    Plot a histogram of the Apple tweet engagement data.

    Parameters:
    tweets (DataFrame): Apple tweet engagement data

    Returns:
    None
    """
    # plot histogram
    tweets["engagement"].hist(bins=100)
    # set title
    plt.title("Apple Tweet Engagement Histogram")
    # set x label
    plt.xlabel("Engagement")
    # set y label
    plt.ylabel("Frequency")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Tweet Engagement Histogram.png"))
    # close plot
    plt.close()


def plot_tweet_engament_boxplot(tweets):
    """
    Plot a boxplot of the Apple tweet engagement data.

    Parameters:
    tweets (DataFrame): Apple tweet engagement data

    Returns:
    None
    """
    # plot boxplot
    sns.boxplot(tweets["engagement"])
    # set title
    plt.title("Apple Tweet Engagement Boxplot")
    # set y label
    plt.ylabel("Engagement")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Tweet Engagement Boxplot.png"))
    # close plot
    plt.close()


def plot_apple_close_scatter(apple):
    """
    Plot a scatter plot of the Apple stock data.

    Parameters:
    apple (DataFrame): Apple stock data

    Returns:
    None
    """

    # plot scatter
    apple.index = pd.to_datetime(apple.index)
    plt.figure(figsize=(15, 10))
    plt.scatter(apple.index, apple["Close"])
    # set title
    plt.title("Apple Close Scatter")
    # set x label
    plt.xlabel("Date")
    # set y label
    plt.ylabel("Close")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Close Scatter.png"))
    # close plot
    plt.close()


def plot_apple_close_histogram(apple):
    """
    Plot a histogram of the Apple stock data.

    Parameters:
    apple (DataFrame): Apple stock data

    Returns:
    None
    """

    # plot histogram
    apple["Close"].hist(bins=100)
    # set title
    plt.title("Apple Close Histogram")
    # set x label
    plt.xlabel("Close")
    # set y label
    plt.ylabel("Frequency")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Close Histogram.png"))
    # close plot
    plt.close()


def plot_apple_close_boxplot(apple):
    """
    Plot a boxplot of the Apple stock data.

    Parameters:
    apple (DataFrame): Apple stock data

    Returns:
    None
    """
    # plot boxplot
    sns.boxplot(apple["Close"])
    # set title
    plt.title("Apple Close Boxplot")
    # set y label
    plt.ylabel("Close")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Close Boxplot.png"))
    # close plot
    plt.close()


def plot_apple_volume_scatter(apple):
    """
    Plot a scatter plot of the Apple stock data.

    Parameters:
    apple (DataFrame): Apple stock data

    Returns:
    None
    """
    # plot scatter
    apple.index = pd.to_datetime(apple.index)
    plt.figure(figsize=(15, 10))
    plt.scatter(apple.index, apple["Volume"])
    # set title
    plt.title("Apple Volume Scatter")
    # set x label
    plt.xlabel("Date")
    # set y label
    plt.ylabel("Volume")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Volume Scatter.png"))
    # close plot
    plt.close()


def plot_apple_volume_histogram(apple):
    """
    Plot a histogram of the Apple stock volume.

    Parameters:
    apple (DataFrame): Apple stock data

    Returns:
    None
    """

    # plot histogram
    apple["Volume"].hist(bins=100)
    # set title
    plt.title("Apple Volume Histogram")
    # set x label
    plt.xlabel("Volume")
    # set y label
    plt.ylabel("Frequency")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Volume Histogram.png"))
    # close plot
    plt.close()


def plot_apple_volume_boxplot(apple):
    """
    Plot a boxplot of the Apple stock volume.

    Parameters:
    apple (DataFrame): Apple stock data

    Returns:
    None
    """

    # plot boxplot
    sns.boxplot(apple["Volume"])
    # set title
    plt.title("Apple Volume Boxplot")
    # set y label
    plt.ylabel("Volume")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Volume Boxplot.png"))
    # close plot
    plt.close()


def plot_tweet_sentiment_scatter(tweets):
    """
    Plot a scatter plot of the Apple tweet sentiment data.

    Parameters:
    tweets (DataFrame): Apple tweet sentiment data

    Returns:
    None
    """
    # plot scatter
    tweets.index = pd.to_datetime(tweets.index)
    plt.figure(figsize=(15, 10))
    plt.scatter(tweets.index, tweets["sentiment"])
    # set title
    plt.title("Apple Tweet Sentiment Scatter")
    # set x label
    plt.xlabel("Date")
    # set y label
    plt.ylabel("Sentiment")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Tweet Sentiment Scatter.png"))
    # close plot
    plt.close()


def plot_tweet_sentiment_histogram(tweets):
    """
    Plot a histogram of the Apple tweet sentiment data.

    Parameters:
    tweets (DataFrame): Apple tweet sentiment data

    Returns:
    None
    """
    # plot histogram
    tweets["sentiment"].hist(bins=100)
    # set title
    plt.title("Apple Tweet Sentiment Histogram")
    # set x label
    plt.xlabel("Sentiment")
    # set y label
    plt.ylabel("Frequency")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Tweet Sentiment Histogram.png"))
    # close plot
    plt.close()


def plot_tweet_sentiment_boxplot(tweets):
    """
    Plot a boxplot of the Apple tweet sentiment data.

    Parameters:
    tweets (DataFrame): Apple tweet sentiment data

    Returns:
    None
    """
    # plot boxplot
    sns.boxplot(tweets["sentiment"])
    # set title
    plt.title("Apple Tweet Sentiment Boxplot")
    # set y label
    plt.ylabel("Sentiment")
    # save plot
    plt.savefig(os.path.join(figures_dir, "Apple Tweet Sentiment Boxplot.png"))
    # close plot
    plt.close()


def plot_all(apple, snp, nasdaq, tweets):
    """
    Plot all figures.

    Parameters:
    apple (DataFrame): Apple stock data
    snp (DataFrame): S&P 500 data
    nasdaq (DataFrame): NASDAQ data
    tweets (DataFrame): Apple tweet sentiment data

    Returns:
    None
    """
    # set index to Date
    apple = apple.set_index("Date")
    snp = snp.set_index("Date")
    nasdaq = nasdaq.set_index("Date")
    tweets = tweets.set_index("Date")
    
    plot_tweet_engament_scatter(tweets)
    plot_tweet_engament_histogram(tweets)
    plot_tweet_engament_boxplot(tweets)
    plot_apple_close_scatter(apple)
    plot_apple_close_histogram(apple)
    plot_apple_close_boxplot(apple)
    plot_apple_volume_scatter(apple)
    plot_apple_volume_histogram(apple)
    plot_apple_volume_boxplot(apple)
    plot_tweet_sentiment_scatter(tweets)
    plot_tweet_sentiment_histogram(tweets)
    plot_tweet_sentiment_boxplot(tweets)
    plot_apple_daily_return_histogram(apple)
    plot_daily_returns(apple, snp, nasdaq)
    plot_apple_daily_return_density(apple)
    plot_snp_histogram(snp)
    plot_nasdaq_histogram(nasdaq)
    plot_apple_boxplot(apple)
