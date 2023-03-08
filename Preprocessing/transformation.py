"""This module provides functions for processing and analyzing stock
and tweet data related to Apple, S&P, and Nasdaq stocks. 

The main functionalities of the module include:
- Process Apple, S&P, and Nasdaq stock data by removing unnecessary columns,
calculating daily returns, and converting the datetime column to just the date
- Calculate adjusted sentiment and engagement scores based on the sentiment and
engagement columns of the tweets data, and weight sentiment by engagement
- Merge the processed stock data and the adjusted tweets data into one dataframe
- Find all the dates on which the stock market was closed between a start and end
date (inclusive)

Example usage:

# Process Apple stock data
apple_data = pd.read_csv('apple_stock_data.csv')
processed_apple_data = process_apple_stock(apple_data)

# Find closed stock market days
closed_days = find_closed_days("2017-04-01", "2022-05-31")

# Transform and combine all the data
transformed_data = transform_and_combine_all_data(processed_apple_data, processed_snp_data,
processed_nasdaq_data, tweets_data)

# Save locally
save_locally(transformed_data, 'C:/Users/MyUser/Documents/stock_data.csv')
""" 

import os
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

# pylint: disable=E1101


def transform_and_combine_all_data(apple, snp, nasdaq, tweets):
    """
    This function performs data transformation and combines all the relevant data into one dataframe

    args:
        apple: a dataframe with apple stock data
        snp: a dataframe with S&P stock data
        nasdaq: a dataframe with Nasdaq stock data
        tweets: a dataframe with tweets data

    return: a dataframe with all the data

    """
    closed_days = find_closed_days("2017-04-01", "2022-05-31")
    # reset the index of the DataFrame
    apple.reset_index(inplace=True)
    snp.reset_index(inplace=True)
    nasdaq.reset_index(inplace=True)
    tweets.reset_index(inplace=True)
    
    tweets = tweets[["Date", "engagement", "sentiment"]]
    # drop volume and daily return columns
    apple = apple[["Date", "Close"]]
    snp = snp[["Date", "Close"]]
    nasdaq = nasdaq[["Date", "Close"]]

    # normalised engagement column
    tweets.loc[:, "norm_engagement"] = tweets["engagement"] / tweets["engagement"].max()
    # normalised sentiment column
    tweets.loc[:, "norm_sentiment"] = tweets["sentiment"] / tweets["sentiment"].max()
    # weight sentiment by engagement
    tweets["weighted_sentiment"] = tweets["norm_sentiment"] * tweets["norm_engagement"]

    # group all columns with the same date
    tweets = tweets.groupby("Date").mean()

    previos_date = None
    weighted_sentiments = []
    sentiments = []
    engagements = []

    # create a new column to store the adjusted sentiment and fill it with NaN
    tweets["adjusted_sentiment"] = np.nan
    tweets["adjusted_engagement"] = np.nan
    tweets["adjusted_weighted_sentiment"] = np.nan

    # move date from index to column
    tweets.reset_index(inplace=True)
    # iterate over the rows of the dataframe
    for i, row in tweets.iterrows():
        # get the date of the current row
        date = row["Date"]
        # if the date is not in the list of closed days and the previous
        # date is in the list of closed days
        if previos_date in closed_days and date not in closed_days:
            # add attributes to lists
            engagements.append(row["norm_engagement"])
            sentiments.append(row["norm_sentiment"])
            weighted_sentiments.append(row["weighted_sentiment"])
            # calculate the average sentiment
            av_sentiment = sum(sentiments) / len(sentiments)
            av_engagement = sum(engagements) / len(engagements)
            av_weighted_sentiment = sum(weighted_sentiments) / len(weighted_sentiments)
            # empty the lists
            sentiments = []
            engagements = []
            weighted_sentiments = []
            # replace the sentiment with the average sentiment
            tweets.at[i, "adjusted_sentiment"] = av_sentiment
            tweets.at[i, "adjusted_engagement"] = av_engagement
            tweets.at[i, "adjusted_weighted_sentiment"] = av_weighted_sentiment
        # if the date is in the list of closed days
        elif date in closed_days:
            # add attributes to lists
            sentiments.append(row["norm_sentiment"])
            engagements.append(row["norm_engagement"])
            weighted_sentiments.append(row["weighted_sentiment"])
        # if the date is not in the list of closed days and the previous date is
        # not in the list of closed days
        else:
            # set the sentiment to the original sentiment
            tweets.at[i, "adjusted_sentiment"] = row["norm_sentiment"]
            tweets.at[i, "adjusted_engagement"] = row["norm_engagement"]
            tweets.at[i, "adjusted_weighted_sentiment"] = row["weighted_sentiment"]
        previos_date = date

    tweets.dropna(inplace=True)
    tweets = tweets.drop(
        [
            "norm_engagement",
            "sentiment",
            "norm_sentiment",
            "engagement",
            "weighted_sentiment",
        ],
        axis=1,
    )

    # calculate the cumulative sum of the sentiment and engagement
    tweets["adjusted_sentiment"] = tweets["adjusted_sentiment"].cumsum()
    tweets["adjusted_engagement"] = tweets["adjusted_engagement"].cumsum()
    tweets["adjusted_weighted_sentiment"] = tweets[
        "adjusted_weighted_sentiment"
    ].cumsum()

    tweets.rename(columns={"adjusted_sentiment": "sentiment"}, inplace=True)
    tweets.rename(columns={"adjusted_engagement": "engagement"}, inplace=True)
    tweets.rename(
        columns={"adjusted_weighted_sentiment": "weighted_sentiment"}, inplace=True
    )

    # merge apple and tweets dataframes
    merged = pd.merge(apple, tweets, on="Date")
    # merge snp and merged dataframes
    merged = pd.merge(snp, merged, on="Date", how="inner", suffixes=(" snp", ""))
    # merge nasdaq and merged dataframes
    merged = pd.merge(
        nasdaq, merged, on="Date", how="inner", suffixes=(" nasdaq", " apple")
    )

    print(merged.info())
    print(merged.describe())

    return merged


def find_closed_days(start_date, end_date):
    """
    Given a start and end date, returns a list of all the dates on which the stock market was closed
    between those two dates (inclusive). Assumes that the stock market is closed on weekends and on
    the following U.S. holidays: New Year's Day, Martin Luther King Jr. Day, Presidents' Day,
    Memorial Day, Independence Day, Labor Day, Thanksgiving Day, and Christmas Day.

    Args:
    start_date : str
        The start date in the format "YYYY-MM-DD"

    end_date : str
        The end date in the format "YYYY-MM-DD"

    Returns:
    list
        A list of all the dates on which the stock market was closed between the start and end dates
        (inclusive) in the format "YYYY-MM-DD"
    """
    # Create a DatetimeIndex with all calendar days between start and end dates
    calendar_days = pd.date_range(start_date, end_date, freq="D")

    # Create a DatetimeIndex with all U.S. federal holidays
    holidays = pd.to_datetime(USFederalHolidayCalendar().holidays(start_date, end_date))

    # Create a list of dates that fall on weekends (Saturday or Sunday)
    weekends = calendar_days[calendar_days.weekday.isin([5, 6])]

    # Combine the list of holidays and weekends to create a list of all
    # the days the stock market was closed
    closed_days = holidays.union(weekends)

    # Convert the list of closed days to a list of date strings in the format "YYYY-MM-DD"
    closed_days_str = [d.strftime("%Y-%m-%d") for d in closed_days]

    return closed_days_str


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
