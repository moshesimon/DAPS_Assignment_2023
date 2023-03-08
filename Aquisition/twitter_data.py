"""This module provides functionality to download tweets related to a
given keyword for a list of usernames, and save the tweet data to a CSV file.
It also provides functionality to combine tweet data from multiple CSV files
into a single Pandas DataFrame, and save the DataFrame to a CSV file locally.

Functions:

get_user_tweets: downloads tweets for a list of usernames and saves the tweet
data to CSV files.
combine_user_tweets: combines individual user tweet data from separate CSV
files into a single Pandas DataFrame.
get_tweets: retrieves tweets that contain a specified keyword and concatenates
them into a single DataFrame.
save_locally: saves a Pandas DataFrame locally as a CSV file.
"""
import os
import twint
import pandas as pd
from config import dataset_dir, usernames


def get_user_tweets(
    keywords="",
    lang="en",
    since=None,
    until=None,
):
    """
    This function downloads tweets that contain a keyword for a list of given usernames.
    It saves the tweets for each username as a separate .csv file in the Datasets folder.
    If a .csv file for a username already exists, the function will not download tweets

    Parameters:
        usernames (list of str): list of Twitter usernames to scrape tweets from
        keywords (str, optional): keywords to search for in tweets (default: '')
        lang (str, optional): language of the tweets (default: 'en')
        since (str, optional): earliest date to search tweets from (default: None)
        until (str, optional): latest date to search tweets until (default: None)
        mention (str, optional): username to mention in the tweets (default: None)
        hashtag (str, optional): hashtag to include in the tweets (default: None)

    Returns:
        None
    """

    for username in usernames:
        tries = 0  # variable to keep track of the number of tries to scrape tweets for each user
        date = until  # variable to keep track of the date of the last tweet
        users_df = []  # variable to store tweets dataframes for each user

        # check if the csv file for the current user already exists
        if not os.path.exists(os.path.join(dataset_dir, username + ".csv")):
            # loop until the earliest date of tweets is after 2017-04-01
            while date > "2017-04-01":
                print("Downloading tweets for user: " + username)
                tries += 1  # increment the number of tries

                # configure twint to scrape tweets
                c = twint.Config()
                c.Search = keywords
                c.Username = username
                c.Lang = lang
                c.Since = since
                c.Until = date
                c.Limit = 10000  # set limit to scrape 10000 tweets
                c.Hide_output = True  # hide output
                c.Pandas = True  # return tweets in pandas dataframe
                twint.run.Search(c)  # scrape tweets
                dataframe = twint.storage.panda.Tweets_df

                # check if tweets were found
                if dataframe.shape[0] == 0:
                    print("No tweets found for user: " + username)
                else:
                    tries = 0  # reset the number of tries
                    
                    users_df.append(
                        dataframe
                    )  # append tweets dataframe to the list of dataframes for the current user
                    print("Tweets found for user: " + username+"----------------1")
                    users_df_to_csv = pd.concat(
                        users_df
                    )  # concatenate dataframes for the current user into one dataframe
                    print("Tweets found for user: " + username+"----------------2")
                    users_df_to_csv.to_csv(
                        os.path.join(dataset_dir, username + ".csv"), index=False
                    )  # save dataframe to csv
                    print("Tweets found for user: " + username+"----------------3")
                    date = dataframe.date.min().split(" ")[
                        0
                    ]  # update the date of the last tweet
                    print("Tweets found for user: " + username+"----------------4")
                    print("Downloaded tweets for user: " + username)
                    print("Number of tweets downloaded: " + str(len(dataframe)))
                    print("Date of last tweet: " + str(date))

                # break the loop if no tweets were found after 10 tries
                print("Number of tries: " + str(tries))
                if tries == 10:
                    print(f"No tweets found for {username} after 10 tries.")
                    break


def combine_user_tweets():
    """
    Combines individual user tweet data from separate CSV files into a single DataFrame.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the concatenated tweet data from all users.

    """
    dataframes = []
    for username in usernames:
        # read the tweet data for the current user from a CSV file
        df = pd.read_csv(os.path.join(dataset_dir, username + ".csv"))
        dataframes.append(df)

    # concatenate the dataframes for all users into one dataframe
    dataframes = pd.concat(dataframes)

    # return the concatenated dataframe
    return dataframes


def get_tweets():
    """
    Retrieves tweets that contain a specified keyword and concatenates them into a single DataFrame.

    Args:
        None

    Returns:
        pandas.DataFrame:
            A DataFrame containing the concatenated tweet data.

    """
    # retrieve tweet data for the specified keyword
    while True:
        try:
            get_user_tweets(keywords="Apple", since="2017-02-01", until="2022-05-31")
            break
        except:
            continue

    # combine the tweet data from all users into a single DataFrame
    combined_tweets = combine_user_tweets()

    # return the concatenated DataFrame
    return combined_tweets


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
