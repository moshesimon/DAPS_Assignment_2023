"""
This module provides functions for processing and analyzing tweets related to Apple Inc.

The module includes the following functions:
- preprocess_tweets(tweets): pre-processes a pandas dataframe of tweets,
removing unnecessary columns, cleaning text, and performing sentiment analysis.
- get_sentiment(tweet, tokenizer, model, config): predicts the sentiment of
a given tweet using a pre-trained sentiment analysis model.
- sentiment_analysis(tweets): performs sentiment analysis on a pandas dataframe of tweets.
- save_locally(data, directory): saves a pandas dataframe locally as a CSV file.

"""
import os
import re
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
)
from scipy.special import softmax
import numpy as np

MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def preprocess_tweets(tweets):
    # write Docstring
    """
    This function preprocesses the tweets and saves them to a csv file.

    Args:
        -Tweets: A pandas dataframe containing the tweets.

    Returns:
        -None
    """
    print("Processing tweets...")
    # remove unnecessary columns
    tweets = tweets[["date", "name", "tweet", "nlikes", "nreplies", "nretweets"]]
    # rename date to Date
    tweets = tweets.rename(columns={"date": "Date"})
    # convert date column to datetime with just date
    tweets["Date"] = pd.to_datetime(tweets["Date"])
    tweets["Date"] = tweets["Date"].apply(lambda x: x.date())
    # set date as index
    tweets = tweets.set_index("Date")
    # sort by date
    tweets = tweets.sort_index(ascending=True)
    # remove dates before 2017-04-01
    tweets = tweets.loc[pd.to_datetime("2017-04-01") :]
    # remove links from tweets
    tweets["tweet"] = tweets["tweet"].apply(lambda x: re.sub(r"http\S+", "", x))
    # remove speech marks from beginning and end of tweets
    tweets["tweet"] = tweets["tweet"].apply(lambda x: x.replace('"', ""))
    # Replace mentions, cashtags and hashtags related to Apple
    tweets["tweet"] = tweets["tweet"].apply(lambda x: x.replace("@Apple", "Apple"))
    tweets["tweet"] = tweets["tweet"].apply(lambda x: x.replace("$aapl", "Apple"))
    tweets["tweet"] = tweets["tweet"].apply(lambda x: x.replace("#Apple", "Apple"))
    # delete all the others
    tweets["tweet"] = tweets["tweet"].apply(lambda x: re.sub(r"@\w+", "", x))
    tweets["tweet"] = tweets["tweet"].apply(lambda x: re.sub(r"\$\w+", "", x))
    tweets["tweet"] = tweets["tweet"].apply(lambda x: re.sub(r"#\w+", "", x))
    # strip whitespace from end of tweets
    tweets["tweet"] = tweets["tweet"].apply(lambda x: x.rstrip())
    # strip whitespace from beginning of tweets
    tweets["tweet"] = tweets["tweet"].apply(lambda x: x.lstrip())
    # Reduce the whitespaces between two words to only one space.
    tweets["tweet"] = tweets["tweet"].apply(lambda x: re.sub(r"\s+", " ", x))
    # remove empty tweets
    tweets = tweets[tweets["tweet"] != ""]
    # join nlikes, nreplies, nretweets columns to a popularity column
    tweets["engagement"] = tweets["nlikes"] + tweets["nreplies"] + tweets["nretweets"]
    tweets = tweets.drop(["nlikes", "nreplies", "nretweets"], axis=1)
    # Perform sentiment analysis
    tweets = sentement_analysis(tweets)
    print(tweets.head())
    print(tweets.info())
    return tweets


def get_sentiment(tweet, tokenizer, model, config):
    """
    This function predicts the sentiment of a given tweet using a
    pre-trained sentiment analysis model.

    Args:
    - tweet: str, required
        The tweet text for which sentiment will be predicted.

    - tokenizer: function, required
        The function to preprocess the tweet text.

    - model: PyTorch model, required
        The pre-trained sentiment analysis model.

    - config: dict, required
        A dictionary that contains the mapping of label IDs to label names.

    Returns:
    - scale: float
        A scale between -1 and 1 representing the sentiment of the tweet,
        where -1 indicates negative sentiment, 1 indicates positive sentiment,
        and 0 indicates neutral sentiment.
    """
    # Preprocess the tweet text
    encoded_input = tokenizer(tweet, return_tensors="pt")
    # Predict sentiment using the pre-trained model
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    # Normalize the scores using softmax
    scores = softmax(scores)
    # Get the ranking of labels
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    score = {}
    # Get the confidence of each label
    for i in range(3):
        score[str(config.id2label[ranking[i]])] = float(scores[ranking[i]])

    positive_confidence = score["positive"]
    negative_confidence = score["negative"]
    neutral_confidence = score["neutral"]
    # Calculate the sentiment scale
    scale = (positive_confidence - negative_confidence) / (
        positive_confidence + negative_confidence + neutral_confidence
    )
    return scale


def sentement_analysis(tweets):
    """
    This function performs sentiment analysis on the tweets in the processed_twitter_news.csv file.
    It saves the processed tweets to the processed_twitter_news.csv file.

    Args:
    - None

    Returns:
    - None
    """

    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    print("Performing sentiment analysis...")
    print("This may take a while...")
    # perform sentiment analysis
    tweets["sentiment"] = tweets["tweet"].apply(
        lambda x: get_sentiment(x, tokenizer, model, config)
    )
    print("Sentiment analysis complete")
    # save processed tweets
    print("Saving processed tweets")
    print(tweets.head())
    print(tweets.info())
    return tweets


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
