"""The project is aimed to analyze and predict the stock prices of Apple using data
from various sources like stocks, twitter, and NASDAQ. The workflow consists of
several stages like data acquisition, data preprocessing, data visualization, data
transformation, data exploration, and data inference.

The main function of the project will execute the complete workflow for the project
and generate predictions using time-series models.

This module contains the main function which executes the complete workflow and
uses functions defined in other modules like Aquisition, Preprocessing, Exploration,
and Inference.

Typical usage example:

    main()
"""

from Aquisition import stock_data, mongo_db, twitter_data
from Preprocessing import (
    tweet_processing,
    stock_processing,
    visualisation,
    transformation,
)
from Exploration import exploration
from Inference import inference
from config import *

retrieve_raw_data_from_monogo = True
retrieve_processed_data_from_monogo = True
retrieve_transformed_data_from_monogo = True


def main():
    """
    Execute the main workflow for the project.

    This function executes the main workflow for the project, which consists of
    several stages: data acquisition, data preprocessing, data visualization,
    data transformation, data exploration, and data inference.

    Data acquisition retrieves raw data from external sources, saves it locally,
    and preprocesses it. Data visualization creates visualizations of the processed
    data. Data transformation combines the processed data into a single dataset and
    saves it. Data exploration analyzes the dataset and tests hypotheses. Data inference
    trains models to predict future stock prices.

    The function can be configured to retrieve processed data from a MongoDB database
    instead of preprocessing it, and to retrieve transformed data from the same database
    instead of transforming it.

    Args:
        None
    
    Returns:
        None

    Raises:
        None

    """

    ###########################################################################
    # DATA ACQUISITION
    ###########################################################################

    print("Acquiring data...")

    if retrieve_raw_data_from_monogo:
        # Get data from MongoDB
        apple = mongo_db.read("apple_stock")
        snp = mongo_db.read("snp_stock")
        nasdaq = mongo_db.read("nasdaq_stock")
        tweets = mongo_db.read("twitter_news")
    else:
        # Acquire data
        apple = stock_data.get_apple_data()
        snp = stock_data.get_snp_data()
        nasdaq = stock_data.get_nasdaq_data()
        tweets = twitter_data.get_tweets()

        print("Saving data to MongoDB...")
        # Save data to MongoDB
        mongo_db.store_data(apple, "apple_stock")
        mongo_db.store_data(snp, "snp_stock")
        mongo_db.store_data(nasdaq, "nasdaq_stock")
        mongo_db.store_data(tweets, "twitter_news")

    print("Saving data locally...")
    # Save data locally
    stock_data.save_locally(apple, apple_stock_dir)
    stock_data.save_locally(snp, snp_stock_dir)
    stock_data.save_locally(nasdaq, nasdaq_stock_dir)
    twitter_data.save_locally(tweets, twitter_news_dir)

    print("Data acquisition complete")

    ###########################################################################
    # DATA PREPROCESSING
    ###########################################################################

    print("Preprocessing data...")

    if retrieve_processed_data_from_monogo:
        # Acquire data from MongoDB
        apple = mongo_db.read("processed_apple_stock")
        snp = mongo_db.read("processed_snp_stock")
        nasdaq = mongo_db.read("processed_nasdaq_stock")
        tweets = mongo_db.read("processed_twitter_news")
    else:
        # Preprocess data
        apple = stock_processing.process_apple_stock(apple)
        snp = stock_processing.process_snp_stock(snp)
        nasdaq = stock_processing.process_nasdaq_stock(nasdaq)
        tweets = tweet_processing.preprocess_tweets(tweets)

        print("Saving processed data to MongoDB...")
        # Save data to MongoDB
        mongo_db.store_data(apple, "processed_apple_stock")
        mongo_db.store_data(snp, "processed_snp_stock")
        mongo_db.store_data(nasdaq, "processed_nasdaq_stock")
        mongo_db.store_data(tweets, "processed_twitter_news")

    print("Saving processed data locally...")
    # Save data locally
    stock_processing.save_locally(apple, processed_apple_stock_dir)
    stock_processing.save_locally(snp, processed_snp_stock_dir)
    stock_processing.save_locally(nasdaq, processed_nasdaq_stock_dir)
    tweet_processing.save_locally(tweets, processed_twitter_news_dir)

    print("Data visualization...")
    visualisation.plot_all(apple, snp, nasdaq, tweets)

    print("Data Transformation...")
    if retrieve_transformed_data_from_monogo:
        # Acquire data from MongoDB
        merged_data = mongo_db.read("merged_data")
    else:
        # Transform data
        merged_data = transformation.transform_and_combine_all_data(
            apple, snp, nasdaq, tweets
        )

        print("Saving transformed data to MongoDB...")
        mongo_db.store_data(merged_data, "merged_data")

    print("Saving transformed data locally...")
    # Save transformed data locally
    transformation.save_locally(merged_data, transformed_dataset_dir)

    ###########################################################################
    # DATA EXPLORATION
    ###########################################################################
    print("Data exploration...")

    exploration.explore_trend_and_seasonality_of_sentiment(merged_data)
    exploration.explore_trend_and_seasonality_of_apple_close(merged_data)
    exploration.explore_correlation_between_sentiment_and_apple_close(merged_data)

    print("Hypothesis testing...")

    exploration.test_yearly_seasonality(merged_data)
    exploration.test_monthly_seasonality(merged_data)
    exploration.test_weekly_seasonality(merged_data)

    print("Data exploration complete")

    ###########################################################################
    # DATA INFERENCE
    ###########################################################################
    print("Data inference...")

    # Train model 1
    print("Training model 1...")
    forecast_1 = inference.train_model_1(merged_data)

    # Train model 2
    print("Training model 2...")
    forecast_2 = inference.train_model_2(merged_data)

    # Visualize model 1
    print("Visualizing model 1...")
    inference.plot_predictions(forecast_1, start_date="2021-04-01", num=1)
    inference.create_joint_plot(forecast_1, num=1)

    # Visualize model 2
    print("Visualizing model 2...")
    inference.plot_predictions(forecast_2, start_date="2021-04-01", num=2)
    inference.create_joint_plot(forecast_2, num=2)

    # Evaluate model 1
    print("Evaluating model 1...")
    inference.evaluate_performence(forecast_1)

    # Evaluate model 2
    print("Evaluating model 2...")
    inference.evaluate_performence(forecast_2)


if __name__ == "__main__":
    main()
