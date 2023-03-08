"""
Module for time series forecasting of Apple stock prices using Facebook Prophet.

This module contains functions to train Prophet models, generate forecasts,
plot predicted values, evaluate model performance, and create visualizations.
Specifically, it contains the following functions:

- `train_model_1`: trains a Prophet model using time series data from a
transformed dataset, and generates a forecast for future values.
- `train_model_2`: trains a Prophet model using time series data from a
transformed dataset, and generates a forecast for future values using additional regressors.
- `plot_predictions`: plots the predicted values for a time series, along
with actual values, using a line plot and shaded areas to indicate confidence intervals.
- `create_joint_plot`: creates a joint plot to visualize the relationship
between predicted and actual values.
- `evaluate_performance`: evaluates the performance of a time series forecast
by calculating several performance metrics, including mean squared error, mean
absolute error, root mean squared error, R-squared, MAPE, and correlation.

Example usage:

  import pandas as pd
  from apple_stock_forecasting import train_model_1, plot_predictions

  # Load and preprocess the data
  data = pd.read_csv('apple_stock.csv')
  transformed_data = preprocess_data(data)

  # Train the model and generate a forecast
  forecast = train_model_1(transformed_data)

  # Plot the predicted values
  plot_predictions(forecast, '2018-01-01', 1)
"""
import os
import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score
from config import figures_dir

def train_model_1(data):
    """
    Trains a Prophet model using time series data from a transformed dataset, and
    generates a forecast for future values.

    Args:
        - data: A Pandas DataFrame containing time series data with a 'Date' column
        and a 'Close apple' column.

    Returns:
        - A Pandas DataFrame containing the forecasted values for the time series,
        along with actual values.

    Example Usage:
        transformed_data = pd.read_csv(transformed_dataset_dir, index_col=0)
        forecast = train_model_1(transformed_data)

    """

    data.set_index("Date", inplace=True)
    # Create a new DataFrame with the closing price of Apple for the training data
    training_data = data.loc["2017-04-01":"2022-04-30", :]

    training_data = pd.DataFrame(
        {"ds": training_data.index, "y": training_data.loc[:,"Close apple"]}
    )

    # create model
    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    # train model
    model.fit(training_data)

    # future data
    future_data = pd.DataFrame({"ds": data.index})

    # predict
    forecast = model.predict(future_data)

    # add actual values
    forecast["actual"] = data.loc[:, "Close apple"].reset_index(drop=True)
    print(forecast.head())
    model.plot_components(forecast)
    plt.savefig(os.path.join(figures_dir, "model_1_components.png"))
    return forecast


def train_model_2(data):
    """
    Trains a Prophet model using time series data from a transformed dataset,
    and generates a forecast for future values using additional regressors.

    Args:
        data: A Pandas DataFrame containing time series data with a 'Date'
        column, a 'Close apple' column, and additional columns for the regressors:
        'weighted_sentiment', 'engagement', 'Close snp', and 'Close nasdaq'.

    Returns:
        A Pandas DataFrame containing the forecasted values for the time series,
        along with actual values.

    Example Usage:
        transformed_data = pd.read_csv(transformed_dataset_dir, index_col=0)
        forecast = train_model_2(transformed_data)
    """

    # Create a new DataFrame for the training data
    training_data = data.loc["2017-04-01":"2022-04-30", :]
    training_data = pd.DataFrame(
        {
            "ds": training_data.index,
            "y": training_data["Close apple"],
            "sentiment": training_data["weighted_sentiment"],
            "engagement": training_data["engagement"],
            "Close snp": training_data["Close snp"],
            "Close nasdaq": training_data["Close nasdaq"],
        }
    )

    # create model
    model = Prophet(
        seasonality_mode="multiplicative",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    # add regressors
    model.add_regressor("sentiment")
    model.add_regressor("engagement")
    model.add_regressor("Close snp")
    model.add_regressor("Close nasdaq")
    # train model
    model.fit(training_data)

    # future data
    future_data = pd.DataFrame(
        {
            "ds": data.index,
            "sentiment": data["weighted_sentiment"],
            "engagement": data["engagement"],
            "Close snp": data["Close snp"],
            "Close nasdaq": data["Close nasdaq"],
        }
    )

    # predict
    forecast = model.predict(future_data)

    # add actual values
    forecast["actual"] = data.loc[:, "Close apple"].reset_index(drop=True)
    print(forecast.head())

    model.plot_components(forecast)
    plt.savefig(os.path.join(figures_dir, "model_2_components.png"))
    return forecast


def plot_predictions(forecast, start_date, num):
    """
    Plots the predicted values for a time series, along with actual values, using a
    line plot and shaded areas to indicate confidence intervals.

    Args:
        forecast: A Pandas DataFrame containing the forecasted values for the time
        series, along with actual values.
        start_date: A string representing the start date for the training data in
        the format 'YYYY-MM-DD'.
        num: An integer representing the number of the prediction plot.

    Returns:
        A Matplotlib Figure object and Axes object.

    Example Usage:
        f, ax = plot_predictions(forecast, '2018-01-01', 1)
    """
    f, ax = plt.subplots(figsize=(14, 8))

    forecast.set_index("ds", inplace=True)

    train = forecast.loc[start_date:"2022-04-30", :]
    ax.plot(train.index, train.actual, color="green")
    ax.plot(train.index, train.yhat, color="steelblue", lw=0.5)
    ax.fill_between(
        train.index, train.yhat_lower, train.yhat_upper, color="steelblue", alpha=0.3
    )

    test = forecast.loc["2022-04-30":, :]
    ax.plot(test.index, test.actual, color="green")
    ax.plot(test.index, test.yhat, color="coral", lw=0.5)
    ax.fill_between(
        test.index, test.yhat_lower, test.yhat_upper, color="coral", alpha=0.3
    )
    ax.axvline(pd.to_datetime("2022-04-30"), color="k", ls="--", alpha=0.7)

    ax.grid(ls=":", lw=0.5)
    plt.savefig(os.path.join(figures_dir, f"predictions_{num}.png"))
    return f, ax


def create_joint_plot(forecast, num, x="yhat", y="actual", title=None):
    """
    Creates a joint plot to visualize the relationship between predicted and actual values.

    Args:
        forecast: A Pandas DataFrame containing the forecasted values for the time series,
        along with actual values.
        num: An integer representing the number of the joint plot.
        x: A string representing the column to use for the x-axis of the plot.
        y: A string representing the column to use for the y-axis of the plot.
        title: A string representing the title of the plot.

    Returns:
        None

    Example Usage:
        create_joint_plot(forecast, 1, x='yhat', y='actual', title='Predicted vs Actual')
    """
    g = sns.jointplot(x=x, y=y, data=forecast, kind="reg", color="b")
    g.fig.set_figwidth(8)
    g.fig.set_figheight(8)

    ax = g.fig.axes[1]
    if title is not None:
        ax.set_title(title, fontsize=16)

    ax = g.fig.axes[0]
    ax.text(
        5000,
        60000,
        "R = {:+4.2f}".format(forecast.loc[:, [y, x]].corr().iloc[0, 1]),
        fontsize=16,
    )
    ax.set_xlabel("Predictions", fontsize=15)
    ax.set_ylabel("Observations", fontsize=15)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.grid(ls=":")
    plt.savefig(os.path.join(figures_dir, f"joint_plot__{num}.png"))


def evaluate_performence(forecast):
    """
    Evaluates the performance of a time series forecast by calculating several performance
    metrics, including mean squared error, mean absolute error, root mean squared error,
    R-squared, MAPE, and correlation.

    Args:
        forecast: A Pandas DataFrame containing the forecasted values for the time series,
        along with actual values.

    Returns:
        None

    Example Usage:
        evaluate_performance(forecast)
    """

    y_test = forecast.loc["2022-04-30":, "actual"]
    y_pred = forecast.loc["2022-04-30":, "yhat"]

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)

    # Calculate the mean absolute error
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate the residuals
    residuals = y_test - y_pred

    # Calculate mean, median, and skewness of the residuals
    mean_residuals = np.mean(residuals)
    median_residuals = np.median(residuals)
    skew_residuals = pd.Series(residuals).skew()

    # Calculate the RMSE
    rmse = np.sqrt(mse)

    # Calculate the R-squared
    r2 = r2_score(y_test, y_pred)

    # MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    # calculate correlation
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    # Print the results
    print(f"Mean squared error: {mse}")
    print(f"Mean absolute error: {mae}")
    print(f"Mean of residuals: {mean_residuals}")
    print(f"Median of residuals: {median_residuals}")
    print(f"Skewness of residuals: {skew_residuals}")
    print(f"RMSE: {rmse}")
    print(f"R-squared: {r2}")
    print(f"MAPE: {mape}")
    print(f"Correlation: {corr}")
