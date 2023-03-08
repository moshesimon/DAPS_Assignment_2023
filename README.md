
# Data Acquisition and Processing Systems (DAPS) (ELEC0136)


### Environment
Here are all the libraries you will need to run the code. 

Twint need to be installed with the special command bellow.

Twint: pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint

Prophet: pip3 install prophet

Mongo: pip3 install pymongo

Numpy: pip3 install numpy

Pandas: pip3 install pandas

Torch: pip3 install torch

YFinance: pip3 install yfinance

Statsmodels: pip install statsmodels

Transformers: pip install transformers

Scipy: pip install scipy

matplotlib: pip install matplotlib

seaborn: pip install matplotlib

sklearn: pip install scikit-learn

Another option is to create a pip env using the requirements.txt file.

There are 3 stages of development with the data.
1) raw data
   This is how the data is taken from its source without any processing 
2) processed data
   This is the data after it has gone through processing
3) transformed and merged
   This is after data transformation and all the data relevant to training is merged into one dataframe.

Each of these stages of data have been saved to mongo database. If you would prefer to collect the data from mongo instead of throught the conventional data acquisition, processing and transformation steps, you will need to set the following boolean values to true:

retrieve_raw_data_from_monogo = False

retrieve_processed_data_from_monogo = False

retrieve_transformed_data_from_monogo = False

You will find these values at the top of the main.py file.

If you want to collect the data as usual you may leave the values as False. 
Just a word of warning, the twitter data can take up to 30 minutes to scrape and an additional 1.5 hours to perform sentiment analysis.
If you want to skip both the scraping and sentiment analysis i would recommend setting the following:

retrieve_raw_data_from_monogo = True

retrieve_processed_data_from_monogo = True

retrieve_transformed_data_from_monogo = False

If you are happy to do the scraping but not the sentiment analysis then you can set the following:

retrieve_raw_data_from_monogo = False

retrieve_processed_data_from_monogo = True

retrieve_transformed_data_from_monogo = False

## Run the code

```
python main.py
```

That's all you need to do!

The code will automaticly take get the raw data and take it through all the sages, saving all the plots to the Figures folder and all the datasets to the Datasets folder.
