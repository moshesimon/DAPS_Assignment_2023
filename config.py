"""Config file for the project, containing paths to relevant
directories and files, as well as various parameters.

The following variables are defined in this module:

- base_dir: The base directory of the project.
- dataset_dir: The directory containing the datasets.
- figures_dir: The directory where figures will be saved.
- apple_stock_dir: The path to the apple stock dataset.
- snp_stock_dir: The path to the S&P 500 stock dataset.
- nasdaq_stock_dir: The path to the NASDAQ stock dataset.
- twitter_news_dir: The path to the raw Twitter news dataset.
- processed_twitter_news_dir: The path to the processed Twitter news dataset.
- processed_apple_stock_dir: The path to the processed apple stock dataset.
- processed_snp_stock_dir: The path to the processed S&P 500 stock dataset.
- processed_nasdaq_stock_dir: The path to the processed NASDAQ stock dataset.
- transformed_dataset_dir: The path to the transformed dataset.
- retrieve_from_mongo: A boolean flag indicating whether to retrieve data from MongoDB or not.
- usernames: A list of Twitter usernames to retrieve news from.
- database_name: The name of the MongoDB database.
"""

import os

base_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(base_dir, "Datasets")
figures_dir = os.path.join(base_dir, "Figures")
apple_stock_dir = os.path.join(dataset_dir, "apple_stock.csv")
snp_stock_dir = os.path.join(dataset_dir, "snp_stock.csv")
nasdaq_stock_dir = os.path.join(dataset_dir, "nasdaq_stock.csv")
twitter_news_dir = os.path.join(dataset_dir, "twitter_news.csv")
processed_twitter_news_dir = os.path.join(dataset_dir, "processed_twitter_news.csv")
processed_apple_stock_dir = os.path.join(dataset_dir, "processed_apple_stock.csv")
processed_snp_stock_dir = os.path.join(dataset_dir, "processed_snp_stock.csv")
processed_nasdaq_stock_dir = os.path.join(dataset_dir, "processed_nasdaq_stock.csv")
transformed_dataset_dir = os.path.join(dataset_dir, "transformed_dataset.csv")


usernames = [
    "BBCBusiness",
    "BBCNews",
    "BI_Europe",
    "Benzinga",
    "business",
    "businessgreen",
    "BloombergTV",
    "bankofamerica",
    "BusinessInsider",
    "CNBC",
    "CNBCnow",
    "CNNBusiness",
    "CreditSuisse",
    "DeutscheBank",
    "Entrepreneur",
    "FastCompany",
    "Fidelity",
    "FinancialTimes",
    "Forbes",
    "FortuneMagazine",
    "GoldmanSachs",
    "GreenBiz",
    "inc",
    "IntesaSanpaolo",
    "Investopedia",
    "InvestorPlace",
    "KKR_Co",
    "ManGroup",
    "MarketWatch",
    "money",
    "MorganStanley",
    "MorningstarInc",
    "NYTimesBusiness",
    "NewYorkFed",
    "Reuters",
    "Schroders",
    "SeekingAlpha",
    "SkyNews",
    "TechCrunch",
    "TheEconomist",
    "UBS",
    "WSJ",
    "YahooFinance",
    "ZeroHedge",
]
