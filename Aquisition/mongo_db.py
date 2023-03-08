"""This module contains utility functions for working with MongoDB databases.

The functions in this module allow the user to establish a connection to a MongoDB server,
read data from a collection in the database and store data in a collection in the database.

Functions:
- connect_server(): Establishes a connection to a MongoDB server and returns a MongoClient object.
- read(collection_name): Retrieves data from a MongoDB collection and returns
it as a Pandas DataFrame.
- store_data(data_df, collection_name): Stores data in a MongoDB collection.
- overwrite_data(data_df, collection_name): Overwrites the data in a MongoDB
collection with new data.

Example usage:

import pandas as pd
from mongodb_utils import connect_server, read, store_data, overwrite_data

# connect to the MongoDB server
client = connect_server()

# retrieve data from a collection
data = read("my_collection")

# store data in a collection
data_df = pd.read_csv("my_data.csv")
store_data(data_df, "my_collection")

# overwrite the data in a collection
new_data_df = pd.read_csv("new_data.csv")
overwrite_data(new_data_df, "my_collection")
"""

import pandas as pd
from pymongo import MongoClient

USERNAME = "new_user_1"
PASSWORD = "oBDt3d6FITgKqZcr"
MONGODB_SERVER_ADDRESS = f"mongodb+srv://{USERNAME}:{PASSWORD}@cluster0.cye1jke.mongodb.net/?retryWrites=true&w=majority"
DATABASE_NAME = "Datasets"


def connect_server():
    """Establishes a connection to a MongoDB server and returns a MongoClient object.

    Args:
        None

    Returns:
        MongoClient:
            A MongoClient object representing the connection to the MongoDB server.

    Raises:
        pymongo.errors.ConnectionFailure:
            If the connection to the MongoDB server cannot be established.
    """
    # Establish connection to MongoDB
    client = MongoClient(MONGODB_SERVER_ADDRESS)
    print("Connected to the server")
    return client


def read(collection_name):
    """
    Retrieves data from a MongoDB collection and returns it as a Pandas DataFrame.

    Args:
        collection_name: str
            The name of the collection to read from.

    Returns:
        pandas.DataFrame or None:
            A DataFrame containing the data from the specified collection, with
            the '_id' column removed.
            If the collection does not exist in the database, returns None.
    """
    # connect to the MongoDB server
    client = connect_server()

    # get a list of collection names in the database
    collections = client[DATABASE_NAME].list_collection_names()

    # check if the specified collection exists in the database
    if collection_name not in collections:
        print(f"{collection_name} does not exist in the database")
        return None

    # retrieve the data from the collection that matches the query
    data = client[DATABASE_NAME][collection_name].find()

    # convert the data to a DataFrame
    df_data = pd.DataFrame(list(data))

    # remove the '_id' column from the DataFrame
    if "_id" in df_data.columns:
        del df_data["_id"]

    # convert the 'Date' column to a datetime object
    if "Date" in df_data.columns:
        df_data["Date"] = pd.to_datetime(df_data["Date"])

    # print some information about the retrieved data
    print(f"Retrieved data from {collection_name}")
    print(df_data.head())

    # return the DataFrame
    return df_data


def store_data(data_df, collection_name):
    """
    Stores data in a MongoDB collection.

    Args:
        data_df: pandas.DataFrame
            The data to store in the collection, represented as a Pandas DataFrame.
        collection_name: str
            The name of the collection to store the data in.

    Returns:
        None

    """
    # connect to the MongoDB server
    client = connect_server()

    # get the database and collection objects
    database = client[DATABASE_NAME]
    collection = database[collection_name]

    # check if the collection is empty
    if collection.count_documents({}) <= 0:
        # convert the data to a dictionary
        data_dict = data_df.to_dict("records")
        # insert the data into the collection
        collection.insert_many(data_dict)
        print(f"Stored data in {collection_name}")
    else:
        print(f"{collection_name} already contains records. Not storing data.")


def overwrite_data(data_df, collection_name):
    """
    Overwrites the data in a MongoDB collection with new data.

    Args:
        data_df: pandas.DataFrame
            The new data to store in the collection, represented as a Pandas DataFrame.
        collection_name: str
            The name of the collection to overwrite the data in.

    Returns:
        None

    """
    # connect to the MongoDB server
    client = connect_server()

    # get the database and collection objects
    database = client[DATABASE_NAME]
    collection = database[collection_name]

    # convert the new data to a dictionary
    data_dict = data_df.to_dict("records")

    # delete the existing data in the collection
    collection.delete_many({})

    # insert the new data into the collection
    collection.insert_many(data_dict)

    # print some information about the overwritten data
    print(f"Overwrote data in {collection_name}")
