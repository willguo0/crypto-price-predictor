
import tensorflow as tf
import numpy as np
import csv
from functools import reduce

def get_data(data_file):
    """
    Read and parse the train line by line, then breaks up the line into price, market cap and total volume.

    :param data_file: Path to the training file.
    :return: list of price, market cap and total volume
    """
    # TODO: load and concatenate training data from training file.
    with open(data_file, 'r') as csvfile:
        prices = []
        market_cap = []
        volume = []
        for row in csv.reader(csvfile, delimiter=','):
            prices.append(row[1])
            market_cap.append(row[2])
            volume.append(row[3])
    prices = prices[1:]
    market_cap = market_cap[1:]
    volume = volume[1:]
    prices = list(map(float, prices))
    market_cap = list(map(float, market_cap))
    volume = list(map(float, volume))
    prices = np.asarray(prices)
    market_cap = np.asarray(market_cap)
    volume = np.asarray(volume)
    return prices, market_cap, volume

prices, market_cap, volume = get_data("./data/ada-usd-max.csv")
print(np.shape(prices))

    